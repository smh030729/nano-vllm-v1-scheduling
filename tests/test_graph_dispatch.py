import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / 'src/nanovllm/engine/graph_runner.py'
spec = importlib.util.spec_from_file_location('graph_runner_under_test', MODULE_PATH)
graph_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_runner)
GraphDispatcher = graph_runner.GraphDispatcher
copy_padded_inputs = graph_runner.copy_padded_inputs
PiecewiseGraphRunner = graph_runner.PiecewiseGraphRunner


class GraphDispatchTests(unittest.TestCase):
    def dispatcher(self, mode='auto'):
        return GraphDispatcher(mode=mode, max_tokens=16, token_sizes=[8, 1, 4, 16, 8], allow_padding=True)

    def test_default_piecewise_requires_an_exact_captured_token_count(self):
        dispatch = GraphDispatcher('auto', 16, [4, 8, 16])
        self.assertEqual(dispatch.select(6, [1, 1, 4], True), 'eager')
        self.assertEqual(dispatch.select(8, [1, 1, 6], True), 'piecewise')
        self.assertEqual(dispatch.select(3, [1, 1, 1], True), 'full')
        self.assertEqual(dispatch.select(3, [1, 1, 1], False), 'eager')

    def test_auto_selects_full_for_uniform_single_token_queries(self):
        self.assertEqual(self.dispatcher().select(3, [1, 1, 1], True), 'full')

    def test_auto_selects_piecewise_for_mixed_query_lengths(self):
        self.assertEqual(self.dispatcher().select(6, [1, 1, 4], True), 'piecewise')

    def test_auto_uses_piecewise_when_full_graph_is_unavailable(self):
        self.assertEqual(self.dispatcher().select(3, [1, 1, 1], False), 'piecewise')

    def test_padding_chooses_smallest_available_bucket(self):
        dispatch = self.dispatcher()
        self.assertEqual(dispatch.token_sizes, (1, 4, 8, 16))
        self.assertEqual(dispatch.token_bucket(5), 8)
        self.assertEqual(dispatch.token_bucket(16), 16)
        self.assertIsNone(dispatch.token_bucket(17))

    def test_full_mode_falls_back_for_variable_queries(self):
        self.assertEqual(self.dispatcher('full').select(6, [1, 1, 4], True), 'eager')
        self.assertEqual(self.dispatcher('full').select(2, [1, 1], False), 'eager')

    def test_piecewise_mode_does_not_switch_to_full(self):
        self.assertEqual(self.dispatcher('piecewise').select(2, [1, 1], True), 'piecewise')

    def test_uncaptured_token_count_and_eager_mode_fall_back(self):
        self.assertEqual(self.dispatcher().select(17, [1, 16], True), 'eager')
        self.assertEqual(self.dispatcher('eager').select(2, [1, 1], True), 'eager')

    def test_invalid_query_metadata_is_rejected(self):
        for count, lengths in [(0, []), (3, [1, 1]), (2, [0, 2]), (2, [-1, 3])]:
            with self.subTest(count=count, lengths=lengths), self.assertRaises(ValueError):
                self.dispatcher().select(count, lengths, True)

    def test_invalid_modes_and_capture_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.dispatcher('pretend_graph')
        for sizes in [[0, 4], [-1, 4]]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                GraphDispatcher('auto', 16, sizes)

    def test_empty_capture_buckets_fall_back_without_inventing_a_bucket(self):
        dispatch = GraphDispatcher('auto', 32, {})
        self.assertEqual(dispatch.token_sizes, ())
        self.assertIsNone(dispatch.token_bucket(1))
        self.assertEqual(dispatch.select(3, [1, 2], False), 'eager')
        self.assertEqual(dispatch.select(2, [1, 1], False), 'eager')
        self.assertEqual(dispatch.select(2, [1, 1], True), 'full')


class GraphPaddingTests(unittest.TestCase):
    def test_reusing_a_larger_bucket_clears_stale_token_and_position_rows(self):
        ids = torch.tensor([90, 91, 92, 93], dtype=torch.int64)
        positions = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        count = copy_padded_inputs(torch.tensor([7, 8]), torch.tensor([5, 6]), ids, positions)
        self.assertEqual(count, 2)
        self.assertEqual(ids.tolist(), [7, 8, 0, 0])
        self.assertEqual(positions.tolist(), [5, 6, 0, 0])

    def test_exact_bucket_does_not_drop_last_input(self):
        ids, positions = torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
        copy_padded_inputs(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2, 3]), ids, positions)
        self.assertEqual(ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(positions.tolist(), [0, 1, 2, 3])

    def test_invalid_shapes_fail_before_mutating_buffers(self):
        for ids, positions in [(torch.arange(5), torch.arange(5)),
                               (torch.arange(2), torch.arange(3)),
                               (torch.ones(1, 2, dtype=torch.int64), torch.arange(2))]:
            dst_ids, dst_positions = torch.full((4,), 99), torch.full((4,), 88)
            with self.subTest(shape=tuple(ids.shape)), self.assertRaises(ValueError):
                copy_padded_inputs(ids, positions, dst_ids, dst_positions)
            self.assertEqual(dst_ids.tolist(), [99] * 4)
            self.assertEqual(dst_positions.tolist(), [88] * 4)


class TinyNorm(torch.nn.Module):
    def forward(self, x, residual=None):
        combined = x if residual is None else x + residual
        normalized = combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-5)
        return normalized if residual is None else (normalized, combined)


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_lens = []
        self.seen_sizes = []

    def forward(self, q, k, v):
        self.seen_sizes.append(q.shape[0])
        assert sum(self.query_lens) == q.shape[0]
        outputs, offset = [], 0
        for length in self.query_lens:
            end = offset + length
            scores = q[offset:end, 0] @ k[offset:end, 0].T / 2
            mask = torch.ones(length, length, dtype=torch.bool).triu(1)
            outputs.append((scores.masked_fill(mask, -torch.inf).softmax(-1) @ v[offset:end, 0]).unsqueeze(1))
            offset = end
        return torch.cat(outputs)


class TinySelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = self.num_kv_heads = 1
        self.head_dim = self.q_size = self.kv_size = 4
        self.qkv_bias = False
        self.qkv_proj, self.o_proj = torch.nn.Linear(4, 12), torch.nn.Linear(4, 4)
        self.q_norm, self.k_norm = TinyNorm(), TinyNorm()
        self.attn = TinyAttention()

    def rotary_emb(self, positions, q, k):
        position = positions[:, None, None] * 0.05
        return q + position, k - position

    def forward(self, positions, hidden):
        q, k, v = self.qkv_proj(hidden).split([4, 4, 4], -1)
        q, k, v = (tensor.reshape(-1, 1, 4) for tensor in (q, k, v))
        q, k = self.rotary_emb(positions, self.q_norm(q), self.k_norm(k))
        return self.o_proj(self.attn(q, k, v).flatten(1))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = TinyNorm(), TinyNorm()
        self.self_attn = TinySelfAttention()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4, 12), torch.nn.SiLU(), torch.nn.Linear(12, 4))

    def forward(self, positions, hidden, residual):
        if residual is None:
            hidden, residual = self.input_layernorm(hidden), hidden
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions, hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.layers = torch.nn.ModuleList([TinyLayer() for _ in range(3)])
        self.norm = TinyNorm()

    def forward(self, ids, positions):
        hidden, residual = self.embed_tokens(ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden, residual)[0]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()

    def forward(self, ids, positions):
        return self.model(ids, positions)


class EagerSegment:
    def __init__(self, function):
        self.function = function

    def replay(self):
        self.function()


class PiecewiseBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyModel()
        self.runner = PiecewiseGraphRunner(self.model, [8])
        state = self.runner._create_buffers(8)
        state.graphs = [EagerSegment(fn) for fn in self.runner._make_segments(state)]
        self.runner.buckets[8] = state

    def compare(self, ids, positions, query_lens):
        for layer in self.model.model.layers:
            layer.self_attn.attn.query_lens = query_lens
        with torch.inference_mode():
            expected = self.model(ids, positions)
            actual = self.runner.run(ids, positions)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for layer in self.model.model.layers:
            self.assertEqual(layer.self_attn.attn.seen_sizes[-1], ids.numel())

    def test_attention_boundaries_preserve_all_layer_residuals(self):
        self.compare(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([100, 0, 1, 2, 3]), [1, 4])
        self.assertEqual(self.runner.stats['segment_replays'], 4)
        self.assertEqual(self.runner.stats['forward_replays'], 1)

    def test_reused_bucket_reads_new_metadata_and_excludes_padding_from_attention(self):
        self.compare(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([0, 1, 0, 1, 2]), [2, 3])
        self.compare(torch.tensor([8, 9]), torch.tensor([7, 20]), [1, 1])
        self.assertEqual(self.runner.buckets[8].attention_out[2:].count_nonzero().item(), 0)
        self.assertEqual(self.runner.stats['forward_replays'], 2)

    def test_empty_piecewise_runner_does_not_touch_cuda_or_capture_fake_buckets(self):
        runner = PiecewiseGraphRunner(self.model, [])
        runner.capture()
        self.assertFalse(runner.supports(1))
        self.assertEqual(runner.buckets, {})
        self.assertEqual(runner.stats['captures'], 0)
        with self.assertRaises(ValueError):
            runner.run(torch.tensor([1]), torch.tensor([0]))


if __name__ == '__main__':
    unittest.main()
