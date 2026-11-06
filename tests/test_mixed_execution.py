import pickle
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.attention import Attention
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context


class ExecutionTests(unittest.TestCase):
    def engine_state(self, engine):
        engine.max_inflight_batches = 1
        engine._pending = deque()
        engine.last_token_events = []
        engine.max_observed_inflight = 0
        engine._reported_finished = set()

    def test_engine_runs_mixed_batch_once(self):
        Sequence.block_size = 4
        engine = LLMEngine.__new__(LLMEngine)
        self.engine_state(engine)
        engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=8,
            eos=99999, num_kvcache_blocks=32, kvcache_block_size=4))
        calls = []
        def run(method, scheduled):
            if method == 'forget':
                return
            self.assertEqual(method, 'submit')
            seqs = scheduled.items
            is_prefill = bool(scheduled.num_prefill_tokens)
            calls.append((is_prefill, [s.seq_id for s in seqs]))
            return SimpleNamespace(result=lambda: [123] * len(seqs))
        engine.model_runner = SimpleNamespace(call=run)
        a = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(a)
        engine.step()
        b = Sequence(list(range(20)), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(b)
        calls.clear()
        outputs, stats = engine.step()
        self.assertEqual(calls, [(True, [a.seq_id, b.seq_id])])
        self.assertEqual(outputs, [(a.seq_id, [123, 123])])
        self.assertEqual((stats.num_decode_tokens, stats.num_prefill_tokens), (1, 7))
        self.assertEqual(b.num_cached_tokens, 7)
        self.assertEqual(b.num_completion_tokens, 0)
        self.assertEqual(stats.decode_seconds, 0.)
        self.assertEqual(stats.prefill_seconds, 0.)
        self.assertGreater(stats.mixed_seconds, 0.)
        self.assertGreaterEqual(stats.elapsed_seconds, stats.mixed_seconds)

    def test_engine_keeps_decode_only_execution_path(self):
        Sequence.block_size = 4
        engine = LLMEngine.__new__(LLMEngine)
        self.engine_state(engine)
        engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=8,
            eos=99999, num_kvcache_blocks=16, kvcache_block_size=4))
        calls = []
        def run(method, scheduled):
            if method == 'forget':
                return
            seqs = scheduled.items
            is_prefill = bool(scheduled.num_prefill_tokens)
            calls.append(is_prefill)
            return SimpleNamespace(result=lambda: [123] * len(seqs))
        engine.model_runner = SimpleNamespace(call=run)
        seq = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(seq)
        _, prefill_stats = engine.step()
        outputs, decode_stats = engine.step()
        self.assertEqual(calls, [True, False])
        self.assertEqual(outputs, [(seq.seq_id, [123, 123])])
        self.assertGreater(prefill_stats.prefill_seconds, 0.)
        self.assertGreater(decode_stats.decode_seconds, 0.)
        self.assertEqual(prefill_stats.mixed_seconds, 0.)
        self.assertEqual(decode_stats.mixed_seconds, 0.)

    def test_worker_serialization_preserves_execution_kind(self):
        seq = Sequence([1, 2, 3])
        restored = pickle.loads(pickle.dumps(seq))
        self.assertTrue(restored.is_prefill)
        self.assertEqual(restored.token_ids, [1, 2, 3])
        seq.is_prefill = False
        seq.num_cached_tokens = 2
        seq.num_scheduled_tokens = 1
        restored = pickle.loads(pickle.dumps(seq))
        self.assertFalse(restored.is_prefill)
        self.assertEqual(restored.token_ids, [1, 2, 3])
        self.assertEqual(restored.last_token, 3)

    def test_mixed_outputs_align_with_final_and_partial_prefills(self):
        Sequence.block_size = 4
        engine = LLMEngine.__new__(LLMEngine)
        self.engine_state(engine)
        engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=6,
            eos=99999, num_kvcache_blocks=32, kvcache_block_size=4))
        calls = []
        def run(method, scheduled):
            if method == 'forget':
                return
            seqs = scheduled.items
            calls.append([s.seq_id for s in seqs])
            return SimpleNamespace(result=lambda: [200 + i for i in range(len(seqs))])
        engine.model_runner = SimpleNamespace(call=run)
        a = Sequence([1, 2], SamplingParams(max_tokens=4, ignore_eos=True))
        engine.scheduler.add(a)
        engine.step()
        b = Sequence([3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
        c = Sequence(list(range(10, 19)), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(b)
        engine.scheduler.add(c)
        calls.clear()
        outputs, stats = engine.step()
        self.assertEqual(calls, [[a.seq_id, b.seq_id, c.seq_id]])
        self.assertEqual(a.completion_token_ids, [200, 200])
        self.assertEqual(outputs, [(b.seq_id, [201])])
        self.assertEqual(c.completion_token_ids, [])
        self.assertEqual(c.num_cached_tokens, 3)
        self.assertEqual((stats.num_decode_tokens, stats.num_prefill_tokens), (1, 5))
        for _ in range(10):
            if engine.is_finished():
                break
            engine.step()
        self.assertTrue(engine.is_finished())
        self.assertEqual(c.num_completion_tokens, 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires GPU for real metadata tensors')
    def test_mixed_metadata_survives_worker_serialization(self):
        Sequence.block_size = 4
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        a, b, c = (Sequence(list(range(10, 15))), Sequence(list(range(20, 26))),
                   Sequence(list(range(30, 39))))
        for seq, cached, scheduled, blocks, prefill in (
            (a, 4, 1, [7, 2], False),
            (b, 5, 1, [5, 6], False),
            (c, 3, 4, [1, 8, 9], True),
        ):
            seq.num_cached_tokens = cached
            seq.num_scheduled_tokens = scheduled
            seq.block_table = blocks
            seq.is_prefill = prefill
        for seqs in ([a, b, c], pickle.loads(pickle.dumps([a, b, c]))):
            with self.subTest(worker=not seqs[0].token_ids):
                try:
                    ids, positions = runner.prepare_prefill(seqs)
                    context = get_context()
                    self.assertEqual(ids.tolist(), [14, 25, 33, 34, 35, 36])
                    self.assertEqual(positions.tolist(), [4, 5, 3, 4, 5, 6])
                    self.assertEqual(context.cu_seqlens_q.tolist(), [0, 1, 2, 6])
                    self.assertEqual(context.cu_seqlens_k.tolist(), [0, 5, 11, 18])
                    self.assertEqual((context.max_seqlen_q, context.max_seqlen_k), (4, 7))
                    self.assertEqual(context.slot_mapping.tolist(), [8, 25, 7, 32, 33, 34])
                    self.assertEqual(context.block_tables.tolist(), [[7, 2, -1], [5, 6, -1], [1, 8, 9]])
                finally:
                    reset_context()

    @unittest.skipUnless(torch.cuda.is_available(), 'requires GPU for FlashAttention')
    def test_mixed_attention_with_shared_prefix_matches_reference(self):
        Sequence.block_size = 256
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 256
        a = Sequence(list(range(257)))
        b = Sequence(list(range(259)))
        a.num_cached_tokens = b.num_cached_tokens = 256
        a.num_scheduled_tokens, b.num_scheduled_tokens = 1, 3
        a.is_prefill = False
        a.block_table, b.block_table = [0, 1], [0, 2]
        generator = torch.Generator(device='cuda').manual_seed(123)
        def randn(shape):
            return torch.randn(shape, device='cuda', dtype=torch.bfloat16, generator=generator)
        attention = Attention(4, 64, 64 ** -0.5, 2)
        attention.k_cache = randn((3, 256, 2, 64))
        attention.v_cache = randn((3, 256, 2, 64))
        prefix_k, prefix_v = attention.k_cache[0].clone(), attention.v_cache[0].clone()
        q, k, v = randn((4, 4, 64)), randn((4, 2, 64)), randn((4, 2, 64))
        try:
            runner.prepare_prefill([a, b])
            actual = attention(q, k, v).float()
            expected = []
            for start, end in ((0, 1), (1, 4)):
                keys = torch.cat((prefix_k, k[start:end])).float().repeat_interleave(2, dim=1)
                values = torch.cat((prefix_v, v[start:end])).float().repeat_interleave(2, dim=1)
                scores = torch.einsum('qhd,khd->hqk', q[start:end].float(), keys) * (64 ** -0.5)
                future = torch.arange(len(keys), device='cuda')[None, :] > (
                    256 + torch.arange(end-start, device='cuda')[:, None])
                scores.masked_fill_(future[None], float('-inf'))
                expected.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), values))
            torch.testing.assert_close(actual, torch.cat(expected), atol=0.01, rtol=0.01)
            self.assertTrue(torch.equal(attention.k_cache[0], prefix_k))
            self.assertTrue(torch.equal(attention.v_cache[0], prefix_v))
            torch.testing.assert_close(attention.k_cache[1, 0], k[0], atol=0, rtol=0)
            torch.testing.assert_close(attention.k_cache[2, :3], k[1:], atol=0, rtol=0)
        finally:
            reset_context()

    @unittest.skipUnless(torch.cuda.is_available(), 'requires GPU for real metadata tensors')
    def test_real_runner_chunk_and_decode_metadata(self):
        Sequence.block_size = 4
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        seq = Sequence(list(range(10, 19)))
        seq.block_table = [7, 2, 10]
        seq.num_cached_tokens = 3
        seq.num_scheduled_tokens = 4
        ids, positions = runner.prepare_prefill([seq])
        context = get_context()
        self.assertEqual(ids.tolist(), [13, 14, 15, 16])
        self.assertEqual(positions.tolist(), [3, 4, 5, 6])
        self.assertEqual(context.slot_mapping.tolist(), [31, 8, 9, 10])
        self.assertEqual(context.cu_seqlens_q.tolist(), [0, 4])
        self.assertEqual(context.cu_seqlens_k.tolist(), [0, 7])
        self.assertEqual(context.block_tables.tolist(), [[7, 2, 10]])
        reset_context()
        ids, positions = runner.prepare_decode([seq])
        context = get_context()
        self.assertEqual(ids.tolist(), [18])
        self.assertEqual(positions.tolist(), [8])
        self.assertEqual(context.slot_mapping.tolist(), [40])
        self.assertEqual(context.context_lens.tolist(), [9])
        reset_context()


if __name__ == '__main__':
    unittest.main()
