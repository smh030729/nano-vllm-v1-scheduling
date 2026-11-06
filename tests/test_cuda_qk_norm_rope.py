import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(scope='module')
def fused_op():
    from nanovllm.ops import load_fused_qk_norm_rope
    load_fused_qk_norm_rope()
    return torch.ops.nanovllm.fused_qk_norm_rope.default


def inputs(tokens=33, dim=128, dtype=torch.bfloat16, padded=False):
    torch.manual_seed(42)
    width = 8 * dim  # four Q heads and two K/V heads
    storage = torch.randn(tokens, width + (16 if padded else 0), device='cuda', dtype=dtype)
    qkv = storage[:, :width]
    positions = torch.randint(0, 1024, (tokens,), device='cuda')
    weights = [torch.randn(dim, device='cuda', dtype=dtype) for _ in range(2)]
    phase = torch.randn(1024, 1, dim // 2, device='cuda')
    cache = torch.cat((phase.cos(), phase.sin()), -1)
    return qkv, positions, *weights, cache, 4, 2, 1e-6


def reference(qkv, positions, qw, kw, cache, qheads, kheads, eps):
    dim = qw.numel()
    q, k, _ = qkv.split([qheads * dim, kheads * dim, kheads * dim], -1)
    cos, sin = cache[positions].chunk(2, -1)
    outputs = []
    for x, heads, weight in ((q, qheads, qw), (k, kheads, kw)):
        x = x.reshape(-1, heads, dim)
        normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps))
        normalized = normalized.to(x.dtype) * weight
        a, b = normalized.float().chunk(2, -1)
        outputs.append(torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype))
    return tuple(outputs)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('dim', [2, 48, 64, 128, 256])
@pytest.mark.parametrize('tokens,padded', [(1, False), (33, True)])
def test_values_and_input_unchanged(fused_op, dtype, dim, tokens, padded):
    args = inputs(tokens, dim, dtype, padded)
    before = args[0].clone()
    actual, expected = fused_op(*args), reference(*args)
    tolerance = {torch.float16: (0.003, 0.004), torch.bfloat16: (0.02, 0.032),
                 torch.float32: (2e-6, 2e-6)}[dtype]
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=tolerance[0], atol=tolerance[1])
        assert result.is_contiguous()
    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


def test_dispatcher_schema_meta_and_compile(fused_op):
    assert torch._C._dispatch_has_kernel_for_dispatch_key('nanovllm::fused_qk_norm_rope', 'CUDA')
    torch.library.opcheck(fused_op, inputs(tokens=3), test_utils=(
        'test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_existing_compiled_layers(fused_op, dtype):
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    args = list(inputs(dtype=dtype))
    qnorm, knorm = [RMSNorm(128).to(device='cuda', dtype=dtype) for _ in range(2)]
    qnorm.weight.copy_(args[2])
    knorm.weight.copy_(args[3])
    rope = RotaryEmbedding(128, 128, 1024, 10000).cuda()
    args[4] = rope.cos_sin_cache
    q, k, _ = args[0].clone().split([512, 256, 256], -1)
    expected = rope(args[1], qnorm(q.reshape(-1, 4, 128)), knorm(k.reshape(-1, 2, 128)))
    tolerance = {torch.float16: (0.003, 0.004), torch.bfloat16: (0.02, 0.032),
                 torch.float32: (2e-6, 2e-6)}[dtype]
    from torch._inductor import config as inductor_config
    for actual, target in zip(fused_op(*args, inductor_config.emulate_precision_casts), expected):
        torch.testing.assert_close(actual, target, rtol=tolerance[0], atol=tolerance[1])


def test_graph_replay_reads_updated_inputs(fused_op):
    args = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_op(*args)
    for position in (0, 1023):
        args[0].normal_()
        args[1].fill_(position)
        graph.replay()
        for result, target in zip(captured, reference(*args)):
            torch.testing.assert_close(result, target, rtol=0.02, atol=0.032)


def test_empty_batch(fused_op):
    q, k = fused_op(*inputs(tokens=0))
    assert q.shape == (0, 4, 128) and k.shape == (0, 2, 128)


def test_invalid_packed_width_rejected(fused_op):
    args = list(inputs())
    args[0] = args[0][:, :-1]
    with pytest.raises(RuntimeError, match='packed QKV width'):
        fused_op(*args)
