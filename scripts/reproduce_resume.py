"""Paired local reproduction of the three resume optimizations (GPU required)."""
import argparse
import atexit
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

from benchmark_v1 import Probe, dump, install_fixed_cache, prompt, quantile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / 'nano-vllm-inference-optimization/src/nano-vllm-upstream'
CACHE = ROOT.parent.parent / 'nano-vllm-queue-aware-cache/src'
FUSION_CASES = [(1, 32, 128), (4, 32, 128), (16, 32, 128),
                (1, 1024, 32), (4, 1024, 32), (1, 3000, 1)]


def cache_trace(tokenizer, unique=False, waves=8):
    # Same family distribution, prompt lengths and seed as bench_queue_eviction.py.
    rng, families, trace = random.Random(20260919), {}, []
    for wave in range(waves):
        requests = []
        for j in range(8):
            index = wave * 8 + j
            family = index if unique else rng.choices(range(8), weights=[4, 3, 2, 1, 1, 1, 1, 1])[0]
            if family not in families:
                text = f'Document {family}: This technical document describes system {family}. '
                text += 'Memory allocation, request scheduling, cache management and performance analysis. '
                ids = tokenizer.encode(text, add_special_tokens=False)
                families[family] = (ids * (4096 // len(ids) + 1))[:4096]
            suffix = [1000 + ((index * 137 + i * 17) % 19000) for i in range(64)]
            requests.append(families[family] + suffix)
        trace.append(requests)
    return trace


def reset_scheduler(engine):
    from nanovllm.engine.scheduler import Scheduler
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids
    engine.scheduler = Scheduler(engine.model_runner.config)


def trace_trial(engine, trace, outputs, baseline, seed):
    import torch
    from nanovllm import SamplingParams
    reset_scheduler(engine)
    runner = engine.model_runner
    original_run, original_graphs = runner.run, runner.graphs
    probe = Probe(engine, baseline)
    torch.manual_seed(seed)
    before = probe.counters()
    probe.collecting = True
    torch.cuda.synchronize()
    start = time.perf_counter()
    logical = 0
    for wave in trace:
        for tokens in wave:
            arrived = time.perf_counter()
            engine.add_request(tokens, SamplingParams(temperature=.8, max_tokens=outputs, ignore_eos=True))
            seq = engine.scheduler.waiting[-1]
            probe.records[seq.seq_id] = dict(logical=logical, seq=seq, arrival=arrived,
                times=[], token_ids=[], finished=None, existing=False,
                expected_outputs=outputs, prompt_tokens=len(tokens))
            logical += 1
        probe.drain()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    probe.collecting = False
    assert not engine.scheduler.block_manager.used_block_ids
    assert all(len(r['times']) == outputs for r in probe.records.values())
    assert not getattr(engine, 'pending_count', 0)
    rows = list(probe.records.values())
    input_tokens = sum(len(p) for wave in trace for p in wave)
    prefill = sum(r['prefill_tokens'] for r in probe.schedule_records)
    result = dict(wall_ms=wall * 1000, output_tok_s=len(rows) * outputs / wall,
        output_tokens=len(rows) * outputs, prefill_tokens=prefill, input_tokens=input_tokens,
        cache_hit_fraction=1 - prefill / input_tokens,
        ttft_p95_ms=quantile([(r['times'][0] - r['arrival']) * 1000 for r in rows], .95),
        scheduler_ms=sum(r['cpu_seconds'] for r in probe.schedule_records) * 1000,
        kv_peak_blocks=probe.peak_blocks,
        runtime_stats={k: probe.counters()[k] - before[k] for k in before},
        trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
        requests=[dict(logical=r['logical'], tokens=r['token_ids'],
            output_times_ms=[(t - start) * 1000 for t in r['times']],
            arrival_ms=(r['arrival'] - start) * 1000) for r in rows])
    if baseline:
        runner.run, runner.graphs = original_run, original_graphs
    return result


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    import torch
    from nanovllm import LLM
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(123)
    baseline = args.arm in ('upstream', 'cache_standalone')
    cache = args.arm in ('cache_standalone', 'cache_on', 'cache_all')
    fused = args.arm in ('fusion_on', 'cache_all')
    is_cache = args.stage == 'cache'
    blocks, budget = (68, 8192) if is_cache else (64, 512)
    config = dict(max_num_seqs=8 if is_cache else 16, max_num_batched_tokens=budget,
        max_model_len=8192 if is_cache else 4096, gpu_memory_utilization=.6,
        enforce_eager=False)
    if baseline:
        install_fixed_cache(blocks)
        if cache:
            config.update(prefix_cache_lookahead=16, uncached_first=True)
    else:
        config.update(num_kvcache_blocks=blocks, cuda_graph_mode='auto', async_scheduling=True,
            prefix_cache_lookahead=16 if cache else 0, uncached_first=cache,
            intrusive_free_queue=cache, fused_qk_norm_rope=fused)
    engine = LLM(args.model, **config)
    atexit.unregister(engine.exit)
    result = dict(stage=args.stage, arm=args.arm, config=config, gpu=torch.cuda.get_device_name(),
                  source=args.source, cases=[], warmups=[], complete=False)
    try:
        if is_cache:
            cases = [('repeated_prefix', cache_trace(engine.tokenizer), 32)]
            if args.controls:
                cases.append(('unique_prefix', cache_trace(engine.tokenizer, unique=True), 32))
        else:
            cases = [(f'b{b}_p{p}_o{o}', [[prompt(p, 12340 + i) for i in range(b)]], o)
                     for b, p, o in FUSION_CASES]
        with torch.inference_mode():
            for name, trace, outputs in cases:
                warm = trace_trial(engine, trace, outputs, baseline, 20260919)
                result['warmups'].append(dict(name=name, **warm))
                for rep in range(args.reps):
                    row = trace_trial(engine, trace, outputs, baseline, 20260919 + rep)
                    result['cases'].append(dict(name=name, rep=rep, **row))
                    dump(args.output, result)
                    print(args.arm, name, rep, row['wall_ms'], row['output_tok_s'], flush=True)
        result['complete'] = True
        dump(args.output, result)
    finally:
        engine.exit()


def micro(args):
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from torch._inductor import config as inductor_config
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    from nanovllm.ops import load_fused_qk_norm_rope
    load_fused_qk_norm_rope()
    torch.manual_seed(42)
    torch._dynamo.config.recompile_limit = 64
    result = dict(complete=False, cases=[],
                  measurement='100 calls per graph, 100 replays per sample; 10 paired rounds')
    with torch.inference_mode():
        for tokens in (1, 4, 16, 32, 128, 512):
            qkv = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16)
            positions = torch.arange(tokens, device='cuda', dtype=torch.int64)
            qw, kw = [RMSNorm(128).to(device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            rope = RotaryEmbedding(128, 128, 4096, 1000000).cuda()

            def unfused():
                q, k, _ = qkv.split((2048, 1024, 1024), -1)
                return rope(positions, qw(q.view(tokens, 16, 128)), kw(k.view(tokens, 8, 128)))

            def fused():
                return torch.ops.nanovllm.fused_qk_norm_rope(qkv, positions, qw.weight, kw.weight,
                    rope.cos_sin_cache, 16, 8, 1e-6, inductor_config.emulate_precision_casts)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(20):
                    unfused(), fused()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            for a, b in zip(unfused(), fused()):
                torch.testing.assert_close(a, b, rtol=.02, atol=.032)
            graphs = {}
            for name, function in (('unfused', unfused), ('fused', fused)):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):
                        output = function()
                graphs[name] = (graph, output)
            timings = {name: [] for name in graphs}
            for pair in range(10):
                order = ('unfused', 'fused') if pair % 2 == 0 else ('fused', 'unfused')
                for name in order:
                    graph = graphs[name][0]
                    for _ in range(10):
                        graph.replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) * 1000 / 10000)
            medians = {k: statistics.median(v) for k, v in timings.items()}
            row = dict(tokens=tokens, rounds_us=timings, median_us=medians,
                       latency_reduction_pct=100 * (1 - medians['fused'] / medians['unfused']))
            result['cases'].append(row)
            dump(args.output, result)
            print(json.dumps(row), flush=True)
    result['complete'] = True
    dump(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cache', 'fusion', 'micro'), required=True)
    parser.add_argument('--arm', default='fusion_off')
    parser.add_argument('--source', default=str(ROOT / 'src'))
    parser.add_argument('--model', default='/home/smh/huggingface/Qwen3-0.6B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    micro(args) if args.stage == 'micro' else worker(args)


if __name__ == '__main__':
    main()
