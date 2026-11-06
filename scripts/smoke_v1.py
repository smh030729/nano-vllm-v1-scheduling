import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/xietaibo/models/Qwen3-0.6B')
    parser.add_argument('--mode', default='auto', choices=('auto', 'eager', 'full', 'piecewise'))
    parser.add_argument('--async-scheduling', action='store_true')
    parser.add_argument('--prefix-cache-lookahead', type=int, default=0)
    parser.add_argument('--uncached-first', action='store_true')
    parser.add_argument('--intrusive-free-queue', action='store_true')
    parser.add_argument('--fused-qk-norm-rope', action='store_true')
    parser.add_argument('--no-mixed-batching', action='store_true')
    parser.add_argument('--no-incremental-kv-allocation', action='store_true')
    args = parser.parse_args()
    engine = LLM(args.model, max_num_seqs=8, max_num_batched_tokens=128,
        max_model_len=4096, gpu_memory_utilization=.6, num_kvcache_blocks=32,
        cuda_graph_mode=args.mode, async_scheduling=args.async_scheduling,
        prefix_cache_lookahead=args.prefix_cache_lookahead, uncached_first=args.uncached_first,
        intrusive_free_queue=args.intrusive_free_queue, mixed_batching=not args.no_mixed_batching,
        fused_qk_norm_rope=args.fused_qk_norm_rope,
        incremental_kv_allocation=not args.no_incremental_kv_allocation,
        cudagraph_token_sizes=(64, 128))
    try:
        prompts = [[100 + i % 1000 for i in range(n)] for n in (33, 513)]
        outputs = engine.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
        assert [len(output['token_ids']) for output in outputs] == [8, 8]
        assert not engine.pending_count
        assert not engine.scheduler.block_manager.used_block_ids
        print(json.dumps(dict(mode=args.mode, asynchronous=args.async_scheduling,
            optimizations=dict(prefix_cache_lookahead=args.prefix_cache_lookahead,
                uncached_first=args.uncached_first, intrusive_free_queue=args.intrusive_free_queue,
                fused_qk_norm_rope=args.fused_qk_norm_rope,
                mixed_batching=not args.no_mixed_batching,
                incremental_kv_allocation=not args.no_incremental_kv_allocation),
            outputs=[output['token_ids'] for output in outputs],
            stats=engine.model_runner.runtime_stats,
            inflight=engine.max_observed_inflight,
            captures=engine.model_runner.piecewise.stats,
            capture_errors=engine.model_runner.piecewise.capture_errors), indent=2))
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
