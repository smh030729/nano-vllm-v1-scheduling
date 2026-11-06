"""GPU correctness checks for the integrated switches; not a timing benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from nanovllm import LLM, SamplingParams


def run_case(model, options):
    torch.manual_seed(42)
    engine = LLM(model, max_num_seqs=4, max_num_batched_tokens=64,
                 max_model_len=768, gpu_memory_utilization=.6, num_kvcache_blocks=8,
                 cudagraph_token_sizes=(64,), **options)
    labels, cold_logits, first_positions = {}, {}, {}
    pressure_prefill_tokens = 0
    output_rows = 0

    def observe(items, logits):
        nonlocal pressure_prefill_tokens, output_rows
        if not torch.isfinite(logits).all().item():
            raise AssertionError('non-finite logits')
        for row, item in enumerate(items):
            label = labels[item.seq_id]
            first_positions.setdefault(label, item.start_pos)
            if label.startswith('pressure') and item.is_prefill:
                pressure_prefill_tokens += item.num_scheduled_tokens
            if item.produces_output:
                output_rows += 1
                if label == 'cold':
                    cold_logits[item.output_index] = logits[row].detach().float().cpu()

    def execute(requests, count):
        ids = []
        for label, prompt in requests:
            seq_id = engine.add_request(prompt, SamplingParams(max_tokens=count, ignore_eos=True))
            labels[seq_id] = label
            ids.append(seq_id)
        completed = {}
        while not engine.is_finished():
            outputs, _ = engine.step()
            completed.update(outputs)
        assert set(completed) == set(ids)
        assert all(tokens == [1000] * count for tokens in completed.values())

    try:
        # Copying diagnostic logits to CPU synchronizes; this is intentionally not a benchmark.
        engine.model_runner.logits_observer = observe
        engine.model_runner.token_override = lambda items, sampled: torch.full_like(sampled, 1000)
        execute([('cold', [77] * 33)], 4)
        for family in (11, 22, 33, 44):
            execute([(f'prime_{family}', [family] * 257)], 1)
        execute([(f'pressure_{family}', [family] * 257) for family in (55, 11, 22, 33)], 4)
        assert output_rows == 24 and len(cold_logits) == 4
        assert not engine.pending_count and not engine.scheduler.requests
        assert not engine.scheduler.block_manager.used_block_ids
        assert not engine.model_runner._last_sampled
        result = dict(options=options, finite_output_rows=output_rows,
                      pressure_prefill_tokens=pressure_prefill_tokens,
                      pressure_prefix_hits={k: v for k, v in first_positions.items()
                                            if k.startswith('pressure')},
                      max_inflight=engine.max_observed_inflight,
                      runtime_stats=dict(engine.model_runner.runtime_stats),
                      capture_errors=dict(engine.model_runner.piecewise.capture_errors),
                      all_resources_released=True)
        return result, cold_logits
    finally:
        engine.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--include-cuda-fusion', action='store_true')
    args = parser.parse_args()
    cache = dict(prefix_cache_lookahead=16, uncached_first=True, intrusive_free_queue=True)
    cases = {
        'switches_off': dict(mixed_batching=False, incremental_kv_allocation=False,
                             async_scheduling=False, cuda_graph_mode='eager'),
        'cache_sync_eager': dict(**cache, async_scheduling=False, cuda_graph_mode='eager'),
        'cache_async_auto': dict(**cache, async_scheduling=True, cuda_graph_mode='auto'),
        'cache_async_separate_full_reservation': dict(**cache, mixed_batching=False,
            incremental_kv_allocation=False, async_scheduling=True, cuda_graph_mode='auto'),
    }
    if args.include_cuda_fusion:
        cases.update(
            fused_sync_eager=dict(**cache, fused_qk_norm_rope=True,
                                  async_scheduling=False, cuda_graph_mode='eager'),
            fused_async_auto=dict(**cache, fused_qk_norm_rope=True,
                                  async_scheduling=True, cuda_graph_mode='auto'))
    if args.case:
        result, logits = run_case(args.model, cases[args.case])
        torch.save(logits, args.output.with_suffix('.pt'))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return
    results, reference = {}, None
    for name in cases:
        output = args.output.with_name(f'{args.output.stem}_{name}.json')
        with output.with_suffix('.log').open('w') as log:
            command = [sys.executable, str(Path(__file__).resolve()), '--model', args.model,
                       '--case', name, '--output', str(output)]
            if args.include_cuda_fusion:
                command.append('--include-cuda-fusion')
            subprocess.run(command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        logits = torch.load(output.with_suffix('.pt'), map_location='cpu', weights_only=True)
        if reference is None:
            reference = logits
        assert set(logits) == set(reference)
        fused = cases[name].get('fused_qk_norm_rope', False)
        # BF16 reductions can change final logits; keep strict equality for old cases.
        rtol, atol = (0.01, 0.125) if fused else (0, 0)
        for position in reference:
            torch.testing.assert_close(logits[position], reference[position], rtol=rtol, atol=atol)
        result['cold_logits_bitwise_equal'] = all(
            torch.equal(logits[p], reference[p]) for p in reference)
        result['cold_logits_max_abs_error'] = max(
            (logits[p] - reference[p]).abs().max().item() for p in reference)
        result['cold_logits_tolerance'] = dict(rtol=rtol, atol=atol)
        results[name] = result
        print(json.dumps({name: result}, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
