"""通过生产 submit 路径检查 KV block 边界、前缀复用及在途生命周期。"""

import argparse
import atexit
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from benchmark_v1 import Probe, capture_metadata, forced_token, prompt


def execution_signature(items, row, mode, bucket):
    return (mode, bucket, tuple(i.num_scheduled_tokens for i in items),
            tuple(i.start_pos + i.num_scheduled_tokens for i in items), row)


def assert_outputs(records, expected):
    for seq_id, tokens in expected.items():
        record = records[seq_id]
        if record['token_ids'] != tokens or list(record['seq'].completion_token_ids) != tokens:
            raise AssertionError(f'accepted output order/count mismatch for request {seq_id}')


def compare_rows(left, right):
    import torch
    if left.keys() != right.keys():
        raise AssertionError('comparison rows do not have the same logical output keys')
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        av, bv = a['row'].float(), b['row'].float()
        finite = bool(torch.isfinite(av).all() and torch.isfinite(bv).all())
        matched = a['history'] == b['history']
        equal = bool(torch.equal(a['row'], b['row']))
        rows.append({'logical_output': list(key), 'matched_execution_history': matched,
                     'bitwise_required': matched, 'bitwise_equal': equal, 'finite': finite,
                     'relative_l2': float((av-bv).norm()/av.norm().clamp_min(1e-12)),
                     'max_abs': float((av-bv).abs().max())})
    return {'passed': all(row['finite'] and (not row['bitwise_required'] or row['bitwise_equal'])
                          for row in rows), 'rows': rows}


class BoundaryProbe(Probe):
    def __init__(self, engine, provenance='cold'):
        super().__init__(engine, False, 120)
        self.rows = {}
        self.histories = {}
        self.provenance = provenance
        self.eos_logical = None
        self.executed = []
        self.previous_counters = self.counters()
        engine.model_runner.logits_observer = self.observe
        engine.model_runner.token_override = self.override

    def add_tokens(self, tokens, logical, output, ignore_eos=True):
        from nanovllm import SamplingParams
        seq_id = self.engine.add_request(tokens, SamplingParams(temperature=.8, max_tokens=output,
                                                               ignore_eos=ignore_eos))
        seq = self.engine.scheduler.requests[seq_id]
        self.records[seq_id] = {'logical': logical, 'seq': seq, 'arrival': time.perf_counter(),
            'times': [], 'token_ids': [], 'finished': None, 'existing': False,
            'expected_outputs': output, 'prompt_tokens': len(tokens)}
        self.histories[seq_id] = [('cache_provenance', self.provenance)]
        return seq

    def observe(self, items, logits):
        runner = self.engine.model_runner
        now = self.counters()
        mode = 'full' if now['full_graph_replays'] > self.previous_counters['full_graph_replays'] else (
            'piecewise' if now['piecewise_graph_replays'] > self.previous_counters['piecewise_graph_replays'] else 'eager')
        self.previous_counters = now
        count = sum(i.num_scheduled_tokens for i in items)
        bucket = (min(n for n in runner.graphs if n >= len(items)) if mode == 'full' else
                  runner.piecewise.token_bucket(count) if mode == 'piecewise' else count)
        for row, (item, logits_row) in enumerate(zip(items, logits)):
            logical = self.records[item.seq_id]['logical']
            signature = execution_signature(items, row, mode, bucket)
            self.histories[item.seq_id].append(signature)
            self.executed.append({'logical': logical, 'generation': item.generation,
                                  'start': item.start_pos, 'count': item.num_scheduled_tokens,
                                  'prefill': item.is_prefill, 'produces_output': item.produces_output,
                                  'output_index': item.output_index, 'mode': mode})
            if item.produces_output:
                key = (logical, item.output_index)
                if key in self.rows:
                    raise AssertionError(f'duplicate submitted logical output {key}')
                self.rows[key] = {'row': logits_row.detach().clone(),
                                  'history': list(self.histories[item.seq_id])}

    def override(self, items, tokens):
        import torch
        values = []
        for item in items:
            logical = self.records[item.seq_id]['logical']
            value = forced_token(logical, max(0, item.output_index))
            if logical == self.eos_logical and item.produces_output and item.output_index == 0:
                value = self.engine.scheduler.eos
            values.append(value)
        return torch.tensor(values, dtype=tokens.dtype, device=tokens.device)

    def result(self):
        import torch
        torch.cuda.synchronize()
        self.rows = {key: {**value, 'row': value['row'].cpu()} for key, value in self.rows.items()}
        manager = self.engine.scheduler.block_manager
        if (self.engine.pending_count or self.engine.scheduler.num_inflight_batches
                or manager.used_block_ids or any(block.ref_count for block in manager.blocks)
                or len(manager.free_block_ids) != len(manager.blocks)):
            raise AssertionError('in-flight ticket or KV block leaked after drain')
        for record in self.records.values():
            seq = record['seq']
            if seq.num_in_flight_tokens or seq.num_output_placeholders or seq.block_table:
                raise AssertionError('request retained in-flight accounting after drain')
        if self.engine.model_runner._last_sampled:
            raise AssertionError('GPU sampled-token feedback cache leaked after drain')
        return {'outputs': {str(r['logical']): r['token_ids'] for r in self.records.values()},
                'executed': self.executed, 'max_inflight': self.engine.max_observed_inflight,
                'no_resource_leak': True}


def reset_scheduler(engine, depth=2, blocks=4):
    from nanovllm.engine.scheduler import Scheduler
    if not engine.is_finished():
        raise AssertionError('cannot reset scheduler before all work drains')
    config = engine.model_runner.config
    settings = {field.name: getattr(config, field.name) for field in fields(config)}
    settings.update(max_inflight_batches=depth, num_kvcache_blocks=blocks,
                    admission_full_sequence=False, long_prefill_token_threshold=0)
    engine.scheduler = Scheduler(SimpleNamespace(**settings))
    engine.max_inflight_batches = depth
    engine.max_observed_inflight = 0
    if hasattr(engine, '_reported_finished'):
        engine._reported_finished.clear()


def normal_case(engine, specifications, provenance='cold'):
    probe = BoundaryProbe(engine, provenance)
    expected = {}
    for tokens, logical, output in specifications:
        seq = probe.add_tokens(tokens, logical, output)
        expected[seq.seq_id] = [forced_token(logical, index) for index in range(output)]
    probe.drain()
    assert_outputs(probe.records, expected)
    return probe, probe.result()


def submit_without_retirement(engine):
    scheduled = engine.scheduler.schedule()
    if not scheduled.items:
        raise AssertionError('expected a schedulable in-flight batch')
    ticket = engine.model_runner.call('submit', scheduled)
    engine._pending.append((scheduled, ticket, time.perf_counter()))
    engine.max_observed_inflight = max(engine.max_observed_inflight, engine.pending_count)


def lifecycle_cases(engine):
    reset_scheduler(engine, blocks=2)
    probe, preempt = normal_case(engine, [(prompt(256, 20), 0, 3), (prompt(256, 21), 1, 3)])
    if not any(row['generation'] > 0 for row in probe.executed):
        raise AssertionError('small-cache workload did not trigger preemption')
    prefill = sum(row['count'] for row in probe.executed if row['prefill'])
    if prefill <= 512:
        raise AssertionError('preempted context was not recomputed')
    preempt['computed_prefill_tokens'] = prefill

    reset_scheduler(engine)
    probe = BoundaryProbe(engine)
    seq = probe.add_tokens(prompt(255, 30), 0, 4, ignore_eos=False)
    probe.eos_logical = 0
    probe.drain()
    assert_outputs(probe.records, {seq.seq_id: [engine.scheduler.eos]})
    eos = probe.result()
    if eos['max_inflight'] != 2:
        raise AssertionError('EOS case did not exercise two in-flight batches')

    reset_scheduler(engine)
    probe = BoundaryProbe(engine)
    seq = probe.add_tokens(prompt(255, 40), 0, 4)
    submit_without_retirement(engine)
    submit_without_retirement(engine)
    if engine.pending_count != 2 or not engine.abort_request(seq.seq_id):
        raise AssertionError('abort did not occur with two real submitted GPU batches')
    probe.flush()
    assert_outputs(probe.records, {seq.seq_id: []})
    abort = probe.result()
    return {'preemption': preempt, 'eos': eos, 'abort': abort}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/xietaibo/models/Qwen3-0.6B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-lifecycle', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    from nanovllm import LLM
    if not torch.cuda.is_available():
        raise RuntimeError('this script requires a real CUDA device')
    torch.manual_seed(123)
    torch._dynamo.config.recompile_limit = 64
    engine = LLM(args.model, max_num_seqs=2, max_num_batched_tokens=512,
                 max_model_len=1024, num_kvcache_blocks=4, gpu_memory_utilization=.6,
                 cuda_graph_mode='auto', cudagraph_token_sizes=(256, 512),
                 async_scheduling=True, max_inflight_batches=2)
    report = {'schema_version': 1, 'model': args.model, 'gpu': torch.cuda.get_device_name(),
              'dtype': str(engine.model_runner.kv_cache.dtype), 'complete': False,
              'note': 'Only matched full execution histories require bitwise logits; otherwise report relative L2.'}
    artifacts = {}
    try:
        with torch.inference_mode():
            specs = [(prompt(255, 10), 0, 4), (prompt(257, 11), 1, 3)]
            reset_scheduler(engine, depth=1)
            sync, report['sync_auto'] = normal_case(engine, specs)
            reset_scheduler(engine, depth=2)
            asynchronous, report['async_auto'] = normal_case(engine, specs)
            report['sync_vs_async'] = compare_rows(sync.rows, asynchronous.rows)
            artifacts.update(sync=sync.rows, asynchronous=asynchronous.rows)
            reset_scheduler(engine, depth=1)
            specs = [(prompt(257, 50), 0, 2)]
            cold, report['prefix_cold'] = normal_case(engine, specs)
            warm, report['prefix_warm'] = normal_case(engine, specs, 'same_confirmed_prefix')
            repeat, report['prefix_repeat'] = normal_case(engine, specs, 'same_confirmed_prefix')
            if warm.executed[0]['start'] != 256 or repeat.executed[0]['start'] != 256:
                raise AssertionError('expected exactly one cached full 256-token prefix block')
            report['cold_vs_prefix'] = compare_rows(cold.rows, warm.rows)
            report['prefix_matched_repeat'] = compare_rows(warm.rows, repeat.rows)
            artifacts.update(cold=cold.rows, warm=warm.rows, repeat=repeat.rows)
            if args.include_lifecycle:
                report['lifecycle'] = lifecycle_cases(engine)
            report['graph_capture'] = capture_metadata(engine.model_runner)
            report['passed'] = all(report[key]['passed'] for key in
                                   ('sync_vs_async', 'cold_vs_prefix', 'prefix_matched_repeat'))
            report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        torch.save(artifacts, args.output.with_suffix('.pt'))
        atexit.unregister(engine.exit)
        engine.exit()
    print(json.dumps({key: report[key] for key in ('complete', 'passed')}, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
