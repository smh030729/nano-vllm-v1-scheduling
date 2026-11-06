"""Single-process benchmark worker; run compare_v1.py for paired experiments."""

import argparse
import atexit
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
WORKLOADS = ('all_decode', 'all_prefill', 'long_decode', 'many_long', 'capacity_pressure')
MODES = ('baseline', 'v1_sync_eager', 'v1_sync_auto', 'v1_async_auto', 'v1_sync_piecewise')
RUNTIME_COUNTERS = ('forward_count', 'eager_count', 'full_graph_replays',
                    'piecewise_graph_replays', 'piecewise_segment_replays',
                    'gpu_feedback_tokens', 'mixed_forward_count')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def prompt(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(100, 10000) for _ in range(length)]


def forced_token(logical, output_index):
    return 700 + logical * 97 + output_index


def record_events(records, events, timestamp):
    for seq_id, token_id in events:
        records[seq_id]['times'].append(timestamp)
        records[seq_id]['token_ids'].append(int(token_id))


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def request_metrics(arrival, times, window_start, existing, finished):
    if not times or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('output events must be nonempty and chronological')
    if times[0] < arrival or finished < times[-1]:
        raise ValueError('arrival, outputs and completion must be chronological')
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:]) if b >= window_start]
    return {
        'existing': existing,
        'total_output_tokens': len(times),
        'window_output_tokens': sum(t >= window_start for t in times),
        'ttft_ms': None if existing else (times[0] - arrival) * 1000,
        'tpot_ms': (times[-1] - times[0]) * 1000 / (len(times) - 1) if len(times) > 1 else None,
        'average_gap_ms': mean_or_none(gaps),
        'max_token_gap_ms': max(gaps) if gaps else None,
        'gap_ms': gaps,
        'e2e_ms': (finished - arrival) * 1000,
        'output_times_from_window_ms': [(t - window_start) * 1000 for t in times],
    }


def counter_delta(before, after):
    result = {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
    if any(value < 0 for value in result.values()):
        raise ValueError('cumulative runtime counter decreased')
    return result


def capture_metadata(runner):
    piecewise = getattr(runner, 'piecewise', None)
    return {
        'configured_token_sizes': list(getattr(runner.config, 'cudagraph_token_sizes', ())),
        'full_buckets': sorted(getattr(runner, 'graphs', {})),
        'piecewise_buckets': sorted(getattr(piecewise, 'buckets', {})),
        'full_capture_errors': list(getattr(runner, 'full_capture_errors', ())),
        'piecewise_capture_errors': dict(getattr(piecewise, 'capture_errors', {})),
    }


def workload_spec(name, blocks, block_size):
    if name not in WORKLOADS:
        raise ValueError(name)
    capacity = blocks * block_size
    longest = min(3000, capacity - block_size - 32)
    if longest < 512:
        raise ValueError('benchmark requires room for at least a 512-token request')
    lengths = {
        'all_decode': [],
        'all_prefill': [min(300, longest), min(500, longest), min(1000, longest)],
        'long_decode': [longest],
        'many_long': [longest] * 4,
        'capacity_pressure': [longest] * max(8, math.ceil(1.5 * capacity / longest)),
    }[name]
    return {'prime_count': 0 if name == 'all_prefill' else 4,
            'prime_length': 32, 'prime_output': 48,
            'new_lengths': lengths, 'new_output': 1 if name == 'all_prefill' else 16}


class Probe:
    def __init__(self, engine, baseline, case_timeout=180):
        self.engine = engine
        self.baseline = baseline
        self.case_timeout = case_timeout
        self.records = {}
        self.collecting = False
        self.force = False
        self.active = []
        self.logits = {}
        self.baseline_events = []
        self.runtime = dict.fromkeys(RUNTIME_COUNTERS, 0)
        self.schedule_records = []
        self.peak_blocks = 0
        self.cpu_step_seconds = 0.
        self.empty_steps = 0
        self.steps = 0
        self._instrument_schedule()
        if baseline:
            self._instrument_baseline()

    def _instrument_schedule(self):
        original = self.engine.scheduler.schedule

        def schedule(*args, **kwargs):
            start = time.perf_counter()
            output = original(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if self.collecting:
                if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], bool):
                    seqs, is_prefill = output
                    total = sum(seq.num_scheduled_tokens for seq in seqs)
                    prefill, decode = (total, 0) if is_prefill else (0, total)
                else:
                    prefill = output.num_prefill_tokens
                    decode = output.num_decode_tokens
                self.schedule_records.append({'prefill_tokens': prefill, 'decode_tokens': decode,
                                              'cpu_seconds': elapsed})
                self.sample_blocks()
            return output

        self.engine.scheduler.schedule = schedule
        manager = self.engine.scheduler.block_manager
        original_allocate = manager._allocate_block

        def allocate(*args, **kwargs):
            result = original_allocate(*args, **kwargs)
            if self.collecting:
                self.sample_blocks()
            return result

        manager._allocate_block = allocate

    def sample_blocks(self):
        used = len(self.engine.scheduler.block_manager.used_block_ids)
        self.peak_blocks = max(self.peak_blocks, used)

    def _instrument_baseline(self):
        runner = self.engine.model_runner
        original_run = runner.run
        original_postprocess = self.engine.scheduler.postprocess
        probe = self

        class GraphProxy:
            def __init__(self, graph):
                self.graph = graph

            def replay(self):
                probe.runtime['full_graph_replays'] += 1
                return self.graph.replay()

        if hasattr(runner, 'graphs'):
            runner.graphs = {key: GraphProxy(graph) for key, graph in runner.graphs.items()}

        def run(seqs, is_prefill):
            self.active = [SimpleNamespace(seq_id=seq.seq_id, output_index=seq.num_completion_tokens,
                sample_output=seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)) for seq in seqs]
            self.runtime['forward_count'] += 1
            mixed = any(seq.is_prefill for seq in seqs) and any(not seq.is_prefill for seq in seqs)
            self.runtime['mixed_forward_count'] += int(mixed)
            graph_before = self.runtime['full_graph_replays']
            tokens = original_run(seqs, is_prefill)
            self.runtime['eager_count'] += int(self.runtime['full_graph_replays'] == graph_before)
            if self.force:
                tokens = [forced_token(self.records[item.seq_id]['logical'], item.output_index)
                          for item in self.active]
            return tokens

        def postprocess(seqs, tokens, *args, **kwargs):
            before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
            result = original_postprocess(seqs, tokens, *args, **kwargs)
            for seq in seqs:
                if seq.num_completion_tokens > before[seq.seq_id]:
                    self.baseline_events.extend((seq.seq_id, token) for token in
                                                seq.completion_token_ids[before[seq.seq_id]:])
            return result

        runner.run = run
        self.engine.scheduler.postprocess = postprocess

    def counters(self):
        values = self.runtime if self.baseline else self.engine.model_runner.runtime_stats
        missing = set(RUNTIME_COUNTERS) - set(values)
        if missing:
            raise ValueError(f'missing runtime counters: {missing}')
        return {key: int(values[key]) for key in RUNTIME_COUNTERS}

    def add(self, length, logical, output, seed):
        from nanovllm import SamplingParams
        tokens = prompt(length, seed + logical)
        arrival = time.perf_counter()
        self.engine.add_request(tokens, SamplingParams(temperature=.8, max_tokens=output, ignore_eos=True))
        seq = self.engine.scheduler.waiting[-1]
        self.records[seq.seq_id] = {'logical': logical, 'seq': seq, 'arrival': arrival,
            'times': [], 'token_ids': [], 'finished': None, 'existing': False,
            'expected_outputs': output, 'prompt_tokens': length}
        return seq

    def step(self, schedule=True):
        self.baseline_events.clear()
        start = time.perf_counter()
        result = self.engine.step() if schedule else self.engine.step(schedule=False)
        end = time.perf_counter()
        events = self.baseline_events if self.baseline else self.engine.last_token_events
        record_events(self.records, events, end)
        for seq_id, _ in result[0]:
            self.records[seq_id]['finished'] = end
        if self.collecting:
            self.cpu_step_seconds += end - start
            self.steps += 1
            self.empty_steps += not bool(events)
            self.sample_blocks()
        return result

    def flush(self):
        while not self.baseline and self.engine.pending_count:
            self.step(schedule=False)

    def drain(self):
        start = time.perf_counter()
        for _ in range(100000):
            if self.engine.is_finished():
                self.flush()
                return
            if time.perf_counter() - start > self.case_timeout:
                raise TimeoutError('workload did not drain within the case timeout')
            self.step()
        raise RuntimeError('workload exceeded 100000 steps')

    def prime(self, seqs):
        start = time.perf_counter()
        depth = None if self.baseline else self.engine.max_inflight_batches
        if depth is not None:
            self.engine.max_inflight_batches = 1
        try:
            while any(seq.num_completion_tokens == 0 for seq in seqs):
                if time.perf_counter() - start > self.case_timeout:
                    raise TimeoutError('prime did not produce an output')
                self.step()
            self.flush()
        finally:
            if depth is not None:
                self.engine.max_inflight_batches = depth

    def _observe_logits(self, items, logits):
        if len(items) != len(logits):
            raise ValueError('logit rows do not match scheduled items')
        for item, row in zip(items, logits):
            record = self.records[item.seq_id]
            if not item.sample_output or item.output_index >= record['expected_outputs']:
                continue
            key = (record['logical'], item.output_index)
            if key in self.logits:
                raise ValueError(f'duplicate numerical-check row {key}')
            self.logits[key] = row.detach().clone()

    def _override_tokens(self, items, tokens):
        import torch
        return torch.tensor([forced_token(self.records[item.seq_id]['logical'], max(0, item.output_index))
                             for item in items], dtype=tokens.dtype, device=tokens.device)

    def correctness(self, destination):
        import torch
        runner = self.engine.model_runner
        self.force = True
        if self.baseline:
            hook = runner.model.lm_head.register_forward_hook(
                lambda module, inputs, output: self._observe_logits(self.active, output))
        else:
            runner.logits_observer = self._observe_logits
            runner.token_override = self._override_tokens
            hook = None
        torch.manual_seed(123)
        first = [self.add(length, logical, 8, 71000) for logical, length in enumerate((33, 61))]
        self.prime(first)
        for logical, length in enumerate((513, 769), 2):
            self.add(length, logical, 8, 71000)
        self.drain()
        continuations = {}
        for record in self.records.values():
            logical = record['logical']
            expected = [forced_token(logical, pos) for pos in range(record['expected_outputs'])]
            actual = list(record['seq'].completion_token_ids)
            if actual != expected or record['token_ids'] != expected:
                raise AssertionError(f'teacher-forced continuation mismatch for {logical}')
            continuations[logical] = actual
        expected_keys = {(logical, pos) for logical, ids in continuations.items() for pos in range(len(ids))}
        if set(self.logits) != expected_keys:
            raise AssertionError('incomplete full-logit rows')
        self.logits = {key: row.cpu() for key, row in self.logits.items()}
        torch.save({'logits': self.logits, 'forced_tokens': continuations}, destination)
        if hook is not None:
            hook.remove()
        else:
            runner.logits_observer = None
            runner.token_override = None
        self.force = False
        return {'logit_rows': len(self.logits), 'passed_local_assertions': True,
                'runtime_stats': self.counters()}

    def case(self, name, rep, blocks, block_size, warm=False):
        import torch
        if not self.engine.is_finished():
            raise RuntimeError('case must start with no active requests')
        self.collecting = False
        self.records.clear()
        spec = workload_spec(name, blocks, block_size)
        seed = 10000 + rep * 100 + WORKLOADS.index(name) * 1000 + (100000 if warm else 0)
        torch.manual_seed(500 + rep)
        active = [self.add(spec['prime_length'], i, spec['prime_output'], seed)
                  for i in range(spec['prime_count'])]
        self.prime(active)
        for record in self.records.values():
            record['existing'] = True
        torch.cuda.synchronize()
        before = self.counters()
        self.schedule_records.clear()
        self.peak_blocks = 0
        self.cpu_step_seconds = 0.
        self.steps = self.empty_steps = 0
        if not self.baseline:
            self.engine.max_observed_inflight = 0
        start = time.perf_counter()
        self.collecting = True
        self.sample_blocks()
        for logical, length in enumerate(spec['new_lengths'], spec['prime_count']):
            self.add(length, logical, spec['new_output'], seed)
        self.drain()
        torch.cuda.synchronize()
        end = time.perf_counter()
        self.collecting = False
        requests = []
        for record in self.records.values():
            if record['finished'] is None or len(record['times']) != record['expected_outputs']:
                raise AssertionError('missing completion or output events')
            if record['token_ids'] != list(record['seq'].completion_token_ids):
                raise AssertionError('accepted events differ from request outputs')
            row = request_metrics(record['arrival'], record['times'], start,
                                  record['existing'], record['finished'])
            row.update(logical=record['logical'], prompt_tokens=record['prompt_tokens'])
            requests.append(row)
        old = [row for row in requests if row['existing']]
        new = [row for row in requests if not row['existing']]
        gaps = [gap for row in requests for gap in row['gap_ms']]
        counters = counter_delta(before, self.counters())
        cache = self.engine.model_runner.kv_cache
        bytes_per_block = cache.numel() * cache.element_size() / blocks
        return {'name': name, 'rep': rep, 'spec': spec, 'wall_ms': (end - start) * 1000,
            'output_tok_s': sum(row['window_output_tokens'] for row in requests) / (end - start),
            'window_output_tokens': sum(row['window_output_tokens'] for row in requests),
            'ttft_mean_ms': mean_or_none([row['ttft_ms'] for row in new]),
            'ttft_p95_ms': quantile([row['ttft_ms'] for row in new], .95),
            'tpot_mean_ms': mean_or_none([row['tpot_ms'] for row in requests]),
            'existing_tpot_ms': mean_or_none([row['tpot_ms'] for row in old]),
            'request_max_gap_mean_ms': mean_or_none([row['max_token_gap_ms'] for row in requests]),
            'existing_max_gap_mean_ms': mean_or_none([row['max_token_gap_ms'] for row in old]),
            'existing_gap_mean_ms': mean_or_none([row['average_gap_ms'] for row in old]),
            'max_token_gap_ms': max(gaps) if gaps else None,
            'gap_p95_ms': quantile(gaps, .95), 'kv_peak_blocks': self.peak_blocks,
            'kv_peak_mib': self.peak_blocks * bytes_per_block / 1024**2,
            'cuda_graph_replays': counters['full_graph_replays'] + counters['piecewise_segment_replays'],
            'async_depth': 0 if self.baseline else int(self.engine.max_observed_inflight),
            'steps': self.steps, 'empty_steps': self.empty_steps,
            'cpu_step_ms': self.cpu_step_seconds * 1000,
            'scheduler_ms': sum(row['cpu_seconds'] for row in self.schedule_records) * 1000,
            'prefill_tokens': sum(row['prefill_tokens'] for row in self.schedule_records),
            'decode_tokens': sum(row['decode_tokens'] for row in self.schedule_records),
            **counters, 'requests': requests}


def install_fixed_cache(blocks):
    import torch
    from nanovllm.engine.model_runner import ModelRunner

    def allocate(runner):
        config, hf = runner.config, runner.config.hf_config
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, 'head_dim', hf.hidden_size // hf.num_attention_heads)
        config.num_kvcache_blocks = blocks
        runner.kv_cache = torch.empty(2, hf.num_hidden_layers, blocks, runner.block_size, heads, dim)
        layers = [module for module in runner.model.modules()
                  if hasattr(module, 'k_cache') and hasattr(module, 'v_cache')]
        if len(layers) != hf.num_hidden_layers:
            raise ValueError('unexpected number of per-layer KV caches')
        for layer_id, module in enumerate(layers):
            module.k_cache = runner.kv_cache[0, layer_id]
            module.v_cache = runner.kv_cache[1, layer_id]

    ModelRunner.allocate_kv_cache = allocate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=os.environ.get('NANOVLLM_MODEL'))
    parser.add_argument('--action', choices=('correctness', 'benchmark'), required=True)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--kv-blocks', type=int, default=64)
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--workloads', choices=WORKLOADS, nargs='+', default=list(WORKLOADS))
    parser.add_argument('--case-timeout', type=int, default=180)
    parser.add_argument('--fused-qk-norm-rope', action='store_true')
    parser.add_argument('--prefix-cache-lookahead', type=int, default=0)
    parser.add_argument('--uncached-first', action='store_true')
    parser.add_argument('--intrusive-free-queue', action='store_true')
    args = parser.parse_args()
    if not args.model:
        parser.error('--model or NANOVLLM_MODEL is required')
    if min(args.budget, args.kv_blocks, args.max_num_seqs, args.max_model_len,
           args.reps, args.case_timeout) <= 0:
        parser.error('numeric settings must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    import nanovllm
    from nanovllm import LLM
    from nanovllm.config import Config
    if not torch.cuda.is_available():
        raise RuntimeError('GPU verification is required; CPU mocks are not benchmark results')
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(123)
    baseline = args.mode == 'baseline'
    config = dict(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.budget,
                  max_model_len=args.max_model_len, gpu_memory_utilization=.6, tensor_parallel_size=1)
    if baseline:
        install_fixed_cache(args.kv_blocks)
        config['enforce_eager'] = False
    else:
        graph = 'eager' if args.mode == 'v1_sync_eager' else (
            'piecewise' if args.mode == 'v1_sync_piecewise' else 'auto')
        config.update(num_kvcache_blocks=args.kv_blocks, cuda_graph_mode=graph,
                      async_scheduling=args.mode == 'v1_async_auto',
                      max_inflight_batches=2 if args.mode == 'v1_async_auto' else 1)
        config.update(fused_qk_norm_rope=args.fused_qk_norm_rope,
                      prefix_cache_lookahead=args.prefix_cache_lookahead,
                      uncached_first=args.uncached_first, intrusive_free_queue=args.intrusive_free_queue)
        known = {field.name for field in fields(Config)}
        if not set(config).issubset(known):
            raise ValueError(f'unsupported V1 config keys: {set(config) - known}')
    engine = LLM(args.model, **config)
    try:
        if engine.model_runner.config.num_kvcache_blocks != args.kv_blocks:
            raise AssertionError('KV cache capacity differs from requested capacity')
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            raise AssertionError(f'expected BF16 KV cache, got {engine.model_runner.kv_cache.dtype}')
        probe = Probe(engine, baseline, args.case_timeout)
        metadata = {'schema_version': 1, **{key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()},
            'import': nanovllm.__file__, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(), 'dtype': str(engine.model_runner.kv_cache.dtype),
            'blocks': engine.model_runner.config.num_kvcache_blocks, 'config': config,
            'graph_capture': capture_metadata(engine.model_runner),
            'logit_capture': 'GPU row clones at execution; CPU transfer only after all in-flight work drains',
            'measurement': 'accepted token events at engine.step return; no HTTP/network latency',
            'tpot_definition': '(last accepted output - first accepted output)/(total outputs-1); includes prime',
            'gap_definition': 'consecutive accepted outputs whose later event is in measurement window; crossing interval included',
            'warmup': 'one independent full workload immediately before its measured repetitions; distinct prompt seed',
            'synchronization': 'case boundaries only; prime uses depth=1 in all modes, then restores configured depth',
            'timing_note': 'cpu_step_ms includes blocking result waits; it is not GPU kernel execution time'}
        with torch.inference_mode():
            if args.action == 'correctness':
                metadata.update(probe.correctness(args.output.with_suffix('.pt')))
            else:
                metadata['cases'] = []
                for name in args.workloads:
                    probe.case(name, 99, args.kv_blocks, engine.model_runner.block_size, warm=True)
                    for rep in range(args.reps):
                        case = probe.case(name, rep, args.kv_blocks, engine.model_runner.block_size)
                        metadata['cases'].append(case)
                        metadata['graph_capture'] = capture_metadata(engine.model_runner)
                        dump(args.output, metadata)
                        print(args.mode, json.dumps({key: value for key, value in case.items()
                                                   if key not in ('requests', 'spec')}), flush=True)
        metadata['graph_capture'] = capture_metadata(engine.model_runner)
        metadata['complete'] = True
        dump(args.output, metadata)
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == '__main__':
    main()
