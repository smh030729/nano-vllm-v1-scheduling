"""CPU-only scheduler/metadata trace for the frozen teacher-forced GPU check."""

import argparse
import atexit
from collections import deque
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from benchmark_v1 import Probe, capture_metadata, dump, forced_token, install_fixed_cache, prompt


ROOT = Path(__file__).resolve().parents[1]


def source_fingerprints():
    sources = {}
    for name in ('src', 'baseline'):
        root = ROOT / name
        sources[name] = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(root.rglob('*'))
                         if path.is_file() and '__pycache__' not in path.parts
                         and path.suffix not in ('.pyc', '.pyo')}
    sources['harness'] = {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                          for name in ('benchmark_v1.py', 'compare_v1.py', 'diagnose_v1.py')}
    return sources


def load_scheduler(source):
    for name in tuple(sys.modules):
        if name == 'nanovllm' or name.startswith('nanovllm.'):
            del sys.modules[name]
    for name, location in [('nanovllm', source / 'nanovllm'),
                           ('nanovllm.engine', source / 'nanovllm/engine')]:
        module = ModuleType(name)
        module.__path__ = [str(location)]
        sys.modules[name] = module
    config = ModuleType('nanovllm.config')
    config.Config = SimpleNamespace
    sys.modules[config.__name__] = config
    scheduler = importlib.import_module('nanovllm.engine.scheduler').Scheduler
    sequence = importlib.import_module('nanovllm.engine.sequence').Sequence
    sampling = importlib.import_module('nanovllm.sampling_params').SamplingParams
    return scheduler, sequence, sampling


def digest(values):
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def trace(mode, budget):
    baseline = mode == 'baseline'
    cls, sequence_cls, sampling_cls = load_scheduler(ROOT / ('baseline' if baseline else 'src'))
    config = SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=budget,
        max_inflight_batches=2 if mode == 'v1_async' else 1, max_model_len=4096,
        eos=999999, kvcache_block_size=256, num_kvcache_blocks=64)
    scheduler = cls(config)
    metadata_builder = None if baseline else importlib.import_module('nanovllm.engine.batch_input').build_batch_metadata
    sequence_cls.block_size = 256
    records, rows, pending = {}, [], deque()

    def add(logical, length):
        seq = sequence_cls(prompt(length, 71000 + logical),
                           sampling_cls(temperature=.8, max_tokens=8, ignore_eos=True))
        scheduler.add(seq)
        records[seq.seq_id] = (logical, seq)

    def submit():
        scheduled = scheduler.schedule()
        items = scheduled.seqs
        if not items:
            return
        entries, token_ids, positions, slots, result_tokens = [], [], [], [], []
        for item in items:
            logical, seq = records[item.seq_id]
            start = item.num_cached_tokens if baseline else item.start_pos
            qlen = item.num_scheduled_tokens
            end = start + qlen
            output_index = seq.num_completion_tokens if baseline else item.output_index
            produces = end == len(seq) if baseline else item.produces_output
            raw = list(seq[start:end]) if baseline else list(item.input_token_ids)
            resolved = [forced_token(logical, start + index - seq.num_prompt_tokens)
                        if token < 0 else token for index, token in enumerate(raw)]
            row_slots = [item.block_table[pos // 256] * 256 + pos % 256 for pos in range(start, end)]
            entries.append(dict(logical=logical, start=start, query_len=qlen, context_len=end,
                output_index=output_index, produces_output=produces, is_prefill=item.is_prefill,
                token_digest=digest(resolved), block_table=list(item.block_table),
                feedback_inputs=sum(token < 0 for token in raw)))
            token_ids.extend(resolved)
            positions.extend(range(start, end))
            slots.extend(row_slots)
            result_tokens.append(forced_token(logical, max(0, output_index)))
        width = max(len(item.block_table) for item in items)
        uniform = all(row['query_len'] == 1 for row in entries)
        any_prefill = any(row['is_prefill'] for row in entries)
        has_cache = any(row['start'] > 0 for row in entries)
        if not baseline:
            built = metadata_builder(items, 256, decode_table_width=16)
            assert built['positions'] == positions and built['slot_mapping'] == slots
            assert built['query_lens'] == [row['query_len'] for row in entries]
        n_tokens = len(token_ids)
        bucket = next((n for n in (64, 128, 256, 512, 1024) if n >= n_tokens and n <= budget), None)
        rows.append(dict(iteration=len(rows), entries=entries, num_input_tokens=n_tokens,
            token_digest=digest(token_ids), position_digest=digest(positions),
            slot_mapping_digest=digest(slots), uniform=uniform,
            logical_block_table_width=width,
            baseline_attention='varlen_paged' if any_prefill and has_cache else (
                'varlen_dense' if any_prefill else 'with_kvcache'),
            baseline_execution='eager' if any_prefill else 'full',
            v1_attention='with_kvcache' if uniform else 'varlen_paged',
            v1_eager_block_table_stride=16 if uniform else width,
            historical_dynamic_decode_stride=width,
            full_graph_block_table_stride=16,
            v1_auto_execution='full' if uniform else ('piecewise' if bucket == n_tokens else 'eager'),
            v1_auto_piecewise_bucket=bucket if not uniform and bucket == n_tokens else None,
            historical_auto_padded_execution='full' if uniform else ('piecewise' if bucket else 'eager'),
            historical_auto_padded_bucket=bucket if not uniform else None))
        pending.append((scheduled, result_tokens))

    def retire():
        scheduled, result_tokens = pending.popleft()
        scheduler.postprocess(scheduled.seqs if baseline else scheduled, result_tokens)

    add(0, 33)
    add(1, 61)
    submit()
    retire()
    add(2, 513)
    add(3, 769)
    for _ in range(100):
        if scheduler.is_finished() and not pending:
            break
        submit()
        if pending and (len(pending) >= config.max_inflight_batches or not scheduler.running):
            retire()
        elif pending and not getattr(scheduler, 'waiting', ()):
            # Engine also retires when schedule() returns empty at the end.
            scheduled = pending[-1][0]
            if all(seq.num_completion_tokens + getattr(seq, 'num_output_placeholders', 0) >= 8
                   for _, seq in records.values() if not seq.is_finished):
                retire()
    else:
        raise RuntimeError('CPU schedule trace did not terminate')
    for logical, seq in records.values():
        assert seq.completion_token_ids == [forced_token(logical, i) for i in range(8)]
    return rows


def install_exact_piecewise_dispatch(runner):
    original = runner.graph_dispatcher.select

    def select(num_tokens, query_lens, full_available):
        selected = original(num_tokens, query_lens, full_available)
        if selected == 'piecewise' and num_tokens not in runner.piecewise.buckets:
            return 'eager'
        return selected

    runner.graph_dispatcher.select = select


def install_dynamic_decode_metadata(runner_module):
    original = runner_module.build_batch_metadata

    def metadata(items, block_size, decode_table_width=None):
        return original(items, block_size)

    runner_module.build_batch_metadata = metadata


def gpu_diagnostic(args):
    fingerprints = source_fingerprints()
    import torch
    baseline = args.gpu_mode == 'baseline_eager'
    source = ROOT / ('baseline' if baseline else 'src')
    sys.path.insert(0, str(source))
    from nanovllm import LLM
    if not args.model or not args.output:
        raise ValueError('--gpu-mode requires --model and --output')
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        raise FileExistsError('diagnostic output must be new; never overwrite formal results')
    config = dict(max_num_seqs=16, max_num_batched_tokens=args.budget,
                  max_model_len=4096, gpu_memory_utilization=.6, tensor_parallel_size=1)
    if baseline:
        install_fixed_cache(args.kv_blocks)
        config['enforce_eager'] = True
    else:
        config.update(num_kvcache_blocks=args.kv_blocks,
                      cuda_graph_mode='auto' if args.gpu_mode.startswith('v1_auto_') else 'eager',
                      cudagraph_allow_padding=args.gpu_mode == 'v1_auto_padded',
                      async_scheduling=False, max_inflight_batches=1)
        if args.gpu_mode == 'v1_eager_wide':
            import nanovllm.engine.model_runner as runner_module
            original = runner_module.build_batch_metadata

            def metadata_with_wide_decode(*values, **kwargs):
                result = original(*values, **kwargs)
                if all(length == 1 for length in result['query_lens']):
                    result['block_tables'] = [row + [0] * (16 - len(row)) for row in result['block_tables']]
                    if any(len(row) != 16 for row in result['block_tables']):
                        raise ValueError('diagnostic assumes max_model_len=4096, block_size=256')
                return result

            runner_module.build_batch_metadata = metadata_with_wide_decode
        elif args.gpu_mode == 'v1_eager_dynamic':
            import nanovllm.engine.model_runner as runner_module
            install_dynamic_decode_metadata(runner_module)
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(123)
    engine = LLM(args.model, **config)
    try:
        if args.gpu_mode == 'v1_auto_exact':
            install_exact_piecewise_dispatch(engine.model_runner)
        probe = Probe(engine, baseline)
        with torch.inference_mode():
            result = probe.correctness(args.output.with_suffix('.pt'))
        result.update(gpu_mode=args.gpu_mode, budget=args.budget, blocks=args.kv_blocks,
            model=args.model, source=str(source), config=config,
            gpu=torch.cuda.get_device_name(), dtype=str(engine.model_runner.kv_cache.dtype),
            graph_capture=capture_metadata(engine.model_runner),
            source_sha256=fingerprints,
            source_changed_during_run=source_fingerprints() != fingerprints,
            note='isolated diagnostic only; production and frozen benchmark are unchanged')
        dump(args.output, result)
        print(json.dumps(result), flush=True)
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


def compare_cpu(left, right, output):
    import torch
    import torch.nn.functional as F
    a, b = [torch.load(path, map_location='cpu', weights_only=True) for path in (left, right)]
    if a['forced_tokens'] != b['forced_tokens'] or set(a['logits']) != set(b['logits']):
        raise ValueError('forced tokens or logit keys differ')
    rows = []
    for key in sorted(a['logits']):
        x, y = a['logits'][key], b['logits'][key]
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('invalid comparison tensors')
        xd, yd = x.double(), y.double()
        rows.append(dict(key=key, bitwise_equal=torch.equal(x, y),
            relative_l2=((xd-yd).norm() / xd.norm().clamp_min(1e-12)).item(),
            cosine=F.cosine_similarity(xd[None], yd[None]).item(),
            max_abs=(xd-yd).abs().max().item()))
    result = dict(left=str(left), right=str(right), row_count=len(rows),
        bitwise_equal_rows=sum(row['bitwise_equal'] for row in rows),
        max_relative_l2=max(row['relative_l2'] for row in rows), rows=rows,
        input_sha256={'left': hashlib.sha256(left.read_bytes()).hexdigest(),
                      'right': hashlib.sha256(right.read_bytes()).hexdigest()},
        comparison_code_sha256=source_fingerprints())
    for side, path in (('left', left), ('right', right)):
        metadata_path = path.with_suffix('.json')
        if metadata_path.is_file():
            result.setdefault('input_metadata_sha256', {})[side] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'comparison_code_sha256')}))
    if output:
        dump(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-mode', choices=('baseline_eager', 'v1_eager_wide', 'v1_auto_exact',
                                              'v1_auto_padded', 'v1_eager_dynamic'))
    parser.add_argument('--model')
    parser.add_argument('--kv-blocks', type=int, default=64)
    parser.add_argument('--compare-left', type=Path)
    parser.add_argument('--compare-right', type=Path)
    args = parser.parse_args()
    if args.gpu_mode:
        gpu_diagnostic(args)
        return
    if args.compare_left or args.compare_right:
        if not args.compare_left or not args.compare_right:
            parser.error('both --compare-left and --compare-right are required')
        compare_cpu(args.compare_left, args.compare_right, args.output)
        return
    traces = {mode: trace(mode, args.budget) for mode in ('baseline', 'v1_sync', 'v1_async')}
    for mode, rows in traces.items():
        print(mode)
        for row in rows:
            entries = ', '.join(f"R{e['logical']}[{e['start']}:{e['context_len']}]"
                f"{'->'+str(e['output_index']) if e['produces_output'] else ''}"
                f"/blocks{e['block_table']}" for e in row['entries'])
            print(row['iteration'], entries, 'tokens', row['num_input_tokens'],
                  'width', row['logical_block_table_width'],
                  'auto', row['v1_auto_execution'], row['v1_auto_piecewise_bucket'])
    comparisons = []
    baseline = traces['baseline']
    for mode in ('v1_sync', 'v1_async'):
        other = traces[mode]
        for index, (a, b) in enumerate(zip(baseline, other)):
            comparisons.append({'mode': mode, 'iteration': index,
                'tokens_equal': a['token_digest'] == b['token_digest'],
                'positions_equal': a['position_digest'] == b['position_digest'],
                'slots_equal': a['slot_mapping_digest'] == b['slot_mapping_digest'],
                'queries_equal': [e['query_len'] for e in a['entries']] == [e['query_len'] for e in b['entries']],
                'order_equal': [e['logical'] for e in a['entries']] == [e['logical'] for e in b['entries']]})
    print('COMPARISONS', json.dumps(comparisons))
    if args.output:
        args.output.write_text(json.dumps({'budget': args.budget, 'traces': traces,
                                          'comparisons': comparisons, 'source_sha256': source_fingerprints()},
                                         indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
