"""Teacher-forced numerical checks and paired GPU performance comparisons."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from benchmark_v1 import MODES, WORKLOADS, dump


REPO = Path(__file__).resolve().parents[1]
METRICS = ('output_tok_s', 'wall_ms', 'ttft_mean_ms', 'ttft_p95_ms', 'tpot_mean_ms',
           'existing_tpot_ms', 'request_max_gap_mean_ms', 'existing_max_gap_mean_ms',
           'existing_gap_mean_ms', 'max_token_gap_ms', 'gap_p95_ms', 'kv_peak_blocks', 'kv_peak_mib',
           'forward_count', 'mixed_forward_count', 'eager_count', 'full_graph_replays',
           'piecewise_graph_replays', 'piecewise_segment_replays', 'cuda_graph_replays',
           'gpu_feedback_tokens', 'async_depth', 'scheduler_ms', 'cpu_step_ms',
           'prefill_tokens', 'decode_tokens', 'steps', 'empty_steps')


def process_order(candidate):
    return [('baseline', 'ab'), (candidate, 'ab'), (candidate, 'ba'), ('baseline', 'ba')]


def percent_change(baseline, candidate):
    return 100 * (candidate / baseline - 1) if baseline and candidate is not None else None


def validate_logit_keys(logits, forced_tokens):
    expected = {(logical, position) for logical, tokens in forced_tokens.items()
                for position in range(len(tokens))}
    if not expected or set(logits) != expected:
        raise ValueError('logit rows must cover every requested output position exactly once')


def fingerprint(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts}


def environment(args):
    import torch
    packages = {}
    for package in ('torch', 'triton', 'transformers', 'flash-attn', 'xxhash'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    model_config = Path(args.model) / 'config.json'
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    return {'schema_version': 1, 'model': args.model,
            'model_config_sha256': hashlib.sha256(model_config.read_bytes()).hexdigest(),
            'gpu': torch.cuda.get_device_name(), 'python': sys.version, 'packages': packages,
            'budgets': args.budgets, 'modes': args.modes, 'workloads': args.workloads,
            'kv_blocks': args.kv_blocks, 'max_num_seqs': args.max_num_seqs,
            'max_model_len': args.max_model_len, 'repetitions_per_process': args.reps,
            'observations_per_mode_per_pair': args.reps * 2,
            'process_order': 'for each candidate: baseline, candidate, candidate, baseline',
            'pairing': 'same workload and prompt seed for each repetition in each process order',
            'warmup': 'one full workload immediately before its measured repetitions in every process',
            'measurement': 'accepted token events at engine.step return, not HTTP streaming',
            'correctness_thresholds': {'relative_l2_max': args.relative_l2, 'cosine_min': args.cosine},
            'source_sha256': {'baseline': fingerprint(args.baseline_source),
                              'v1': fingerprint(args.v1_source)},
            'harness_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), REPO / 'scripts/benchmark_v1.py')}}


def correctness_path(output, mode, budget):
    return output / f'correctness_{mode}_b{budget}.json'


def benchmark_path(output, candidate, mode, budget, order):
    return output / f'benchmark_{candidate}_{mode}_b{budget}_{order}.json'


def run_worker(args, mode, budget, action, destination):
    if args.resume and destination.is_file():
        data = json.loads(destination.read_text(encoding='utf-8'))
        if not data.get('complete'):
            print('RESTART incomplete', destination.name, flush=True)
        else:
            if data['blocks'] != args.kv_blocks or data['schema_version'] != 1:
                raise ValueError('cached run has incompatible settings')
            if action == 'correctness':
                if not data.get('passed_local_assertions') or not destination.with_suffix('.pt').exists():
                    raise ValueError('cached numerical run is incomplete')
            elif len(data['cases']) != args.reps * len(args.workloads):
                raise ValueError('cached benchmark case count differs')
            print('REUSE', destination.name, flush=True)
            return data
    elif destination.exists() and not args.overwrite:
        raise FileExistsError(f'{destination}; use --resume or --overwrite')
    source = args.baseline_source if mode == 'baseline' else args.v1_source
    command = [sys.executable, str(REPO / 'scripts/benchmark_v1.py'), '--mode', mode,
        '--source-root', str(source), '--model', args.model, '--output', str(destination),
        '--action', action, '--budget', str(budget), '--kv-blocks', str(args.kv_blocks),
        '--max-num-seqs', str(args.max_num_seqs), '--max-model-len', str(args.max_model_len),
        '--reps', str(args.reps), '--case-timeout', str(args.case_timeout),
        '--workloads', *args.workloads]
    print('START', destination.name, flush=True)
    log_path = destination.with_suffix('.log')
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    if result.returncode:
        print(log_path.read_text(encoding='utf-8', errors='replace')[-16000:], flush=True)
        raise RuntimeError(f'{destination.name}: exit {result.returncode}')
    data = json.loads(destination.read_text(encoding='utf-8'))
    if not data.get('complete') or data['blocks'] != args.kv_blocks:
        raise ValueError('worker did not produce a complete matching result')
    print('DONE', destination.name, flush=True)
    return data


def compare_logits(args, candidate, budget, environment_hash):
    import torch
    import torch.nn.functional as F
    baseline, optimized = [torch.load(correctness_path(args.output_dir, mode, budget).with_suffix('.pt'),
                                     map_location='cpu', weights_only=True)
                           for mode in ('baseline', candidate)]
    for data in (baseline, optimized):
        validate_logit_keys(data['logits'], data['forced_tokens'])
    if baseline['forced_tokens'] != optimized['forced_tokens']:
        raise ValueError('teacher-forced continuations differ')
    if set(baseline['logits']) != set(optimized['logits']):
        raise ValueError('full-logit row keys differ')
    rows = []
    for key in sorted(baseline['logits']):
        x, y = baseline['logits'][key].float(), optimized['logits'][key].float()
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError(f'logit row shape mismatch: {key}')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f'nonfinite logits: {key}')
        rows.append({'key': key, 'relative_l2': ((x-y).norm() / x.norm().clamp_min(1e-12)).item(),
                     'cosine': F.cosine_similarity(x[None], y[None]).item(),
                     'max_abs': (x-y).abs().max().item(),
                     'top1_equal': x.argmax().item() == y.argmax().item(),
                     'bitwise_equal': torch.equal(x, y)})
    result = {'candidate': candidate, 'budget': budget, 'rows': rows,
        'forced_tokens_equal': True, 'logit_rows': len(rows),
        'max_relative_l2': max(row['relative_l2'] for row in rows),
        'min_cosine': min(row['cosine'] for row in rows),
        'top1_equal_rows': sum(row['top1_equal'] for row in rows),
        'bitwise_equal_rows': sum(row['bitwise_equal'] for row in rows),
        'thresholds': {'relative_l2_max': args.relative_l2, 'cosine_min': args.cosine},
        'environment_sha256': environment_hash,
        'passed': all(row['relative_l2'] <= args.relative_l2 and row['cosine'] >= args.cosine for row in rows)}
    dump(args.output_dir / f'comparison_{candidate}_b{budget}.json', result)
    print('LOGITS', candidate, budget, result['passed'], 'rows', len(rows),
          'max_relative_l2', result['max_relative_l2'], 'top1', result['top1_equal_rows'], flush=True)
    return result


def summarize(args):
    rows = []
    for budget in args.budgets:
        for candidate in args.modes:
            paired = {}
            for mode in ('baseline', candidate):
                paired[mode] = {}
                for order in ('ab', 'ba'):
                    path = benchmark_path(args.output_dir, candidate, mode, budget, order)
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if not data.get('complete'):
                        raise ValueError(f'incomplete result: {path}')
                    for case in data['cases']:
                        key = (order, case['name'], case['rep'])
                        if key in paired[mode]:
                            raise ValueError(f'duplicate case: {key}')
                        paired[mode][key] = case
            if set(paired['baseline']) != set(paired[candidate]):
                raise ValueError('paired cases differ')
            for workload in args.workloads:
                row = {'budget': budget, 'workload': workload, 'candidate': candidate}
                keys = sorted(key for key in paired['baseline'] if key[1] == workload)
                if len(keys) != args.reps * 2:
                    raise ValueError('wrong number of observations')
                for mode, label in (('baseline', 'baseline'), (candidate, 'optimized')):
                    data = [paired[mode][key] for key in keys]
                    summary = {'observations': len(data), 'median': {}, 'range': {}}
                    for metric in METRICS:
                        values = [case[metric] for case in data if case[metric] is not None]
                        summary['median'][metric] = statistics.median(values) if values else None
                        summary['range'][metric] = [min(values), max(values)] if values else None
                    row[label] = summary
                row['change_pct'] = {metric: percent_change(row['baseline']['median'][metric],
                                                           row['optimized']['median'][metric])
                                     for metric in METRICS}
                row['paired_change_pct'] = {}
                for metric in METRICS:
                    values = [percent_change(paired['baseline'][key][metric], paired[candidate][key][metric])
                              for key in keys]
                    values = [value for value in values if value is not None]
                    row['paired_change_pct'][metric] = {'values': values,
                        'mean': statistics.mean(values) if values else None,
                        'median': statistics.median(values) if values else None}
                rows.append(row)
    dump(args.output_dir / 'v1_summary.json', rows)
    columns = ('output_tok_s', 'ttft_mean_ms', 'existing_tpot_ms', 'existing_max_gap_mean_ms',
               'kv_peak_blocks', 'forward_count', 'full_graph_replays', 'piecewise_graph_replays', 'async_depth')
    lines = ['Median of matched workload repetitions in separate ABBA processes.',
             'Positive relative change means larger; latency regressions are retained.',
             'Token timestamps are engine.step accepted-result times, not network streaming times.',
             'TPOT includes priming; token gaps include the interval crossing the measurement boundary.',
             'budget workload pair mode ' + ' '.join(columns)]
    for row in rows:
        for mode, label in (('baseline', 'baseline'), (row['candidate'], 'optimized')):
            values = row[label]['median']
            lines.append(' '.join([str(row['budget']), row['workload'], row['candidate'], mode] +
                                  ['-' if values[key] is None else f'{values[key]:.3f}' for key in columns]))
    report = '\n'.join(lines) + '\n'
    (args.output_dir / 'v1_summary.txt').write_text(report, encoding='utf-8')
    print(report, end='')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('all', 'correctness', 'benchmark', 'summarize'), default='all')
    parser.add_argument('--model', default=os.environ.get('NANOVLLM_MODEL'))
    parser.add_argument('--baseline-source', type=Path, default=REPO / 'baseline')
    parser.add_argument('--v1-source', type=Path, default=REPO / 'src')
    parser.add_argument('--output-dir', type=Path, default=REPO / 'results/v1-comparison')
    parser.add_argument('--modes', choices=MODES[1:], nargs='+',
                        default=['v1_sync_eager', 'v1_sync_auto', 'v1_async_auto'])
    parser.add_argument('--workloads', choices=WORKLOADS, nargs='+', default=list(WORKLOADS))
    parser.add_argument('--budgets', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--kv-blocks', type=int, default=64)
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--relative-l2', type=float, default=.03)
    parser.add_argument('--cosine', type=float, default=.999)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--case-timeout', type=int, default=180)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for key in ('output_dir', 'baseline_source', 'v1_source'):
        setattr(args, key, getattr(args, key).resolve())
    if min([*args.budgets, args.reps, args.kv_blocks, args.max_num_seqs,
            args.max_model_len, args.timeout, args.case_timeout]) <= 0:
        parser.error('numeric settings must be positive')
    if args.relative_l2 <= 0 or not 0 < args.cosine <= 1:
        parser.error('invalid numerical thresholds')
    for values in (args.budgets, args.modes, args.workloads):
        if len(set(values)) != len(values):
            parser.error('budgets, modes and workloads must not contain duplicates')
    if args.action == 'summarize':
        summarize(args)
        return
    if not args.model:
        parser.error('--model or NANOVLLM_MODEL is required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = environment(args)
    env_path = args.output_dir / 'v1_environment.json'
    if env_path.exists() and not args.overwrite:
        if json.loads(env_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('source/harness/config changed; use a new output directory or --overwrite')
    dump(env_path, metadata)
    env_hash = hashlib.sha256(env_path.read_bytes()).hexdigest()
    if args.action in ('all', 'correctness'):
        failed = []
        for budget in args.budgets:
            run_worker(args, 'baseline', budget, 'correctness', correctness_path(args.output_dir, 'baseline', budget))
            for mode in args.modes:
                run_worker(args, mode, budget, 'correctness', correctness_path(args.output_dir, mode, budget))
                if not compare_logits(args, mode, budget, env_hash)['passed']:
                    failed.append((mode, budget))
        if failed:
            raise AssertionError(f'numerical validation failed; performance blocked for {failed}')
    if args.action in ('all', 'benchmark'):
        for budget in args.budgets:
            for candidate in args.modes:
                path = args.output_dir / f'comparison_{candidate}_b{budget}.json'
                check = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if not check.get('passed') or check.get('environment_sha256') != env_hash:
                    raise ValueError(f'run matching correctness checks first: {candidate}, {budget}')
                for mode, order in process_order(candidate):
                    destination = benchmark_path(args.output_dir, candidate, mode, budget, order)
                    run_worker(args, mode, budget, 'benchmark', destination)
        summarize(args)
    print('FINISHED', args.action, flush=True)


if __name__ == '__main__':
    main()
