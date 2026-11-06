"""Run isolated ABBA experiments and preserve raw data, clocks and source hashes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from benchmark_v1 import dump
from reproduce_resume import ROOT, UPSTREAM, CACHE


def jobs(model, output):
    worker = ROOT / 'scripts/benchmark_v1.py'
    common = ['--model', model, '--action', 'benchmark', '--reps', '3']
    plans = []
    for group, budget, base, extra in (
        ('schedule512', 512, UPSTREAM, []),
        ('schedule1024', 1024, UPSTREAM, []),
        ('mixed_baseline512', 512, ROOT / 'baseline', []),
        ('combined512', 512, UPSTREAM,
         ['--fused-qk-norm-rope', '--prefix-cache-lookahead', '16',
          '--uncached-first', '--intrusive-free-queue'])):
        for index, arm in enumerate(('A', 'B', 'B', 'A')):
            name = f'{group}_{index}_{arm}'
            command = [sys.executable, str(worker), *common, '--budget', str(budget),
                '--mode', 'baseline' if arm == 'A' else 'v1_async_auto',
                '--source-root', str(base if arm == 'A' else ROOT / 'src')]
            if arm == 'B':
                command += extra
            plans.append(dict(name=name, group=group, arm=arm, command=command))
    worker = ROOT / 'scripts/reproduce_resume.py'
    for index, arm in enumerate(('A', 'B', 'B', 'A')):
        plans.append(dict(name=f'fusion_{index}_{arm}', group='fusion', arm=arm,
            command=[sys.executable, str(worker), '--stage', 'fusion', '--model', model,
                     '--arm', 'fusion_off' if arm == 'A' else 'fusion_on']))
    for pair in range(6):
        for arm in (('A', 'B') if pair % 2 == 0 else ('B', 'A')):
            plans.append(dict(name=f'cache_friend_{pair}_{arm}', group='cache_friend', arm=arm,
                command=[sys.executable, str(worker), '--stage', 'cache', '--model', model,
                    '--arm', 'upstream' if arm == 'A' else 'cache_standalone',
                    '--source', str(UPSTREAM if arm == 'A' else CACHE), '--reps', '1']))
    # Separate the marginal cache effect in the integrated engine from all three on.
    for index, arm in enumerate(('A', 'B', 'C', 'C', 'B', 'A')):
        plans.append(dict(name=f'cache_integrated_{index}_{arm}', group='cache_integrated', arm=arm,
            command=[sys.executable, str(worker), '--stage', 'cache', '--model', model,
                '--arm', dict(A='cache_off', B='cache_on', C='cache_all')[arm],
                '--reps', '3', '--controls']))
    plans.append(dict(name='micro', group='micro', arm='micro',
        command=[sys.executable, str(worker), '--stage', 'micro']))
    for plan in plans:
        plan['output'] = str(output / (plan['name'] + '.json'))
        plan['command'] += ['--output', plan['output']]
    return plans


def telemetry(stop, path):
    with path.open('w') as stream:
        stream.write('timestamp,pstate,graphics_MHz,memory_MHz,temperature_C,power_W,gpu_util_pct,memory_MiB\n')
        while not stop.is_set():
            sample = subprocess.run(['nvidia-smi',
                '--query-gpu=timestamp,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            stream.write(sample.stdout)
            stream.flush()
            stop.wait(.5)


def summarize(output, plans):
    grouped, summary = {}, {}
    keys = ('output_tok_s', 'wall_ms', 'prefill_tokens', 'ttft_p95_ms', 'existing_tpot_ms',
            'existing_max_gap_mean_ms', 'kv_peak_blocks', 'scheduler_ms')
    for plan in plans:
        data = json.loads(Path(plan['output']).read_text())
        if not data.get('complete'):
            raise ValueError(f'incomplete: {plan["name"]}')
        if plan['group'] == 'micro':
            summary['micro'] = data
            continue
        for case in data['cases']:
            grouped.setdefault((plan['group'], case['name']), {}).setdefault(plan['arm'], []).append(case)
    for (group, name), arms in grouped.items():
        row = dict(arms={})
        for arm, cases in arms.items():
            row['arms'][arm] = dict(observations=len(cases), median={}, range={})
            for key in keys:
                values = [c[key] for c in cases if c.get(key) is not None]
                if values:
                    row['arms'][arm]['median'][key] = statistics.median(values)
                    row['arms'][arm]['range'][key] = [min(values), max(values)]
        row['change_pct'] = {}
        for arm in arms:
            if arm == 'A':
                continue
            a, b = row['arms']['A']['median'], row['arms'][arm]['median']
            row['change_pct'][arm] = {k: 100 * (b[k] / a[k] - 1)
                                      for k in a if k in b and a[k] != 0}
        summary.setdefault(group, {})[name] = row
    dump(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/smh/huggingface/Qwen3-0.6B')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans = jobs(args.model, args.output)
    manifest = {}
    for label, root in [('integrated', ROOT / 'src'), ('upstream', UPSTREAM / 'nanovllm'),
                        ('mixed', ROOT / 'baseline/nanovllm'), ('cache', CACHE / 'nanovllm'),
                        ('harness', ROOT / 'scripts')]:
        manifest[label] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.suffix in ('.py', '.cu', '.cpp')}
    metadata = dict(model=args.model, requested_graphics_MHz=2400,
        lock_confirmed_by_user=True, source_sha256=manifest,
        packages={p: importlib.metadata.version(p) for p in
                  ('torch', 'triton', 'flash-attn', 'transformers', 'xxhash')},
        python=sys.version, jobs=plans)
    destination = args.output / 'environment.json'
    if destination.exists() and json.loads(destination.read_text()) != metadata:
        raise RuntimeError('source or settings changed; use a new output directory')
    dump(destination, metadata)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, args.output / 'telemetry.csv'), daemon=True)
    monitor.start()
    try:
        for i, plan in enumerate(plans):
            path = Path(plan['output'])
            if path.exists() and json.loads(path.read_text()).get('complete'):
                print('SKIP', plan['name'], flush=True)
                continue
            print(f'[{i+1}/{len(plans)}] START {plan["name"]}', flush=True)
            start = time.perf_counter()
            with path.with_suffix('.log').open('w') as stream:
                subprocess.run(plan['command'], stdout=stream, stderr=subprocess.STDOUT, check=True)
            print('DONE', plan['name'], round(time.perf_counter() - start, 2), 's', flush=True)
        summarize(args.output, plans)
    finally:
        stop.set()
        monitor.join()


if __name__ == '__main__':
    main()
