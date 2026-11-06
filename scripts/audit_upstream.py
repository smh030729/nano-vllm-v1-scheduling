"""Audit the completed original nano-vLLM versus asynchronous V1 experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


REPO = Path(__file__).resolve().parents[1]
COMMIT = 'bb823b3e06983d71485a8e1f23715ebd87d98ef8'
CANDIDATE = 'v1_async_auto'
WORKLOADS = ('all_decode', 'all_prefill', 'long_decode', 'many_long', 'capacity_pressure')
LABELS = dict(zip(WORKLOADS, ('纯 Decode', '纯 Prefill', '1 条长 Prompt', '4 条长 Prompt', 'KV 容量压力')))
METRICS = ('output_tok_s', 'wall_ms', 'ttft_mean_ms', 'ttft_p95_ms', 'tpot_mean_ms',
           'existing_tpot_ms', 'request_max_gap_mean_ms', 'existing_max_gap_mean_ms',
           'existing_gap_mean_ms', 'max_token_gap_ms', 'gap_p95_ms', 'kv_peak_blocks', 'kv_peak_mib',
           'forward_count', 'mixed_forward_count', 'eager_count', 'full_graph_replays',
           'piecewise_graph_replays', 'piecewise_segment_replays', 'cuda_graph_replays',
           'gpu_feedback_tokens', 'async_depth', 'scheduler_ms', 'cpu_step_ms',
           'prefill_tokens', 'decode_tokens', 'steps', 'empty_steps')
RESOURCES = ('forward_count', 'prefill_tokens', 'decode_tokens', 'kv_peak_blocks',
             'kv_peak_mib', 'scheduler_ms', 'cpu_step_ms', 'async_depth',
             'full_graph_replays', 'piecewise_graph_replays', 'gpu_feedback_tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*.py'))
            if '__pycache__' not in p.parts}


def path_key(value):
    value = str(value).replace('\\', '/').rstrip('/')
    if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        value = value[5] + ':' + value[6:]
    return value.lower() if len(value) > 1 and value[1] == ':' else value


def close(actual, expected, label):
    if expected is None or actual is None:
        require(actual is expected, f'{label}: null mismatch ({actual}, {expected})')
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f'{label}: length mismatch')
        for i, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, f'{label}[{i}]')
    else:
        require(isinstance(actual, (int, float)) and math.isfinite(actual), f'{label}: nonfinite/non-numeric')
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7),
                f'{label}: {actual} != independently calculated {expected}')


def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def change(base, new):
    return (new / base - 1) * 100 if base and new is not None else None


def expected_spec(name):
    lengths = {'all_decode': [], 'all_prefill': [300, 500, 1000],
               'long_decode': [3000], 'many_long': [3000] * 4,
               'capacity_pressure': [3000] * 9}[name]
    return {'prime_count': 0 if name == 'all_prefill' else 4, 'prime_length': 32,
            'prime_output': 48, 'new_lengths': lengths,
            'new_output': 1 if name == 'all_prefill' else 16}


def audit_case(case, label):
    spec = case['spec']
    require(spec == expected_spec(case['name']), f'{label}: unexpected workload specification')
    expected = {i: (spec['prime_length'], spec['prime_output'], True)
                for i in range(spec['prime_count'])}
    expected.update({i: (length, spec['new_output'], False)
                     for i, length in enumerate(spec['new_lengths'], spec['prime_count'])})
    requests = {r['logical']: r for r in case['requests']}
    require(len(requests) == len(case['requests']) and set(requests) == set(expected),
            f'{label}: missing/duplicate logical requests')
    for logical, (prompt_tokens, outputs, existing) in expected.items():
        row = requests[logical]
        tag = f'{label}/request{logical}'
        require((row['prompt_tokens'], row['total_output_tokens'], row['existing']) ==
                (prompt_tokens, outputs, existing), f'{tag}: request work differs')
        times = row['output_times_from_window_ms']
        require(len(times) == outputs and all(math.isfinite(t) for t in times), f'{tag}: output events missing')
        require(all(a <= b for a, b in zip(times, times[1:])), f'{tag}: non-monotonic outputs')
        require(times[-1] <= case['wall_ms'] + 1e-7, f'{tag}: output after measurement end')
        window = sum(t >= 0 for t in times)
        require(window == outputs - int(existing), f'{tag}: incorrect priming/window boundary')
        close(row['window_output_tokens'], window, f'{tag}/window_output_tokens')
        gaps = [b - a for a, b in zip(times, times[1:]) if b >= 0]
        close(row['gap_ms'], gaps, f'{tag}/gap_ms')
        close(row['max_token_gap_ms'], max(gaps) if gaps else None, f'{tag}/max_gap')
        close(row['average_gap_ms'], average(gaps), f'{tag}/average_gap')
        close(row['tpot_ms'], (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
              f'{tag}/tpot')
        if existing:
            require(row['ttft_ms'] is None, f'{tag}: existing request TTFT should be excluded')
        else:
            require(0 <= row['ttft_ms'] <= row['e2e_ms'], f'{tag}: invalid TTFT')
    old = [r for r in requests.values() if r['existing']]
    new = [r for r in requests.values() if not r['existing']]
    gaps = [gap for r in requests.values() for gap in r['gap_ms']]
    total = sum(r['window_output_tokens'] for r in requests.values())
    require(case['wall_ms'] > 0, f'{label}: invalid wall time')
    calculated = {'window_output_tokens': total, 'output_tok_s': total * 1000 / case['wall_ms'],
                  'ttft_mean_ms': average([r['ttft_ms'] for r in new]),
                  'ttft_p95_ms': quantile([r['ttft_ms'] for r in new], .95),
                  'tpot_mean_ms': average([r['tpot_ms'] for r in requests.values()]),
                  'existing_tpot_ms': average([r['tpot_ms'] for r in old]),
                  'request_max_gap_mean_ms': average([r['max_token_gap_ms'] for r in requests.values()]),
                  'existing_max_gap_mean_ms': average([r['max_token_gap_ms'] for r in old]),
                  'existing_gap_mean_ms': average([r['average_gap_ms'] for r in old]),
                  'max_token_gap_ms': max(gaps) if gaps else None, 'gap_p95_ms': quantile(gaps, .95)}
    for key, value in calculated.items():
        close(case[key], value, f'{label}/{key}')
    require(0 <= case['kv_peak_blocks'] <= 64, f'{label}: invalid KV peak')
    require(case['forward_count'] == case['eager_count'] + case['full_graph_replays'] +
            case['piecewise_graph_replays'], f'{label}: forward accounting differs')
    require(case['cuda_graph_replays'] == case['full_graph_replays'] + case['piecewise_segment_replays'],
            f'{label}: graph accounting differs')
    require(case['prefill_tokens'] >= sum(spec['new_lengths']), f'{label}: Prefill tokens undercounted')
    minimum_decode = spec['prime_count'] * (spec['prime_output'] - 1) + len(spec['new_lengths']) * (spec['new_output'] - 1)
    require(case['prefill_tokens'] + case['decode_tokens'] >= sum(spec['new_lengths']) + minimum_decode,
            f'{label}: total computed tokens undercounted')
    return {i: (r['prompt_tokens'], r['total_output_tokens'], r['window_output_tokens'], r['existing'])
            for i, r in requests.items()}


def audit(args):
    artifacts = {}
    def load(path):
        path = Path(path)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        artifacts[str(path)] = digest(path)
        return value

    manifest = load(args.upstream_manifest)
    require(manifest['commit'] == COMMIT, 'unexpected original upstream commit')
    require(len(manifest['sha256']) == 26, 'expected original 26-file manifest')
    for name, expected in manifest['sha256'].items():
        path = (args.upstream_source / name).resolve()
        require(path.is_relative_to(args.upstream_source), f'manifest path escapes source: {name}')
        require(digest(path) == expected, f'original upstream file changed: {name}')
    env_path = args.results / 'v1_environment.json'
    env = load(env_path)
    require(env['budgets'] == [512, 1024] and env['modes'] == [CANDIDATE] and
            env['workloads'] == list(WORKLOADS), 'audit expects both budgets and all five workloads for async auto')
    require((env['kv_blocks'], env['max_num_seqs'], env['max_model_len'],
             env['repetitions_per_process'], env['observations_per_mode_per_pair']) == (64, 16, 4096, 3, 6),
            'unexpected controlled experiment settings')
    require(env['process_order'] == 'for each candidate: baseline, candidate, candidate, baseline',
            'experiment does not declare ABBA order')
    thresholds = {'relative_l2_max': .03, 'cosine_min': .999}
    require(env['correctness_thresholds'] == thresholds, 'numerical thresholds changed')
    for label, root in (('baseline', args.upstream_source), ('v1', args.v1_source)):
        require(env['source_sha256'][label] == fingerprint(root), f'{label}: source hashes differ from run')
    require(set(env['harness_sha256']) == {'benchmark_v1.py', 'compare_v1.py'}, 'unexpected harness manifest')
    for name, expected in env['harness_sha256'].items():
        require(digest(REPO / 'scripts' / name) == expected, f'measurement harness changed: {name}')

    def worker(path, mode, budget, action):
        data = load(path)
        require(data.get('complete') is True, f'incomplete worker: {path.name}')
        expected = {'schema_version': 1, 'mode': mode, 'budget': budget, 'action': action,
                    'blocks': 64, 'kv_blocks': 64, 'max_num_seqs': 16, 'max_model_len': 4096,
                    'reps': 3, 'workloads': list(WORKLOADS), 'model': env['model'], 'gpu': env['gpu'],
                    'torch': env['packages']['torch'], 'dtype': 'torch.bfloat16'}
        for key, value in expected.items():
            require(data.get(key) == value, f'{path.name}: mismatched {key}')
        source = args.upstream_source if mode == 'baseline' else args.v1_source
        require(path_key(data['source_root']) == path_key(source), f'{path.name}: wrong source directory')
        require(path_key(data['import']) == path_key(source / 'nanovllm/__init__.py'), f'{path.name}: wrong imported package')
        config = data['config']
        for key, value in {'max_num_seqs': 16, 'max_num_batched_tokens': budget,
                           'max_model_len': 4096, 'tensor_parallel_size': 1}.items():
            require(config[key] == value, f'{path.name}: config differs for {key}')
        if mode == CANDIDATE:
            require(config['async_scheduling'] is True and config['max_inflight_batches'] == 2 and
                    config['cuda_graph_mode'] == 'auto', f'{path.name}: wrong candidate configuration')
        else:
            require(config['enforce_eager'] is False, f'{path.name}: baseline Graph disabled')
        return data

    numerical = []
    for budget in env['budgets']:
        for mode in ('baseline', CANDIDATE):
            path = args.results / f'correctness_{mode}_b{budget}.json'
            data = worker(path, mode, budget, 'correctness')
            require(data['passed_local_assertions'] is True and data['logit_rows'] == 32,
                    f'{path.name}: local correctness incomplete')
            tensor_path = path.with_suffix('.pt')
            require(tensor_path.is_file() and tensor_path.stat().st_size > 0, f'{path.name}: logits tensor missing')
            artifacts[str(tensor_path)] = digest(tensor_path)
        path = args.results / f'comparison_{CANDIDATE}_b{budget}.json'
        data = load(path)
        require(data['candidate'] == CANDIDATE and data['budget'] == budget and
                data['environment_sha256'] == digest(env_path) and data['thresholds'] == thresholds,
                f'{path.name}: comparison provenance differs')
        require(data['forced_tokens_equal'] is True and data['passed'] is True,
                f'{path.name}: numerical validation failed')
        rows = data['rows']
        require(len(rows) == 32 and {tuple(r['key']) for r in rows} == {(i, j) for i in range(4) for j in range(8)},
                f'{path.name}: incomplete/duplicate logits rows')
        for row in rows:
            require(all(math.isfinite(row[k]) for k in ('relative_l2', 'cosine', 'max_abs')),
                    f'{path.name}: nonfinite logits comparison')
            require(0 <= row['relative_l2'] <= .03 and .999 <= row['cosine'] <= 1.000001 and row['max_abs'] >= 0,
                    f'{path.name}: tolerance exceeded')
        calculated = {'logit_rows': len(rows), 'max_relative_l2': max(r['relative_l2'] for r in rows),
                      'min_cosine': min(r['cosine'] for r in rows),
                      'top1_equal_rows': sum(r['top1_equal'] for r in rows),
                      'bitwise_equal_rows': sum(r['bitwise_equal'] for r in rows)}
        for key, value in calculated.items():
            close(data[key], value, f'{path.name}/{key}')
        numerical.append({'budget': budget, **calculated, 'passed_tolerance': True})

    raw_summary = load(args.results / 'v1_summary.json')
    summaries = {(row['budget'], row['workload'], row['candidate']): row for row in raw_summary}
    require(len(summaries) == len(raw_summary) == 10, 'expected exactly ten distinct summary rows')
    calculated_rows, work_differences, files = [], [], []
    checked_values, pair_count = 0, 0
    for budget in env['budgets']:
        pair = {'baseline': {}, CANDIDATE: {}}
        for mode, order in (('baseline', 'ab'), (CANDIDATE, 'ab'), (CANDIDATE, 'ba'), ('baseline', 'ba')):
            path = args.results / f'benchmark_{CANDIDATE}_{mode}_b{budget}_{order}.json'
            data = worker(path, mode, budget, 'benchmark')
            require(len(data['cases']) == 15, f'{path.name}: incomplete observation count')
            local_keys = set()
            for case in data['cases']:
                key = (order, case['name'], case['rep'])
                require(key not in pair[mode] and key not in local_keys, f'{path.name}: duplicate case')
                require(case['name'] in WORKLOADS and case['rep'] in range(3), f'{path.name}: unexpected case')
                signature = audit_case(case, f'{path.name}/{key}')
                require(case['async_depth'] == (0 if mode == 'baseline' else 2), f'{path.name}/{key}: wrong async depth')
                pair[mode][key] = (case, signature)
                local_keys.add(key)
            require(local_keys == {(order, name, rep) for name in WORKLOADS for rep in range(3)},
                    f'{path.name}: cases do not cover workloads/repetitions')
            files.append({'budget': budget, 'mode': mode, 'order': order, 'file': path.name,
                          'complete': True, 'cases': len(local_keys)})
        require(set(pair['baseline']) == set(pair[CANDIDATE]), 'paired case keys differ')
        for key in sorted(pair['baseline']):
            base, left = pair['baseline'][key]
            new, right = pair[CANDIDATE][key]
            require(base['spec'] == new['spec'] and left == right and
                    base['window_output_tokens'] == new['window_output_tokens'], f'{budget}/{key}: paired work differs')
            pair_count += 1
            work_differences.append({'budget': budget, 'order': key[0], 'workload': key[1], 'rep': key[2],
                                    'resources': {metric: {'baseline': base[metric], 'candidate': new[metric],
                                                'difference': new[metric] - base[metric]} for metric in RESOURCES}})
        for name in WORKLOADS:
            stored = summaries[(budget, name, CANDIDATE)]
            keys = sorted(key for key in pair['baseline'] if key[1] == name)
            require(len(keys) == 6, 'each side must have six observations')
            row = {'budget': budget, 'workload': name, 'candidate': CANDIDATE}
            for mode, label in (('baseline', 'baseline'), (CANDIDATE, 'optimized')):
                require(stored[label]['observations'] == 6, 'summary observation count differs')
                require(set(stored[label]['median']) == set(METRICS), 'summary metric set differs')
                result = {'observations': 6, 'median': {}, 'range': {}}
                for metric in METRICS:
                    values = [pair[mode][key][0][metric] for key in keys if pair[mode][key][0][metric] is not None]
                    result['median'][metric] = statistics.median(values) if values else None
                    result['range'][metric] = [min(values), max(values)] if values else None
                    for kind in ('median', 'range'):
                        close(stored[label][kind][metric], result[kind][metric], f'{budget}/{name}/{label}/{kind}/{metric}')
                        checked_values += 1
                row[label] = result
            row['change_pct'] = {}
            for metric in METRICS:
                relative = change(row['baseline']['median'][metric], row['optimized']['median'][metric])
                close(stored['change_pct'][metric], relative, f'{budget}/{name}/change/{metric}')
                row['change_pct'][metric] = relative
                values = [change(pair['baseline'][key][0][metric], pair[CANDIDATE][key][0][metric]) for key in keys]
                values = [v for v in values if v is not None]
                for kind, value in {'values': values, 'mean': average(values),
                                    'median': statistics.median(values) if values else None}.items():
                    close(stored['paired_change_pct'][metric][kind], value, f'{budget}/{name}/paired/{metric}/{kind}')
                    checked_values += 1
                checked_values += 1
            calculated_rows.append(row)
    return {'schema_version': 1, 'passed': True, 'errors': [], 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
            'baseline_commit': COMMIT, 'candidate': CANDIDATE, 'original_files_unchanged': 26,
            'source_hashes_match': True, 'harness_hashes_match': True, 'processes': files,
            'performance_processes': len(files), 'observations': len(files) * 15, 'paired_cases': pair_count,
            'observations_per_side_per_workload': 6, 'checked_summary_values': checked_values,
            'computed_token_counts_equal': all(item['resources'][metric]['difference'] == 0
                for item in work_differences for metric in ('prefill_tokens', 'decode_tokens')),
            'numerical_checks': numerical, 'summaries': calculated_rows,
            'resource_differences': work_differences, 'artifact_sha256': artifacts,
            'audit_script_sha256': digest(__file__), 'environment': env,
            'limitations': ['ABBA roles and file completeness are verified against the hashed driver; no independent GPU process trace.',
                           'Numerical audit recomputes row statistics from comparison JSON and hashes the saved tensor files; it does not rerun the model.',
                           'Different forward counts and computed token totals are permitted and reported as strategy costs.',
                           'TTFT, TPOT and gaps are engine-step accepted outputs, not HTTP streaming; no significance claim.']}


def report(result):
    env = result['environment']
    lines = ['# 原始 nano-vLLM 与新版异步调度对比', '',
             f'基线：原始 nano-vLLM `{COMMIT}`；新版：统一调度、增量 KV 与 `async auto`。',
             '基线已支持分块 Prefill，但每步仅执行 Prefill 或 Decode。本次不是上一版混合单 forward 基线。', '',
             f"环境：{env['gpu']} / Qwen3-0.6B / BF16 / PyTorch {env['packages']['torch']}。",
             '两版均为 64 个 KV block（每块 256 token）、最多 16 请求；预算 512 / 1024。',
             '混合负载先让 4 条请求进入 Decode，再注入 1 / 4 / 9 条 3000-token Prompt；每条已有请求共生成 48 token，新混合请求各生成 16 token。',
             '纯 Prefill 为 300 / 500 / 1000-token Prompt，各只生成 1 token；纯 Decode 只有上述 4 条已有请求。含已有请求的负载从各请求首个输出后开始测量。',
             '每组独立进程按原始→新版→新版→原始执行，各进程每类负载先预热、再测 3 次；每侧 6 次取中位数。',
             '共 8 个性能进程、120 次观测、60 对匹配负载。性能基线仅由测试脚本固定 KV 容量，原始 26 文件字节不变。', '',
             '表中为原始 → 新版；吞吐只计输出 token/s，延迟单位 ms，变化为两侧中位数之比。所有负收益保留。', '',
             '| 预算 | 负载 | 输出吞吐 | 吞吐变化 | 新请求 TTFT | 已有请求 TPOT | 已有请求 max gap 均值 |',
             '|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.2f}'
    def pair(row, metric):
        return f"{number(row['baseline']['median'][metric])} → {number(row['optimized']['median'][metric])}"
    for row in result['summaries']:
        lines.append(f"| {row['budget']} | {LABELS[row['workload']]} | {pair(row, 'output_tok_s')} | "
                     f"{row['change_pct']['output_tok_s']:+.2f}% | {pair(row, 'ttft_mean_ms')} | "
                     f"{pair(row, 'existing_tpot_ms')} | {pair(row, 'existing_max_gap_mean_ms')} |")
    regressions = [f"预算 {row['budget']}、{LABELS[row['workload']]}场景 TTFT 增加 {row['change_pct']['ttft_mean_ms']:.2f}%"
                   for row in result['summaries']
                   if row['change_pct']['ttft_mean_ms'] is not None and row['change_pct']['ttft_mean_ms'] > 0]
    if regressions:
        lines += ['', '保留的退化：' + '；'.join(regressions) + '。不将这些观测解释为统计显著差异。']
    lines += ['', '数值验证采用相同后续 token，对齐每条请求的输出位置；沿用相对 L2 ≤ 3%、cosine ≥ 0.999 门槛。', '']
    for item in result['numerical_checks']:
        lines.append(f"- 预算 {item['budget']}：{item['logit_rows']} 行完整 logits 容差通过；最大相对 L2 "
                     f"{item['max_relative_l2'] * 100:.4f}%，最低 cosine {item['min_cosine']:.9f}，"
                     f"top-1 一致 {item['top1_equal_rows']}/{item['logit_rows']}，逐位一致 "
                     f"{item['bitwise_equal_rows']}/{item['logit_rows']}。")
    lines += ['', '**此处是容差通过，不代表全部 logits 逐位一致或全部 argmax 一致。**', '',
              '执行量与资源峰值（六次观测中位数）：', '',
              '| 预算 | 负载 | forward 次数 | 实际 Prefill token | 实际 Decode token | 在用 KV block 峰值 | 调度 CPU ms |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in result['summaries']:
        lines.append(f"| {row['budget']} | {LABELS[row['workload']]} | {pair(row, 'forward_count')} | "
                     f"{pair(row, 'prefill_tokens')} | {pair(row, 'decode_tokens')} | "
                     f"{pair(row, 'kv_peak_blocks')} | {pair(row, 'scheduler_ms')} |")
    work_note = ('本次全部 60 对观测的实际 Prefill / Decode 计算 token 数也相同；forward 次数可因调度不同。'
                 if result['computed_token_counts_equal'] else
                 '本次实际计算 token 数存在差异，逐对差值保留在审计明细；forward 次数可因调度不同。')
    lines += ['', '两版请求、Prompt 长度、目标输出数和测量窗口输出数逐对一致。' + work_note,
              '异步深度为 2；在用 KV block 峰值不是进程显存峰值，KV tensor 仍预分配。',
              'TPOT 从测量前的首个输出计算；max gap 先取每条已有请求最大相邻输出间隔，再求请求均值，包含跨测量起点的间隔。',
              '时间来自 engine.step() 接收结果，不含 HTTP/网络。本机受控负载结果不等于线上效果，也不宣称统计显著。',
              '审计核对源码/测量脚本哈希、ABBA 文件完整性、逐请求事件、配对工作量以及全部汇总中位数与百分比；数值误差统计由比较 JSON 重算，保存的 tensor 文件另记录哈希。',
              '[独立审计明细](upstream_audit.json) · [测量汇总](v1_summary.json) · [实验来源](upstream_provenance.json)', '',
              '在项目根目录、相同 Linux / WSL CUDA 环境下复现（替换模型路径，使用新的输出目录）：', '',
              '```bash',
              'python scripts/compare_v1.py --action all --model /path/to/Qwen3-0.6B \\',
              '  --baseline-source ../nano-vllm-inference-optimization/src/nano-vllm-upstream \\',
              '  --modes v1_async_auto --budgets 512 1024 --kv-blocks 64 --max-num-seqs 16 \\',
              '  --reps 3 --output-dir results/upstream-reproduction',
              '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--upstream-manifest', type=Path, required=True)
    parser.add_argument('--upstream-source', type=Path, required=True)
    parser.add_argument('--v1-source', type=Path, default=REPO / 'src')
    args = parser.parse_args()
    for name in ('results', 'upstream_manifest', 'upstream_source', 'v1_source'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = audit(args)
        markdown = report(result)
        code = 0
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        result = {'passed': False, 'errors': [f'{type(exc).__name__}: {exc}'], 'baseline_commit': COMMIT,
                  'audited_at_utc': datetime.now(timezone.utc).isoformat()}
        markdown = '# 原始 nano-vLLM 对照审计未通过\n\n' + result['errors'][0] + '\n'
        code = 1
    if args.results.is_dir():
        (args.results / 'upstream_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (args.results / 'upstream_comparison.md').write_text(markdown, encoding='utf-8')
    if code:
        print('AUDIT FAILED:', result['errors'][0], file=sys.stderr)
    else:
        print(f"AUDIT PASSED: {result['performance_processes']} processes, {result['observations']} observations, "
              f"{result['paired_cases']} pairs, {result['checked_summary_values']} summary values; "
              f"original {result['original_files_unchanged']} files unchanged")
    return code


if __name__ == '__main__':
    raise SystemExit(main())
