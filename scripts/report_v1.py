"""Read-only audit of paired benchmark JSON, with optional Chinese report output."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
NAMES = {'all_decode': '纯 Decode', 'all_prefill': '纯 Prefill',
         'long_decode': '长 Prompt + Decode', 'many_long': '多长 Prompt + Decode',
         'capacity_pressure': '等待需求超过 KV 容量'}
METRICS = ('output_tok_s', 'wall_ms', 'ttft_mean_ms', 'ttft_p95_ms', 'tpot_mean_ms',
           'existing_tpot_ms', 'existing_gap_mean_ms', 'max_token_gap_ms', 'gap_p95_ms',
           'existing_max_gap_mean_ms', 'request_max_gap_mean_ms', 'kv_peak_blocks', 'kv_peak_mib',
           'prefill_tokens', 'decode_tokens', 'forward_count', 'full_graph_replays',
           'piecewise_graph_replays', 'piecewise_segment_replays', 'cuda_graph_replays',
           'gpu_feedback_tokens', 'async_depth', 'mixed_forward_count', 'eager_count',
           'scheduler_ms', 'cpu_step_ms', 'steps', 'empty_steps')


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def change(before, after):
    return 100 * (after / before - 1) if before and after is not None else None


def same_number(a, b):
    if a is None or b is None:
        return a is b
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-5)


def request_signature(case):
    return sorted((row['logical'], row['existing'], row['prompt_tokens'],
                   row['total_output_tokens'], row['window_output_tokens']) for row in case['requests'])


def audit_case(case, blocks, mode):
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def number(actual, expected, message):
        check(same_number(actual, expected), message)

    spec, requests = case['spec'], case['requests']
    expected_outputs = spec['prime_count'] * (spec['prime_output'] - 1) + len(spec['new_lengths']) * spec['new_output']
    expected_requests = spec['prime_count'] + len(spec['new_lengths'])
    check(len(requests) == expected_requests, '请求数量不匹配')
    check(sorted(row['logical'] for row in requests) == list(range(expected_requests)), '请求编号缺失或重复')
    check(case['window_output_tokens'] == expected_outputs, '测量窗口输出 token 数不匹配')
    check(sum(row['window_output_tokens'] for row in requests) == expected_outputs, '逐请求输出计数不匹配')
    check(case['wall_ms'] > 0, '总耗时必须为正')
    number(case['output_tok_s'], expected_outputs / (case['wall_ms'] / 1000), '吞吐计算不匹配')
    for row in requests:
        logical = row['logical']
        old = logical < spec['prime_count']
        expected = spec['prime_output'] if old else spec['new_output']
        prompt_length = spec['prime_length'] if old else spec['new_lengths'][logical - spec['prime_count']]
        times = row['output_times_from_window_ms']
        check(row['existing'] == old, f'请求{logical}的新旧状态错误')
        check(row['prompt_tokens'] == prompt_length, f'请求{logical}的Prompt长度错误')
        check(row['total_output_tokens'] == expected == len(times), f'请求{logical}的完整输出计数错误')
        check(row['window_output_tokens'] == expected - int(old), f'请求{logical}的窗口输出计数错误')
        check(sum(t >= 0 for t in times) == row['window_output_tokens'], f'请求{logical}的时间戳计数错误')
        check(all(b >= a for a, b in zip(times, times[1:])), f'请求{logical}的时间戳不单调')
        gaps = [b - a for a, b in zip(times, times[1:]) if b >= 0]
        check(len(gaps) == len(row['gap_ms']) and all(same_number(a, b) for a, b in zip(gaps, row['gap_ms'])),
              f'请求{logical}的gap序列不匹配')
        number(row['tpot_ms'], (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
               f'请求{logical}的TPOT不匹配')
        number(row['average_gap_ms'], mean(gaps), f'请求{logical}的平均gap不匹配')
        number(row['max_token_gap_ms'], max(gaps) if gaps else None, f'请求{logical}的最大gap不匹配')
    old = [row for row in requests if row['existing']]
    new = [row for row in requests if not row['existing']]
    number(case['existing_tpot_ms'], mean([row['tpot_ms'] for row in old]), '已有请求TPOT汇总错误')
    number(case['existing_max_gap_mean_ms'], mean([row['max_token_gap_ms'] for row in old]), '已有请求max gap均值错误')
    number(case['request_max_gap_mean_ms'], mean([row['max_token_gap_ms'] for row in requests]), '请求max gap均值错误')
    number(case['ttft_mean_ms'], mean([row['ttft_ms'] for row in new]), '新请求TTFT汇总错误')
    check(0 < case['kv_peak_blocks'] <= blocks, 'KV逻辑占用超容量或为空')
    check(case['forward_count'] == case['eager_count'] + case['full_graph_replays'] + case['piecewise_graph_replays'],
          'forward数与实际执行路径计数不一致')
    check(case['cuda_graph_replays'] == case['full_graph_replays'] + case['piecewise_segment_replays'],
          '真实Graph replay总数不匹配')
    check(0 <= case['mixed_forward_count'] <= case['forward_count'], '混合forward计数不合法')
    if case['piecewise_graph_replays']:
        check(case['piecewise_segment_replays'] % case['piecewise_graph_replays'] == 0,
              '每次分段forward的Graph数量不一致')
    else:
        check(case['piecewise_segment_replays'] == 0, '无分段forward但存在segment replay')
    if mode == 'baseline':
        check(case['async_depth'] == 0, 'baseline不应记录异步深度')
    else:
        expected_depth = 2 if mode == 'v1_async_auto' else 1
        check(case['async_depth'] == expected_depth, '实际在途深度与模式不一致')
    if mode == 'v1_sync_eager':
        check(case['cuda_graph_replays'] == 0, '同步eager不应重放Graph')
    minimum_work = sum(spec['new_lengths']) + expected_outputs - len(spec['new_lengths'])
    check(case['prefill_tokens'] + case['decode_tokens'] >= minimum_work, '模型计算token数少于必需输入')
    return errors


def audit_results(directory, allow_incomplete=False):
    environment_path = directory / 'v1_environment.json'
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    environment_hash = hashlib.sha256(environment_path.read_bytes()).hexdigest()
    errors, warnings, pending, pairs, source_changes = [], [], [], [], []
    cases_by_run, graph_evidence = {}, []
    repetitions = environment['repetitions_per_process']
    expected_cases = {(name, rep) for name in environment['workloads'] for rep in range(repetitions)}
    for label, root in (('baseline', ROOT / 'baseline'), ('v1', ROOT / 'src')):
        for relative, expected in environment['source_sha256'][label].items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                source_changes.append(str(path))
    for name, expected in environment['harness_sha256'].items():
        path = ROOT / 'scripts' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            source_changes.append(str(path))
    if source_changes:
        errors.append('当前源码或冻结harness与实验指纹不一致')
    for budget in environment['budgets']:
        for candidate in environment['modes']:
            check_path = directory / f'comparison_{candidate}_b{budget}.json'
            if not check_path.is_file():
                errors.append(f'缺少数值验证: {check_path.name}')
            else:
                check = json.loads(check_path.read_text(encoding='utf-8'))
                if not check.get('passed') or check.get('environment_sha256') != environment_hash:
                    errors.append(f'数值验证未通过或指纹不符: {check_path.name}')
            for mode in ('baseline', candidate):
                for order in ('ab', 'ba'):
                    path = directory / f'benchmark_{candidate}_{mode}_b{budget}_{order}.json'
                    if not path.is_file():
                        pending.append(path.name)
                        continue
                    try:
                        data = json.loads(path.read_text(encoding='utf-8'))
                    except json.JSONDecodeError:
                        if allow_incomplete:
                            pending.append(path.name)
                            continue
                        raise
                    if not data.get('complete'):
                        pending.append(path.name)
                        continue
                    expected_meta = {'mode': mode, 'budget': budget, 'blocks': environment['kv_blocks'],
                        'kv_blocks': environment['kv_blocks'], 'max_num_seqs': environment['max_num_seqs'],
                        'max_model_len': environment['max_model_len'], 'reps': repetitions,
                        'gpu': environment['gpu'], 'dtype': 'torch.bfloat16', 'model': environment['model']}
                    for key, value in expected_meta.items():
                        if data.get(key) != value:
                            errors.append(f'{path.name}: {key}不匹配')
                    cases = {(case['name'], case['rep']): case for case in data['cases']}
                    if set(cases) != expected_cases or len(cases) != len(data['cases']):
                        errors.append(f'{path.name}: workload或重复数不完整/重复')
                    for key, case in cases.items():
                        errors.extend(f'{path.name}/{key}: {message}'
                                      for message in audit_case(case, environment['kv_blocks'], mode))
                    capture = data['graph_capture']
                    if capture['full_capture_errors'] or capture['piecewise_capture_errors']:
                        warnings.append(f'{path.name}: 存在Graph捕获失败，需结合实际replay解释')
                    if any(case['full_graph_replays'] for case in cases.values()) and not capture['full_buckets']:
                        errors.append(f'{path.name}: 存在full replay但没有成功捕获bucket')
                    if any(case['piecewise_graph_replays'] for case in cases.values()) and not capture['piecewise_buckets']:
                        errors.append(f'{path.name}: 存在piecewise replay但没有成功捕获bucket')
                    cases_by_run[(budget, candidate, mode, order)] = cases
                    graph_evidence.append({'file': path.name, 'mode': mode, 'budget': budget,
                        'full_replays': sum(case['full_graph_replays'] for case in cases.values()),
                        'piecewise_forwards': sum(case['piecewise_graph_replays'] for case in cases.values()),
                        'segment_replays': sum(case['piecewise_segment_replays'] for case in cases.values()),
                        'gpu_feedback_tokens': sum(case['gpu_feedback_tokens'] for case in cases.values()),
                        'depths': sorted(set(case['async_depth'] for case in cases.values())), 'capture': capture})
            for order in ('ab', 'ba'):
                a = cases_by_run.get((budget, candidate, 'baseline', order))
                b = cases_by_run.get((budget, candidate, candidate, order))
                if a is None or b is None:
                    continue
                for key in expected_cases & set(a) & set(b):
                    x, y = a[key], b[key]
                    if x['spec'] != y['spec'] or request_signature(x) != request_signature(y):
                        errors.append(f'{candidate}/{budget}/{order}/{key}: 配对请求或输出数量不一致')
                    pairs.append({'candidate': candidate, 'budget': budget, 'order': order,
                        'workload': key[0], 'rep': key[1],
                        'output_tokens': [x['window_output_tokens'], y['window_output_tokens']],
                        'prefill_tokens': [x['prefill_tokens'], y['prefill_tokens']],
                        'decode_tokens': [x['decode_tokens'], y['decode_tokens']],
                        'forward_count': [x['forward_count'], y['forward_count']],
                        'kv_peak_blocks': [x['kv_peak_blocks'], y['kv_peak_blocks']],
                        'extra_prefill_above_prompt': [x['prefill_tokens'] - sum(x['spec']['new_lengths']),
                                                       y['prefill_tokens'] - sum(y['spec']['new_lengths'])]})
    observations = sum(len(cases) for cases in cases_by_run.values())
    expected_observations = len(environment['budgets']) * len(environment['modes']) * 4 * len(expected_cases)
    summaries = []
    for budget in environment['budgets']:
        for candidate in environment['modes']:
            if any((budget, candidate, mode, order) not in cases_by_run
                   for mode in ('baseline', candidate) for order in ('ab', 'ba')):
                continue
            for workload in environment['workloads']:
                row = {'candidate': candidate, 'budget': budget, 'workload': workload}
                for mode, label in (('baseline', 'baseline'), (candidate, 'optimized')):
                    cases = [cases_by_run[(budget, candidate, mode, order)][(workload, rep)]
                             for order in ('ab', 'ba') for rep in range(repetitions)]
                    row[label] = {metric: median([case[metric] for case in cases]) for metric in METRICS}
                    row[label]['observations'] = len(cases)
                row['change_pct'] = {metric: change(row['baseline'][metric], row['optimized'][metric])
                                     for metric in METRICS}
                summaries.append(row)
    summary_path = directory / 'v1_summary.json'
    summary_validation = {'present': summary_path.is_file(), 'rows_checked': 0, 'medians_checked': 0}
    if summary_path.is_file():
        stored = json.loads(summary_path.read_text(encoding='utf-8'))
        by_key = {(row['budget'], row['candidate'], row['workload']): row for row in stored}
        expected_keys = {(row['budget'], row['candidate'], row['workload']) for row in summaries}
        if len(by_key) != len(stored) or set(by_key) != expected_keys:
            errors.append('v1_summary.json的候选/预算/负载行不完整或重复')
        for row in summaries:
            key = (row['budget'], row['candidate'], row['workload'])
            target = by_key.get(key)
            if target is None:
                continue
            summary_validation['rows_checked'] += 1
            for label in ('baseline', 'optimized'):
                if target[label]['observations'] != row[label]['observations']:
                    errors.append(f'{key}/{label}: summary观测数不匹配')
                for metric, value in target[label]['median'].items():
                    if metric not in row[label] or not same_number(value, row[label][metric]):
                        errors.append(f'{key}/{label}/{metric}: summary中位数与独立计算不匹配')
                    summary_validation['medians_checked'] += 1
            for metric, value in target['change_pct'].items():
                if metric not in row['change_pct'] or not same_number(value, row['change_pct'][metric]):
                    errors.append(f'{key}/{metric}: summary变化百分比与独立计算不匹配')
    elif observations == expected_observations:
        pending.append('v1_summary.json')
    result = {'complete': not pending and observations == expected_observations,
        'valid': not errors, 'completed_observations': observations,
        'expected_observations': expected_observations, 'completed_processes': len(cases_by_run),
        'environment_sha256': environment_hash, 'errors': errors, 'warnings': warnings,
        'pending': pending, 'source_changes': source_changes, 'pairs': pairs,
        'summary_validation': summary_validation,
        'graph_evidence': graph_evidence, 'summaries': summaries, 'environment': environment}
    return result


def format_number(value):
    return '—' if value is None else f'{value:.2f}'


def arrow(row, key):
    a, b = row['baseline'][key], row['optimized'][key]
    if key in ('kv_peak_blocks', 'prefill_tokens', 'forward_count') and a is not None and b is not None:
        return f'{a:g} → {b:g}'
    return '—' if a is None and b is None else f'{format_number(a)} → {format_number(b)}'


def render_markdown(audit):
    if not audit['complete'] or not audit['valid']:
        raise ValueError('only complete, valid experiments can produce the final README table')
    env = audit['environment']
    n = env['observations_per_mode_per_pair']
    rows = audit['summaries']
    main = [row for row in rows if row['candidate'] == 'v1_async_auto']
    lines = [f"{env['gpu']} / Qwen3-0.6B / BF16，固定 {env['kv_blocks']} 个 KV block、"
             f"最多 {env['max_num_seqs']} 个请求。每个候选分别执行独立进程 ABBA 对照，"
             f"每类负载独立预热，表中为 {n} 次观测的中位数。箭头表示配对基线 → 新版本。", '',
        '| Token预算 | 负载 | 输出吞吐 tok/s | 吞吐变化 | 新请求TTFT ms | 已有Decode TPOT ms | 已有Decode max gap均值 ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in main:
        lines.append(f"| {row['budget']} | {NAMES[row['workload']]} | {arrow(row, 'output_tok_s')} | "
            f"{row['change_pct']['output_tok_s']:+.2f}% | {arrow(row, 'ttft_mean_ms')} | "
            f"{arrow(row, 'existing_tpot_ms')} | {arrow(row, 'existing_max_gap_mean_ms')} |")
    regressions = []
    for row in main:
        for key, label in (('ttft_mean_ms', 'TTFT'), ('existing_tpot_ms', '已有Decode TPOT'),
                           ('existing_max_gap_mean_ms', '已有Decode max gap均值')):
            delta = row['change_pct'][key]
            if delta is not None and delta > 0:
                regressions.append(f"预算{row['budget']}的{NAMES[row['workload']]}：{label} +{delta:.2f}%")
    if regressions:
        lines += ['', '保留的延迟退化：' + '；'.join(regressions) + '。']
    lines += ['', '同步消融：各列均相对该模式自己的 ABBA 配对基线，数值为输出吞吐变化；负值完整保留。', '',
        '| Token预算 | 负载 | 同步 eager | 同步 auto | 异步 auto |', '| --- | --- | ---: | ---: | ---: |']
    for budget in env['budgets']:
        for workload in env['workloads']:
            selected = {row['candidate']: row for row in rows if row['budget'] == budget and row['workload'] == workload}
            values = [f"{selected[mode]['change_pct']['output_tok_s']:+.2f}%" if mode in selected else '—'
                      for mode in ('v1_sync_eager', 'v1_sync_auto', 'v1_async_auto')]
            lines.append(f"| {budget} | {NAMES[workload]} | " + ' | '.join(values) + ' |')
    lines += ['', '同步 eager 关闭了 Graph，因此该列同时包含新调度与执行模式差异，不能单独解释为调度收益。', '',
        '| Token预算 | 负载 | KV峰值 block（基线 → 异步） | Prefill计算token（基线 → 异步） | Forward次数（基线 → 异步） |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in main:
        lines.append(f"| {row['budget']} | {NAMES[row['workload']]} | {arrow(row, 'kv_peak_blocks')} | "
                     f"{arrow(row, 'prefill_tokens')} | {arrow(row, 'forward_count')} |")
    pressure = [pair for pair in audit['pairs'] if pair['workload'] == 'capacity_pressure']
    if pressure and max(max(pair['kv_peak_blocks']) for pair in pressure) < env['kv_blocks']:
        lines += ['', '容量场景的等待总需求超过缓存容量，但本次实际占用未触及容量上限；'
                  '不能据此声称该性能负载触发了真实抢占。抢占正确性应引用独立边界测试。']
    if all(pair['prefill_tokens'][0] == pair['prefill_tokens'][1]
           and pair['decode_tokens'][0] == pair['decode_tokens'][1]
           and pair['forward_count'][0] == pair['forward_count'][1] for pair in audit['pairs']):
        lines += ['', '全部配对观测的 Prefill/Decode 计算token数和 forward 次数一致。'
                  '本次性能实验比较的是相同计算量下的执行效率。']
    if all(pair['extra_prefill_above_prompt'] == [0, 0] for pair in audit['pairs']):
        lines += ['各负载未观察到额外 Prefill 重算；缓存峰值达到容量也不等于发生了抢占。']
    async_runs = [item for item in audit['graph_evidence'] if item['mode'] == 'v1_async_auto']
    if async_runs:
        lines += ['', '异步组全部正式观测中，实际执行 '
            f"{sum(item['full_replays'] for item in async_runs)} 次完整 Graph replay、"
            f"{sum(item['piecewise_forwards'] for item in async_runs)} 次分段 forward"
            f"（{sum(item['segment_replays'] for item in async_runs)} 次分段 Graph replay），"
            f"GPU 直接反馈 {sum(item['gpu_feedback_tokens'] for item in async_runs)} 个输入 token，"
            f"最大在途 batch 数为 {max(max(item['depths']) for item in async_runs)}。"]
    lines += ['', f"审计通过：{audit['completed_processes']} 个性能进程，{audit['completed_observations']} 条正式观测；"
              '配对请求与输出数量一致，执行路径计数与实际 Graph replay 计数相符。', '',
        '统计口径：token时间戳取 Engine.step 回收真实输出时刻，不含 HTTP/网络。TTFT只统计新请求；'
        '已有请求的TPOT从注入长Prompt前的首个输出起计算；max gap先对每个已有Decode请求取最大相邻输出间隔，再求请求均值，'
        '包含跨越测量起点的间隔。纯Prefill每请求只生成1个token，TPOT不适用。'
        'KV峰值是逻辑占用的block数，启动时预分配的KV显存容量在各模式间相同。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--markdown-output', type=Path)
    args = parser.parse_args()
    audit = audit_results(args.results, args.allow_incomplete)
    if args.audit_output:
        args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ('complete', 'valid', 'completed_observations',
          'expected_observations', 'completed_processes', 'errors', 'warnings', 'pending')}, ensure_ascii=False))
    if not audit['valid'] or (not audit['complete'] and not args.allow_incomplete):
        raise SystemExit(1)
    if args.audit_only:
        return
    markdown = render_markdown(audit)
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding='utf-8')
    else:
        print(markdown)


if __name__ == '__main__':
    main()
