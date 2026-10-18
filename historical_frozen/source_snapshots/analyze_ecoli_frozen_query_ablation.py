"""Validate saved ablation predictions and summarize every fixed configuration."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from run_ecoli_frozen_query_ablation import (
    HERE, ROOT, OUT, DRUGS, THRESHOLDS, SEEDS, BUDGETS, TOL,
    read, write, sha, stem, original, raw_labels, check_roles, check_trace,
)


def independent_metrics(y, p, baseline):
    # Deliberately recompute counts directly from artifact arrays.
    resistant = y == 1
    susceptible = y == 0
    squared = np.square(p - y)
    return dict(n=int(y.size), resistant=int(resistant.sum()), susceptible=int(susceptible.sum()),
                brier=float(np.sum(squared) / y.size),
                brier_R=float(np.sum(squared[resistant]) / resistant.sum()) if resistant.any() else None,
                brier_S=float(np.sum(squared[susceptible]) / susceptible.sum()) if susceptible.any() else None,
                false_susceptible=int(np.count_nonzero((p < .5) & resistant)),
                false_susceptible_rate=float(np.count_nonzero((p < .5) & resistant) / resistant.sum()) if resistant.any() else None,
                false_resistant=int(np.count_nonzero((p >= .5) & susceptible)),
                false_resistant_rate=float(np.count_nonzero((p >= .5) & susceptible) / susceptible.sum()) if susceptible.any() else None,
                classification_fixed=int(np.count_nonzero(((baseline >= .5) != y) & ((p >= .5) == y))),
                classification_harmed=int(np.count_nonzero(((baseline >= .5) == y) & ((p >= .5) != y))),
                resistant_fixed=int(np.count_nonzero(resistant & (baseline < .5) & (p >= .5))),
                resistant_harmed=int(np.count_nonzero(resistant & (baseline >= .5) & (p < .5))))


def main():
    manifest = read(OUT / 'manifest.json')
    result_path = OUT / 'results.json'
    results = read(result_path)
    assert manifest['status'] == 'complete' and manifest['results_sha256'] == sha(result_path)
    binding = results['binding']
    assert binding == manifest['binding']
    assert binding['followup_code_sha256'] == sha(HERE / 'run_ecoli_frozen_query_ablation.py')
    assert binding['protocol_sha256'] == sha(HERE / 'FROZEN_QUERY_ABLATION_PROTOCOL.md')
    assert binding['original_validation_sha256'] == sha(HERE / 'ecoli_full_validation.json')
    full_validation = read(HERE / 'ecoli_full_validation.json')
    assert len(full_validation['predictions']) == 360
    for artifact in full_validation['predictions']:
        assert sha(artifact['path']) == artifact['sha256']
    for name, h in binding['formal_code_hashes'].items():
        assert sha(HERE / name) == h
    for name, h in binding['input_hashes'].items():
        assert sha(ROOT / name) == h
    for p, h in binding['original_artifact_hashes'].items():
        assert sha(p) == h
    assert len(results['artifacts']) == 180
    assert all(lib['num_threads'] == 1 for lib in manifest['numerical_libraries'])
    for artifact in results['artifacts']:
        assert sha(artifact['npz']) == artifact['npz_sha256']
        assert sha(artifact['json']) == artifact['json_sha256']
        js = read(artifact['json'])
        assert js['binding'] == binding and js['prediction_sha256'] == artifact['npz_sha256']
    indexed = {(r['drug'], r['threshold'], r['seed'], r['budget'], r['strategy']): r for r in results['metrics']}
    ci = {(r['drug'], r['threshold'], r['seed'], r['budget']): r for r in results['contrasts']}
    assert len(indexed) == len(results['metrics']) == 720
    assert len(ci) == len(results['contrasts']) == 360
    for artifact in results['artifacts']:
        for row in read(artifact['json'])['metrics']:
            key=(row['drug'],row['threshold'],row['seed'],row['budget'],row['strategy'])
            assert row == indexed[key]
    labels = raw_labels()
    max_metric_error = max_replay_error = max_identity_error = 0.
    checked = 0

    def compare(saved, actual):
        nonlocal max_metric_error
        for key, value in actual.items():
            if value is None:
                assert saved[key] is None
            else:
                error = abs(saved[key] - value)
                max_metric_error = max(max_metric_error, error)
                assert error <= TOL, (key, saved[key], value)
        weighted = sum(actual['brier_' + cls] * actual[n] for cls, n in (('R', 'resistant'), ('S', 'susceptible')) if actual[n]) / actual['n']
        assert abs(weighted - actual['brier']) <= TOL

    for drug in DRUGS:
        for t in THRESHOLDS:
            for seed in SEEDS:
                arrays, traces = {}, {}
                for strategy in ('decision_diagnostic', 'random_gene'):
                    p = original(drug, t, strategy, seed)
                    arrays[strategy] = np.load(str(p) + '.npz', allow_pickle=False)
                    traces[strategy] = read(str(p) + '.json')
                    check_roles(arrays[strategy], drug, labels)
                    check_trace(traces[strategy], arrays[strategy], drug, labels)
                ra, da = arrays['random_gene'], arrays['decision_diagnostic']
                for key in ('labels', 'audit_labels', 'source_samples', 'query_pool_samples', 'audit_samples', 'evaluation_samples',
                            'source_clusters', 'query_pool_clusters', 'audit_clusters', 'evaluation_clusters'):
                    assert np.array_equal(ra[key], da[key])
                y, base = ra['labels'], ra['p_b0']
                fitted = {}
                for strategy in ('diagnostic_path_gene', 'random_path_gene_replay'):
                    fitted[strategy] = np.load(str(OUT / stem(drug, t, strategy, seed)) + '.npz', allow_pickle=False)
                    a = fitted[strategy]
                    assert np.array_equal(a['labels'], y) and np.array_equal(a['evaluation_samples'], ra['evaluation_samples'])
                    assert np.array_equal(a['evaluation_clusters'], ra['evaluation_clusters'])
                    assert np.array_equal(a['audit_samples'], ra['audit_samples']) and np.array_equal(a['audit_labels'], ra['audit_labels'])
                    assert np.max(np.abs(a['p_b0'] - base)) <= TOL
                    for b in BUDGETS:
                        r = indexed[(drug, t, seed, b, strategy)]
                        assert r['total_target_assay_cost'] == b + len(ra['audit_samples'])
                        assert r['source_n'] == len(ra['source_samples'])
                        trace_strategy = 'decision_diagnostic' if strategy == 'diagnostic_path_gene' else 'random_gene'
                        expected = next(x for x in traces[trace_strategy]['results'] if x['budget'] == b)
                        assert r['queried_samples'] == expected['queried_samples']
                        p = a[f'p_b{b}']
                        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
                        audit_p = a[f'audit_p_b{b}']
                        assert len(audit_p) == len(ra['audit_samples'])
                        assert np.all(np.isfinite(audit_p)) and np.all((audit_p >= 0) & (audit_p <= 1))
                        compare(r, independent_metrics(y, p, base))
                        assert r['resistant_fixed'] - r['resistant_harmed'] == int(((y==1) & (base<.5)).sum()) - r['false_susceptible']
                        assert r['classification_fixed'] - r['classification_harmed'] == int(((base>=.5)!=y).sum()) - int(((p>=.5)!=y).sum())
                        if strategy == 'random_path_gene_replay':
                            for key in (f'p_b{b}', f'audit_p_b{b}'):
                                error = float(np.max(np.abs(a[key] - ra[key]))) if len(a[key]) else 0.
                                max_replay_error = max(max_replay_error, error)
                                assert error <= TOL
                for b in BUDGETS:
                    c = ci[(drug, t, seed, b)]
                    assert c['total_target_assay_cost'] == b + len(ra['audit_samples'])
                    predictions = dict(diagnostic=da[f'p_b{b}'], fixed_path_gene=fitted['diagnostic_path_gene'][f'p_b{b}'], random_gene=ra[f'p_b{b}'])
                    for name, p in predictions.items():
                        compare(c['metrics'][name], independent_metrics(y, p, base))
                    d, g, r = [float(np.mean((predictions[name] - y) ** 2)) for name in ('diagnostic', 'fixed_path_gene', 'random_gene')]
                    for key, actual in (('query_delta', g-r), ('action_delta', d-g), ('total_delta', d-r)):
                        assert abs(c[key] - actual) <= TOL
                    error = abs(c['query_delta'] + c['action_delta'] - c['total_delta'])
                    max_identity_error = max(max_identity_error, error)
                    assert error <= TOL
                    expected = next(x for x in traces['decision_diagnostic']['results'] if x['budget'] == b)
                    assert c['original_action'] == expected['action']
                    assert abs(d - expected['brier']) <= TOL
                    checked += 1
                for a in list(arrays.values()) + list(fitted.values()):
                    a.close()

    summaries = []
    for drug in DRUGS:
        for t in THRESHOLDS:
            for b in BUDGETS:
                group = [ci[(drug, t, seed, b)] for seed in SEEDS]
                entry = dict(drug=drug, threshold=t, budget=b, seeds=len(group),
                             mean_total_assay_cost=float(np.mean([c['total_target_assay_cost'] for c in group])),
                             actions=dict(Counter(c['original_action'] for c in group)))
                for key in ('query_delta', 'action_delta', 'total_delta'):
                    entry[key] = float(np.mean([c[key] for c in group]))
                    entry[key + '_better_seeds'] = sum(c[key] < -1e-12 for c in group)
                    entry[key + '_worse_seeds'] = sum(c[key] > 1e-12 for c in group)
                entry['metrics'] = {}
                for name in ('diagnostic', 'fixed_path_gene', 'random_gene'):
                    entry['metrics'][name] = {}
                    for metric in group[0]['metrics'][name]:
                        values = [c['metrics'][name][metric] for c in group if c['metrics'][name][metric] is not None]
                        entry['metrics'][name][metric] = float(np.mean(values)) if values else None
                        if metric in ('brier_R', 'brier_S', 'false_susceptible_rate', 'false_resistant_rate'):
                            entry['metrics'][name][metric + '_defined_seeds'] = len(values)
                summaries.append(entry)
    validation = dict(status='passed', model_trajectories=180, model_metric_rows=720,
                      configuration_budget_contrasts=checked, configurations=90,
                      max_metric_error=max_metric_error, max_random_replay_error=max_replay_error,
                      max_decomposition_identity_error=max_identity_error,
                      coverage=True, raw_AST_match=True, query_prefix_and_cost_match=True,
                      sample_and_component_isolation=True, original_hashes_unchanged=True,
                      original_prediction_files_checked=360, numerical_threads_match=True,
                      results_sha256=sha(result_path), validator_sha256=sha(Path(__file__)),
                      protocol_sha256=binding['protocol_sha256'])
    write(HERE / 'ecoli_frozen_query_validation.json', validation)
    write(HERE / 'ecoli_frozen_query_analysis.json', dict(designation=results['designation'], summaries=summaries,
                                                       validation=validation, original_results= str(result_path)))
    fields = ['drug','threshold','budget','query_delta','action_delta','total_delta',
              'query_delta_better_seeds','action_delta_better_seeds','total_delta_better_seeds','mean_total_assay_cost']
    with (HERE / 'ecoli_frozen_query_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in summaries)
    lines = ['# 冻结查询路径消融：选样还是修复？', '',
             '这是已查看正式结果之后追加的探索性诊断，不是新盲态验证。原有正式方法、数据与结果保持原哈希。', '',
             '完成180条模型轨迹、720个模型预算指标行、360组逐配置对照。随机查询重放的最大概率误差：'
             f'{max_replay_error:.3g}；指标重算最大误差：{max_metric_error:.3g}；分解恒等式误差：{max_identity_error:.3g}。', '',
             '初次重放因线程环境不同未达到固定误差标准，其部分产物已单独保留并排除。'
             '随后按原流水线单线程环境重算，误差标准仍为1e-10；没有更改正式模型或原有结果。', '',
             'G：在诊断查询路径上重训同一个序列sketch逻辑回归；D：原诊断控制器；R：原随机查询逻辑回归。'
             '选样差=G−R，动作差=D−G，总差=D−R；Brier差值小于0为改善。'
             '这只是在固定已生成路径上的分解，不是因果机制识别。', '',
             '下表沿用既有主组件阈值0.99，展示预算25的五划分描述均值；全部阈值、预算和逐配置结果保存在JSON/CSV中。', '',
             '| 药物 | 选样差 G−R | 动作差 D−G | 总差 D−R | 总差改善划分/5 |',
             '|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['threshold'] == .99 and r['budget'] == 25:
            lines.append(f"| {r['drug']} | {r['query_delta']:+.5f} | {r['action_delta']:+.5f} | {r['total_delta']:+.5f} | {r['total_delta_better_seeds']}/5 |")
    lines += ['', '## 全预算与两阈值的等药物描述均值', '',
              '| 组件阈值 | 查询预算 | 选样差 | 动作差 | 总差 |', '|---|---:|---:|---:|---:|']
    for t in THRESHOLDS:
        for b in BUDGETS:
            group = [s for s in summaries if s['threshold'] == t and s['budget'] == b]
            means = [np.mean([s[k] for s in group]) for k in ('query_delta','action_delta','total_delta')]
            lines.append(f'| {t} | {b} | {means[0]:+.5f} | {means[1]:+.5f} | {means[2]:+.5f} |')
    lines += ['', '## 类别损失和假敏感错误并非同一个终点', '',
              '固定0.99/预算25；类别损失为D减R的五划分描述均值；假敏感为各划分计数的均值。'
              'D/G/R的评估样本相同。符号不能转化为独立临床事件或临床安全结论。', '',
              '| 药物 | 耐药类Brier差 | 敏感类Brier差 | 耐药n范围 | 耐药类有效划分 | D假敏感数（率） | G假敏感数（率） | R假敏感数（率） |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['threshold'] == .99 and s['budget'] == 25:
            d,g,r = [s['metrics'][name] for name in ('diagnostic','fixed_path_gene','random_gene')]
            def difference(key):
                return f"{d[key]-r[key]:+.5f}" if d[key] is not None and r[key] is not None else '未定义'
            def false_sensitive(m):
                rate = f"{m['false_susceptible_rate']:.1%}" if m['false_susceptible_rate'] is not None else '未定义'
                return f"{m['false_susceptible']:.1f}（{rate}）"
            resistant_n = [ci[(s['drug'],.99,seed,25)]['metrics']['diagnostic']['resistant'] for seed in SEEDS]
            lines.append(f"| {s['drug']} | {difference('brier_R')} | {difference('brier_S')} | "
                         f"{min(resistant_n)}–{max(resistant_n)} | {d['brier_R_defined_seeds']}/5 | "
                         f"{false_sensitive(d)} | {false_sensitive(g)} | {false_sensitive(r)} |")
    main_summary = [s for s in summaries if s['threshold']==.99 and s['budget']==25]
    offset = [s['drug'] for s in main_summary if s['query_delta'] < -1e-12 and s['action_delta'] > 1e-12]
    opposite = [s['drug'] for s in main_summary if s['query_delta'] > 1e-12 and s['action_delta'] < -1e-12]
    lines += ['', '## 下一步应解决的具体问题', '',
              '在固定主展示条件下，选样差改善而动作差恶化的药物：'+(', '.join(offset) if offset else '无')+'。'
              '选样差恶化而动作差改善的药物：'+(', '.join(opposite) if opposite else '无')+'。'
              '这里的符号是五划分均值上的探索性分类，不表示统计显著、原因已被识别或可直接部署。', '',
              '下一方法版本应把“是否执行修复”与“获取哪些标签”分开验证，并明确耐药类损失、'
              '假敏感风险和标注成本；不能只优化总体Brier后宣称可靠性已经恢复。'
              '任何新动作准则须在新版本协议中冻结，使用独立确认数据验证，而不能回头挑本轮最好的动作。']
    lines += ['', '## 解释边界', '',
              '收益和损害都保留；评估阈值固定0.5，不为降低假敏感率重新调阈值。耐药/敏感类Brier、'
              '假敏感计数以及相对预算0的修正/新引入错误均在JSON中。无耐药样本的类指标为null，并记录有效划分数。', '',
              '五个划分重复使用同一队列，平均计数不是独立患者事件，也不用于临床显著性判断。'
              '成本仍计入原有审计样本，审计标签不参与重训。', '',
              '当前特征是序列sketch而非AMR注释；组件不是严格ANI/SNP。MIC及历史断点协调、'
              '可靠患者/实验室隔离、AMR专用强基线复现和新盲态外部确认仍未完成。'
              '这些计算不证明临床安全、稳定SOTA或NCS/Nature录用。', '',
              f'独立产物目录：`{OUT}`。原始完整对照见 `ECOLI_FULL_RESULTS.md`；'
              '本消融不替代原有ExtraTrees及不确定性查询对照。']
    (HERE / 'ECOLI_FROZEN_QUERY_ABLATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(validation)


if __name__ == '__main__':
    main()
