"""Summarize every fixed strategy and paired split, retaining adverse outcomes."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from validate_msdeepamr_reexecution import sha

ROOT = Path(__file__).resolve().parent
RUN = ncs_paths.data_path('experiments/msdeepamr_reproduction/reexecution_B_v3')


def main():
    validation = json.loads((ROOT / 'msdeepamr_reexecution_validation.json').read_text())
    assert validation['status'] == 'passed', 'Full 120-config validation is required before scientific summary'
    assert validation['manifest_sha256'] == sha(RUN / 'manifest.json')
    rows = []
    for name, digest in validation['result_sha256'].items():
        assert sha(RUN / name) == digest
        rows.append(json.loads((RUN / name).read_text()))
    groups = defaultdict(list)
    for row in rows: groups[row['species'], row['strategy']].append(row)
    summary = []
    metric_keys = ('AUROC', 'AUPRC_trapezoid', 'average_precision', 'Brier', 'F1', 'MCC')
    for (species, strategy), group in sorted(groups.items()):
        item = {'species': species, 'drug': group[0]['drug'], 'strategy': strategy, 'splits': len(group),
                'test_R_range': [min(r['metrics']['R'] for r in group), max(r['metrics']['R'] for r in group)],
                'test_S_range': [min(r['metrics']['S'] for r in group), max(r['metrics']['S'] for r in group)]}
        for key in metric_keys:
            values = [r['metrics'][key] for r in group if r['metrics'][key] is not None]
            item[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)),
                         'min': float(min(values)), 'max': float(max(values)), 'defined': len(values)}
        item['FN_sum_across_overlapping_tests'] = sum(r['metrics']['FN'] for r in group)
        item['FP_sum_across_overlapping_tests'] = sum(r['metrics']['FP'] for r in group)
        summary.append(item)
    by_key = {(r['species'], r['seed'], r['strategy']): r for r in rows}
    paired = []
    for species in ('e_coli', 'k_pneumoniae', 's_aureus'):
        for seed in range(10):
            for strategy in ('local_only', 'frozen_features', 'all_layers'):
                a, b = by_key[species, seed, strategy], by_key[species, seed, 'source_only']
                assert all(a[k] == b[k] for k in ('train_rows', 'test_rows', 'train_codes', 'test_codes', 'train_y', 'test_y'))
                paired.append({'species': species, 'seed': seed, 'strategy': strategy, 'control': 'source_only',
                               'deltas': {key: a['metrics'][key] - b['metrics'][key] for key in metric_keys},
                               'FN_delta': a['metrics']['FN'] - b['metrics']['FN'],
                               'FP_delta': a['metrics']['FP'] - b['metrics']['FP']})
    result = {'status': 'complete', 'configs': len(rows), 'summary_rows': len(summary), 'paired_rows': len(paired),
              'summary': summary, 'paired': paired, 'failed_configs': validation['failed_configs'],
              'input_sha256': {str(ROOT / 'msdeepamr_reexecution_validation.json'): sha(ROOT / 'msdeepamr_reexecution_validation.json'),
                               str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
              'exact_paper_reproduction': False,
              'limitations': [
                  'Ten repeated splits share target isolates; mean/std are split variability, not independent-patient inference.',
                  'Test R denominators can be very small, especially K. pneumoniae ceftriaxone and S. aureus oxacillin.',
                  'Only B site supplied; C/D and original ten split identities not reproduced.',
                  'Migrated runtime and paper/notebook hyperparameter disagreement are disclosed.',
                  'Public retrospective reexecution does not close the holder-blind confirmation gap.',
              ]}
    out = ROOT / 'msdeepamr_reexecution_analysis.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# MSDeepAMR DRIAMS-B 开放资产重执行', '',
             f"固定配置：{len(rows)}；完整汇总：{len(summary)}；同测试样本配对：{len(paired)}。", '',
             '三个任务、十个本地固定拆分、四策略；完整原论文精确复现=false。', '',
             '|任务|策略|拆分|测试R范围|AUROC均值±SD|AUPRC均值±SD|Brier均值±SD|', '|---|---|---:|---|---|---|---|']
    for r in summary:
        values = [f"{r[k]['mean']:.4f} ± {r[k]['std']:.4f}" for k in ('AUROC', 'AUPRC_trapezoid', 'Brier')]
        lines.append(f"|{r['species']} {r['drug']}|{r['strategy']}|{r['splits']}|{r['test_R_range']}|" + '|'.join(values) + '|')
    lines += ['', '## 全部配对（策略−source_only；Brier负值较好）', '',
              '|任务|seed|策略|ΔAUROC|ΔAUPRC|ΔBrier|ΔFN|ΔFP|', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"|{r['species']}|{r['seed']}|{r['strategy']}|{r['deltas']['AUROC']:.4f}|{r['deltas']['AUPRC_trapezoid']:.4f}|{r['deltas']['Brier']:.4f}|{r['FN_delta']}|{r['FP_delta']}|")
    lines += ['', '## 限制', '', *('- ' + text for text in result['limitations'])]
    (ROOT / 'MSDEEPAMR_REEXECUTION_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'configs': len(rows), 'summary_rows': len(summary), 'paired_rows': len(paired)}))


if __name__ == '__main__': main()
