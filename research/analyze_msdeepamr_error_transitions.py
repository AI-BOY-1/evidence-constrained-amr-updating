"""All fixed-threshold pairwise error migrations, explicitly post hoc."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ncs_paths.data_path('experiments/msdeepamr_reproduction/reexecution_B_v3')
STRATEGIES = ('source_only', 'local_only', 'frozen_features', 'all_layers')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'execution.lock').exists(), 'Formal prediction execution still active'
    validation_path = ROOT / 'msdeepamr_reexecution_validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['status'] == 'passed' and validation['metrics_replayed'] == 120
    assert validation['manifest_sha256'] == sha(RUN / 'manifest.json')
    records = {}
    for name, digest in validation['result_sha256'].items():
        path = RUN / name
        assert sha(path) == digest
        row = json.loads(path.read_text())
        assert row['status'] == 'complete'
        records[row['species'], row['seed'], row['strategy']] = row
    paired, samples = [], []
    groups = defaultdict(list)
    for species in ('e_coli', 'k_pneumoniae', 's_aureus'):
        for seed in range(10):
            for control, treatment in itertools.combinations(STRATEGIES, 2):
                a, b = records[species, seed, control], records[species, seed, treatment]
                for key in ('train_rows', 'test_rows', 'train_codes', 'test_codes', 'train_y', 'test_y', 'excluded_missing_rows'):
                    assert a[key] == b[key], (species, seed, key)
                y = np.asarray(a['test_y'])
                pa, pb = np.asarray(a['probability']), np.asarray(b['probability'])
                wrong_a, wrong_b = (pa > .5) != y, (pb > .5) != y
                corrected = wrong_a & ~wrong_b
                introduced = ~wrong_a & wrong_b
                item = {'species': species, 'drug': a['drug'], 'seed': seed, 'control': control,
                        'treatment': treatment, 'n': len(y), 'R': int(y.sum()), 'S': int((y == 0).sum()),
                        'corrected_errors': int(corrected.sum()), 'introduced_errors': int(introduced.sum()),
                        'both_correct': int((~wrong_a & ~wrong_b).sum()), 'both_wrong': int((wrong_a & wrong_b).sum()),
                        'corrected_false_sensitive': int((corrected & (y == 1)).sum()),
                        'introduced_false_sensitive': int((introduced & (y == 1)).sum()),
                        'corrected_false_resistant': int((corrected & (y == 0)).sum()),
                        'introduced_false_resistant': int((introduced & (y == 0)).sum()),
                        'delta_AUROC': b['metrics']['AUROC'] - a['metrics']['AUROC'],
                        'delta_AUPRC': b['metrics']['AUPRC_trapezoid'] - a['metrics']['AUPRC_trapezoid'],
                        'delta_Brier': b['metrics']['Brier'] - a['metrics']['Brier'],
                        'delta_FN': b['metrics']['FN'] - a['metrics']['FN'],
                        'delta_FP': b['metrics']['FP'] - a['metrics']['FP']}
                assert sum(item[k] for k in ('corrected_errors', 'introduced_errors', 'both_correct', 'both_wrong')) == len(y)
                assert item['delta_FN'] == item['introduced_false_sensitive'] - item['corrected_false_sensitive']
                assert item['delta_FP'] == item['introduced_false_resistant'] - item['corrected_false_resistant']
                assert item['delta_FN'] + item['delta_FP'] == item['introduced_errors'] - item['corrected_errors']
                item['higher_AUROC_more_false_sensitive'] = item['delta_AUROC'] > 0 and item['delta_FN'] > 0
                item['lower_Brier_more_false_sensitive'] = item['delta_Brier'] < 0 and item['delta_FN'] > 0
                paired.append(item)
                groups[species, control, treatment].append(item)
                for i in range(len(y)):
                    samples.append({'species': species, 'seed': seed, 'control': control, 'treatment': treatment,
                                    'original_row': a['test_rows'][i], 'code': a['test_codes'][i], 'y': int(y[i]),
                                    'control_probability': float(pa[i]), 'treatment_probability': float(pb[i]),
                                    'control_wrong': bool(wrong_a[i]), 'treatment_wrong': bool(wrong_b[i]),
                                    'transition': 'corrected' if corrected[i] else 'introduced' if introduced[i]
                                                  else 'both_wrong' if wrong_a[i] else 'both_correct'})
    assert len(paired) == 180
    summary = []
    for (species, control, treatment), rows in sorted(groups.items()):
        assert len(rows) == 10
        item = {'species': species, 'control': control, 'treatment': treatment, 'splits': 10,
                'test_R_range': [min(r['R'] for r in rows), max(r['R'] for r in rows)]}
        for k in ('delta_AUROC', 'delta_AUPRC', 'delta_Brier', 'delta_FN', 'delta_FP'):
            item[k + '_mean'] = float(np.mean([r[k] for r in rows]))
        for k in ('corrected_errors', 'introduced_errors', 'corrected_false_sensitive', 'introduced_false_sensitive',
                  'corrected_false_resistant', 'introduced_false_resistant'):
            item[k + '_sum_over_overlapping_splits'] = sum(r[k] for r in rows)
        for k in ('higher_AUROC_more_false_sensitive', 'lower_Brier_more_false_sensitive'):
            item[k + '_splits'] = sum(r[k] for r in rows)
        summary.append(item)
    out = {'status': 'complete_posthoc_descriptive_analysis', 'created_utc': datetime.now(timezone.utc).isoformat(),
           'paired_rows': len(paired), 'summary_rows': len(summary), 'sample_pair_rows': len(samples),
           'pairings': paired, 'summary': summary, 'sample_transitions': samples,
           'input_sha256': {str(validation_path): sha(validation_path),
                            str(ROOT / 'MSDEEPAMR_ERROR_TRANSITION_PLAN.md'): sha(ROOT / 'MSDEEPAMR_ERROR_TRANSITION_PLAN.md'),
                            str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
           'result_sha256': validation['result_sha256'], 'threshold': '>0.5', 'posthoc': True,
           'limitations': ['All paired observations share a finite retrospective target cohort.',
                           'Repeated-split sums are not counts of unique patients or independent events.',
                           'R denominators are small; no causal or clinical safety conclusion.',
                           'This does not execute the project counterexample-repair strategy on spectra.']}
    path = ROOT / 'msdeepamr_error_transitions.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines = ['# MSDeepAMR固定阈值错误转移（事后完整分析）', '',
             f"180个同样本配对，18个完整组合汇总，逐样本配对行数{len(samples)}。阈值原样p>0.5，全部策略对照保留。", '',
             '后策略减前策略；Brier负值较好。累计纠正/引入是共享样本的重复拆分计数，不是独立患者事件。', '',
             '|任务|前策略|后策略|测试R|ΔAUROC均值|ΔBrier均值|纠正/引入总错|排序改善且FN增加拆分|Brier改善且FN增加拆分|',
             '|---|---|---|---|---:|---:|---|---:|---:|']
    for r in summary:
        lines.append(f"|{r['species']}|{r['control']}|{r['treatment']}|{r['test_R_range']}|{r['delta_AUROC_mean']:+.4f}|{r['delta_Brier_mean']:+.4f}|{r['corrected_errors_sum_over_overlapping_splits']}/{r['introduced_errors_sum_over_overlapping_splits']}|{r['higher_AUROC_more_false_sensitive_splits']}/10|{r['lower_Brier_more_false_sensitive_splits']}/10|")
    lines += ['', '## 全部配对', '', '|任务|seed|前策略|后策略|R|纠正错|新引入错|ΔFN|ΔFP|ΔAUROC|ΔBrier|',
              '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"|{r['species']}|{r['seed']}|{r['control']}|{r['treatment']}|{r['R']}|{r['corrected_errors']}|{r['introduced_errors']}|{r['delta_FN']}|{r['delta_FP']}|{r['delta_AUROC']:+.4f}|{r['delta_Brier']:+.4f}|")
    lines += ['', *('- ' + note for note in out['limitations'])]
    (ROOT / 'MSDEEPAMR_ERROR_TRANSITIONS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('status', 'paired_rows', 'summary_rows', 'sample_pair_rows')}))


if __name__ == '__main__':
    main()
