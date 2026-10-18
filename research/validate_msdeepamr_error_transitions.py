"""Independent sample-wise replay of the complete post hoc transition report."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import itertools
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
RUN = ncs_paths.data_path('experiments/msdeepamr_reproduction/reexecution_B_v3')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path = ROOT / 'msdeepamr_error_transitions.json'
    result = json.loads(path.read_text())
    assert result['status'] == 'complete_posthoc_descriptive_analysis' and result['posthoc'] is True
    assert result['threshold'] == '>0.5'
    formal = json.loads((ROOT / 'msdeepamr_reexecution_validation.json').read_text())
    assert formal['status'] == 'passed' and formal['manifest_sha256'] == sha(RUN / 'manifest.json')
    assert result['result_sha256'] == formal['result_sha256']
    for p, digest in result['input_sha256'].items():
        assert sha(Path(p)) == digest
    records = {}
    for name, digest in result['result_sha256'].items():
        p = RUN / name
        assert sha(p) == digest
        row = json.loads(p.read_text())
        records[row['species'], row['seed'], row['strategy']] = row
    strategies = ('source_only', 'local_only', 'frozen_features', 'all_layers')
    expected = {(s, seed, a, b) for s in ('e_coli', 'k_pneumoniae', 's_aureus') for seed in range(10)
                for a, b in itertools.combinations(strategies, 2)}
    pairings = {(r['species'], r['seed'], r['control'], r['treatment']): r for r in result['pairings']}
    assert len(pairings) == len(result['pairings']) == result['paired_rows'] == 180
    assert set(pairings) == expected
    observed_samples = {}
    for r in result['sample_transitions']:
        key = r['species'], r['seed'], r['control'], r['treatment'], r['original_row']
        assert key not in observed_samples
        observed_samples[key] = r
    total_samples = 0
    for key, r in pairings.items():
        species, seed, control, treatment = key
        a, b = records[species, seed, control], records[species, seed, treatment]
        assert all(a[k] == b[k] for k in ('test_rows', 'test_codes', 'test_y', 'train_rows', 'train_codes', 'train_y'))
        assert r['n'] == len(a['test_y']) and r['R'] == sum(a['test_y']) and r['S'] == r['n'] - r['R']
        counts = {k: 0 for k in ('corrected_errors', 'introduced_errors', 'both_correct', 'both_wrong',
                                'corrected_false_sensitive', 'introduced_false_sensitive',
                                'corrected_false_resistant', 'introduced_false_resistant')}
        for row_id, code, y, pa, pb in zip(a['test_rows'], a['test_codes'], a['test_y'], a['probability'], b['probability']):
            wrong_a, wrong_b = int(pa > .5) != y, int(pb > .5) != y
            transition = ('both_wrong' if wrong_a else 'both_correct') if wrong_a == wrong_b else ('corrected' if wrong_a else 'introduced')
            count_key = transition + '_errors' if transition in ('corrected', 'introduced') else transition
            counts[count_key] += 1
            if transition in ('corrected', 'introduced'):
                counts[transition + ('_false_sensitive' if y == 1 else '_false_resistant')] += 1
            sample_key = (*key, row_id)
            sample = observed_samples[sample_key]
            assert sample['code'] == code and sample['y'] == y
            assert sample['control_probability'] == pa and sample['treatment_probability'] == pb
            assert sample['control_wrong'] == wrong_a and sample['treatment_wrong'] == wrong_b
            assert sample['transition'] == transition
            total_samples += 1
        assert all(r[k] == v for k, v in counts.items())
        for label, metric in [('delta_AUROC', 'AUROC'), ('delta_AUPRC', 'AUPRC_trapezoid'),
                              ('delta_Brier', 'Brier'), ('delta_FN', 'FN'), ('delta_FP', 'FP')]:
            assert math.isclose(r[label], b['metrics'][metric] - a['metrics'][metric], rel_tol=0, abs_tol=1e-12)
        assert r['higher_AUROC_more_false_sensitive'] == (r['delta_AUROC'] > 0 and r['delta_FN'] > 0)
        assert r['lower_Brier_more_false_sensitive'] == (r['delta_Brier'] < 0 and r['delta_FN'] > 0)
    assert total_samples == len(observed_samples) == result['sample_pair_rows']
    expected_groups = {(s, a, b) for s in ('e_coli', 'k_pneumoniae', 's_aureus')
                       for a, b in itertools.combinations(strategies, 2)}
    summaries = {(r['species'], r['control'], r['treatment']): r for r in result['summary']}
    assert set(summaries) == expected_groups and len(summaries) == result['summary_rows'] == 18
    for (s, a, b), summary in summaries.items():
        rows = [pairings[s, seed, a, b] for seed in range(10)]
        assert summary['splits'] == 10 and summary['test_R_range'] == [min(r['R'] for r in rows), max(r['R'] for r in rows)]
        for k in ('delta_AUROC', 'delta_AUPRC', 'delta_Brier', 'delta_FN', 'delta_FP'):
            assert math.isclose(summary[k + '_mean'], math.fsum(r[k] for r in rows) / 10, rel_tol=0, abs_tol=1e-12)
        for k in ('corrected_errors', 'introduced_errors', 'corrected_false_sensitive', 'introduced_false_sensitive',
                  'corrected_false_resistant', 'introduced_false_resistant'):
            assert summary[k + '_sum_over_overlapping_splits'] == sum(r[k] for r in rows)
        for k in ('higher_AUROC_more_false_sensitive', 'lower_Brier_more_false_sensitive'):
            assert summary[k + '_splits'] == sum(r[k] for r in rows)
    out = {'status': 'passed', 'paired_rows_replayed': 180, 'summary_rows_replayed': 18,
           'sample_pair_rows_replayed': total_samples, 'result_sha256': sha(path),
           'report_sha256': sha(ROOT / 'MSDEEPAMR_ERROR_TRANSITIONS.md'), 'posthoc': True,
           'input_sha256': {str(path): sha(path), str(Path(__file__).resolve()): sha(Path(__file__).resolve())}}
    (ROOT / 'msdeepamr_error_transitions_validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('status', 'paired_rows_replayed', 'summary_rows_replayed', 'sample_pair_rows_replayed')}))


if __name__ == '__main__':
    main()
