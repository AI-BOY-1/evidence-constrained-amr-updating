"""Independent replay of B-cohort labels, partitions, coverage and probabilities."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

ROOT = Path(__file__).resolve().parent
RUN = ncs_paths.data_path('experiments/msdeepamr_reproduction/reexecution_B_v3')
REPO = RUN.parent / 'repo'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def independently_measure(y, prob):
    pred = prob > .5
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    positive, negative = prob[y == 1], prob[y == 0]
    roc = float(((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) /
                (len(positive) * len(negative))) if len(positive) and len(negative) else None
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    precision, recall, _ = precision_recall_curve(y, prob)
    return {'n': len(y), 'R': int(y.sum()), 'S': int((y == 0).sum()),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUROC': roc,
            'AUPRC_trapezoid': float(auc(recall, precision)), 'average_precision': float(average_precision_score(y, prob)),
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'MCC': (tp * tn - fp * fn) / denom if denom else 0., 'Brier': float(np.mean((prob - y) ** 2))}


def main():
    assert not (RUN / 'execution.lock').exists(), 'Cannot certify a running pipeline'
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for path, digest in manifest['binding'].items(): assert sha(path) == digest, path
    rows = {p.name: json.loads(p.read_text()) for p in RUN.glob('*_seed*_*.json')}
    expected = {f'{species}_seed{seed}_{strategy}.json' for species in ('e_coli', 'k_pneumoniae', 's_aureus')
                for seed in range(10) for strategy in ('source_only', 'local_only', 'frozen_features', 'all_layers')}
    assert not set(rows) - expected
    checks = []
    replayed = []
    full_predictions = {}
    for species in ('e_coli', 'k_pneumoniae', 's_aureus'):
        drug = 'Oxacillin' if species == 's_aureus' else 'Ceftriaxone'
        df = pd.read_csv(REPO / 'datasets/driams_b' / f'{species}_driams_b_bin3_2000_20000Da.csv')
        features = [str(i) for i in range(2000, 20000, 3)]
        valid = df[drug].notna() & df[features].notna().all(axis=1)
        frame = df[valid]
        y = frame[drug].to_numpy(int)
        assert not frame.code.duplicated().any(), 'Duplicate target sample code requires prespecified grouping'
        full_path = RUN / f'{species}_source_full.json'
        if full_path.exists():
            full = json.loads(full_path.read_text())
            assert full['binding'] == manifest['binding']
            assert full['original_rows'] == frame.index.tolist()
            assert full['codes'] == frame.code.astype(str).tolist()
            assert full['y'] == y.tolist()
            assert full['excluded_missing_rows'] == df.index[~valid].tolist()
            full_prob = np.asarray(full['probability'])
            assert len(full_prob) == len(y) and np.isfinite(full_prob).all()
            assert ((0 <= full_prob) & (full_prob <= 1)).all()
            for key, value in independently_measure(y, full_prob).items():
                assert np.isclose(value, full['metrics'][key], rtol=0, atol=1e-12)
            full_predictions[full_path.name] = sha(full_path)
            checks.append(full_path.name + ':original_labels_and_metric_replay')
        for seed in range(10):
            tr, ts = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed, stratify=y)
            for strategy in ('source_only', 'local_only', 'frozen_features', 'all_layers'):
                name = f'{species}_seed{seed}_{strategy}.json'
                if name not in rows: continue
                row = rows[name]
                assert row['binding'] == manifest['binding']
                assert row['train_rows'] == frame.index[tr].tolist() and row['test_rows'] == frame.index[ts].tolist()
                assert row['train_codes'] == frame.code.iloc[tr].astype(str).tolist()
                assert row['test_codes'] == frame.code.iloc[ts].astype(str).tolist()
                assert row['train_y'] == y[tr].tolist() and row['test_y'] == y[ts].tolist()
                assert row['excluded_missing_rows'] == df.index[~valid].tolist()
                assert not set(row['train_codes']) & set(row['test_codes'])
                checks.append(name + ':split_and_original_labels')
                if row['status'] != 'complete': continue
                prob = np.array(row['probability'])
                assert len(prob) == len(ts) and np.isfinite(prob).all() and ((0 <= prob) & (prob <= 1)).all()
                m = independently_measure(y[ts], prob)
                for key, value in m.items():
                    assert np.isclose(value, row['metrics'][key], rtol=0, atol=1e-12), (name, key)
                if strategy == 'source_only':
                    assert row['epochs_run'] == 0
                    assert full_path.exists()
                    assert np.allclose(prob, full_prob[ts], rtol=0, atol=1e-5)
                elif strategy in ('frozen_features', 'all_layers'): assert row['epochs_run'] == 10
                else: assert 1 <= row['epochs_run'] <= 100
                replayed.append(name)
                checks.append(name + ':metric_replay')
    failures = [name for name, row in rows.items() if row['status'] == 'failed']
    complete = set(rows) == expected and len(replayed) == 120 and len(full_predictions) == 3 and not failures
    out = {'status': 'passed' if complete else 'partial_replay_passed_full_coverage_incomplete',
           'expected_configs': 120, 'observed_configs': len(rows), 'metrics_replayed': len(replayed),
           'failed_configs': failures, 'missing_configs': sorted(expected - set(rows)),
           'checks': checks, 'manifest_sha256': sha(RUN / 'manifest.json'),
           'result_sha256': {name: sha(RUN / name) for name in rows},
           'full_source_prediction_sha256': full_predictions,
           'exact_paper_reproduction': False,
           'limitations': ['Only B external tasks/local ten splits/migrated runtime, not full original paper reproduction.',
                           'Repeated split scores are correlated; their spread is not a patient-level confidence interval.',
                           'No newly collected labels or independent holder-controlled blind validation.']}
    (ROOT / 'msdeepamr_reexecution_validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('status', 'expected_configs', 'observed_configs', 'metrics_replayed', 'failed_configs')}))


if __name__ == '__main__': main()
