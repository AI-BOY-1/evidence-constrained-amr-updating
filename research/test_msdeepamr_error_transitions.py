"""Synthetic correctness fixtures in temporary storage, never scientific outputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc, confusion_matrix
import analyze_msdeepamr_error_transitions as analysis
import validate_msdeepamr_error_transitions as validation


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    tn, fp, fn, tp = confusion_matrix(y, p > .5, labels=[0, 1]).ravel()
    return {'AUROC': float(roc_auc_score(y, p)), 'AUPRC_trapezoid': float(auc(recall, precision)),
            'Brier': float(brier_score_loss(y, p)), 'FN': int(fn), 'FP': int(fp)}


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('Corrupt fixture should be rejected')


def main():
    with tempfile.TemporaryDirectory(prefix='msdeepamr_transition_fixture_') as directory:
        root = Path(directory)
        run = root / 'synthetic_run'
        run.mkdir()
        (root / 'MSDEEPAMR_ERROR_TRANSITION_PLAN.md').write_text('SYNTHETIC FIXTURE; NOT SCIENTIFIC RESULTS')
        (run / 'manifest.json').write_text('{"status":"synthetic_fixture"}')
        analysis.ROOT = validation.ROOT = root
        analysis.RUN = validation.RUN = run
        y = np.array([1, 1, 0, 0])
        probabilities = {'source_only': [.1, .9, .9, .1], 'local_only': [.9, .1, .1, .9],
                         'frozen_features': [.9, .9, .1, .1], 'all_layers': [.1, .1, .9, .9]}
        hashes = {}
        for species in ('e_coli', 'k_pneumoniae', 's_aureus'):
            for seed in range(10):
                for strategy, prob in probabilities.items():
                    row = {'status': 'complete', 'species': species, 'seed': seed, 'strategy': strategy,
                           'drug': 'synthetic_fixture', 'train_rows': [4, 5], 'train_codes': ['t4', 't5'],
                           'train_y': [0, 1], 'test_rows': [0, 1, 2, 3], 'test_codes': ['r0', 'r1', 's0', 's1'],
                           'test_y': y.tolist(), 'excluded_missing_rows': [], 'probability': prob,
                           'metrics': metrics(y, np.asarray(prob))}
                    p = run / f'{species}_seed{seed}_{strategy}.json'
                    p.write_text(json.dumps(row))
                    hashes[p.name] = sha(p)
        v = {'status': 'passed', 'metrics_replayed': 120, 'manifest_sha256': sha(run / 'manifest.json'),
             'result_sha256': hashes, 'synthetic_fixture': True}
        (root / 'msdeepamr_reexecution_validation.json').write_text(json.dumps(v))
        analysis.main()
        validation.main()
        result_path = root / 'msdeepamr_error_transitions.json'
        result = json.loads(result_path.read_text())
        pair = next(r for r in result['pairings'] if r['species'] == 'e_coli' and r['seed'] == 0
                    and r['control'] == 'source_only' and r['treatment'] == 'local_only')
        assert pair['corrected_errors'] == pair['introduced_errors'] == 2
        assert pair['corrected_false_sensitive'] == pair['introduced_false_sensitive'] == 1
        assert pair['corrected_false_resistant'] == pair['introduced_false_resistant'] == 1
        assert result['sample_pair_rows'] == 720
        result['summary'][0]['delta_FN_mean'] += 1
        result_path.write_text(json.dumps(result))
        rejected(validation.main)
        result['summary'][0]['delta_FN_mean'] -= 1
        result['sample_transitions'][0]['y'] = 1 - result['sample_transitions'][0]['y']
        result_path.write_text(json.dumps(result))
        rejected(validation.main)
        result['sample_transitions'][0]['y'] = 1 - result['sample_transitions'][0]['y']
        result['sample_transitions'].append(result['sample_transitions'][0].copy())
        result_path.write_text(json.dumps(result))
        rejected(validation.main)
    print(json.dumps({'status': 'passed', 'fixtures': 4,
                      'checks': ['known_directional_transitions', 'corrupt_summary_rejected',
                                 'corrupt_original_label_rejected', 'duplicate_sample_pair_rejected'],
                      'scientific_results': False, 'temporary_fixture_storage_only': True}))


if __name__ == '__main__':
    main()
