"""Run the frozen, plate-isolated own photo-to-distance experiment."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ncs_paths.data_path('experiments/combiant_public_v1')
SUB = BASE / 'replication_subset'
DIST = ('A', 'B', 'C', 'AB', 'BC', 'AC')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_features(path, side):
    with Image.open(path) as im:
        im = im.convert('RGB')
        width, height = im.size
        edge = min(width, height)
        left, top = (width - edge) // 2, (height - edge) // 2
        im = im.crop((left, top, left + edge, top + edge)).resize((side, side), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    radial = np.sqrt((np.indices((side, side))[0] - (side - 1) / 2) ** 2 +
                     (np.indices((side, side))[1] - (side - 1) / 2) ** 2) / side
    global_features = []
    for channel in (arr[..., 0], arr[..., 1], arr[..., 2], gray):
        global_features.extend([float(channel.mean()), float(channel.std()),
                                 float(np.quantile(channel, .1)), float(np.quantile(channel, .5)),
                                 float(np.quantile(channel, .9))])
    global_features.extend([float((gray * radial).mean()), float((gray * radial ** 2).mean())])
    return np.concatenate([arr.reshape(-1), np.asarray(global_features, dtype=np.float32)])


def metric_rows(y, pred, fold, method):
    errors = pred - y
    rows = []
    for j, distance in enumerate(DIST):
        finite = np.isfinite(errors[:, j])
        absolute = np.abs(errors[finite, j])
        rows.append({'fold': fold, 'method': method, 'distance': distance, 'n': len(y), 'finite_n': len(absolute),
            'bias_mm': float(errors[finite, j].mean()) if len(absolute) else None, 'mae_mm': float(absolute.mean()) if len(absolute) else None,
            'median_absolute_mm': float(np.median(absolute)) if len(absolute) else None,
            'p95_absolute_mm': float(np.quantile(absolute, .95)) if len(absolute) else None, 'max_absolute_mm': float(absolute.max()) if len(absolute) else None,
            'over_1mm': int((absolute > 1).sum()), 'over_2mm': int((absolute > 2).sum()),
            'over_3mm': int((absolute > 3).sum())})
    complete = np.isfinite(errors).all(axis=1)
    macro = np.abs(errors[complete]).mean(axis=1)
    rows.append({'fold': fold, 'method': method, 'distance': 'macro', 'n': len(y), 'finite_n': len(macro),
        'bias_mm': float(errors[complete].mean()), 'mae_mm': float(macro.mean()),
        'median_absolute_mm': float(np.median(macro)), 'p95_absolute_mm': float(np.quantile(macro, .95)),
        'max_absolute_mm': float(macro.max()), 'over_1mm': int((macro > 1).sum()),
        'over_2mm': int((macro > 2).sum()), 'over_3mm': int((macro > 3).sum())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('pilot', 'full'), default='full')
    args = parser.parse_args()
    OUT = BASE / ('own_vision_pilot_v1' if args.scope == 'pilot' else 'own_vision_v1')
    OUT.mkdir(exist_ok=True)
    pairing_path = BASE / 'pairing_audit.json'
    pairing = json.loads(pairing_path.read_text(encoding='utf-8'))
    acquisition = json.loads((BASE / 'image_acquisition_manifest.json').read_text(encoding='utf-8'))
    if args.scope == 'full' and (acquisition.get('status') != 'evaluation_inputs_acquired_pending_audit' or len(acquisition.get('members', {})) != 299):
        raise RuntimeError(f"Full input image acquisition is required before own experiment: {acquisition.get('status')} {len(acquisition.get('members', {}))}/299")
    snapshot = OUT / 'acquisition_snapshot.json'
    if snapshot.exists():
        acquisition = json.loads(snapshot.read_text(encoding='utf-8'))
    else:
        snapshot.write_text(json.dumps(acquisition, indent=2), encoding='utf-8')
    binding = {str(p): sha(p) for p in (pairing_path, snapshot, ROOT / 'COMBIANT_OWN_VISION_PROTOCOL.md', Path(__file__),
                  BASE / 'paired_distances.csv', SUB / 'CombiANT_raw_data/combiANT_user_measurements.csv', SUB / 'inference/output.csv')}
    if (OUT / 'results.json').exists():
        previous = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
        for path, value in previous['binding'].items():
            assert sha(Path(path)) == value, 'Changed completed experiment binding: ' + path
        for field, name in [('metrics_sha256', 'metrics.csv'), ('predictions_sha256', 'predictions.csv'),
                            ('review_sha256', 'review_controls.csv'), ('features_sha256', 'features_targets.npz')]:
            assert sha(OUT / name) == previous[field]
        print(json.dumps({'status': 'existing_complete_verified', 'scope': args.scope}))
        return
    rows = []
    with (BASE / 'paired_distances.csv').open(newline='', encoding='utf-8') as stream:
        table = list(csv.DictReader(stream))
    by_name = {name: [r for r in table if r['name'] == name] for name in {r['name'] for r in table}}
    for pair in pairing['pairs']:
        if pair['member'] not in acquisition['members']:
            if args.scope == 'pilot':
                continue
            raise RuntimeError('Paired input missing: ' + pair['member'])
        assert sha(SUB / pair['member']) == acquisition['members'][pair['member']]['sha256']
        binding[str(SUB / pair['member'])] = acquisition['members'][pair['member']]['sha256']
        distances = by_name[pair['name']]
        rows.append({**pair, 'target': [float(next(r['manual_mm'] for r in distances if r['distance'] == d) or 'nan') for d in DIST]})
    X32 = np.asarray([image_features(SUB / r['member'], 32) for r in rows], dtype=np.float32)
    X16 = np.asarray([image_features(SUB / r['member'], 16) for r in rows], dtype=np.float32)
    y = np.asarray([r['target'] for r in rows], dtype=np.float64)
    complete_target = np.isfinite(y).all(axis=1)
    groups = np.asarray([r['plate'] for r in rows])
    if args.scope == 'full':
        assert len(np.unique(groups)) == 98 and len(rows) == 279
    assert len(np.unique(groups)) >= 10
    splitter = GroupKFold(n_splits=5)
    results, predictions, review_rows, plans = [], [], [], []
    all_predictions = {}
    for fold, (train, test) in enumerate(splitter.split(X32, y, groups), 1):
        assert not set(groups[train]) & set(groups[test])
        plans.append({'fold': fold, 'train_names': [rows[i]['name'] for i in train], 'test_names': [rows[i]['name'] for i in test],
                      'train_plates': sorted(set(groups[train].tolist())), 'test_plates': sorted(set(groups[test].tolist()))})
        fit_train = train[complete_target[train]]
        mean_pred = np.repeat(np.nanmean(y[train], axis=0, keepdims=True), len(test), axis=0)
        sw = np.asarray([[float(next(r['software_mm'] for r in by_name[rows[i]['name']] if r['distance'] == d) or 'nan') for d in DIST] for i in test])
        for method, pred in [('train_plate_mean', mean_pred), ('published_software_reference', sw)]:
            results += metric_rows(y[test], pred, fold, method)
            all_predictions.setdefault(method, []).append((test, pred))
        for method, features, model in (
                ('own_ridge_32px', X32, Ridge(alpha=100.0)),
                ('own_ridge_16px', X16, Ridge(alpha=100.0)),
                ('own_extra_trees_32px', X32, ExtraTreesRegressor(n_estimators=128, min_samples_leaf=3,
                 max_features=1.0, random_state=7, n_jobs=1))):
            model.fit(features[fit_train], y[fit_train])
            pred = model.predict(features[test])
            results += metric_rows(y[test], pred, fold, method)
            all_predictions.setdefault(method, []).append((test, pred))
            uncertainty = np.std([tree.predict(features[test]) for tree in model.estimators_], axis=0).mean(axis=1) if method == 'own_extra_trees_32px' else np.zeros(len(test))
            for i, p, score in zip(test, pred, uncertainty):
                predictions.append({'fold': fold, 'method': method, 'name': rows[i]['name'], 'plate': rows[i]['plate'],
                    'user': rows[i]['user'], 'risk_score': float(score), **{f'target_{d}': float(y[i, j]) for j, d in enumerate(DIST)},
                    **{f'prediction_{d}': float(p[j]) for j, d in enumerate(DIST)}})
            if method == 'own_extra_trees_32px':
                test_plates = sorted(set(groups[test].tolist()))
                gray = X32[test, :3072].reshape(-1, 32, 32, 3).mean(axis=3)
                lap = gray[:, 2:, 1:-1] + gray[:, :-2, 1:-1] + gray[:, 1:-1, 2:] + gray[:, 1:-1, :-2] - 4 * gray[:, 1:-1, 1:-1]
                quality = -lap.var(axis=(1, 2))
                scores = {'uncertainty': {p: float(uncertainty[groups[test] == p].mean()) for p in test_plates},
                          'image_quality': {p: float(quality[groups[test] == p].mean()) for p in test_plates}}
                errors = np.abs(pred - y[test])
                for strategy in ('uncertainty', 'image_quality', 'random'):
                    for seed in range(100) if strategy == 'random' else [-1]:
                        ranked = list(np.random.default_rng(seed).permutation(test_plates)) if strategy == 'random' else sorted(test_plates, key=lambda p: (-scores[strategy][p], p))
                        for fraction in (0.0, .1, .25, .5):
                            k = int(np.floor(len(test_plates) * fraction))
                            reviewed = ranked[:k]
                            keep = ~np.isin(groups[test], reviewed)
                            review_rows.append({'fold': fold, 'strategy': strategy, 'seed': seed, 'budget_fraction': fraction,
                                'test_plates': len(test_plates), 'reviewed_plates': k, 'reviewed_ids': ';'.join(map(str, reviewed)),
                                'original_distance_n': int(np.isfinite(errors).sum()), 'unreviewed_distance_n': int(np.isfinite(errors[keep]).sum()),
                                'unresolved_mae_mm': float(np.nansum(errors[keep]) / np.isfinite(errors).sum()),
                                'unreviewed_mae_mm': float(np.nanmean(errors[keep])),
                                'original_over_2mm': int((errors > 2).sum()),
                                'unreviewed_over_2mm': int((errors[keep] > 2).sum())})
        if args.scope == 'full':
            train_user1 = train[np.asarray([rows[i]['user'] == 1 and complete_target[i] for i in train])]
            test_user23 = test[np.asarray([rows[i]['user'] in (2, 3) for i in test])]
            assert len(train_user1) and len(test_user23)
            for method, model in [('operator_transfer_ridge', Ridge(alpha=100.0)),
                ('operator_transfer_extra_trees', ExtraTreesRegressor(n_estimators=128, min_samples_leaf=3, random_state=7, n_jobs=1))]:
                model.fit(X32[train_user1], y[train_user1]); p = model.predict(X32[test_user23])
                results += metric_rows(y[test_user23], p, fold, method)
                all_predictions.setdefault(method, []).append((test_user23, p))
                for i, prediction in zip(test_user23, p):
                    predictions.append({'fold': fold, 'method': method, 'name': rows[i]['name'], 'plate': rows[i]['plate'],
                        'user': rows[i]['user'], 'risk_score': 0.0, **{f'target_{d}': float(y[i,j]) for j,d in enumerate(DIST)},
                        **{f'prediction_{d}': float(prediction[j]) for j,d in enumerate(DIST)}})
    for method, parts in all_predictions.items():
        indexes = np.concatenate([ix for ix, _ in parts]); pred = np.concatenate([pr for _, pr in parts])
        results += metric_rows(y[indexes], pred, 'pooled', method)
    np.savez_compressed(OUT / 'features_targets.npz', X32=X32, X16=X16, targets=y, plates=groups,
                        names=np.asarray([r['name'] for r in rows]), users=np.asarray([r['user'] for r in rows]))
    result = {'status': 'complete', 'binding': binding, 'rows': len(rows), 'plates': len(np.unique(groups)),
        'scope': args.scope, 'available_user_counts': {str(u): sum(r['user'] == u for r in rows) for u in (1,2,3)},
        'incomplete_manual_targets': [{'name': rows[i]['name'], 'missing_distances': [d for j,d in enumerate(DIST) if not np.isfinite(y[i,j])]}
                                      for i in range(len(rows)) if not complete_target[i]],
        'missing_unique_paired_images': [r['member'] for r in pairing['pairs'] if r['member'] not in acquisition['members']],
        'features_32': int(X32.shape[1]), 'features_16': int(X16.shape[1]), 'folds': 5,
        'metric_rows': len(results), 'metrics': results, 'fold_plan': plans, 'review_rows': len(review_rows),
        'limitations': ['Exploratory public-data experiment; no external or blind cohort.',
            'Two or three photographs of a plate are grouped into one test fold.',
            'Manual distances are references, not perfect labels; outputs are not S/I/R.',
            'This is a visual branch separate from the genomic NCS experiments.']}
    (OUT / 'binding.json').write_text(json.dumps(binding, indent=2), encoding='utf-8')
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    with (OUT / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    with (OUT / 'review_controls.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0])); writer.writeheader(); writer.writerows(review_rows)
    result['metrics_sha256'] = sha(OUT / 'metrics.csv'); result['predictions_sha256'] = sha(OUT / 'predictions.csv')
    result['review_sha256'] = sha(OUT / 'review_controls.csv'); result['features_sha256'] = sha(OUT / 'features_targets.npz')
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'rows': result['rows'], 'plates': result['plates'],
                      'metric_rows': result['metric_rows'], 'metrics_sha256': result['metrics_sha256']}))


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    with threadpool_limits(limits=1):
        main()
