"""Exploratory label-informed action control with sealed audit/evaluation labels."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer

from counterexample_guided_repair import compact_sparse, entropy_score, feature_dict, load_records, metrics
from decision_repair_core import EXPERTS, ExpertBank, action_matrix, diverse_indices, query_scores, update_weights
from run_ncs_experiments import DEFAULT_DRUGS, cohorts

ROOT = Path(__file__).resolve().parent
DEST = ncs_paths.data_path('experiments/completion/decision_repair')
SEEDS = (7, 17, 27, 37, 47)
BUDGETS = (0, 10, 25, 50)
STRATEGIES = {
    'random_gene': ('random', 'gene'),
    'uncertainty_gene': ('uncertainty', 'gene'),
    'disagreement_gene': ('disagreement', 'gene'),
    'diversity_gene': ('diversity', 'gene'),
    'random_diagnostic': ('random', 'diagnostic'),
    'uncertainty_diagnostic': ('uncertainty', 'diagnostic'),
    'information_diagnostic': ('information', 'diagnostic'),
    'decision_diagnostic': ('decision', 'diagnostic'),
    'decision_gene': ('decision', 'gene'),
    'random_tree': ('random', 'tree'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def mask(rows):
    return [replace(r, label=-1) for r in rows]


def audit_partition(pool):
    audit = [r for r in pool if int(hashlib.sha256(r.sample.encode()).hexdigest()[:8], 16) % 4 == 0]
    query = [r for r in pool if r not in audit]
    return query, audit


def tree_predict(training, evaluation, audit, seed):
    vec = DictVectorizer(sparse=True)
    x = compact_sparse(vec.fit_transform([feature_dict(r, 'gene') for r in training]))
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2).fit(x, [r.label for r in training])
    def predict(rows):
        if not rows:
            return np.asarray([], dtype=float)
        return model.predict_proba(compact_sparse(vec.transform([feature_dict(r, 'gene') for r in rows])))[:, list(model.classes_).index(1)]
    return predict(evaluation), predict(audit)


def trajectory(source, feature_pool, evaluation_features, audit_features, reveal, strategy, seed, budgets):
    """No audit/evaluation labels are accepted by this function."""
    assert all(r.label == -1 for r in feature_pool + evaluation_features + audit_features)
    objective, repair = STRATEGIES[strategy]
    rng = np.random.default_rng(seed)
    available, queried, events, snapshots = list(feature_pool), [], [], []
    bank = ExpertBank(source, seed)
    weights = np.full(len(EXPERTS), 1 / len(EXPERTS))
    while True:
        count = len(queried)
        pool_p = bank.predict(available)
        if count in budgets:
            matrix = action_matrix(pool_p) if len(available) else np.zeros((len(EXPERTS), len(EXPERTS)))
            action = int(np.argmin(weights @ matrix)) if repair == 'diagnostic' else 2
            if repair == 'tree':
                pe, pa = tree_predict(source + queried, evaluation_features, audit_features, seed)
                action_name = 'extra_trees'
            else:
                pe, pa = bank.predict(evaluation_features)[action], bank.predict(audit_features)[action]
                action_name = EXPERTS[action]
            snapshots.append({'budget': count, 'evaluation_p': pe, 'audit_p': pa,
                'action': action_name, 'evidence_weights': weights.tolist(),
                'queried_samples': [r.sample for r in queried]})
        if count >= max(budgets):
            break
        size = min(5, max(budgets) - count, len(available))
        if objective == 'diversity':
            selected = diverse_indices(source + queried, available, size)
        else:
            if objective == 'random':
                scores = rng.random(len(available))
            elif objective == 'uncertainty':
                scores = entropy_score(pool_p[2])
            elif objective == 'disagreement':
                scores = np.abs(pool_p[2] - pool_p[3])
            else:
                scores = query_scores(pool_p, weights, objective)
            selected = np.argsort(-(scores + rng.uniform(0, 1e-12, len(scores))))[:size]
        chosen = [available[i] for i in selected]
        # Cache predictions before accessing any new labels.
        prior = pool_p[:, selected].copy()
        observed = [reveal(r.sample) for r in chosen]
        labels = np.asarray([r.label for r in observed])
        weights = update_weights(weights, prior, labels)
        queried.extend(observed)
        events.append({'query_start': count, 'query_n': size, 'samples': [r.sample for r in chosen],
            'pre_reveal_predictions': prior.tolist(), 'revealed_labels': labels.tolist(),
            'post_reveal_evidence_weights': weights.tolist()})
        used = {r.sample for r in chosen}
        available = [r for r in available if r.sample not in used]
        bank.update(queried)
    return snapshots, events


def selective_metrics(y, p, fraction):
    n = max(1, int(np.ceil(fraction * len(y))))
    selected = np.argsort(-np.abs(p - 0.5), kind='stable')[:n]
    ys, ps = y[selected], p[selected]
    sensitive = ps < 0.5
    resistant = ys == 1
    false_s = int((sensitive & resistant).sum())
    return {'requested_coverage': fraction, 'coverage': len(selected) / len(y),
        'accepted_n': len(selected), 'accepted_R': int(resistant.sum()),
        'accepted_sensitive_n': int(sensitive.sum()), 'false_sensitive_n': false_s,
        'false_sensitive_among_R': false_s / resistant.sum() if resistant.any() else None,
        'resistant_among_sensitive': false_s / sensitive.sum() if sensitive.any() else None,
        'brier': float(np.mean((ps - ys) ** 2))}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'decision_repair_manifest.json'
    manifest = {'status': 'running', 'designation': 'exploratory post-result method development',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
        'packages': {x: version(x) for x in ('numpy', 'scipy', 'scikit-learn')},
        'code_hashes': {name: sha(ROOT / name) for name in ('run_decision_repair.py', 'decision_repair_core.py', 'counterexample_guided_repair.py')},
        'protocol_sha256': sha(ROOT / 'COMPLETION_PROTOCOL_2026-10-08.md'),
        'input_sha256': sha(ROOT / 'cleaned_human_kp.tsv'), 'completed_trajectories': 0}
    write_json(manifest_path, manifest)
    output = {'results': [], 'events': [], 'trajectory_audits': [], 'risk_coverage': [], 'skipped': [],
        'strategies': STRATEGIES, 'seeds': list(SEEDS), 'budgets': list(BUDGETS),
        'limitations': ['Prediction-expert evidence is not causal fault identification.',
            'Decision value freezes current candidate prediction rules; static five-sample batching is an approximation.',
            'Audit labels are withheld from all selection/training, but full fixed audit cost is charged.',
            'These generic baselines do not claim replication of published active-testing/domain-adaptation algorithms.',
            'Previously evaluated internal cohorts are reused for exploratory development.']}
    try:
        for drug in DEFAULT_DRUGS:
            records = load_records(ROOT / 'cleaned_human_kp.tsv', drug)
            for protocol in ('forward_2019', 'forward_2017'):
                source, full_pool, evaluation = cohorts(records, protocol)
                pool, audit = audit_partition(full_pool)
                eligible = [b for b in BUDGETS if b <= len(pool)]
                groups = [set(r.sample for r in g) for g in (source, pool, audit, evaluation)]
                assert all(not (groups[i] & groups[j]) for i in range(4) for j in range(i))
                if len(pool) < 10:
                    output['skipped'].append({'drug': drug, 'protocol': protocol, 'reason': 'query_pool_smaller_than_10'})
                    continue
                oracle = {r.sample: r for r in pool}
                y, ya = np.asarray([r.label for r in evaluation]), np.asarray([r.label for r in audit])
                for strategy in STRATEGIES:
                    for seed in SEEDS:
                        revealed = []
                        def reveal(sample):
                            assert sample in oracle and sample not in revealed
                            revealed.append(sample)
                            return oracle[sample]
                        snapshots, events = trajectory(source, mask(pool), mask(evaluation), mask(audit), reveal, strategy, seed, eligible)
                        arrays = {'labels': y, 'audit_labels': ya,
                            'evaluation_samples': np.asarray([r.sample for r in evaluation]),
                            'evaluation_projects': np.asarray([r.project for r in evaluation]),
                            'audit_samples': np.asarray([r.sample for r in audit])}
                        for snapshot in snapshots:
                            b, p, pa = snapshot['budget'], snapshot['evaluation_p'], snapshot['audit_p']
                            arrays[f'p_b{b}'], arrays[f'audit_p_b{b}'] = p, pa
                            metadata = {'drug': drug, 'protocol': protocol, 'strategy': strategy, 'seed': seed, 'budget': b}
                            output['results'].append({**metadata, **metrics(y, p), 'action': snapshot['action'],
                                'evidence_weights': snapshot['evidence_weights'], 'audit_n': len(audit),
                                'total_target_assay_cost': b + len(audit),
                                'audit_brier': float(np.mean((pa - ya) ** 2)) if len(audit) else None,
                                'queried_samples': snapshot['queried_samples']})
                            for coverage in (0.25, 0.5, 0.75, 0.9, 1.0):
                                output['risk_coverage'].append({**metadata, **selective_metrics(y, p, coverage)})
                        artifact = DEST / f'{drug}_{protocol}_{strategy}_{seed}.npz'
                        np.savez_compressed(artifact, **arrays)
                        output['events'].extend({'drug': drug, 'protocol': protocol, 'strategy': strategy, 'seed': seed, **event} for event in events)
                        assert len(revealed) == len(set(revealed)) == max(eligible)
                        output['trajectory_audits'].append({'drug': drug, 'protocol': protocol, 'strategy': strategy,
                            'seed': seed, 'source_n': len(source), 'query_pool_n': len(pool), 'audit_n': len(audit),
                            'evaluation_n': len(evaluation), 'budgets': eligible, 'prediction_file': str(artifact),
                            'labels_masked_for_selection': True, 'four_way_biosample_disjoint': True})
                        manifest['completed_trajectories'] += 1
                write_json(ROOT / 'decision_repair_results.json', output)
                write_json(manifest_path, manifest)
                print(json.dumps({'drug': drug, 'protocol': protocol, 'trajectories': manifest['completed_trajectories'],
                    'query_pool': len(pool), 'audit': len(audit), 'eligible_budgets': eligible}), flush=True)
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
            result_rows=len(output['results']), query_events=len(output['events']),
            prediction_files=len(output['trajectory_audits']), risk_coverage_rows=len(output['risk_coverage']),
            output_sha256=sha(ROOT / 'decision_repair_results.json'))
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
