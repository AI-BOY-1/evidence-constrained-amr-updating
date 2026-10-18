"""Post-result frozen-path ablation; never changes the formal batch."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Match the completed formal subprocess environment before loading numerical DLLs.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import sourmash
from threadpoolctl import threadpool_info
from counterexample_guided_repair import ProbabilityModel, Record

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/NCS_data/experiments/ecoli_bvbrc')
OUT = ROOT / 'frozen_query_ablation_v1'
DRUGS = ('meropenem', 'imipenem', 'ertapenem', 'ceftazidime', 'ciprofloxacin',
         'trimethoprim/sulfamethoxazole', 'amikacin', 'gentamicin', 'cefepime')
THRESHOLDS = (.99, .95)
SEEDS = (7, 17, 27, 37, 47)
BUDGETS = (0, 10, 25, 50)
TOL = 1e-10


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, data):
    p = Path(p)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(p)


def stem(drug, threshold, strategy, seed):
    return f'{drug.replace("/", "-")}_t{threshold}_{strategy}_{seed}'


def original(drug, threshold, strategy, seed):
    return ROOT / 'repair_full_scaled5000' / drug.replace('/', '-') / stem(drug, threshold, strategy, seed)


def raw_labels():
    grouped = defaultdict(set)
    with (ROOT / 'ecoli_human_sr_pairs.tsv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            grouped[(r['sequence_id'] or r['assembly_accession'], r['antibiotic'])].add(r['label'])
    return {k: int(next(iter(v)) == 'Resistant') for k, v in grouped.items()
            if len(v) == 1 and v <= {'Susceptible', 'Resistant'}}


def metrics(y, p, baseline):
    assert len(y) == len(p) == len(baseline) and len(y)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    r, s = y == 1, y == 0
    pred, base = p >= .5, baseline >= .5
    wrong, prior_wrong = pred != y, base != y
    loss = (p - y) ** 2
    return dict(n=len(y), resistant=int(r.sum()), susceptible=int(s.sum()),
                brier=float(loss.mean()),
                brier_R=float(loss[r].mean()) if r.any() else None,
                brier_S=float(loss[s].mean()) if s.any() else None,
                false_susceptible=int((r & ~pred).sum()),
                false_susceptible_rate=float((r & ~pred).sum() / r.sum()) if r.any() else None,
                false_resistant=int((s & pred).sum()),
                false_resistant_rate=float((s & pred).sum() / s.sum()) if s.any() else None,
                classification_fixed=int((prior_wrong & ~wrong).sum()),
                classification_harmed=int((~prior_wrong & wrong).sum()),
                resistant_fixed=int((r & ~base & pred).sum()),
                resistant_harmed=int((r & base & ~pred).sum()))


def check_roles(arrays, drug, labels):
    sample_sets, cluster_sets = [], []
    for role in ('source', 'query_pool', 'audit', 'evaluation'):
        ids = arrays[role + '_samples'].tolist()
        clusters = arrays[role + '_clusters'].tolist()
        assert len(ids) == len(set(ids)) == len(clusters)
        assert all((s, drug) in labels for s in ids)
        sample_sets.append(set(ids)); cluster_sets.append(set(clusters))
    for i in range(4):
        for j in range(i):
            assert not sample_sets[i] & sample_sets[j]
            assert not cluster_sets[i] & cluster_sets[j]
    for role in ('evaluation', 'audit'):
        key = 'labels' if role == 'evaluation' else 'audit_labels'
        assert arrays[key].tolist() == [labels[(s, drug)] for s in arrays[role + '_samples']]


def check_trace(trace, arrays, drug, labels):
    path = []
    for event in trace['events']:
        assert event['query_start'] == len(path)
        assert event['query_n'] == len(event['samples'])
        assert event['revealed_labels'] == [labels[(s, drug)] for s in event['samples']]
        path.extend(event['samples'])
    assert len(path) == 50 == len(set(path))
    assert set(path) <= set(arrays['query_pool_samples'])
    by_budget = {r['budget']: r for r in trace['results']}
    assert set(by_budget) == set(BUDGETS)
    for b, r in by_budget.items():
        assert r['queried_samples'] == path[:b]
        assert r['total_target_assay_cost'] == b + len(arrays['audit_samples'])
    return path, by_budget


def provenance():
    validation_path = HERE / 'ecoli_full_validation.json'
    v = read(validation_path)
    assert v['status'] == 'passed'
    for name, value in v['code_hashes'].items():
        assert sha(HERE / name) == value, ('method changed', name)
    for name, value in v['input_hashes'].items():
        assert sha(ROOT / name) == value, ('input changed', name)
    prediction_hashes = {str(Path(x['path'])): x['sha256'] for x in v['predictions']}
    originals = {}
    for drug in DRUGS:
        for t in THRESHOLDS:
            for seed in SEEDS:
                for strategy in ('decision_diagnostic', 'random_gene'):
                    p = original(drug, t, strategy, seed)
                    # Do not with_suffix a stem containing .99: append extensions instead.
                    npz, js = Path(str(p) + '.npz'), Path(str(p) + '.json')
                    assert sha(npz) == prediction_hashes[str(npz)]
                    tr = read(js)
                    assert tr['code_hashes'] == v['code_hashes'] and tr['input_hashes'] == v['input_hashes']
                    originals[str(npz)] = sha(npz); originals[str(js)] = sha(js)
    return dict(original_validation_sha256=sha(validation_path),
                formal_code_hashes=v['code_hashes'], input_hashes=v['input_hashes'],
                original_artifact_hashes=originals,
                followup_code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(HERE / 'FROZEN_QUERY_ABLATION_PROTOCOL.md'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    binding = provenance()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        old = read(manifest_path)
        assert old['binding'] == binding, 'Refuse resume after provenance changes; use a new version directory.'
        if old['status'] == 'complete':
            assert sha(OUT / 'results.json') == old['results_sha256']
            print('Already complete with matching provenance; no fits repeated.', flush=True)
            return
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    binding=binding, designation='post-result exploratory fixed-query-path ablation',
                    configurations=90, planned_model_trajectories=180, planned_metric_rows=720,
                    packages={n: version(n) for n in ('numpy', 'scikit-learn', 'sourmash')},
                    numerical_environment={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    numerical_libraries=threadpool_info())
    assert all(lib['num_threads'] == 1 for lib in manifest['numerical_libraries'])
    write(manifest_path, manifest)
    labels = raw_labels()
    tokens = {}
    for sig in sourmash.load_file_as_signatures(str(ROOT / 'ecoli_assemblies_k31_scaled1000.sig')):
        sample = Path(sig.filename or sig.name).stem
        assert sample not in tokens
        tokens[sample] = tuple('sketch31::' + str(h) for h in sorted(sig.minhash.downsample(scaled=5000).hashes))
    assert len(tokens) == 3516
    rows, contrasts, artifacts = [], [], []
    done = 0
    for drug in DRUGS:
        for t in THRESHOLDS:
            for seed in SEEDS:
                paths = {s: original(drug, t, s, seed) for s in ('decision_diagnostic', 'random_gene')}
                originals = {s: np.load(str(p) + '.npz', allow_pickle=False) for s, p in paths.items()}
                traces = {s: read(str(p) + '.json') for s, p in paths.items()}
                da, ra = originals['decision_diagnostic'], originals['random_gene']
                for key in ra.files:
                    if not key.startswith(('p_b', 'audit_p_b')):
                        assert np.array_equal(da[key], ra[key]), key
                for a in originals.values():
                    check_roles(a, drug, labels)
                baseline = ra['p_b0']; y = ra['labels']
                fitted = {}
                for strategy, a in originals.items():
                    query, budgets = check_trace(traces[strategy], a, drug, labels)
                    name = 'diagnostic_path_gene' if strategy == 'decision_diagnostic' else 'random_path_gene_replay'
                    target = OUT / stem(drug, t, name, seed)
                    npz_path, json_path = Path(str(target) + '.npz'), Path(str(target) + '.json')
                    if json_path.exists():
                        saved = read(json_path)
                        assert saved['binding'] == binding
                        assert sha(npz_path) == saved['prediction_sha256']
                        fitted[name] = {b: np.load(npz_path, allow_pickle=False)[f'p_b{b}'] for b in BUDGETS}
                        rows.extend(saved['metrics'])
                    else:
                        def records(ids, clusters, masked=False):
                            return [Record(str(s), str(c), 0, '', tokens[str(s)], ('__SEQUENCE_SKETCH__',),
                                           -1 if masked else labels[(str(s), drug)]) for s, c in zip(ids, clusters)]
                        source = records(a['source_samples'], a['source_clusters'])
                        evaluation = records(a['evaluation_samples'], a['evaluation_clusters'], True)
                        audit = records(a['audit_samples'], a['audit_clusters'], True)
                        query_map = dict(zip(a['query_pool_samples'], a['query_pool_clusters']))
                        reveals = records(query, [query_map[s] for s in query])
                        arrays = {k: a[k] for k in ('labels', 'audit_labels', 'evaluation_samples', 'evaluation_clusters', 'audit_samples')}
                        model_rows, replay_error = [], 0.
                        fitted[name] = {}
                        for b in BUDGETS:
                            model = ProbabilityModel('gene', seed).fit(source + reveals[:b])
                            assert all(r.label == -1 for r in evaluation + audit)
                            p, pa = model.predict(evaluation), model.predict(audit)
                            arrays[f'p_b{b}'], arrays[f'audit_p_b{b}'] = p, pa
                            fitted[name][b] = p
                            if strategy == 'random_gene':
                                errors = [float(np.max(np.abs(p - a[f'p_b{b}'])))]
                                if len(pa):
                                    errors.append(float(np.max(np.abs(pa - a[f'audit_p_b{b}']))))
                                replay_error = max(replay_error, *errors)
                                assert replay_error <= TOL, (drug, t, seed, b, replay_error)
                            model_rows.append(dict(drug=drug, threshold=t, seed=seed, budget=b,
                                                   strategy=name, query_strategy=strategy,
                                                   queried_samples=query[:b], source_n=len(source),
                                                   audit_n=len(audit), total_target_assay_cost=b + len(audit),
                                                   **metrics(y, p, baseline)))
                        np.savez_compressed(npz_path, **arrays)
                        write(json_path, dict(binding=binding, prediction_sha256=sha(npz_path),
                                              metrics=model_rows, max_random_replay_error=replay_error,
                                              labels_masked=True, sample_and_component_disjoint=True))
                        rows.extend(model_rows)
                    artifacts.append(dict(npz=str(npz_path), npz_sha256=sha(npz_path),
                                          json=str(json_path), json_sha256=sha(json_path)))
                for b in BUDGETS:
                    d, g, r = da[f'p_b{b}'], fitted['diagnostic_path_gene'][b], ra[f'p_b{b}']
                    values = {s: metrics(y, p, baseline) for s, p in (('diagnostic', d), ('fixed_path_gene', g), ('random_gene', r))}
                    contrasts.append(dict(drug=drug, threshold=t, seed=seed, budget=b,
                                          original_action=next(x['action'] for x in traces['decision_diagnostic']['results'] if x['budget'] == b),
                                          query_delta=values['fixed_path_gene']['brier'] - values['random_gene']['brier'],
                                          action_delta=values['diagnostic']['brier'] - values['fixed_path_gene']['brier'],
                                          total_delta=values['diagnostic']['brier'] - values['random_gene']['brier'],
                                          metrics=values, total_target_assay_cost=b + len(ra['audit_samples'])))
                for a in originals.values():
                    a.close()
                done += 1
                manifest.update(completed_configurations=done, metric_rows=len(rows), contrast_rows=len(contrasts))
                write(manifest_path, manifest)
                print(json.dumps(dict(drug=drug, threshold=t, seed=seed, configurations=done, metric_rows=len(rows))), flush=True)
    assert done == 90 and len(rows) == 720 and len(contrasts) == 360 and len(artifacts) == 180
    result = dict(binding=binding, metrics=rows, contrasts=contrasts, artifacts=artifacts,
                  designation=manifest['designation'])
    write(OUT / 'results.json', result)
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(), results_sha256=sha(OUT / 'results.json'))
    write(manifest_path, manifest)
    print(json.dumps(dict(status='complete', metric_rows=len(rows), contrasts=len(contrasts))), flush=True)


if __name__ == '__main__':
    main()
