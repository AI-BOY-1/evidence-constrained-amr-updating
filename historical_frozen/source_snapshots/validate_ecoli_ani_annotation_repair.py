"""Independently reconstruct components, AST labels, metrics and query evidence."""
import csv
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from run_ecoli_repair import DRUGS, SEEDS, STRATEGIES, measured_metrics
from decision_repair_core import EXPERTS, update_weights

HERE = Path(__file__).resolve().parent
DATA = Path('D:/NCS_data/experiments/ecoli_bvbrc')
DEST = DATA / 'ani_annotation_repair_v1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def groups(names, screen, rows, integrity, threshold):
    adjacency = {n: set() for n in names}
    def connect(a, b):
        adjacency[a].add(b)
        adjacency[b].add(a)
    for pair in screen['pairs']:
        for key in ('a_to_b', 'b_to_a'):
            d = pair.get(key)
            if d and d['identity'] >= threshold and d['query_fraction'] >= .8 and d['reference_fraction'] >= .8:
                connect(pair['a'], pair['b'])
    for g in integrity['duplicate_sequence_groups']:
        available = sorted(set(g) & set(names))
        for n in available[1:]:
            connect(available[0], n)
    bs = defaultdict(set)
    for r in rows:
        n = r['sequence_id'] or r['assembly_accession']
        if n in adjacency and r['biosample_accession']:
            bs[r['biosample_accession']].add(n)
    for g in bs.values():
        available = sorted(g)
        for n in available[1:]:
            connect(available[0], n)
    clusters = {}
    for n in sorted(names):
        if n in clusters:
            continue
        pending, members = [n], {n}
        while pending:
            for other in adjacency[pending.pop()] - members:
                members.add(other)
                pending.append(other)
        root = min(members)
        clusters.update({s: root for s in members})
    return clusters


def main():
    manifest = read(DEST / 'manifest.json')
    result = read(DEST / 'results.json')
    binding = read(DEST / 'binding.json')
    assert manifest['status'] == result['status'] == 'complete'
    assert manifest['binding'] == binding and sha(DEST / 'results.json') == manifest['result_sha256']
    for name, value in binding['code'].items():
        assert sha(HERE / name) == value
    for key, p in [('screen_sha256', HERE / 'ecoli_full_skani_screen.json'),
                   ('annotation_manifest_sha256', DATA / 'acquired_amr_v1/manifest.json'),
                   ('pair_sha256', DATA / 'ecoli_human_sr_pairs.tsv'),
                   ('integrity_sha256', DATA / 'ecoli_fasta_integrity.json'),
                   ('signature_sha256', DATA / 'ecoli_assemblies_k31_scaled1000.sig'),
                   ('protocol_sha256', HERE / 'SCIENCE_EXTENSION_PROTOCOL.md')]:
        assert sha(p) == binding[key]
    for n, digest in binding['annotation_sha256'].items():
        assert sha(DATA / 'acquired_amr_v1' / (n + '.json')) == digest
    integrity = read(DATA / 'ecoli_fasta_integrity.json')
    names = sorted(r['sequence_id'] for r in integrity['inventory'] if r['status'] == 'valid')
    screen = read(HERE / 'ecoli_full_skani_screen.json')
    assert set(names) == set(binding['annotation_sha256']) and len(names) == 3516
    with (DATA / 'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    labels, by, bs_labels, exact_labels = {}, defaultdict(list), defaultdict(set), defaultdict(set)
    exact = {n: g[0] for g in integrity['duplicate_sequence_groups'] for n in g}
    for r in rows:
        n, d = r['sequence_id'] or r['assembly_accession'], r['antibiotic']
        if n in names:
            by[(n, d)].append(r)
            exact_labels[(exact.get(n, n), d)].add(r['label'])
        if r['biosample_accession']:
            bs_labels[(r['biosample_accession'], d)].add(r['label'])
    eligible = {}
    for drug in DRUGS:
        eligible[drug] = sorted(n for (n, d), rs in by.items() if d == drug
            and len({r['label'] for r in rs}) == 1
            and len(exact_labels[(exact.get(n, n), d)]) == 1
            and all(len(bs_labels[(r['biosample_accession'], d)]) == 1 for r in rs if r['biosample_accession']))
        labels.update({(n, drug): int(by[(n, drug)][0]['label'] == 'Resistant') for n in eligible[drug]})
    partitions = {t: groups(names, screen, rows, integrity, t) for t in (.99, .999)}
    skip = {(r['drug'], r['threshold'], r['seed'], r['feature']): r for r in result['skipped']}
    expected = set(product(DRUGS, (.99, .999), SEEDS, ('homolog', 'sketch'), STRATEGIES))
    seen, errors, all_rows = set(), [], []
    roles = ('source', 'query_pool', 'audit', 'evaluation')
    paired_roles = {}
    for artifact in result['predictions']:
        p = Path(artifact['path'])
        assert sha(p) == artifact['sha256']
        shard = read(p.with_suffix('.json'))
        assert shard['prediction_sha256'] == sha(p) and shard['binding_sha256'] == sha(DEST / 'binding.json')
        first = shard['rows'][0]
        drug, threshold, seed, feature, strategy = (first[k] for k in ('drug','ANI_threshold','seed','feature','strategy'))
        config = (drug, threshold, seed, feature, strategy)
        assert config not in seen and config in expected
        seen.add(config)
        cluster = partitions[threshold]
        buckets = defaultdict(list)
        for n in eligible[drug]:
            h = int(hashlib.sha256(f'{seed}:{cluster[n]}'.encode()).hexdigest()[:8], 16) % 10
            role = 'source' if h < 4 else 'query_pool' if h < 7 else 'audit' if h == 7 else 'evaluation'
            buckets[role].append(n)
        with np.load(p, allow_pickle=False) as z:
            for role in roles:
                assert z[role + '_samples'].tolist() == buckets[role]
                assert z[role + '_clusters'].tolist() == [cluster[n] for n in buckets[role]]
            assert z['labels'].tolist() == [labels[(n, drug)] for n in buckets['evaluation']]
            assert z['audit_labels'].tolist() == [labels[(n, drug)] for n in buckets['audit']]
            sets = [set(z[r+'_clusters'].tolist()) for r in roles]
            assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
            role_key = (drug, threshold, seed)
            signature = tuple(tuple(buckets[r]) for r in roles)
            assert paired_roles.setdefault(role_key, signature) == signature
            budgets = [b for b in (0,10,25,50) if b <= len(buckets['query_pool'])]
            assert [r['budget'] for r in shard['rows']] == budgets
            weights = np.full(len(EXPERTS), 1 / len(EXPERTS))
            histories = {0: ([], weights.copy())}
            queried = []
            for event in shard['events']:
                samples = event['samples']
                assert event['query_start'] == len(queried) and event['query_n'] == len(samples)
                assert len(samples) == len(set(samples)) and not set(samples) & set(queried)
                assert set(samples) <= set(buckets['query_pool'])
                y = [labels[(n, drug)] for n in samples]
                assert y == event['revealed_labels']
                prior = np.asarray(event['pre_reveal_predictions'])
                assert prior.shape == (len(EXPERTS), len(samples)) and np.all(np.isfinite(prior))
                weights = update_weights(weights, prior, np.asarray(y))
                assert np.allclose(weights, event['post_reveal_evidence_weights'], atol=1e-12, rtol=0)
                queried.extend(samples)
                histories[len(queried)] = (queried.copy(), weights.copy())
            assert len(queried) == max(budgets)
            for r in shard['rows']:
                b = r['budget']
                assert tuple(r[k] for k in ('drug','ANI_threshold','seed','feature','strategy')) == config
                assert r['queried_samples'] == histories[b][0]
                assert np.allclose(r['evidence_weights'], histories[b][1], atol=1e-12, rtol=0)
                assert r['total_target_assay_cost'] == b + len(buckets['audit'])
                for key, role in [('source_n','source'),('query_n','query_pool'),('audit_n','audit'),('evaluation_n','evaluation')]:
                    assert r[key] == len(buckets[role])
                pred = z[f'p_b{b}']
                assert pred.shape == z['labels'].shape and np.all(np.isfinite(pred)) and np.all((pred>=0)&(pred<=1))
                audit_p = z[f'audit_p_b{b}']
                assert audit_p.shape == z['audit_labels'].shape and np.all(np.isfinite(audit_p)) and np.all((audit_p>=0)&(audit_p<=1))
                for k, value in measured_metrics(z['labels'], pred).items():
                    if value is None or not np.isfinite(value):
                        assert r[k] is None
                    else:
                        error = abs(float(value) - r[k]); errors.append(error); assert error < 1e-10
        all_rows.extend(shard['rows'])
    for key, s in skip.items():
        drug, t, seed, feature = key
        c = partitions[t]; bucket = defaultdict(list)
        for n in eligible[drug]:
            h = int(hashlib.sha256(f'{seed}:{c[n]}'.encode()).hexdigest()[:8],16)%10
            bucket[0 if h<4 else 1 if h<7 else 2 if h==7 else 3].append(n)
        assert s['sizes'] == [len(bucket[i]) for i in range(4)]
        source_classes = {labels[(n,drug)] for n in bucket[0]}
        has_features = any(read(DATA/'acquired_amr_v1'/(n+'.json'))['qualifying_reference_tokens'] for n in bucket[0]) if feature=='homolog' else bool(bucket[0])
        assert len(source_classes)<2 or len(bucket[1])<10 or len(bucket[3])<10 or not has_features
        for strategy in STRATEGIES:
            config = (drug,t,seed,feature,strategy)
            assert config not in seen
            seen.add(config)
    assert seen == expected and all_rows == result['results']
    assert manifest['trajectories'] == len(result['predictions']) and manifest['metric_rows'] == len(all_rows)
    out = {'status':'passed','prediction_files':len(result['predictions']),'metric_rows':len(all_rows),
           'skipped_configurations':len(skip),'max_metric_error':max(errors,default=0),
           'checks':{'input_code_hashes':True,'configuration_coverage':True,'raw_AST_labels':True,
                     'independent_components_and_roles':True,'feature_comparison_same_roles':True,
                     'query_weight_replay':True,'cost_and_metric_recomputation':True},
           'validator_sha256':sha(Path(__file__)),'result_sha256':sha(DEST/'results.json'),
           'limitations':['Approximate screen may miss pairs; no SNP/patient independence claim.',
                          'Event weight replay is not a complete model-training replay.']}
    (HERE/'ecoli_ani_annotation_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out),flush=True)


if __name__ == '__main__':
    main()
