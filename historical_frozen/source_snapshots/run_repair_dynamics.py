"""Frozen prediction transitions and audit-only stopping interventions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/NCS_data/experiments/ecoli_bvbrc')
STRATEGIES = ('decision_diagnostic', 'random_gene', 'uncertainty_gene', 'random_tree')
DRUGS = ('meropenem', 'imipenem', 'ertapenem', 'ceftazidime', 'ciprofloxacin',
         'trimethoprim/sulfamethoxazole', 'amikacin', 'gentamicin', 'cefepime')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def measure(y, p):
    r = y == 1
    return {'n': len(y), 'R': int(r.sum()), 'brier': float(np.mean((p-y)**2)),
            'FS': int(np.sum(r & (p < .5))), 'FR': int(np.sum(~r & (p >= .5)))}


def transition(y, before, after):
    a = before >= .5
    b = after >= .5
    e0, e1 = a != y, b != y
    fs0, fs1 = (y == 1) & ~a, (y == 1) & ~b
    fr0, fr1 = (y == 0) & a, (y == 0) & b
    delta = (after-y)**2 - (before-y)**2
    return {'brier_delta': float(delta.mean()),
            'errors_before': int(e0.sum()), 'errors_after': int(e1.sum()),
            'corrected': int((e0 & ~e1).sum()), 'introduced': int((~e0 & e1).sum()),
            'FS_before': int(fs0.sum()), 'FS_after': int(fs1.sum()),
            'FS_corrected': int((fs0 & ~fs1).sum()), 'FS_introduced': int((~fs0 & fs1).sum()),
            'FR_before': int(fr0.sum()), 'FR_after': int(fr1.sum()),
            'FR_corrected': int((fr0 & ~fr1).sum()), 'FR_introduced': int((~fr0 & fr1).sum()),
            'loss_increased_n': int((delta > 0).sum()),
            'loss_decreased_n': int((delta < 0).sum())}


def audit_gate(y, probabilities, directional):
    budgets = sorted(probabilities)
    if not len(y) or (directional and not np.any(y == 1)):
        return {'status': 'infeasible', 'reason': 'empty_audit' if not len(y) else 'no_audit_R'}
    chosen = consumed = 0
    events = []
    for budget in budgets[1:]:
        old = measure(y, probabilities[chosen])
        new = measure(y, probabilities[budget])
        accepted = new['brier'] < old['brier'] and (not directional or new['FS'] <= old['FS'])
        consumed = budget
        events.append({'attempted_budget': budget, 'previous_checkpoint': chosen,
                       'accepted': bool(accepted), 'audit_brier_delta': new['brier']-old['brier'],
                       'audit_FS_delta': new['FS']-old['FS']})
        if not accepted:
            break
        chosen = budget
    return {'status': 'complete', 'selected_budget': chosen, 'consumed_budget': consumed,
            'events': events}


def summarize(rows, keys, values):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    result = []
    for key, rs in sorted(groups.items()):
        row = {**dict(zip(keys, key)), 'partition_n': len(rs)}
        for field in values:
            row['mean_'+field] = float(np.mean([r[field] for r in rs]))
        row['evaluation_R_range'] = [min(r['R'] for r in rs), max(r['R'] for r in rs)]
        result.append(row)
    return result


def main(dataset):
    out = ROOT / ('repair_dynamics_'+dataset+'_v1')
    out.mkdir(exist_ok=True)
    vp = HERE / ('ecoli_full_validation.json' if dataset == 'full' else 'ecoli_ani_annotation_validation.json')
    validation = read(vp)
    assert validation['status'] == 'passed', 'Inputs require independent validation'
    if dataset == 'full':
        artifacts = validation['predictions']
        skipped = []
    else:
        rp = ROOT / 'ani_annotation_repair_v1/results.json'
        assert validation['result_sha256'] == sha(rp)
        raw = read(rp)
        artifacts = raw['predictions']
        skipped = raw['skipped']
    binding = {'dataset': dataset, 'prior_validation_sha256': sha(vp),
               'protocol_sha256': sha(HERE/'REPAIR_DYNAMICS_PROTOCOL.md'),
               'code_sha256': sha(Path(__file__)), 'predictions': artifacts}
    mp = out/'manifest.json'
    if mp.exists():
        previous = read(mp)
        assert previous['binding'] == binding, 'Refuse mixed dynamics versions'
        if previous['status'] == 'complete':
            assert previous['result_sha256'] == sha(out/'results.json')
            print(json.dumps({'status': 'existing_complete', 'dataset': dataset}))
            return
    manifest = {'status': 'running', 'binding': binding}
    write(mp, manifest)
    transitions, gates, snapshots, inputs = [], [], [], []
    aligned = defaultdict(dict)
    for artifact in artifacts:
        p = Path(artifact['path'])
        assert sha(p) == artifact['sha256']
        meta = read(p.with_suffix('.json'))
        rows = meta['results'] if dataset == 'full' else meta['rows']
        info = rows[0]
        key = {'drug': info['drug'], 'threshold': info.get('threshold', info.get('ANI_threshold')),
               'feature': 'sketch' if dataset == 'full' else info['feature'],
               'seed': info['seed'], 'strategy': info['strategy']}
        with np.load(p, allow_pickle=False) as z:
            z = {n: z[n].copy() for n in z.files}
        y, ya = z['labels'], z['audit_labels']
        budgets = sorted(int(k[3:]) for k in z if k.startswith('p_b'))
        assert budgets[0] == 0
        role_samples, role_clusters = [], []
        for role in ('source', 'query_pool', 'audit', 'evaluation'):
            role_samples.append(set(z[role+'_samples'].tolist()))
            role_clusters.append(set(z[role+'_clusters'].tolist()))
        for i in range(4):
            for j in range(i):
                assert not role_samples[i] & role_samples[j]
                assert not role_clusters[i] & role_clusters[j]
        for b in budgets:
            assert np.isfinite(z[f'p_b{b}']).all() and np.isfinite(z[f'audit_p_b{b}']).all()
            snapshots.append({**key, 'budget': b, **measure(y, z[f'p_b{b}']),
                              'audit_n': len(ya), 'audit_R': int(ya.sum())})
        for b0, b1 in itertools.combinations(budgets, 2):
            pa, pb = z[f'p_b{b0}'], z[f'p_b{b1}']
            ta = transition(ya, z[f'audit_p_b{b0}'], z[f'audit_p_b{b1}']) if len(ya) else None
            te = transition(y, pa, pb)
            transitions.append({**key, 'from_budget': b0, 'to_budget': b1,
                'adjacent': budgets.index(b1) == budgets.index(b0)+1,
                'n': len(y), 'R': int(y.sum()), 'audit_n': len(ya), 'audit_R': int(ya.sum()),
                **te, 'audit_brier_delta': ta['brier_delta'] if ta else None,
                'audit_FS_delta': ta['FS_after']-ta['FS_before'] if ta else None})
        for rule in ('brier_only', 'brier_and_FS'):
            replay = audit_gate(ya, {b: z[f'audit_p_b{b}'] for b in budgets}, rule == 'brier_and_FS')
            row = {**key, 'rule': rule, 'n': len(y), 'R': int(y.sum()),
                   'audit_n': len(ya), 'audit_R': int(ya.sum()), **replay}
            if replay['status'] == 'complete':
                chosen, used = replay['selected_budget'], replay['consumed_budget']
                a, c = measure(y, z[f'p_b{chosen}']), measure(y, z[f'p_b{used}'])
                row.update(selected=a, same_cost_control=c,
                           total_target_assay_cost=used+len(ya),
                           brier_delta=a['brier']-c['brier'], FS_delta=a['FS']-c['FS'],
                           FR_delta=a['FR']-c['FR'])
            gates.append(row)
        group = (key['drug'], key['threshold'], key['feature'], key['seed'])
        aligned[group][key['strategy']] = z
        inputs.append({'path': str(p), 'sha256': artifact['sha256'],
                       'sidecar_sha256': sha(p.with_suffix('.json'))})
    overlap, shared_samples = [], []
    for group, files in sorted(aligned.items()):
        assert set(files) == set(STRATEGIES)
        base = files[STRATEGIES[0]]
        for z in files.values():
            for k in ('labels', 'evaluation_samples', 'evaluation_clusters'):
                assert np.array_equal(z[k], base[k])
        y = base['labels']
        budgets = sorted(int(k[3:]) for k in base if k.startswith('p_b'))
        for b in budgets:
            misses = np.array([(files[s][f'p_b{b}'] < .5) & (y == 1) for s in STRATEGIES])
            count = misses.sum(axis=0)
            overlap.append({**dict(zip(('drug', 'threshold', 'feature', 'seed'), group)),
                'budget': b, 'n': len(y), 'R': int(y.sum()),
                'all_four_FS': int((count == 4).sum()),
                'at_least_one_FS': int((count > 0).sum()),
                'miss_histogram_R': [int(((count == k) & (y == 1)).sum()) for k in range(5)]})
            for i in np.flatnonzero(y == 1):
                shared_samples.append({**dict(zip(('drug', 'threshold', 'feature', 'seed'), group)),
                    'budget': b, 'sample': str(base['evaluation_samples'][i]),
                    'component': str(base['evaluation_clusters'][i]), 'strategies_FS': int(count[i])})
    tsummary = summarize(transitions, ('drug','threshold','feature','strategy','from_budget','to_budget'),
                         ('brier_delta','FS_corrected','FS_introduced','FR_corrected','FR_introduced'))
    gsummary = summarize([r for r in gates if r['status']=='complete'],
                         ('drug','threshold','feature','strategy','rule'),
                         ('brier_delta','FS_delta','FR_delta','consumed_budget','selected_budget','audit_R'))
    osummary = summarize(overlap, ('drug','threshold','feature','budget'), ('all_four_FS','at_least_one_FS','R'))
    result = {'status':'complete','dataset':dataset,'inputs':inputs,'transitions':transitions,
              'gates':gates,'snapshots':snapshots,'overlap':overlap,'shared_samples':shared_samples,
              'transition_summary':tsummary,'gate_summary':gsummary,'overlap_summary':osummary,
              'prior_skipped':skipped,
              'limitations':['Post-result exploratory frozen-path interventions, not new model training.',
                'All partitions reuse public cohorts; counts are not independent clinical events.',
                'Shared misses do not establish irreducible information deficit or a mechanism.',
                'Audit-gated rollback charges rejected measurements and can fail on evaluation data.',
                'Original sketch partitions have cross-role ANI links; ANI versions are separate.']}
    write(out/'results.json', result)
    for name, data in [('transitions',transitions),('snapshots',snapshots),('gate_summary',gsummary),
                       ('transition_summary',tsummary),('overlap_summary',osummary),('shared_samples',shared_samples)]:
        if data:
            with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    manifest.update(status='complete',result_sha256=sha(out/'results.json'),prediction_files=len(inputs),
                    transitions=len(transitions),gate_replays=len(gates),overlap_configs=len(overlap))
    write(mp,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='binding'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('full','ani'),default='full')
    main(parser.parse_args().dataset)
