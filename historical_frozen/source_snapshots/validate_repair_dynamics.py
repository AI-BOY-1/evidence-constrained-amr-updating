"""Independent replay of the transition, overlap and stopping artifacts."""
import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/NCS_data/experiments/ecoli_bvbrc')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def check_metrics(y,p,reported):
    expected={'n':len(y),'R':int(np.sum(y==1)),'brier':float(np.mean(np.square(p-y))),
              'FS':int(np.sum((y==1)&(p<.5))),'FR':int(np.sum((y==0)&(p>=.5)))}
    for k,v in expected.items():
        assert abs(v-reported[k])<1e-12,(k,v,reported[k])


def main(dataset):
    dest=ROOT/f'repair_dynamics_{dataset}_v1'
    m=load(dest/'manifest.json');r=load(dest/'results.json')
    assert m['status']==r['status']=='complete'
    assert m['result_sha256']==digest(dest/'results.json')
    assert m['binding']['code_sha256']==digest(HERE/'run_repair_dynamics.py')
    assert m['binding']['protocol_sha256']==digest(HERE/'REPAIR_DYNAMICS_PROTOCOL.md')
    vp=HERE/('ecoli_full_validation.json' if dataset=='full' else 'ecoli_ani_annotation_validation.json')
    v=load(vp);assert v['status']=='passed' and digest(vp)==m['binding']['prior_validation_sha256']
    expected_artifacts=v['predictions'] if dataset=='full' else load(ROOT/'ani_annotation_repair_v1/results.json')['predictions']
    assert expected_artifacts==m['binding']['predictions']
    if dataset=='ani':assert v['result_sha256']==digest(ROOT/'ani_annotation_repair_v1/results.json')
    inputs={a['path']:a for a in r['inputs']}
    assert set(inputs)=={a['path'] for a in expected_artifacts}
    ids=lambda row:tuple(row[k] for k in ('drug','threshold','feature','seed','strategy'))
    transitions={(*ids(t),t['from_budget'],t['to_budget']):t for t in r['transitions']}
    snapshots={(*ids(t),t['budget']):t for t in r['snapshots']}
    gates={(*ids(t),t['rule']):t for t in r['gates']}
    assert len(transitions)==len(r['transitions']) and len(gates)==len(r['gates'])
    groups=defaultdict(dict);nt=ng=ns=0
    for artifact in expected_artifacts:
        p=Path(artifact['path']);assert digest(p)==artifact['sha256']==inputs[str(p)]['sha256']
        assert digest(p.with_suffix('.json'))==inputs[str(p)]['sidecar_sha256']
        meta=load(p.with_suffix('.json'));info=(meta['results'] if dataset=='full' else meta['rows'])[0]
        key=(info['drug'],info.get('threshold',info.get('ANI_threshold')),
             'sketch' if dataset=='full' else info['feature'],info['seed'],info['strategy'])
        with np.load(p,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
        y=z['labels'];ya=z['audit_labels']
        assert np.isin(y,[0,1]).all() and np.isin(ya,[0,1]).all()
        roles=('source','query_pool','audit','evaluation')
        for role in roles:
            samples=z[role+'_samples'];assert len(samples)==len(set(samples.tolist()))
        for a,b in itertools.combinations(roles,2):
            for field in ('samples','clusters'):
                assert not set(z[a+'_'+field])&set(z[b+'_'+field])
        budgets=sorted(int(k[3:]) for k in z if k.startswith('p_b'))
        assert budgets[0]==0 and set(budgets)<={0,10,25,50}
        for b in budgets:
            for prefix,label in (('p_b',y),('audit_p_b',ya)):
                probs=z[prefix+str(b)];assert len(probs)==len(label)
                assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
            check_metrics(y,z[f'p_b{b}'],snapshots[(*key,b)]);ns+=1
        for a,b in itertools.combinations(budgets,2):
            t=transitions[(*key,a,b)];pa=z[f'p_b{a}'];pb=z[f'p_b{b}']
            right0=(pa>=.5)==y;right1=(pb>=.5)==y
            assert t['corrected']==int(np.sum(~right0&right1))
            assert t['introduced']==int(np.sum(right0&~right1))
            assert t['errors_before']==int(np.sum(~right0)) and t['errors_after']==int(np.sum(~right1))
            assert t['errors_after']-t['errors_before']==t['introduced']-t['corrected']
            for cls,tag in ((1,'FS'),(0,'FR')):
                wrong0=(y==cls)&~right0;wrong1=(y==cls)&~right1
                for suffix,value in [('before',wrong0.sum()),('after',wrong1.sum()),
                                     ('corrected',np.sum(wrong0&~wrong1)),('introduced',np.sum(~wrong0&wrong1))]:
                    assert t[tag+'_'+suffix]==int(value)
            ld=np.square(pb-y)-np.square(pa-y)
            assert abs(t['brier_delta']-float(ld.mean()))<1e-12
            assert t['loss_increased_n']==int((ld>0).sum()) and t['loss_decreased_n']==int((ld<0).sum())
            assert t['n']==len(y) and t['R']==int(y.sum())
            assert t['adjacent']==(budgets.index(b)-budgets.index(a)==1)
            if len(ya):
                da=np.square(z[f'audit_p_b{b}']-ya)-np.square(z[f'audit_p_b{a}']-ya)
                assert abs(t['audit_brier_delta']-float(da.mean()))<1e-12
            else:assert t['audit_brier_delta'] is None
            nt+=1
        for rule in ('brier_only','brier_and_FS'):
            g=gates[(*key,rule)];ng+=1
            impossible=not len(ya) or (rule=='brier_and_FS' and not np.any(ya==1))
            if impossible:
                assert g['status']=='infeasible';continue
            assert g['status']=='complete'
            chosen=consumed=0;events=[]
            for b in budgets[1:]:
                old=z[f'audit_p_b{chosen}'];new=z[f'audit_p_b{b}']
                d=float(np.square(new-ya).mean()-np.square(old-ya).mean())
                dfs=int(np.sum((ya==1)&(new<.5))-np.sum((ya==1)&(old<.5)))
                accept=d<0 and (rule=='brier_only' or dfs<=0)
                events.append((b,chosen,bool(accept),d,dfs));consumed=b
                if not accept:break
                chosen=b
            assert (chosen,consumed)==(g['selected_budget'],g['consumed_budget'])
            assert g['total_target_assay_cost']==consumed+len(ya)
            assert len(events)==len(g['events'])
            for a,b in zip(events,g['events']):
                assert a[:3]==(b['attempted_budget'],b['previous_checkpoint'],b['accepted'])
                assert abs(a[3]-b['audit_brier_delta'])<1e-12 and a[4]==b['audit_FS_delta']
            check_metrics(y,z[f'p_b{chosen}'],g['selected'])
            check_metrics(y,z[f'p_b{consumed}'],g['same_cost_control'])
            for field,metric in [('brier_delta','brier'),('FS_delta','FS'),('FR_delta','FR')]:
                assert abs(g[field]-(g['selected'][metric]-g['same_cost_control'][metric]))<1e-12
        groups[key[:4]][key[4]]=z
    assert nt==len(transitions) and ng==len(gates) and ns==len(snapshots)
    overlap={tuple(o[k] for k in ('drug','threshold','feature','seed','budget')):o for o in r['overlap']}
    samples={tuple(o[k] for k in ('drug','threshold','feature','seed','budget','sample')):o for o in r['shared_samples']}
    no=nh=0
    for key,files in groups.items():
        assert set(files)=={'decision_diagnostic','random_gene','uncertainty_gene','random_tree'}
        base=next(iter(files.values()));y=base['labels']
        for z in files.values():
            for role in roles:
                for field in ('samples','clusters'):assert np.array_equal(z[role+'_'+field],base[role+'_'+field])
            assert np.array_equal(z['labels'],y) and np.array_equal(z['audit_labels'],base['audit_labels'])
        for b in sorted(int(k[3:]) for k in base if k.startswith('p_b')):
            nmiss=np.sum([z[f'p_b{b}']<.5 for z in files.values()],axis=0)
            hist=[int(np.sum((y==1)&(nmiss==k))) for k in range(5)]
            o=overlap[(*key,b)];assert hist==o['miss_histogram_R']
            assert sum(hist)==o['R']==int(y.sum()) and o['all_four_FS']==hist[4]
            assert o['at_least_one_FS']==sum(hist[1:]);no+=1
            for i in np.flatnonzero(y==1):
                s=samples[(*key,b,str(base['evaluation_samples'][i]))]
                assert s['strategies_FS']==int(nmiss[i]) and s['component']==str(base['evaluation_clusters'][i]);nh+=1
    assert no==len(overlap)==len(r['overlap']) and nh==len(samples)==len(r['shared_samples'])
    # Recompute every exported descriptive summary from its complete row set.
    for name,source,keys,values in [
        ('transition_summary',r['transitions'],('drug','threshold','feature','strategy','from_budget','to_budget'),
         ('brier_delta','FS_corrected','FS_introduced','FR_corrected','FR_introduced')),
        ('gate_summary',[g for g in r['gates'] if g['status']=='complete'],('drug','threshold','feature','strategy','rule'),
         ('brier_delta','FS_delta','FR_delta','consumed_budget','selected_budget','audit_R')),
        ('overlap_summary',r['overlap'],('drug','threshold','feature','budget'),('all_four_FS','at_least_one_FS','R'))]:
        buckets=defaultdict(list)
        for row in source:buckets[tuple(row[k] for k in keys)].append(row)
        assert len(buckets)==len(r[name])
        for row in r[name]:
            rs=buckets[tuple(row[k] for k in keys)];assert row['partition_n']==len(rs)
            for v in values:assert abs(row['mean_'+v]-float(np.mean([a[v] for a in rs])))<1e-12
            assert row['evaluation_R_range']==[min(a['R'] for a in rs),max(a['R'] for a in rs)]
    validation={'status':'passed','dataset':dataset,'result_sha256':digest(dest/'results.json'),
                'manifest_sha256':digest(dest/'manifest.json'),'validator_sha256':digest(__file__),
                'prediction_files':len(inputs),'transitions':nt,'gate_replays':ng,'overlap_configs':no,
                'resistant_sample_observations':nh,
                'checks':{'prior_hashes':True,'role_isolation':True,'transition_conservation':True,
                          'audit_only_stopping_replay':True,'rejected_query_cost':True,
                          'all_strategy_sample_alignment':True,'summary_recalculation':True}}
    (dest/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('full','ani'),default='full')
    main(parser.parse_args().dataset)
