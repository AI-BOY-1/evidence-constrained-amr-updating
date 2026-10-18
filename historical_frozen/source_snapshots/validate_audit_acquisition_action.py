"""Independent label, isolation, selection, metric and action-cost replay."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=Path('D:/NCS_data/experiments/ecoli_bvbrc')
OUT=DATA/'audit_acquisition_action_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(x,y):
    if isinstance(x,dict):
        assert set(x)==set(y)
        for k in x:equal(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for a,b in zip(x,y):equal(a,b)
    elif isinstance(x,float):assert abs(x-y)<1e-12,(x,y)
    else:assert x==y,(x,y)
def metrics(y,p):
    r=y==1;s=~r
    # Freeze NumPy mean arithmetic for strict-zero decisions; Python sum has
    # a different reduction order and can change a tie by 5e-17.
    return {'n':len(y),'R':int(sum(r)),'S':int(sum(s)),'brier':float(np.mean((p-y)**2)),
            'FS':int(sum(r&(p<.5))),'FR':int(sum(s&(p>=.5))),
            'R_brier':float(sum((p[r]-1)**2)/sum(r)) if sum(r) else None,
            'S_brier':float(sum(p[s]**2)/sum(s)) if sum(s) else None}
def flips(y,a,b):
    err0=np.not_equal(a>=.5,y);err1=np.not_equal(b>=.5,y)
    return {k:int(sum(g)) for k,g in {
        'corrected':err0&~err1,'introduced':~err0&err1,'FS_corrected':(y==1)&err0&~err1,
        'FS_introduced':(y==1)&~err0&err1,'FR_corrected':(y==0)&err0&~err1,
        'FR_introduced':(y==0)&~err0&err1}.items()}
def independent_order(samples,components,p,seed,method):
    ranks={i:hashlib.sha256(('audit-acquisition-v1:'+str(seed)+':'+str(s)).encode()).hexdigest() for i,s in enumerate(samples)}
    ranked=sorted(range(len(samples)),key=lambda i:(ranks[i],str(samples[i])))
    if method=='random':return ranked
    if method=='predicted_R':return sorted(ranked,key=lambda i:-float(p[i]))
    groups={c:[i for i in ranked if components[i]==c] for c in set(components)}
    keys=sorted(groups,key=lambda c:hashlib.sha256(('audit-component-v1:'+str(seed)+':'+str(c)).encode()).hexdigest())
    result=[];depth=0
    while len(result)<len(samples):
        for c in keys:
            if depth<len(groups[c]):result.append(groups[c][depth])
        depth+=1
    return result
def main():
    result=read(OUT/'results.json');binding=read(OUT/'binding.json')
    assert result['status']=='complete_pending_validation'
    assert result['binding_sha256']==sha(OUT/'binding.json')
    for p,d in binding['inputs'].items():assert sha(p)==d
    original=read(DATA/'ani_annotation_repair_v1/results.json')
    assert result['inherited_skips']==original['skipped']
    assert len(result['artifacts'])==len(original['predictions'])==680
    ast=defaultdict(set)
    with (DATA/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            ast[(row['sequence_id'] or row['assembly_accession'],row['antibiotic'])].add(int(row['label']=='Resistant'))
    source_artifacts={a['path']:a['sha256'] for a in original['predictions']}
    conditions=skips=actions=0;observed=set();hashes={}
    for a in result['artifacts']:
        path=Path(a['path']);assert sha(path)==a['sha256'];r=read(path);p=Path(r['source_npz'])
        assert str(p) in source_artifacts and str(p) not in observed;observed.add(str(p))
        assert sha(p)==r['source_npz_sha256']==source_artifacts[str(p)]
        assert r['binding_sha256']==result['binding_sha256'];assert sha(p.with_suffix('.json'))==r['source_metadata_sha256']
        meta=read(p.with_suffix('.json'));z=np.load(p,allow_pickle=False);drug=r['identity']['drug']
        for k in r['identity']:assert r['identity'][k]==meta['rows'][0][k]
        role_sets=[set(z[role+'_samples']) for role in ('source','query_pool','audit','evaluation')]
        comps=[set(z[role+'_clusters']) for role in ('source','query_pool','audit','evaluation')]
        assert all(not role_sets[i]&role_sets[j] and not comps[i]&comps[j] for i in range(4) for j in range(i))
        for key,labelkey in [('audit_samples','audit_labels'),('evaluation_samples','labels')]:
            for sample,y in zip(z[key],z[labelkey]):assert ast[(str(sample),drug)]=={int(y)}
        event_ids=[n for ev in meta['events'] for n in ev['samples']]
        assert len(event_ids)==len(set(event_ids)) and set(event_ids)<=role_sets[1]
        y=z['labels'];base=z['p_b0'];ya=z['audit_labels'];ab=z['audit_p_b0']
        samples=z['audit_samples'];cs=z['audit_clusters'];lookup={str(s):i for i,s in enumerate(samples)}
        base_score=metrics(y,base)
        seen=set()
        for c in r['conditions']:
            key=(c['query_budget'],c['audit_size'],c['acquisition']);assert key not in seen;seen.add(key)
            b,n,method=key;assert b in (10,25,50) and n in (5,10,25,50)
            expected=independent_order(samples,cs,ab,r['identity']['seed'],method)[:n]
            ix=np.array([lookup[s] for s in c['audit_samples']]);assert list(ix)==expected
            assert c['audit_components']==[str(cs[i]) for i in ix]
            q=next(row for row in meta['rows'] if row['budget']==b)['queried_samples']
            assert c['query_samples']==q and len(q)==len(set(q))==b and set(q)<=role_sets[1]
            anew=z[f'audit_p_b{b}'];new=z[f'p_b{b}'];oldm=metrics(ya[ix],ab[ix]);newm=metrics(ya[ix],anew[ix])
            g=[ix[cs[ix]==cid] for cid in sorted(set(cs[ix]))];rg=[i[ya[i]==1] for i in g if np.any(ya[i]==1)]
            e={'old':oldm,'new':newm,'components':len(g),'R_components':len(rg),
               'brier_delta':newm['brier']-oldm['brier'],'FS_delta':newm['FS']-oldm['FS'],
               'component_brier_delta':float(np.mean([np.mean((anew[i]-ya[i])**2-(ab[i]-ya[i])**2) for i in g])),
               'component_FS_rate_delta':float(np.mean([np.mean(anew[i]<.5)-np.mean(ab[i]<.5) for i in rg])) if rg else None}
            equal(c['audit_evidence'],e);new_score=metrics(y,new)
            assert [a['rule'] for a in c['actions']]==['retain','update','audit_loss','audit_direction','component_direction']
            for action in c['actions']:
                rule=action['rule'];status='fixed' if rule in ('retain','update') else 'eligible'
                accept=rule=='update'
                if rule=='audit_loss':accept=e['brier_delta']<0
                if rule in ('audit_direction','component_direction'):
                    if oldm['R']==0:status='infeasible_no_audit_R_fallback_retain';accept=False
                    elif rule=='audit_direction':accept=e['brier_delta']<0 and e['FS_delta']<=0
                    else:accept=e['component_brier_delta']<0 and e['component_FS_rate_delta']<=0
                assert action['accepted']==accept and action['decision_status']==status,(path.name,key,rule,action['accepted'],accept,action['decision_status'],status,e,c['audit_evidence'])
                assert action['selected_budget']==(b if accept else 0)
                assert action['total_new_label_cost']==b+n and action['source_labels_sunk_cost']==len(z['source_samples'])
                score=new_score if accept else base_score;pred=new if accept else base
                equal(action['evaluation'],score);equal(action['change_from_frozen'],flips(y,base,pred))
                equal(action['evaluation_brier_delta_from_frozen'],score['brier']-base_score['brier'])
                assert action['evaluation_FS_delta_from_frozen']==score['FS']-base_score['FS']
                equal(action['oracle_brier_regret'],max(0.,score['brier']-min(base_score['brier'],new_score['brier'])))
                actions+=1
            conditions+=1
        for s in r['skipped_conditions']:
            key=(s['query_budget'],s['audit_size'],s['acquisition']);assert key not in seen;seen.add(key)
            assert f'p_b{s["query_budget"]}' not in z.files or s['audit_size']>len(ya)
            assert s['rules_skipped']==['retain','update','audit_loss','audit_direction','component_direction'];skips+=1
        assert seen=={(b,n,m) for b in (10,25,50) for n in (5,10,25,50) for m in ('random','predicted_R','component_coverage')}
        hashes[str(path)]=sha(path)
    assert conditions+skips==24480 and conditions==result['conditions'] and skips==result['skipped_conditions'] and actions==result['action_rows']
    v={'status':'passed','result_sha256':sha(OUT/'results.json'),'validator_sha256':sha(Path(__file__)),
       'files_checked':len(observed),'conditions_checked':conditions,'skipped_conditions_checked':skips,
       'action_rows_checked':actions,'artifacts_sha256':hashes,
       'numerical_note':'Strict-zero acceptance uses the frozen NumPy mean definition; alternate Python-sum reduction gave a -5.55e-17 apparent improvement at one observed tie. No threshold or source result changed.',
       'checks':{'original_AST_labels':True,'four_role_sample_component_isolation':True,
                 'query_only_label_access':True,'label_blind_sampling_order':True,'complete_configuration_coverage':True,
                 'audit_and_evaluation_metric_replay':True,'action_selection_and_full_cost_replay':True}}
    (OUT/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print(json.dumps({k:v[k] for k in ('status','files_checked','conditions_checked','skipped_conditions_checked','action_rows_checked')}))
if __name__=='__main__':main()
