"""Check permutations, transport membership and measured control invariants."""
from itertools import product
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from decision_repair_core import EXPERTS, update_weights
from run_reliability_controls import DATA, OUT, HERE, DRUGS, SEEDS, CONDITIONS, STRATEGIES, ROLES, population, category, period, sha, read
from validate_ecoli_ani_annotation_repair import groups as independent_components


def expected_roles(records,condition,seed):
    assigned={};reasons=defaultdict(int);candidate=defaultdict(list)
    for n,v in sorted(records.items()):
        c=v['component'];h=int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10
        if condition.startswith('time_'):
            if not v['year']:reasons['unknown_or_ambiguous_date']+=1;continue
            a,b=(2012,2014) if condition=='time_2012' else (2014,2017)
            candidate[c].append((n,'source' if v['year']<=a else 'query_pool' if v['year']<=b else 'target'))
        elif '_to_' in condition:
            source,target=condition.split('_to_');s=category(v['standard'])
            if s not in (source,target):reasons['mixed_or_unknown_standard']+=1;continue
            candidate[c].append((n,'source' if s==source else 'target'))
        else:
            assigned[n]=ROLES[0 if h<4 else 1 if h<7 else 2 if h==7 else 3]
    for c,rs in candidate.items():
        if len({g for n,g in rs})>1:reasons['component_spans_windows_or_standards']+=len(rs);continue
        h=int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10
        for n,g in rs:
            if g!='target':assigned[n]=g
            elif condition.startswith('time_'):assigned[n]='audit' if h<3 else 'evaluation'
            else:assigned[n]='query_pool' if h<4 else 'audit' if h<6 else 'evaluation'
    return assigned,dict(reasons)


def expected_donors(group,records,condition,role,rng):
    buckets=defaultdict(list);donors={n:n for n in group}
    for n in group:
        v=records[n]
        key=v['component'] if condition=='label_within_component' else (category(v['standard']),period(v['year'])) if condition=='features_within_context' else 'all'
        buckets[key].append(n)
    if role in ('source','query_pool') and condition in ('label_global','label_within_component','features_within_context'):
        for members in buckets.values():
            order=rng.permutation(len(members))
            donors.update({n:members[int(i)] for n,i in zip(members,order)})
    return donors


def main():
    m=read(OUT/'manifest.json');raw=read(OUT/'results.json');binding=read(OUT/'binding.json')
    assert m['status']==raw['status']=='complete' and m['result_sha256']==sha(OUT/'results.json') and m['binding_sha256']==sha(OUT/'binding.json')
    for n,digest in binding['code'].items():assert sha(HERE/n)==digest
    for p,digest in binding['inputs'].items():assert sha(Path(p))==digest
    for n,digest in binding['annotations'].items():assert sha(DATA/'acquired_amr_v1'/(n+'.json'))==digest
    names,rows,inv,meta,tokens=population();screen=read(HERE/'ecoli_full_skani_screen.json')
    clusters={t:independent_components(names,screen,rows,inv,t) for t in (.99,.999)}
    plans={};plan_configs=set();changed=defaultdict(lambda:defaultdict(int))
    for artifact in raw['plans']:
        p=Path(artifact['path']);assert sha(p)==artifact['sha256'];x=read(p)
        drug,t,feature,seed,condition=(x[k] for k in ('drug','threshold','feature','seed','condition'))
        config=(drug,t,feature,seed,condition);assert config not in plan_configs;plan_configs.add(config)
        records={n:{**v,'component':clusters[t][n]} for (n,d),v in meta.items() if d==drug}
        assigned,reasons=expected_roles(records,condition,seed)
        entries={v['sample']:v for v in x['records']};assert set(entries)==set(assigned)
        assert x['excluded']==reasons
        expected_seed=int(hashlib.sha256(f'{condition}:{drug}:{t}:{seed}:{feature}'.encode()).hexdigest()[:16],16)
        assert x['permutation_seed']==expected_seed
        rng=np.random.default_rng(expected_seed)
        for role in ROLES:
            group=sorted(n for n in assigned if assigned[n]==role);assert len(group)==x['sizes'][ROLES.index(role)]
            donors=expected_donors(group,records,condition,role,rng)
            assert {entries[n]['donor'] for n in group}==set(group)
            for n in group:
                a=entries[n];v=records[n];donor=donors[n];d=records[donor]
                assert a['role']==role and a['component']==v['component'] and a['year']==v['year'] and a['standard']==v['standard']
                assert a['original_label']==v['label'] and a['donor']==donor
                synthetic=d['label'] if condition.startswith('label_') and role in ('source','query_pool') else v['label']
                assert a['training_label']==synthetic and a['label_changed']==(synthetic!=v['label'])
                feature_changed=tokens[feature][n]!=tokens[feature][donor] if condition=='features_within_context' and role in ('source','query_pool') else False
                assert a['features_changed']==feature_changed
                changed[condition]['labels_changed']+=int(a['label_changed']);changed[condition]['vectors_changed']+=int(feature_changed)
        plans[str(p)]={'data':x,'entries':entries,'config':config}
    assert plan_configs==set(product(DRUGS,(.99,.999),('homolog','sketch'),SEEDS,CONDITIONS))
    seen=set();all_metrics=[];errors=[]
    for artifact in raw['predictions']:
        p=Path(artifact['path']);assert sha(p)==artifact['sha256'];x=read(p.with_suffix('.json'))
        assert x['prediction_sha256']==sha(p) and x['binding_sha256']==sha(OUT/'binding.json')
        plan=plans[x['plan_path']];assert x['plan_sha256']==sha(Path(x['plan_path']))
        entries=plan['entries'];conf=plan['config'];first=x['rows'][0];strategy=first['strategy'];config=conf+(strategy,)
        assert config not in seen;seen.add(config)
        with np.load(p,allow_pickle=False) as z:
            for role in ROLES:
                samples=sorted(n for n,v in entries.items() if v['role']==role)
                assert z[role+'_samples'].tolist()==samples
                assert z[role+'_clusters'].tolist()==[entries[n]['component'] for n in samples]
            assert z['labels'].tolist()==[entries[n]['original_label'] for n in z['evaluation_samples']]
            assert z['audit_labels'].tolist()==[entries[n]['original_label'] for n in z['audit_samples']]
            sets=[set(z[r+'_clusters'].tolist()) for r in ROLES];assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
            budgets=[b for b in (0,10,25,50) if b<=len(z['query_pool_samples'])]
            assert [r['budget'] for r in x['rows']]==budgets
            weights=np.full(len(EXPERTS),1/len(EXPERTS));queried=[];histories={0:([],weights.copy())}
            for ev in x['events']:
                assert ev['query_start']==len(queried) and ev['query_n']==len(ev['samples'])
                assert len(set(ev['samples']))==len(ev['samples']) and not set(ev['samples'])&set(queried)
                assert all(entries[n]['role']=='query_pool' for n in ev['samples'])
                lab=[entries[n]['training_label'] for n in ev['samples']];assert lab==ev['revealed_labels']
                prior=np.asarray(ev['pre_reveal_predictions']);assert prior.shape==(len(EXPERTS),len(lab)) and np.all(np.isfinite(prior)) and np.all((prior>=0)&(prior<=1))
                weights=update_weights(weights,prior,np.array(lab));assert np.allclose(weights,ev['post_reveal_evidence_weights'],atol=1e-12,rtol=0)
                queried.extend(ev['samples']);histories[len(queried)]=(queried.copy(),weights.copy())
            assert len(queried)==max(budgets)
            y=z['labels'];a=(y==1)
            for r in x['rows']:
                assert tuple(r[k] for k in ('drug','threshold','feature','seed','condition'))==conf and r['strategy']==strategy
                b=r['budget'];assert r['queried_samples']==histories[b][0] and np.allclose(r['evidence_weights'],histories[b][1],atol=1e-12,rtol=0)
                assert r['sizes']==plan['data']['sizes'] and r['total_target_assay_cost']==b+len(z['audit_labels'])
                pred=z[f'p_b{b}'];assert pred.shape==y.shape and np.all(np.isfinite(pred)) and np.all((pred>=0)&(pred<=1))
                ap=z[f'audit_p_b{b}'];assert ap.shape==z['audit_labels'].shape and np.all(np.isfinite(ap)) and np.all((ap>=0)&(ap<=1))
                calc={'n':len(y),'resistant':int(a.sum()),'brier':float(np.mean((pred-y)**2)),
                      'false_susceptible':int((a&(pred<.5)).sum()),'false_resistant':int((~a&(pred>=.5)).sum()),
                      'false_susceptible_rate':float(np.mean(pred[a]<.5)) if a.any() else None,
                      'false_resistant_rate':float(np.mean(pred[~a]>=.5)) if (~a).any() else None,
                      'resistant_brier':float(np.mean((pred[a]-1)**2)) if a.any() else None,
                      'susceptible_brier':float(np.mean(pred[~a]**2)) if (~a).any() else None,
                      'calibration_in_large':float(np.mean(pred)-np.mean(y)),
                      'auroc':float(roc_auc_score(y,pred)) if len(set(y))>1 else None,
                      'auprc':float(average_precision_score(y,pred)) if len(set(y))>1 else None}
                ece=0
                bins=np.clip(np.floor(pred*10).astype(int),0,9)
                for k in range(10):
                    ix=bins==k
                    if ix.any():ece+=float(ix.mean()*abs(pred[ix].mean()-y[ix].mean()))
                calc['ece10']=ece
                for k,value in calc.items():
                    if value is None:assert r[k] is None
                    else:error=abs(value-r[k]);errors.append(error);assert error<1e-10
        all_metrics.extend(x['rows'])
    for s in raw['skipped']:
        conf=tuple(s[k] for k in ('drug','threshold','feature','seed','condition'))
        config=conf+(s['strategy'],);assert config not in seen;seen.add(config)
        plan=next(v for v in plans.values() if v['config']==conf);v=plan['entries']
        source=[n for n,r in v.items() if r['role']=='source'];sizes=plan['data']['sizes']
        assert s['sizes']==sizes
        reason='insufficient_source_classes' if len({v[n]['training_label'] for n in source})<2 else 'query_or_evaluation_below_10' if sizes[1]<10 or sizes[3]<10 else 'empty_source_features' if not any(tokens[conf[2]][v[n]['donor'] if conf[4]=='features_within_context' else n] for n in source) else None
        assert reason==s['reason'] and reason is not None
    assert seen==set(c+(s,) for c in plan_configs for s in STRATEGIES)
    assert all_metrics==raw['rows'] and m['metric_rows']==len(all_metrics) and m['trajectories']==len(raw['predictions'])
    out={'status':'passed','prediction_files':len(raw['predictions']),'metric_rows':len(all_metrics),'skips':len(raw['skipped']),
         'configuration_coverage':len(seen),'max_metric_error':max(errors,default=0),'changed_records':{k:dict(v) for k,v in changed.items()},
         'result_sha256':sha(OUT/'results.json'),'validator_sha256':sha(Path(__file__)),
         'checks':{'hashes':True,'independent_components_and_roles':True,'permutation_rng_and_bijections':True,
                   'AST_and_transformed_label_provenance':True,'temporal_standard_membership':True,'query_weights':True,'cost_and_metrics':True}}
    write=HERE/'reliability_controls_validation.json';write.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
