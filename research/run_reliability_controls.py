"""Version-bound negative controls and prospective-direction stress tests."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import json
from collections import defaultdict, Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
import sourmash
from counterexample_guided_repair import Record, parse_year
from run_decision_repair import trajectory, mask, write_json
from run_ecoli_repair import DRUGS, SEEDS, measured_metrics
from run_ecoli_ani_annotation_repair import components

HERE=Path(__file__).resolve().parent
DATA=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=DATA/'reliability_controls_v1'
STRATEGIES=('decision_diagnostic','random_gene')
CONDITIONS=('label_global','label_within_component','features_within_context',
            'time_2012','time_2014','EUCAST_to_CLSI','CLSI_to_EUCAST')
ROLES=('source','query_pool','audit','evaluation')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def category(s):
    vals={x.strip().upper() for x in s.replace(';',',').split(',') if x.strip()}
    return next(iter(vals)) if len(vals)==1 and vals <= {'EUCAST','CLSI'} else 'mixed' if vals else 'unknown'


def period(year):return 'unknown' if not year else '<2015' if year<2015 else '2015-2019' if year<2020 else '>=2020'


def extra_metrics(y,p):
    result=measured_metrics(y,p)
    result.update(false_resistant=int(((y==0)&(p>=.5)).sum()),
        resistant_brier=float(np.mean((p[y==1]-1)**2)) if np.any(y==1) else None,
        susceptible_brier=float(np.mean(p[y==0]**2)) if np.any(y==0) else None,
        calibration_in_large=float(np.mean(p)-np.mean(y)),
        ece10=sum(float(np.sum(ix)/len(y)*abs(np.mean(p[ix])-np.mean(y[ix])))
                  for b in range(10) if np.any(ix:=(np.minimum((p*10).astype(int),9)==b))))
    return result


def population():
    inv=read(DATA/'ecoli_fasta_integrity.json')
    names=sorted(r['sequence_id'] for r in inv['inventory'] if r['status']=='valid')
    with (DATA/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    by=defaultdict(list);bs=defaultdict(set);ex=defaultdict(set)
    exact={n:g[0] for g in inv['duplicate_sequence_groups'] for n in g};available=set(names)
    for r in rows:
        n=r['sequence_id'] or r['assembly_accession'];d=r['antibiotic']
        if n in available:by[(n,d)].append(r);ex[(exact.get(n,n),d)].add(r['label'])
        if r['biosample_accession']:bs[(r['biosample_accession'],d)].add(r['label'])
    meta={}
    for (n,d),rs in sorted(by.items()):
        if d not in DRUGS or len({r['label'] for r in rs})!=1 or len(ex[(exact.get(n,n),d)])!=1:continue
        if any(len(bs[(r['biosample_accession'],d)])>1 for r in rs if r['biosample_accession']):continue
        years={parse_year(r['collection_date']) for r in rs}
        year=(next(iter(years)) or 0) if len(years)==1 else 0
        standards=';'.join(sorted({r['standards'] for r in rs}))
        meta[(n,d)]={'year':year,'standard':standards,'label':int(rs[0]['label']=='Resistant')}
    tokens={'sketch':{},'homolog':{}}
    for sig in sourmash.load_file_as_signatures(str(DATA/'ecoli_assemblies_k31_scaled1000.sig')):
        tokens['sketch'][Path(sig.filename or sig.name).stem]=tuple('sketch31::'+str(h) for h in sorted(sig.minhash.downsample(scaled=5000).hashes))
    for n in names:
        x=read(DATA/'acquired_amr_v1'/(n+'.json'))
        tokens['homolog'][n]=tuple(sorted({'homolog::'+t.split('_')[0] for t in x['qualifying_reference_tokens']}))
    return names,rows,inv,meta,tokens


def partition(records,condition,seed):
    role=lambda c:int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10
    excluded=Counter();groups=[[],[],[],[]]
    if condition.startswith('label_') or condition=='features_within_context':
        for r in records:
            h=role(r.project);groups[0 if h<4 else 1 if h<7 else 2 if h==7 else 3].append(r)
    else:
        candidates=defaultdict(list)
        for r in records:
            if condition.startswith('time_'):
                if not r.year:excluded['unknown_or_ambiguous_date']+=1;continue
                cutoff=2012 if condition=='time_2012' else 2014;end=2014 if cutoff==2012 else 2017
                group=0 if r.year<=cutoff else 1 if r.year<=end else 2
            else:
                s,t=condition.split('_to_');context=category(r.country)
                if context not in (s,t):excluded['mixed_or_unknown_standard']+=1;continue
                group=0 if context==s else 1
            candidates[r.project].append((r,group))
        for c,rs in sorted(candidates.items()):
            if len({g for r,g in rs})>1:
                excluded['component_spans_windows_or_standards']+=len(rs);continue
            h=role(c)
            for r,g in rs:
                if condition.startswith('time_'):index=g if g<2 else 2 if h<3 else 3
                else:index=0 if g==0 else 1 if h<4 else 2 if h<6 else 3
                groups[index].append(r)
        groups=[sorted(g,key=lambda r:r.sample) for g in groups]
    return groups,dict(excluded)


def transform(groups,condition,drug,threshold,seed,feature):
    rngseed=int(hashlib.sha256(f'{condition}:{drug}:{threshold}:{seed}:{feature}'.encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(rngseed);result=[list(g) for g in groups];maps=[]
    for role,g in zip(ROLES,groups):
        donors={r.sample:r.sample for r in g}
        buckets=defaultdict(list)
        if role in ('source','query_pool'):
            for i,r in enumerate(g):
                key=r.project if condition=='label_within_component' else (category(r.country),period(r.year)) if condition=='features_within_context' else 'all'
                buckets[key].append(i)
            if condition in ('label_global','label_within_component','features_within_context'):
                for indices in buckets.values():
                    shuffled=rng.permutation(indices)
                    for i,j in zip(indices,shuffled):
                        donors[g[i].sample]=g[int(j)].sample
                        result[ROLES.index(role)][i]=replace(g[i],features=g[int(j)].features) if condition=='features_within_context' else replace(g[i],label=g[int(j)].label)
        lookup={r.sample:r for r in result[ROLES.index(role)]}
        maps.extend({'sample':r.sample,'component':r.project,'role':role,'year':r.year,'standard':r.country,
                     'original_label':r.label,'training_label':lookup[r.sample].label,'donor':donors[r.sample],
                     'label_changed':r.label!=lookup[r.sample].label,'features_changed':r.features!=lookup[r.sample].features}
                    for r in g)
    return result,maps,rngseed


def main():
    OUT.mkdir(exist_ok=True)
    screen=read(HERE/'ecoli_full_skani_screen.json');ann=read(DATA/'acquired_amr_v1/manifest.json')
    assert screen['status']=='complete' and screen['queries_completed']==3516 and ann['status']=='complete' and ann['completed']==3516
    names,raw,inv,meta,tokens=population()
    source_names=('run_reliability_controls.py','run_decision_repair.py','decision_repair_core.py','counterexample_guided_repair.py','run_ecoli_repair.py','run_ecoli_ani_annotation_repair.py')
    inputs=[HERE/'ecoli_full_skani_screen.json',DATA/'acquired_amr_v1/manifest.json',DATA/'ecoli_human_sr_pairs.tsv',DATA/'ecoli_fasta_integrity.json',DATA/'ecoli_assemblies_k31_scaled1000.sig',HERE/'RELIABILITY_CONTROLS_PROTOCOL.md']
    binding={'code':{n:sha(HERE/n) for n in source_names},'inputs':{str(p):sha(p) for p in inputs},
             'annotations':{n:sha(DATA/'acquired_amr_v1'/(n+'.json')) for n in names}}
    bp=OUT/'binding.json'
    if bp.exists():assert read(bp)==binding,'Refuse mixed control versions'
    else:write_json(bp,binding)
    manifest={'status':'running','expected_configurations':2520,'binding_sha256':sha(bp),
              'designation':'post-result exploratory permutation and transport stress tests'}
    write_json(OUT/'manifest.json',manifest);metrics=[];skips=[];artifacts=[];plan=[]
    for t in (.99,.999):
        clusters=components(names,screen,t,raw,inv)
        for drug in DRUGS:
            for feature in ('homolog','sketch'):
                records=[Record(n,clusters[n],v['year'],v['standard'],tokens[feature][n],('__HOMOLOGY_CANDIDATES__' if feature=='homolog' else '__SEQUENCE_SKETCH__',),v['label'])
                         for (n,d),v in sorted(meta.items()) if d==drug]
                for seed in SEEDS:
                    for condition in CONDITIONS:
                        groups,excluded=partition(records,condition,seed)
                        transformed,mapping,rngseed=transform(groups,condition,drug,t,seed,feature)
                        sizes=list(map(len,groups));source,pool,audit,evaluation=transformed
                        config={'drug':drug,'threshold':t,'feature':feature,'seed':seed,'condition':condition}
                        stem=f'{drug.replace("/","-")}_ani{t}_{feature}_{seed}_{condition}'
                        pp=OUT/(stem+'_plan.json')
                        descriptor={**config,'sizes':sizes,'excluded':excluded,'permutation_seed':rngseed,'records':mapping}
                        if pp.exists():assert read(pp)==descriptor
                        else:write_json(pp,descriptor)
                        plan.append({'path':str(pp),'sha256':sha(pp)})
                        reason='insufficient_source_classes' if len({r.label for r in source})<2 else 'query_or_evaluation_below_10' if len(pool)<10 or len(evaluation)<10 else 'empty_source_features' if not any(r.features for r in source) else None
                        if reason:
                            skips.extend({**config,'strategy':s,'sizes':sizes,'reason':reason,'excluded':excluded} for s in STRATEGIES);continue
                        sets=[{r.project for r in g} for g in groups]
                        assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
                        for strategy in STRATEGIES:
                            jp=OUT/(stem+'_'+strategy+'.json');pz=jp.with_suffix('.npz')
                            if jp.exists():
                                shard=read(jp);assert shard['binding_sha256']==sha(bp) and shard['prediction_sha256']==sha(pz) and shard['plan_sha256']==sha(pp)
                                metrics.extend(shard['rows']);artifacts.append({'path':str(pz),'sha256':sha(pz)});continue
                            budgets=[b for b in (0,10,25,50) if b<=len(pool)];oracle={r.sample:r for r in pool};queried=[]
                            def reveal(n):
                                assert n in oracle and n not in queried
                                queried.append(n);return oracle[n]
                            snapshots,events=trajectory(source,mask(pool),mask(evaluation),mask(audit),reveal,strategy,seed,budgets)
                            arrays={'labels':np.array([r.label for r in evaluation]),'audit_labels':np.array([r.label for r in audit])}
                            for role,g in zip(ROLES,groups):
                                arrays[role+'_samples']=np.array([r.sample for r in g]);arrays[role+'_clusters']=np.array([r.project for r in g])
                            result=[]
                            for snap in snapshots:
                                b=snap['budget'];p=snap['evaluation_p'];arrays[f'p_b{b}']=p;arrays[f'audit_p_b{b}']=snap['audit_p']
                                result.append({**config,'strategy':strategy,'budget':b,'sizes':sizes,'total_target_assay_cost':b+len(audit),
                                               'queried_samples':snap['queried_samples'],'evidence_weights':snap['evidence_weights'],'action':snap['action'],**extra_metrics(arrays['labels'],p)})
                            np.savez_compressed(pz,**arrays)
                            write_json(jp,{'binding_sha256':sha(bp),'plan_sha256':sha(pp),'prediction_sha256':sha(pz),'plan_path':str(pp),'rows':result,'events':events})
                            metrics.extend(result);artifacts.append({'path':str(pz),'sha256':sha(pz)})
                        manifest.update(trajectories=len(artifacts),metric_rows=len(metrics),skipped_configurations=len(skips));write_json(OUT/'manifest.json',manifest)
                    print(json.dumps({'threshold':t,'drug':drug,'feature':feature,'seed':seed,'trajectories':len(artifacts),'skips':len(skips)}),flush=True)
    write_json(OUT/'results.json',{'status':'complete','rows':metrics,'skipped':skips,'predictions':artifacts,'plans':plan,
                                  'limitations':['Exploratory retrospective tests, no blind/clinical evidence.','Synthetic-label controls are not real phenotype results.','Temporal and standard transport may be confounded.','One permutation per seed is not a permutation significance test.']})
    manifest.update(status='complete',trajectories=len(artifacts),metric_rows=len(metrics),skipped_configurations=len(skips),result_sha256=sha(OUT/'results.json'))
    assert len(artifacts)+len(skips)==2520
    write_json(OUT/'manifest.json',manifest)


if __name__=='__main__':main()
