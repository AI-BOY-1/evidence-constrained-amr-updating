"""Replay frozen stratum membership and metrics without the audit metric code."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
import numpy as np
from counterexample_guided_repair import parse_year

HERE=Path(__file__).resolve().parent
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=ROOT/'reliability_debt_audit_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def measurements(y,p):
    n=len(y);r=int(np.count_nonzero(y));s=n-r
    fs=int(np.count_nonzero((y==1)&(p<.5)));fr=int(np.count_nonzero((y==0)&(p>=.5)))
    ece=0
    if n:
        for b in range(10):
            ix=np.clip(np.floor(p*10).astype(int),0,9)==b
            if ix.any():ece+=float(ix.sum()/n*abs(p[ix].mean()-y[ix].mean()))
    return dict(n=n,R=r,brier=float(np.dot(p-y,p-y)/n) if n else None,
        mean_probability=float(p.sum()/n) if n else None,
        calibration_in_large=float((p.sum()-r)/n) if n else None,ece10=ece if n else None,
        false_susceptible=fs,false_susceptible_rate=fs/r if r else None,
        false_resistant=fr,false_resistant_rate=fr/s if s else None)


def main():
    ref=read(HERE/'reliability_debt_audit.json');manifest=read(OUT/'manifest.json');data=read(OUT/'results.json')
    assert ref['version']==2 and ref['manifest_sha256']==sha(OUT/'manifest.json')
    assert ref['result_sha256']==manifest['result_sha256']==sha(OUT/'results.json')
    assert manifest['status']==data['status']=='complete'
    assert manifest['code_sha256']==sha(HERE/'run_reliability_debt_audit.py')
    assert manifest['inputs']['pair_sha256']==sha(ROOT/'ecoli_human_sr_pairs.tsv')
    assert manifest['inputs']['full_manifest_sha256']==sha(ROOT/'repair_full_scaled5000/manifest.json')
    original=read(HERE/'ecoli_full_validation.json');assert original['status']=='passed'
    pairs=defaultdict(list)
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            pairs[(r['sequence_id'] or r['assembly_accession'],r['antibiotic'])].append(r)
    metrics={};errors=[];groups={};count=0
    def compare(expected,actual):
        assert expected.keys()==actual.keys()
        for k,v in expected.items():
            if v is None:assert actual[k] is None,(k,actual[k])
            else:
                error=abs(v-actual[k]);errors.append(error);assert error<1e-10,(k,error)
    for a in data['prediction_artifacts']:
        p=Path(a['path']);assert a['sha256']==sha(p)
        shard=read(p.with_suffix('.json'));first=shard['results'][0]
        drug,t,seed,strategy=(first[k] for k in ('drug','threshold','seed','strategy'))
        config=(drug,t,seed,strategy);assert config not in groups
        with np.load(p,allow_pickle=False) as z:
            samples=z['evaluation_samples'].tolist();y=z['labels']
            sizes=Counter(c for role in ('source','query_pool','audit','evaluation') for c in z[role+'_clusters'].tolist())
            members={('all','all'):list(range(len(y)))}
            for field,levels in manifest['strata'].items():
                for level in levels:members[(field,level)]=[]
            for i,(n,c) in enumerate(zip(samples,z['evaluation_clusters'].tolist())):
                rs=pairs[(n,drug)];assert rs
                assert {r['label'] for r in rs}=={'Resistant' if y[i] else 'Susceptible'}
                standards={s.strip().upper() for r in rs for s in r['standards'].replace(';',',').split(',') if s.strip()}
                std='EUCAST' if standards=={'EUCAST'} else 'CLSI' if standards=={'CLSI'} else 'mixed' if standards=={'EUCAST','CLSI'} else 'unknown'
                years={parse_year(v) for r in rs for v in r['collection_date'].split(';')}
                year=next(iter(years)) if len(years)==1 else None
                era='unknown' if year is None else '<2015' if year<2015 else '2015-2019' if year<2020 else '>=2020'
                size='singleton' if sizes[c]==1 else '2-9' if sizes[c]<10 else '>=10'
                for field,level in [('standard',std),('collection_period',era),('component_size',size)]:members[(field,level)].append(i)
            groups[config]=samples
            for budget in (0,10,25,50):
                if f'p_b{budget}' not in z.files:continue
                pred=z[f'p_b{budget}'];assert pred.shape==y.shape and np.all(np.isfinite(pred))
                for (field,level),ix in members.items():
                    metrics[config+(budget,field,level)]=measurements(y[ix],pred[ix]);count+=1
    expected_configs=set(product({a[0] for a in groups},(.99,.95),(7,17,27,37,47),('decision_diagnostic','random_gene','uncertainty_gene','random_tree')))
    assert len(expected_configs)==360 and set(groups)==expected_configs
    actual={}
    for r in data['stratified_rows']:
        k=tuple(r[n] for n in ('drug','threshold','seed','strategy','budget','stratum','level'))
        assert k not in actual;actual[k]=r
        compare(metrics[k],{n:r[n] for n in metrics[k]})
    assert set(actual)==set(metrics) and count==manifest['metric_rows']
    expected_contrasts={};omitted={}
    for k,left in metrics.items():
        drug,t,seed,strategy,budget,field,level=k
        if strategy!='decision_diagnostic' or budget not in (25,50):continue
        right=metrics[(drug,t,seed,'random_gene',budget,field,level)]
        assert groups[(drug,t,seed,strategy)]==groups[(drug,t,seed,'random_gene')]
        ck=(drug,t,seed,budget,field,level)
        if left['n']<10:omitted[ck]=left['n'];continue
        assert left['n']==right['n'] and left['R']==right['R']
        expected_contrasts[ck]=dict(n=left['n'],R=left['R'],brier_delta=left['brier']-right['brier'],
            false_susceptible_delta=left['false_susceptible']-right['false_susceptible'],
            false_susceptible_rate_delta=left['false_susceptible_rate']-right['false_susceptible_rate'] if left['R'] else None)
    seen=set();buckets=defaultdict(list)
    for r in data['contrasts']:
        k=tuple(r[n] for n in ('drug','threshold','seed','budget','stratum','level'));assert k not in seen;seen.add(k)
        compare(expected_contrasts[k],{n:r[n] for n in expected_contrasts[k]})
        buckets[(k[0],k[1],k[3],k[4],k[5])].append(expected_contrasts[k])
    assert seen==set(expected_contrasts)
    seen=set()
    for r in data['omitted_contrasts']:
        k=tuple(r[n] for n in ('drug','threshold','seed','budget','stratum','level'));assert k not in seen;seen.add(k)
        assert r['n']==omitted[k] and r['reason']=='fewer_than_10_evaluation_records'
    assert seen==set(omitted)
    seen=set()
    for r in data['summaries']:
        k=tuple(r[n] for n in ('drug','threshold','budget','stratum','level'));assert k not in seen;seen.add(k)
        rs=buckets[k];ds=[v['brier_delta'] for v in rs]
        calc=dict(partition_n=len(rs),mean_brier_delta=float(np.mean(ds)),median_brier_delta=float(np.median(ds)),
            lower_loss_partitions=sum(d<0 for d in ds),mean_false_susceptible_delta=float(np.mean([v['false_susceptible_delta'] for v in rs])),
            mean_n=float(np.mean([v['n'] for v in rs])),mean_R=float(np.mean([v['R'] for v in rs])))
        compare(calc,{n:r[n] for n in calc});assert np.allclose(r['brier_range'],[min(ds),max(ds)],atol=1e-10,rtol=0)
    assert seen==set(buckets)
    vp=HERE/'reliability_debt_validation.json';previous=read(vp) if vp.exists() else {}
    png=OUT/'reliability_debt_strata.png';digest=sha(png)
    review=previous.get('visual_review',{})
    if review.get('figure_sha256')!=digest:review={'status':'pending','figure_sha256':digest}
    result=dict(status='passed',prediction_files=len(groups),stratified_rows=count,contrasts=len(expected_contrasts),
        summaries=len(buckets),max_metric_error=max(errors,default=0),result_sha256=sha(OUT/'results.json'),
        validator_sha256=sha(Path(__file__)),figure_sha256=digest,visual_review=review,
        checks=['source_and_input_hashes','original_AST_labels','strata_including_empty','sample_pairing','metrics','contrasts_and_summaries'],
        limitations=['Descriptive strata using drug-eligible sketch component sizes; not causal effects or strict lineage validation.'])
    vp.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
