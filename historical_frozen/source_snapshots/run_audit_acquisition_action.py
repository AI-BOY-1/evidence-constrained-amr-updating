"""Label-blind audit acquisition and action replay on frozen ANI predictions."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=Path('D:/NCS_data/experiments/ecoli_bvbrc')
SOURCE=DATA/'ani_annotation_repair_v1'
OUT=DATA/'audit_acquisition_action_v1'
BUDGETS=(10,25,50)
SIZES=(5,10,25,50)
ACQUISITIONS=('random','predicted_R','component_coverage')
RULES=('retain','update','audit_loss','audit_direction','component_direction')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()

def acquisition_order(samples,clusters,base_p,seed,method):
    """No label argument and no evaluation information."""
    ranks=[hashlib.sha256(f'audit-acquisition-v1:{seed}:{s}'.encode()).hexdigest() for s in samples]
    random_order=sorted(range(len(samples)),key=lambda i:(ranks[i],str(samples[i])))
    if method=='random':return random_order
    if method=='predicted_R':return sorted(random_order,key=lambda i:-float(base_p[i]))
    buckets=defaultdict(list)
    for i in random_order:buckets[str(clusters[i])].append(i)
    keys=sorted(buckets,key=lambda c:hashlib.sha256(f'audit-component-v1:{seed}:{c}'.encode()).hexdigest())
    return [buckets[c][level] for level in range(max(map(len,buckets.values()),default=0))
            for c in keys if level<len(buckets[c])]

def score(y,p):
    y=np.asarray(y);p=np.asarray(p);r=y==1;s=y==0
    return {'n':len(y),'R':int(r.sum()),'S':int(s.sum()),'brier':float(np.mean((p-y)**2)),
            'FS':int(np.sum(r&(p<.5))),'FR':int(np.sum(s&(p>=.5))),
            'R_brier':float(np.mean((p[r]-1)**2)) if r.any() else None,
            'S_brier':float(np.mean(p[s]**2)) if s.any() else None}

def change(y,old,new):
    err0=(old>=.5)!=y;err1=(new>=.5)!=y
    return {'corrected':int(np.sum(err0&~err1)),'introduced':int(np.sum(~err0&err1)),
            'FS_corrected':int(np.sum((y==1)&err0&~err1)),
            'FS_introduced':int(np.sum((y==1)&~err0&err1)),
            'FR_corrected':int(np.sum((y==0)&err0&~err1)),
            'FR_introduced':int(np.sum((y==0)&~err0&err1))}

def audit_evidence(y,old,new,clusters):
    a,b=score(y,old),score(y,new)
    groups=[np.asarray(clusters)==c for c in sorted(set(clusters))]
    equal_component_loss=float(np.mean([np.mean((new[g]-y[g])**2-(old[g]-y[g])**2) for g in groups]))
    resistant_groups=[g&(y==1) for g in groups if np.any(g&(y==1))]
    eqfs=float(np.mean([np.mean(new[g]<.5)-np.mean(old[g]<.5) for g in resistant_groups])) if resistant_groups else None
    return {'old':a,'new':b,'components':len(groups),'R_components':len(resistant_groups),
            'brier_delta':b['brier']-a['brier'],'FS_delta':b['FS']-a['FS'],
            'component_brier_delta':equal_component_loss,'component_FS_rate_delta':eqfs}

def decide(e,rule):
    if rule=='retain':return False,'fixed'
    if rule=='update':return True,'fixed'
    if rule=='audit_loss':return e['brier_delta']<0,'eligible'
    if e['old']['R']==0:return False,'infeasible_no_audit_R_fallback_retain'
    if rule=='audit_direction':return e['brier_delta']<0 and e['FS_delta']<=0,'eligible'
    return e['component_brier_delta']<0 and e['component_FS_rate_delta']<=0,'eligible'

def replay(path,metadata,binding_sha):
    z=np.load(path,allow_pickle=False)
    y=z['labels'];ya=z['audit_labels'];samples=z['audit_samples'];clusters=z['audit_clusters']
    old=z['p_b0'];audit0=z['audit_p_b0'];base=score(y,old)
    info=metadata['rows'][0]
    identity={k:info[k] for k in ('drug','ANI_threshold','feature','strategy','seed')}
    orders={a:acquisition_order(samples,clusters,audit0,info['seed'],a) for a in ACQUISITIONS}
    conditions=[];skips=[]
    for b in BUDGETS:
        for n in SIZES:
            for method in ACQUISITIONS:
                key={'query_budget':b,'audit_size':n,'acquisition':method}
                if f'p_b{b}' not in z.files or n>len(ya):
                    skips.append({**key,'reason':'missing_query_budget' if f'p_b{b}' not in z.files else 'audit_pool_smaller_than_fixed_size',
                                  'audit_pool_n':len(ya),'rules_skipped':list(RULES)});continue
                ix=np.asarray(orders[method][:n]);new=z[f'p_b{b}'];audnew=z[f'audit_p_b{b}']
                e=audit_evidence(ya[ix],audit0[ix],audnew[ix],clusters[ix])
                budgetrow=next(r for r in metadata['rows'] if r['budget']==b)
                actions=[]
                newscore=score(y,new);newchange=change(y,old,new)
                for rule in RULES:
                    accept,status=decide(e,rule);v=newscore if accept else base
                    t=newchange if accept else change(y,old,old)
                    actions.append({'rule':rule,'decision_status':status,'accepted':bool(accept),
                                    'selected_budget':b if accept else 0,'total_new_label_cost':b+n,
                                    'source_labels_sunk_cost':len(z['source_samples']),
                                    'evaluation':v,'change_from_frozen':t,
                                    'evaluation_brier_delta_from_frozen':v['brier']-base['brier'],
                                    'evaluation_FS_delta_from_frozen':v['FS']-base['FS'],
                                    'oracle_brier_regret':max(0.,v['brier']-min(base['brier'],newscore['brier']))})
                conditions.append({**key,'audit_samples':[str(samples[i]) for i in ix],
                                   'audit_components':[str(clusters[i]) for i in ix],
                                   'query_samples':budgetrow['queried_samples'],'audit_evidence':e,'actions':actions})
    return {'identity':identity,'binding_sha256':binding_sha,'source_npz':str(path),
            'source_npz_sha256':sha(path),'source_metadata_sha256':sha(path.with_suffix('.json')),
            'evaluation_n':len(y),'evaluation_R':int(y.sum()),'audit_pool_n':len(ya),
            'conditions':conditions,'skipped_conditions':skips}

def main(limit=None):
    OUT.mkdir(exist_ok=True)
    v=read(HERE/'ecoli_ani_annotation_validation.json');assert v['status']=='passed'
    assert v['result_sha256']==sha(SOURCE/'results.json')
    data=read(SOURCE/'results.json');assert len(data['predictions'])==680
    inputs=[HERE/'ecoli_ani_annotation_validation.json',SOURCE/'results.json',SOURCE/'binding.json',
            HERE/'AUDIT_ACQUISITION_ACTION_PROTOCOL_V1.md',HERE/'run_audit_acquisition_action.py']
    inputs += [HERE/n for n in ('run_decision_repair.py','decision_repair_core.py','counterexample_guided_repair.py')]
    binding={'inputs':{str(p):sha(p) for p in inputs},'predictions':data['predictions']}
    bp=OUT/'binding.json'
    if bp.exists():assert read(bp)==binding,'Refuse mixed audit-acquisition versions'
    else:write(bp,binding)
    bs=sha(bp);mp=OUT/'manifest.json'
    prior=read(mp) if mp.exists() else {}
    if prior.get('status')=='complete':
        assert prior['binding_sha256']==bs and prior['result_sha256']==sha(OUT/'results.json')
        print(json.dumps({'status':'existing_complete'}));return
    lock=OUT/'run.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Lock present; inspect owner before rerunning')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    manifest={'status':'running','pid':os.getpid(),'binding_sha256':bs,'expected_files':680,
              'started_utc':prior.get('started_utc',utc()),'designation':'posthoc_frozen_audit_acquisition_intervention'}
    write(mp,manifest);artifacts=[];conds=skips=actions=0
    try:
        for i,a in enumerate(data['predictions']):
            if limit is not None and i>=limit:break
            p=Path(a['path']);assert sha(p)==a['sha256']
            meta=read(p.with_suffix('.json'));assert meta['prediction_sha256']==a['sha256']
            dest=OUT/(p.stem+'.json')
            if dest.exists():
                r=read(dest);assert r['binding_sha256']==bs and r['source_npz_sha256']==a['sha256']
                assert r['source_metadata_sha256']==sha(p.with_suffix('.json'))
            else:
                r=replay(p,meta,bs);write(dest,r)
            artifacts.append({'path':str(dest),'sha256':sha(dest)})
            conds+=len(r['conditions']);skips+=len(r['skipped_conditions']);actions+=sum(len(c['actions']) for c in r['conditions'])
            manifest.update(completed_files=len(artifacts),conditions=conds,skipped_conditions=skips,action_rows=actions,
                            last_file=p.name,updated_utc=utc());write(mp,manifest)
            if (i+1)%40==0:print(json.dumps({'files':i+1,'conditions':conds,'actions':actions}),flush=True)
        result={'status':'partial_fixture_run' if limit is not None else 'complete_pending_validation',
                'binding_sha256':bs,'artifacts':artifacts,'conditions':conds,'action_rows':actions,
                'skipped_conditions':skips,'inherited_skips':data['skipped'],
                'inherited_skipped_trajectories':4*len(data['skipped']),
                'expected_conditions_from_completed_trajectories':680*3*4*3,
                'limitations':['Retrospective frozen predictions, not new training or blind confirmation.',
                               'Query path fixed; audit size changes do not retrain under reallocated total budget.',
                               'All candidates within a path are charged its consumed query and selected audit labels.',
                               'Correlated configuration counts, not independent patients or significance.']}
        write(OUT/'results.json',result)
        manifest.update(status=result['status'],result_sha256=sha(OUT/'results.json'),finished_utc=utc());write(mp,manifest)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),updated_utc=utc());write(mp,manifest);raise
    finally:lock.unlink()
    print(json.dumps({k:manifest[k] for k in ('status','completed_files','conditions','skipped_conditions','action_rows')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args();main(args.limit)
