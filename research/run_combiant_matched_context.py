"""Matched plates and counts; fixed post-result acquisition-context experiment."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BASE=ncs_paths.data_path('experiments/combiant_public_v1/own_vision_v1')
OUT=BASE.parent/'matched_context_v1'
DIST=('A','B','C','AB','BC','AC')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def estimator(method):
    return Ridge(alpha=100.) if method=='ridge' else ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=1.,random_state=7,n_jobs=1)

def summarize(records):
    summaries=[];paired=[]
    for method in ('ridge','extra_trees'):
        for regime in ('user1','user2','user3','mixed'):
            for user in ('all','1','2','3'):
                rs=[r for r in records if r['method']==method and r['regime']==regime and (user=='all' or str(r['user'])==user)]
                for distance in (*DIST,'macro'):
                    vals=[]
                    for r in rs:
                        e=np.abs(np.asarray(r['prediction'])-np.asarray(r['target']))
                        if distance=='macro':
                            if not np.isfinite(e).all():continue
                            value=float(e.mean())
                        else:
                            value=float(e[DIST.index(distance)])
                            if not np.isfinite(value):continue
                        vals.append((r['plate'],value))
                    if not vals:continue
                    plates=sorted({p for p,v in vals});vs=[v for p,v in vals]
                    summaries.append({'method':method,'regime':regime,'test_user':user,'distance':distance,
                        'n':len(vs),'plates':len(plates),'mae_mm':float(np.mean(vs)),
                        'plate_equal_mae_mm':float(np.mean([np.mean([v for p,v in vals if p==plate]) for plate in plates])),
                        'p95_absolute_mm':float(np.quantile(vs,.95)),'max_absolute_mm':max(vs),'over_2mm':sum(v>2 for v in vs)})
        by={(r['regime'],r['name']):r for r in records if r['method']==method}
        for user in (1,2,3):
            for regime in ('user1','user2','user3','mixed'):
                if regime==f'user{user}':continue
                rs=[r for r in records if r['method']==method and r['regime']==regime and r['user']==user]
                for distance in (*DIST,'macro'):
                    values=[]
                    for r in rs:
                        ref=by[f'user{user}',r['name']]
                        assert r['target']==ref['target'] or np.array_equal(r['target'],ref['target'],equal_nan=True)
                        e=np.abs(np.asarray(r['prediction'])-np.asarray(r['target']))
                        b=np.abs(np.asarray(ref['prediction'])-np.asarray(ref['target']))
                        if distance=='macro':
                            if not np.isfinite(e).all():continue
                            v,w=float(e.mean()),float(b.mean())
                        else:
                            v,w=float(e[DIST.index(distance)]),float(b[DIST.index(distance)])
                            if not np.isfinite(v+w):continue
                        values.append((r['plate'],v,w))
                    plates=sorted({p for p,v,w in values})
                    paired.append({'method':method,'regime':regime,'reference':f'user{user}','test_user':user,'distance':distance,
                        'n':len(values),'plates':len(plates),'delta_mm':float(np.mean([v-w for p,v,w in values])),
                        'plate_equal_delta_mm':float(np.mean([np.mean([v-w for p,v,w in values if p==plate]) for plate in plates])),
                        'improved':sum(v<w for p,v,w in values),'worsened':sum(v>w for p,v,w in values),
                        'new_over_2mm':sum(v>2 and w<=2 for p,v,w in values),'removed_over_2mm':sum(v<=2 and w>2 for p,v,w in values)})
    return summaries,paired

def main():
    OUT.mkdir(exist_ok=True)
    lock=OUT/'run.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        inputs=[BASE/'results.json',BASE/'validation.json',BASE/'features_targets.npz',HERE/'COMBIANT_MATCHED_CONTEXT_PROTOCOL.md',Path(__file__)]
        binding={str(p):sha(p) for p in inputs}
        if (OUT/'results.json').exists():
            assert read(OUT/'results.json')['binding']==binding,'Completed source binding changed'
            print('Existing matched experiment retained');return
        previous=read(BASE/'results.json');validation=read(BASE/'validation.json')
        assert validation['status']=='passed' and validation['result_sha256']==sha(BASE/'results.json')
        assert previous['features_sha256']==sha(BASE/'features_targets.npz')
        with np.load(BASE/'features_targets.npz',allow_pickle=False) as z:
            X=z['X32'];y=z['targets'];names=z['names'].tolist();plates=z['plates'];users=z['users']
        complete=np.isfinite(y).all(axis=1)
        plans=[];records=[];fits=0
        for plan in previous['fold_plan']:
            fold=plan['fold'];test=np.array([i for i,n in enumerate(names) if n in plan['test_names']])
            train=np.array([i for i,n in enumerate(names) if n in plan['train_names']])
            maps={u:{int(plates[i]):int(i) for i in train if int(users[i])==u and complete[i]} for u in (1,2,3)}
            common=sorted(set(maps[1])&set(maps[2])&set(maps[3]));assert common
            user_order=np.random.default_rng(20261010+fold).permutation([1,2,3]).tolist()
            choices={f'user{u}':[maps[u][p] for p in common] for u in (1,2,3)}
            choices['mixed']=[maps[user_order[j%3]][p] for j,p in enumerate(common)]
            plans.append({'fold':fold,'common_train_plates':common,'test_names':[names[i] for i in test],
                'train_names':{k:[names[i] for i in indexes] for k,indexes in choices.items()},'mixed_user_order':user_order})
            for regime,ix in choices.items():
                assert len(ix)==len(common) and not set(plates[ix])&set(plates[test])
                for method in ('ridge','extra_trees'):
                    model=estimator(method);model.fit(X[ix],y[ix]);pred=model.predict(X[test]);fits+=1
                    for i,p in zip(test,pred):
                        records.append({'fold':fold,'method':method,'regime':regime,'name':names[i],
                            'plate':int(plates[i]),'user':int(users[i]),'target':y[i].tolist(),'prediction':p.tolist()})
                    print(json.dumps({'fold':fold,'regime':regime,'method':method,'training_n':len(ix),'fits':fits}),flush=True)
        summary,paired=summarize(records)
        (OUT/'predictions.json').write_text(json.dumps(records),encoding='utf-8')
        result={'status':'computed_pending_independent_validation','binding':binding,'fits':fits,'plans':plans,
            'prediction_rows':len(records),'prediction_sha256':sha(OUT/'predictions.json'),'summaries':summary,'paired':paired,
            'designation':'post-result matched-count and matched-plate exploratory experiment'}
        (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        for filename,rs in (('summary.csv',summary),('paired.csv',paired)):
            with (OUT/filename).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
        print(json.dumps({'fits':fits,'summary_rows':len(summary),'paired_rows':len(paired)}))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
