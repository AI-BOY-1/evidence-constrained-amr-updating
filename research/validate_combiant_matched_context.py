"""Rebuild matched plans, retrain all fits and recompute every comparison."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

BASE=ncs_paths.data_path('experiments/combiant_public_v1/own_vision_v1')
OUT=BASE.parent/'matched_context_v1'
DIST=('A','B','C','AB','BC','AC')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    r=load(OUT/'results.json');records=load(OUT/'predictions.json');old=load(BASE/'results.json')
    for p,d in r['binding'].items():assert sha(Path(p))==d,p
    assert sha(OUT/'predictions.json')==r['prediction_sha256']
    with np.load(BASE/'features_targets.npz',allow_pickle=False) as z:
        X=z['X32'];y=z['targets'];names=z['names'].tolist();plates=z['plates'];users=z['users']
    lookup={(x['fold'],x['method'],x['regime'],x['name']):x for x in records}
    assert len(lookup)==len(records)==r['prediction_rows']==279*8
    fits=0;maximum=0.;errors={}
    for p in r['plans']:
        orig=next(x for x in old['fold_plan'] if x['fold']==p['fold'])
        eligible=[i for i,n in enumerate(names) if n in orig['train_names'] and np.isfinite(y[i]).all()]
        eligible_plates=[]
        for plate in sorted(set(plates[eligible].tolist())):
            if all(sum(plates[i]==plate and users[i]==u for i in eligible)==1 for u in (1,2,3)):eligible_plates.append(plate)
        assert eligible_plates==p['common_train_plates']
        order=np.random.default_rng(20261010+p['fold']).permutation([1,2,3]).tolist()
        assert order==p['mixed_user_order']
        test=[i for i,n in enumerate(names) if n in orig['test_names']]
        assert p['test_names']==[names[i] for i in test]
        for regime in ('user1','user2','user3','mixed'):
            ix=[]
            for j,plate in enumerate(eligible_plates):
                u=order[j%3] if regime=='mixed' else int(regime[-1])
                ix.append(next(i for i in eligible if plates[i]==plate and users[i]==u))
            assert [names[i] for i in ix]==p['train_names'][regime]
            assert not set(plates[ix])&set(plates[test])
            for method in ('ridge','extra_trees'):
                model=Ridge(alpha=100.) if method=='ridge' else ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=1.,random_state=7,n_jobs=1)
                model.fit(X[ix],y[ix]);pred=model.predict(X[test]);fits+=1
                for i,pr in zip(test,pred):
                    row=lookup[p['fold'],method,regime,names[i]]
                    assert row['plate']==plates[i] and row['user']==users[i]
                    assert np.array_equal(row['target'],y[i],equal_nan=True)
                    err=float(np.max(np.abs(pr-np.array(row['prediction']))));maximum=max(maximum,err);assert err<1e-7
                    errors[method,regime,names[i]]=np.abs(pr-y[i])
    assert fits==40==r['fits']
    maxmetric=0.
    def check(row,field,value):
        nonlocal maxmetric
        e=abs(float(row[field])-float(value));maxmetric=max(maxmetric,e);assert e<1e-10,(field,e)
    def value(e,d):return float(sum(e)/6) if d=='macro' else float(e[DIST.index(d)])
    assert len(r['summaries'])==224 and len(r['paired'])==126
    assert len({(x['method'],x['regime'],x['test_user'],x['distance']) for x in r['summaries']})==224
    assert len({(x['method'],x['regime'],x['test_user'],x['distance']) for x in r['paired']})==126
    for row in r['summaries']:
        indexes=[i for i,n in enumerate(names) if row['test_user']=='all' or str(users[i])==row['test_user']]
        vals=[]
        for i in indexes:
            e=errors[row['method'],row['regime'],names[i]]
            if row['distance']=='macro' and not np.isfinite(e).all():continue
            q=value(e,row['distance'])
            if np.isfinite(q):vals.append((int(plates[i]),q))
        ps=set(p for p,q in vals);a=[q for p,q in vals]
        assert row['n']==len(a) and row['plates']==len(ps)
        for field,v in [('mae_mm',np.mean(a)),('p95_absolute_mm',np.percentile(a,95)),('max_absolute_mm',max(a)),
            ('over_2mm',sum(q>2 for q in a)),('plate_equal_mae_mm',np.mean([np.mean([q for p,q in vals if p==plate]) for plate in ps]))]:check(row,field,v)
    for row in r['paired']:
        assert row['reference']==f"user{row['test_user']}"
        vals=[]
        for i,n in enumerate(names):
            if users[i]!=row['test_user']:continue
            e=errors[row['method'],row['regime'],n];b=errors[row['method'],row['reference'],n]
            if row['distance']=='macro' and not np.isfinite(e).all():continue
            q,w=value(e,row['distance']),value(b,row['distance'])
            if np.isfinite(q+w):vals.append((int(plates[i]),q,w))
        ps=set(p for p,q,w in vals)
        assert row['n']==len(vals) and row['plates']==len(ps)
        for field,v in [('delta_mm',np.mean([q-w for p,q,w in vals])),
            ('plate_equal_delta_mm',np.mean([np.mean([q-w for p,q,w in vals if p==plate]) for plate in ps])),
            ('improved',sum(q<w for p,q,w in vals)),('worsened',sum(q>w for p,q,w in vals)),
            ('new_over_2mm',sum(q>2 and w<=2 for p,q,w in vals)),('removed_over_2mm',sum(q<=2 and w>2 for p,q,w in vals))]:check(row,field,v)
    v={'status':'passed','result_sha256':sha(OUT/'results.json'),'predictions_sha256':sha(OUT/'predictions.json'),
        'validator_sha256':sha(Path(__file__)),'fits_replayed':fits,'prediction_rows':len(records),
        'summaries_recomputed':len(r['summaries']),'paired_recomputed':len(r['paired']),
        'max_prediction_error':maximum,'max_metric_error':maxmetric,
        'checks':{'equal_training_plates_and_counts':True,'original_fold_sample_and_plate_isolation':True,
            'mixed_view_assignment':True,'raw_targets':True,'all_models_retrained':True,'full_comparison_coverage':True}}
    (OUT/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
