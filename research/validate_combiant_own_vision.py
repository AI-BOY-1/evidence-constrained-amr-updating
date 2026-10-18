"""Independently replay own visual fits, raw targets, held-out metrics and review budgets."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ncs_paths.data_path('experiments/combiant_public_v1')
D = ('A','B','C','AB','BC','AC')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def features(path, side):
    with Image.open(path) as original:
        original=original.convert('RGB'); w,h=original.size;edge=min(w,h)
        left,top=(w-edge)//2,(h-edge)//2
        arr=np.asarray(original.crop((left,top,left+edge,top+edge)).resize((side,side),Image.Resampling.BILINEAR),dtype=np.float32)/255
    gray=arr.mean(axis=-1)
    summaries=[]
    for channel in [arr[:,:,0],arr[:,:,1],arr[:,:,2],gray]:
        summaries += [channel.mean(),channel.std(),np.quantile(channel,.1),np.quantile(channel,.5),np.quantile(channel,.9)]
    rr,cc=np.mgrid[0:side,0:side];rad=np.sqrt((rr-(side-1)/2)**2+(cc-(side-1)/2)**2)/side
    summaries += [(gray*rad).mean(),(gray*rad**2).mean()]
    return np.concatenate((arr.ravel(),np.asarray(summaries,dtype=np.float32)))


def near(a,b):
    if b is None:
        assert a is None
    else:
        assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=('pilot','full'),default='full');args=ap.parse_args()
    out=BASE/('own_vision_pilot_v1' if args.scope=='pilot' else 'own_vision_v1')
    rp=out/'results.json';result=json.loads(rp.read_text(encoding='utf-8'))
    for raw,value in result['binding'].items():
        assert sha(Path(raw))==value,raw
    for field,name in [('features_sha256','features_targets.npz'),('predictions_sha256','predictions.csv'),
                       ('metrics_sha256','metrics.csv'),('review_sha256','review_controls.csv')]:
        assert sha(out/name)==result[field]
    with np.load(out/'features_targets.npz',allow_pickle=False) as archive:
        values={name:archive[name] for name in archive.files}
    names=values['names'].tolist();index={name:i for i,name in enumerate(names)}
    assert len(index)==len(names)==result['rows']
    targets=values['targets'];groups=values['plates'];users=values['users']
    manual={r['name']:r for r in csv_rows(BASE/'replication_subset/CombiANT_raw_data/combiANT_user_measurements.csv')}
    published={Path(r['name']).stem:r for r in csv_rows(BASE/'replication_subset/inference/output.csv') if Path(r['name']).stem in index}
    pair=json.loads((BASE/'pairing_audit.json').read_text(encoding='utf-8'))
    photo={r['name']:BASE/'replication_subset'/r['member'] for r in pair['pairs']}
    for name,i in index.items():
        assert int(float(manual[name]['plate']))==groups[i]
        for j,distance in enumerate(D):
            v=manual[name][distance+'_user']
            expected=float(v) if v else float('nan')
            assert (np.isnan(expected) and np.isnan(targets[i,j])) or expected==targets[i,j]
        for side in (16,32):
            np.testing.assert_array_equal(features(photo[name],side),values['X'+str(side)][i])
    predictions=csv_rows(out/'predictions.csv')
    keyed={(r['fold'],r['method'],r['name']):r for r in predictions}
    assert len(keyed)==len(predictions)
    metrics=[]; reviewed=csv_rows(out/'review_controls.csv'); review_checks=0
    pooled={};seen=[];max_error=0.0
    for plan in result['fold_plan']:
        fold=plan['fold']; train=np.asarray([index[n] for n in plan['train_names']]);test=np.asarray([index[n] for n in plan['test_names']])
        assert not set(groups[train]) & set(groups[test])
        assert set(groups[train]) == set(plan['train_plates']) and set(groups[test]) == set(plan['test_plates'])
        assert set(train)|set(test)==set(range(len(names))) and not set(train)&set(test)
        seen.extend(test.tolist())
        complete=train[np.isfinite(targets[train]).all(axis=1)]
        tasks=[('own_ridge_32px',32,Ridge(alpha=100.0),complete,test),
               ('own_ridge_16px',16,Ridge(alpha=100.0),complete,test),
               ('own_extra_trees_32px',32,ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=1.,random_state=7,n_jobs=1),complete,test)]
        if args.scope=='full':
            transfer_train=complete[users[complete]==1];transfer_test=test[np.isin(users[test],(2,3))]
            tasks += [('operator_transfer_ridge',32,Ridge(alpha=100.),transfer_train,transfer_test),
                      ('operator_transfer_extra_trees',32,ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,random_state=7,n_jobs=1),transfer_train,transfer_test)]
        for method,side,model,fit,eval_indices in tasks:
            x=values['X'+str(side)];model.fit(x[fit],targets[fit]);pred=model.predict(x[eval_indices])
            pooled.setdefault(method,[]).append((eval_indices,pred));metrics.append((str(fold),method,eval_indices,pred))
            for ix,p in zip(eval_indices,pred):
                saved=keyed[(str(fold),method,names[ix])]
                assert int(saved['plate'])==groups[ix] and int(saved['user'])==users[ix]
                saved_targets=np.asarray([float(saved['target_'+d]) for d in D])
                np.testing.assert_equal(saved_targets,targets[ix])
                restored=np.array([float(saved['prediction_'+d]) for d in D])
                max_error=max(max_error,float(np.max(np.abs(p-restored))))
                np.testing.assert_allclose(p,restored,rtol=1e-9,atol=1e-9)
            if method=='own_extra_trees_32px':
                risk=np.asarray([e.predict(x[test]) for e in model.estimators_]).std(axis=0).mean(axis=1)
                for ix,score in zip(test,risk):near(keyed[(str(fold),method,names[ix])]['risk_score'],score)
                gray=x[test,:3072].reshape(-1,32,32,3).mean(axis=3)
                lap=gray[:,2:,1:-1]+gray[:,:-2,1:-1]+gray[:,1:-1,2:]+gray[:,1:-1,:-2]-4*gray[:,1:-1,1:-1]
                quality=-lap.var(axis=(1,2));plates=sorted(set(groups[test].tolist()));err=np.abs(pred-targets[test])
                for row in [r for r in reviewed if r['fold']==str(fold)]:
                    strategy=row['strategy'];seed=int(row['seed'])
                    if strategy=='random':rank=np.random.default_rng(seed).permutation(plates).tolist()
                    else:
                        score=risk if strategy=='uncertainty' else quality
                        rank=sorted(plates,key=lambda p:(-float(score[groups[test]==p].mean()),p))
                    count=int(np.floor(len(plates)*float(row['budget_fraction'])))
                    ids=rank[:count];assert row['reviewed_ids']==';'.join(map(str,ids))
                    assert int(row['reviewed_plates'])==count
                    keep=~np.isin(groups[test],ids)
                    assert int(row['original_distance_n'])==np.isfinite(err).sum()
                    assert int(row['unreviewed_distance_n'])==np.isfinite(err[keep]).sum()
                    near(row['unresolved_mae_mm'],np.nansum(err[keep])/np.isfinite(err).sum())
                    near(row['unreviewed_mae_mm'],np.nanmean(err[keep]))
                    assert int(row['original_over_2mm'])==(err>2).sum()
                    assert int(row['unreviewed_over_2mm'])==(err[keep]>2).sum()
                    review_checks+=1
        mean=np.repeat(np.nanmean(targets[train],axis=0,keepdims=True),len(test),axis=0)
        sw=np.array([[float(published[names[i]][d]) for d in D] for i in test])
        for method,pred in [('train_plate_mean',mean),('published_software_reference',sw)]:
            pooled.setdefault(method,[]).append((test,pred));metrics.append((str(fold),method,test,pred))
    assert sorted(seen)==list(range(len(names)))
    for method,parts in pooled.items():metrics.append(('pooled',method,np.concatenate([x for x,_ in parts]),np.concatenate([p for _,p in parts])))
    metric_map={(str(r['fold']),r['method'],r['distance']):r for r in result['metrics']}
    checked=0
    for fold,method,ix,pred in metrics:
        delta=pred-targets[ix]
        for j,distance in enumerate(list(D)+['macro']):
            if distance=='macro':
                use=np.isfinite(delta).all(axis=1);e=np.abs(delta[use]).mean(axis=1);bias=float(delta[use].mean())
            else:
                use=np.isfinite(delta[:,j]);e=np.abs(delta[use,j]);bias=float(delta[use,j].mean()) if len(e) else None
            saved=metric_map[(fold,method,distance)]
            assert saved['n']==len(ix) and saved['finite_n']==len(e)
            for field,val in [('bias_mm',bias),('mae_mm',float(e.mean()) if len(e) else None),
                ('median_absolute_mm',float(np.median(e)) if len(e) else None),('p95_absolute_mm',float(np.quantile(e,.95)) if len(e) else None),
                ('max_absolute_mm',float(e.max()) if len(e) else None)]:near(saved[field],val)
            for cutoff in (1,2,3):assert saved[f'over_{cutoff}mm']==int((e>cutoff).sum())
            checked+=1
    assert checked==result['metric_rows'] and review_checks==len(reviewed)==result['review_rows']
    validation={'status':'passed','scope':args.scope,'result_sha256':sha(rp),'rows':len(names),'plates':len(set(groups.tolist())),
        'raw_targets_and_features_replayed':True,'fits_replayed':len(metrics)-len(pooled)-10,
        'sample_and_plate_isolation_passed':True,'metric_rows_recomputed':checked,'review_rows_recomputed':review_checks,
        'max_prediction_replay_error':max_error,'validator_sha256':sha(Path(__file__)),
        'limitations':['Pilot snapshot is operator-imbalanced and incomplete.' if args.scope=='pilot' else 'Public retrospective data; not a blind external confirmation.',
                       'Review is perfect-reference simulation, not an implemented human intervention.']}
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8');print(json.dumps(validation))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
