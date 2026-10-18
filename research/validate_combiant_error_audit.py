"""Independent row-wise validation of the frozen post-result error audit."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=ncs_paths.data_path('experiments/combiant_public_v1/own_vision_v1')
OUT=BASE/'error_audit_v1'
DIST=('A','B','C','AB','BC','AC')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    a=load(OUT/'results.json');r=load(BASE/'results.json');v=load(BASE/'validation.json')
    assert v['status']=='passed' and sha(BASE/'results.json')==v['result_sha256']
    for p,d in a['binding'].items():assert sha(Path(p))==d,p
    with np.load(BASE/'features_targets.npz',allow_pickle=False) as z:
        names=z['names'].tolist();targets=z['targets'].copy()
    raw=list(csv.DictReader((BASE.parent/'paired_distances.csv').open(encoding='utf-8',newline='')))
    software={(x['name'],x['distance']):float(x['software_mm'] or 'nan') for x in raw}
    rows=list(csv.DictReader((BASE/'predictions.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==a['prediction_rows']
    records={}
    for x in rows:
        k=tuple(x[f] for f in ('name','plate','user','fold'))
        y=np.array([float(x['target_'+d]) for d in DIST])
        assert np.array_equal(y,targets[names.index(x['name'])],equal_nan=True)
        pred=np.array([float(x['prediction_'+d]) for d in DIST])
        assert (x['method'],k) not in records
        records[x['method'],k]=np.abs(pred-y)
        if x['method']=='own_extra_trees_32px':
            plan=next(p for p in r['fold_plan'] if str(p['fold'])==x['fold'])
            train=targets[[names.index(n) for n in plan['train_names']]]
            records['train_plate_mean',k]=np.abs(np.nanmean(train,axis=0)-y)
            records['published_software_reference',k]=np.abs(np.array([software[x['name'],d] for d in DIST])-y)
    assert len(records)-len(rows)==a['baseline_rows_reconstructed']
    expected={(m,u,d) for m in {m for m,k in records}
        for u in ('all',*sorted({k[2] for mm,k in records if mm==m})) for d in (*DIST,'macro')}
    assert {(x['method'],x['user'],x['distance']) for x in a['summaries']}==expected
    maximum=0.0
    def check(actual,expected):
        nonlocal maximum
        error=abs(float(actual)-float(expected));maximum=max(maximum,error)
        assert error<1e-10,(actual,expected)
    for x in a['summaries']:
        values=[]
        for (m,k),err in records.items():
            if m!=x['method'] or (x['user']!='all' and k[2]!=x['user']):continue
            if x['distance']=='macro':
                if not np.isfinite(err).all():continue
                value=float(sum(err)/6)
            else:
                value=float(err[DIST.index(x['distance'])])
                if not np.isfinite(value):continue
            values.append((k[1],value))
        plates={p for p,e in values};e=[e for p,e in values]
        assert x['n']==len(e) and x['plates']==len(plates)
        for field,val in [('mae_mm',sum(e)/len(e)),('p95_absolute_mm',np.percentile(e,95)),
            ('max_absolute_mm',max(e)),('over_2mm',sum(t>2 for t in e)),
            ('plate_equal_mae_mm',np.mean([np.mean([q for p,q in values if p==plate]) for plate in plates]))]:check(x[field],val)
    assert len(a['paired'])==18
    for x in a['paired']:
        pairs=[]
        for (m,k),err in records.items():
            if m!=x['transfer'] or (x['user']!='all' and k[2]!=x['user']):continue
            other=records[x['reference'],k]
            if not np.isfinite(err).all() or not np.isfinite(other).all():continue
            pairs.append((k[1],float(sum(err)/6),float(sum(other)/6)))
        plates={p for p,t,e in pairs};d=[t-e for p,t,e in pairs]
        assert len(pairs)==x['n'] and len(plates)==x['plates']
        for field,val in [('transfer_mae_mm',np.mean([t for p,t,e in pairs])),
            ('reference_mae_mm',np.mean([e for p,t,e in pairs])),('paired_delta_mm',np.mean(d)),
            ('plate_equal_delta_mm',np.mean([np.mean([t-e for p,t,e in pairs if p==plate]) for plate in plates])),
            ('improved_photos',sum(t<e for p,t,e in pairs)),('worsened_photos',sum(t>e for p,t,e in pairs)),
            ('unchanged_photos',sum(t==e for p,t,e in pairs)),
            ('new_over_2mm',sum(t>2 and e<=2 for p,t,e in pairs)),
            ('removed_over_2mm',sum(t<=2 and e>2 for p,t,e in pairs))]:check(x[field],val)
    out={'status':'passed','result_sha256':sha(OUT/'results.json'),'validator_sha256':sha(Path(__file__)),
        'summaries_recomputed':len(a['summaries']),'paired_comparisons_recomputed':len(a['paired']),
        'raw_targets_matched':True,'identical_test_photo_keys':True,'source_hashes_passed':True,
        'maximum_numeric_error':maximum,'limitations':a['limitations']}
    (OUT/'validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out))

if __name__=='__main__':main()
