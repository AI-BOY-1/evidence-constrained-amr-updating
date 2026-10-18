"""Validate exported figure provenance and all displayed sample counts."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(dataset):
    dest=ROOT/f'repair_dynamics_{dataset}_v1'
    r=read(dest/'results.json');v=read(dest/'validation.json');f=read(dest/'figure_validation.json')
    assert v['status']=='passed' and v['result_sha256']==f['result_sha256']==sha(dest/'results.json')
    assert f['code_sha256']==sha(HERE/'analyze_repair_dynamics.py')
    for ext,value in f['exports'].items():assert value==sha(dest/('repair_dynamics_atlas.'+ext))
    with Image.open(dest/'repair_dynamics_atlas.png') as im:
        assert im.size==(4000,4800)
        assert all(a<b for a,b in im.convert('RGB').getextrema())
    expected=[x for x in r['shared_samples'] if x['seed']==7 and x['budget']==25
              and x['threshold']==.99 and x['feature']=='sketch']
    key=lambda x:(x['drug'],x['sample'])
    assert {key(x):x for x in expected}=={key(x):x for x in f['sample_display']}
    t=[x for x in r['transitions'] if x['adjacent']]
    assert f['headline']=={'adjacent_transitions':len(t),
        'improved_brier_increased_FS':sum(x['brier_delta']<0 and x['FS_after']>x['FS_before'] for x in t),
        'introduced_FS_transitions':sum(x['FS_introduced']>0 for x in t),
        'both_introduced_and_corrected':sum(x['introduced']>0 and x['corrected']>0 for x in t),
        'brier_worsened':sum(x['brier_delta']>0 for x in t)}
    for rule,c in f['gate_counts'].items():
        a=[x for x in r['gates'] if x['rule']==rule];b=[x for x in a if x['status']=='complete']
        s=[x for x in b if x['selected_budget']<x['consumed_budget']]
        assert c=={'configured':len(a),'feasible':len(b),'infeasible':len(a)-len(b),'rollback':len(s),
                   'lower_eval_brier':sum(x['brier_delta']<0 for x in s),
                   'higher_eval_brier':sum(x['brier_delta']>0 for x in s),
                   'equal_eval_brier':sum(x['brier_delta']==0 for x in s),
                   'increased_eval_FS':sum(x['FS_delta']>0 for x in s),
                   'decreased_eval_FS':sum(x['FS_delta']<0 for x in s)}
    f.update(status='source_and_export_checks_passed',validator_sha256=sha(__file__),
             sample_values_checked=len(expected),
             limitations=['Source and export hashes are checked; this is not independent pixel reconstruction.'])
    review=dest/'visual_review.json'
    if review.exists():
        vr=read(review)
        if vr['figure_sha256']==f['exports']['png'] and vr['status']=='passed':
            f['visual_review']=vr;f['status']='passed'
    (dest/'figure_validation.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
    print(json.dumps({'status':f['status'],'sample_values_checked':len(expected),'visual_review':f['visual_review'].get('status')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('full','ani'),default='full')
    main(p.parse_args().dataset)
