#!/usr/bin/env python
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from ncs_amr import Record, run_trajectory

def make_records():
    rows=[]
    for i in range(80):
        role='source' if i<40 else 'query_pool' if i<60 else 'audit' if i<70 else 'evaluation'
        label=-1 if role!='source' else int(i%3==0)
        rows.append(Record(f's{i:03}',f'c{i:03}',2010+i%10,'EUCAST',tuple(f'k{(i+j)%17}' for j in range(2)),label))
    return {role:[r for i,r in enumerate(rows) if (('source' if i<40 else 'query_pool' if i<60 else 'audit' if i<70 else 'evaluation')==role)] for role in ('source','query_pool','audit','evaluation')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='demo_output');args=ap.parse_args()
    groups=make_records();oracle={r.sample:Record(r.sample,r.component,r.year,r.standards,r.features,int((int(r.sample[1:])%3)==0)) for r in groups['query_pool']}
    revealed=[]
    def reveal(sample):
        if sample in revealed: raise RuntimeError('duplicate query')
        revealed.append(sample); return oracle[sample]
    snapshots,events=run_trajectory(groups,reveal,seed=7,budgets=(0,5,10),batch_size=5)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'demo_result.json').write_text(json.dumps({'status':'synthetic_smoke_test','snapshots':[{'budget':x['budget'],'action':x['action'],'n':len(x['evaluation_p'])} for x in snapshots],'events':len(events)},indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','out':str(out),'snapshots':len(snapshots),'events':len(events)}))
if __name__=='__main__':main()
