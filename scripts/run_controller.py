#!/usr/bin/env python
"""Run portable controller on caller-owned JSONL; no data download."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ncs_amr import Record,run_trajectory
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--query-budget',type=int,nargs='+',default=[0,10,25,50]);ap.add_argument('--strategy',choices=('decision_diagnostic','random_gene','uncertainty_gene','random_tree'),default='decision_diagnostic');args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('Refuse to overwrite an existing result')
    groups={k:[] for k in ('source','query_pool','audit','evaluation')}
    for line in Path(args.input).read_text(encoding='utf-8').splitlines():
        if line.strip():
            x=json.loads(line);role=x.pop('role');groups[role].append(Record.from_mapping(x))
    original_pool=list(groups['query_pool'])
    if any(r.label not in (0,1) for r in original_pool):raise ValueError('query_pool requires recorded labels in the offline input; they are masked until reveal')
    oracle={r.sample:r for r in original_pool};groups['query_pool']=[r.masked() for r in original_pool];groups['audit']=[r.masked() for r in groups['audit']];groups['evaluation']=[r.masked() for r in groups['evaluation']]
    seen=[]
    def reveal(s):
        if s in seen:raise RuntimeError('duplicate query')
        seen.append(s);return oracle[s]
    snaps,events=run_trajectory(groups,reveal,args.seed,tuple(sorted(set(args.query_budget))),strategy=args.strategy)
    import hashlib,platform
    from importlib.metadata import version
    root=Path(__file__).resolve().parents[1]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src/ncs_amr').glob('*.py')}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'status':'portable_run','seed':args.seed,'strategy':args.strategy,'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'code_sha256':hashes,'python':platform.python_version(),'packages':{n:version(n) for n in ('numpy','scipy','scikit-learn')},'snapshots':[{'budget':x['budget'],'action':x['action'],'evaluation_p':x['evaluation_p'].tolist(),'audit_p':x['audit_p'].tolist(),'evidence_weights':x['evidence_weights'],'queried_samples':x['queried_samples']} for x in snaps],'events':events},indent=2),encoding='utf-8')
if __name__=='__main__':main()
