"""Reconstruct pairing and all summary tables without analyzer imports."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ncs_paths.data_path('experiments/ecoli_bvbrc/audit_acquisition_action_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):
    if isinstance(a,float):assert abs(a-float(b))<1e-12
    elif isinstance(a,bool):assert str(a)==b
    else:assert str(a)==str(b)
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def checksummaries(raw,table,keys,kind):
    groups=defaultdict(list)
    for r in raw:groups[tuple(str(r[k]) for k in keys)].append(r)
    assert len(groups)==len(table);seen=set()
    for t in table:
        key=tuple(str(t[k]) for k in keys);assert key in groups and key not in seen;seen.add(key)
        rs=groups[key]
        if kind=='paired':
            vals={'paired_configurations':len(rs),'mean_brier_delta':float(np.mean([r['delta'] for r in rs])),
                  'lower_loss':sum(r['delta']<0 for r in rs),'higher_loss':sum(r['delta']>0 for r in rs),
                  'equal_loss':sum(r['delta']==0 for r in rs),'FS_increased':sum(r['FS_delta']>0 for r in rs),
                  'FS_decreased':sum(r['FS_delta']<0 for r in rs),
                  'lower_loss_more_FS':sum(r['delta']<0 and r['FS_delta']>0 for r in rs)}
        else:
            vals={'configurations':len(rs),'eligible':sum(r['eligible'] for r in rs),'accepted':sum(r['accepted'] for r in rs),
                  'audit_R_zero':sum(r['audit_R']==0 for r in rs),'mean_brier_delta_from_frozen':float(np.mean([r['delta'] for r in rs])),
                  'eval_loss_higher':sum(r['delta']>0 for r in rs),'eval_FS_increased':sum(r['FS_delta']>0 for r in rs),
                  'any_new_FS':sum(r['FS_introduced']>0 for r in rs),
                  'lower_loss_more_FS':sum(r['delta']<0 and r['FS_delta']>0 for r in rs),
                  'mean_oracle_regret':float(np.mean([r['regret'] for r in rs])),
                  'audit_R_min':min(r['audit_R'] for r in rs),'audit_R_max':max(r['audit_R'] for r in rs),
                  'eval_R_min':min(r['evaluation_R'] for r in rs),'eval_R_max':max(r['evaluation_R'] for r in rs),
                  'mean_audit_components':float(np.mean([r['audit_components'] for r in rs]))}
        for k,x in vals.items():eq(x,t[k])
def main():
    analysis=read(HERE/'audit_acquisition_action_analysis.json');v=read(OUT/'validation.json')
    assert v['status']=='passed' and v['result_sha256']==sha(OUT/'results.json')
    for p,d in {**analysis['input_sha256'],**analysis['outputs_sha256']}.items():assert sha(p)==d
    assert analysis['code_sha256']==sha(HERE/'analyze_audit_acquisition_action.py')
    raw=read(OUT/'results.json');rows=[];rulepairs=[];samplingpairs=[];seen_skips=defaultdict(int)
    for art in raw['artifacts']:
        r=read(art['path']);assert sha(art['path'])==art['sha256'];lookup={}
        for c in r['conditions']:
            common={**r['identity'],'query_budget':c['query_budget'],'audit_size':c['audit_size'],
                    'acquisition':c['acquisition'],'audit_R':c['audit_evidence']['old']['R'],
                    'audit_components':c['audit_evidence']['components'],'evaluation_R':r['evaluation_R']}
            aa={a['rule']:a for a in c['actions']}
            for rule,a in aa.items():
                row={**common,'rule':rule,'eligible':not a['decision_status'].startswith('infeasible'),
                     'accepted':a['accepted'],'delta':a['evaluation_brier_delta_from_frozen'],
                     'FS_delta':a['evaluation_FS_delta_from_frozen'],'FS_introduced':a['change_from_frozen']['FS_introduced'],
                     'regret':a['oracle_brier_regret'],'cost':a['total_new_label_cost']}
                rows.append(row);lookup[(c['query_budget'],c['audit_size'],c['acquisition'],rule)]=row
                if rule not in ('retain','update'):
                    for control in ('retain','update'):
                        assert a['total_new_label_cost']==aa[control]['total_new_label_cost']
                        rulepairs.append({**common,'rule':rule,'control':control,
                                          'delta':a['evaluation']['brier']-aa[control]['evaluation']['brier'],
                                          'FS_delta':a['evaluation']['FS']-aa[control]['evaluation']['FS']})
        for (b,n,acq,rule),a in lookup.items():
            if acq!='random':
                ref=lookup[(b,n,'random',rule)];assert a['cost']==ref['cost']
                samplingpairs.append({**r['identity'],'query_budget':b,'audit_size':n,'acquisition':acq,'rule':rule,
                                      'control':'random_acquisition','delta':a['delta']-ref['delta'],'FS_delta':a['FS_delta']-ref['FS_delta']})
        for s in r['skipped_conditions']:seen_skips[(s['query_budget'],s['audit_size'],s['reason'])]+=1
    for name,rs in [('rule_pairs.csv',rulepairs),('sampling_pairs.csv',samplingpairs)]:
        table=csvrows(OUT/name);assert len(table)==len(rs)
        for a,b in zip(rs,table):
            assert set(a)==set(b)
            for k,x in a.items():eq(x,b[k])
    checksummaries(rows,csvrows(OUT/'summary.csv'),('query_budget','audit_size','acquisition','rule'),'metrics')
    checksummaries(rows,csvrows(OUT/'detailed.csv'),('drug','ANI_threshold','feature','strategy','query_budget','audit_size','acquisition','rule'),'metrics')
    checksummaries(rows,analysis['global_rule_summary'],('rule',),'metrics')
    checksummaries(rulepairs,csvrows(OUT/'rule_pair_summary.csv'),('query_budget','audit_size','acquisition','rule','control'),'paired')
    checksummaries(samplingpairs,csvrows(OUT/'sampling_pair_summary.csv'),('query_budget','audit_size','acquisition','rule','control'),'paired')
    checksummaries(samplingpairs,analysis['global_sampling_summary'],('acquisition','rule','control'),'paired')
    checksummaries(samplingpairs,csvrows(OUT/'drug_sampling_summary.csv'),('drug','acquisition','rule','control'),'paired')
    checksummaries(samplingpairs,analysis['drug_sampling_summary'],('drug','acquisition','rule','control'),'paired')
    assert len(rows)==analysis['action_rows']==87120
    assert len(rulepairs)==analysis['rule_pair_rows'] and len(samplingpairs)==analysis['sampling_pair_rows']
    for s in analysis['skips']:assert seen_skips[(s['query_budget'],s['audit_size'],s['reason'])]==s['conditions']
    validation={'status':'passed','analysis_sha256':sha(HERE/'audit_acquisition_action_analysis.json'),
                'report_sha256':sha(HERE/'AUDIT_ACQUISITION_ACTION_RESULTS.md'),'validator_sha256':sha(Path(__file__)),
                'rule_pairs_checked':len(rulepairs),'sampling_pairs_checked':len(samplingpairs),
                'checks':{'complete_summary_recompute':True,'complete_pairs_same_roles_and_cost':True,'all_skips_retained':True}}
    (HERE/'audit_acquisition_analysis_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))
if __name__=='__main__':main()
