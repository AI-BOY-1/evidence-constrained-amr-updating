"""Report all frozen ANI/feature contrasts, including skipped and negative cases."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
DATA=ncs_paths.data_path('experiments/ecoli_bvbrc/ani_annotation_repair_v1')


def main():
    raw=json.loads((DATA/'results.json').read_text(encoding='utf-8'))
    validation=json.loads((HERE/'ecoli_ani_annotation_validation.json').read_text(encoding='utf-8'))
    assert validation['status']=='passed'
    assert validation['result_sha256']==hashlib.sha256((DATA/'results.json').read_bytes()).hexdigest()
    by={(r['drug'],r['ANI_threshold'],r['feature'],r['seed'],r['strategy'],r['budget']):r for r in raw['results']}
    contrasts=[]
    for key,r in by.items():
        drug,t,feature,seed,strategy,b=key
        if strategy!='decision_diagnostic':continue
        for control in ('random_gene','uncertainty_gene','random_tree'):
            c=by.get((drug,t,feature,seed,control,b))
            if c is None:continue
            contrasts.append({'drug':drug,'ANI_threshold':t,'feature':feature,'seed':seed,'budget':b,
                'control':control,'brier_delta':r['brier']-c['brier'],
                'false_susceptible_rate_delta':None if r['false_susceptible_rate'] is None else r['false_susceptible_rate']-c['false_susceptible_rate'],
                'evaluation_n':r['evaluation_n'],'total_target_assay_cost':r['total_target_assay_cost']})
    grouped=defaultdict(list)
    for r in contrasts:
        grouped[(r['drug'],r['ANI_threshold'],r['feature'],r['budget'],r['control'])].append(r)
    summary=[]
    for key,rs in sorted(grouped.items()):
        vals=[r['brier_delta'] for r in rs]
        summary.append(dict(zip(('drug','ANI_threshold','feature','budget','control'),key),
            partition_n=len(rs),mean_brier_delta=float(np.mean(vals)),partitions_lower_loss=sum(v<0 for v in vals),
            partition_delta_range=[min(vals),max(vals)],evaluation_n_range=[min(r['evaluation_n'] for r in rs),max(r['evaluation_n'] for r in rs)]))
    feature_contrasts=[]
    for key,r in by.items():
        drug,t,feature,seed,strategy,b=key
        if feature!='homolog':continue
        c=by.get((drug,t,'sketch',seed,strategy,b))
        if c:feature_contrasts.append({'drug':drug,'ANI_threshold':t,'seed':seed,'strategy':strategy,'budget':b,
            'homolog_minus_sketch_brier':r['brier']-c['brier'],'evaluation_n':r['evaluation_n']})
    out={'status':'complete','result_sha256':validation['result_sha256'],'trajectory_count':len(raw['predictions']),
         'metric_rows':len(raw['results']),'skipped':raw['skipped'],'cohort_audits':raw['cohort_audits'],
         'contrasts':contrasts,'summary':summary,'feature_contrasts':feature_contrasts,
         'limitations':['All tested configurations and skips retained; no threshold selection.',
                        'Repeated partitions share a retrospective finite cohort; not independent replication.',
                        'Approximate ANI-screen components do not establish SNP or patient independence.']}
    (HERE/'ecoli_ani_annotation_analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    for name,rs in [('ani_feature_contrasts',contrasts),('ani_feature_summary',summary),('homolog_sketch_contrasts',feature_contrasts)]:
        if rs:
            with (HERE/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    lines=['# ANI-screen and acquired-homolog sensitivity','',
           f"Verified trajectories: {len(raw['predictions'])}; metric rows: {len(raw['results'])}; skipped feature/partition configurations: {len(raw['skipped'])}.",'',
           'Brier difference = diagnostic controller minus same-input control; lower is better. All partitions share one retrospective cohort.','',
           '|Drug|ANI threshold|Input|Budget|Control|Mean difference|Lower-loss partitions|Evaluation n range|',
           '|---|---:|---|---:|---|---:|---:|---|']
    for r in summary:
        lines.append(f"|{r['drug']}|{r['ANI_threshold']}|{r['feature']}|{r['budget']}|{r['control']}|{r['mean_brier_delta']:+.6f}|{r['partitions_lower_loss']}/{r['partition_n']}|{r['evaluation_n_range']}|")
    lines+=['','## Fixed-rule skips','',json.dumps(raw['skipped'],indent=2),'',
            'No clinical safety, mechanism, holder-blind confirmation or stable SOTA conclusion follows from these computations.']
    (HERE/'ECOLI_ANI_ANNOTATION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'complete','trajectories':len(raw['predictions']),'metric_rows':len(raw['results']),'skips':len(raw['skipped'])}),flush=True)


if __name__=='__main__':main()
