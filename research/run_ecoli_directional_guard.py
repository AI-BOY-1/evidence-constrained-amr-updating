"""Post-result fixed-prediction directional guard with no label-based tuning.

A guarded repair cannot create a new susceptible prediction on a sample that
the original baseline predicted resistant. This relative property does NOT
establish absolute safety, nor eliminate baseline false-susceptible errors.
"""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv, hashlib, json
from pathlib import Path
import numpy as np
from run_ecoli_frozen_query_ablation import DRUGS,THRESHOLDS,SEEDS,BUDGETS,original,metrics,raw_labels,check_roles,provenance,sha

HERE=Path(__file__).resolve().parent
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=ROOT/'directional_guard_v1'

def guard(base, candidate):
    return np.where((base>=.5)&(candidate<.5),.5,candidate)

def main():
    OUT.mkdir(exist_ok=True)
    binding=provenance(); binding['guard_code_sha256']=sha(Path(__file__))
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf-8'))['binding']==binding
    labels=raw_labels();rows=[];artifacts=[]
    for drug in DRUGS:
        for t in THRESHOLDS:
            for seed in SEEDS:
                stem=original(drug,t,'decision_diagnostic',seed)
                arr=np.load(str(stem)+'.npz',allow_pickle=False);check_roles(arr,drug,labels)
                random=np.load(str(original(drug,t,'random_gene',seed))+'.npz',allow_pickle=False)
                y=arr['labels'];base=random['p_b0']
                tr=json.loads(Path(str(stem)+'.json').read_text(encoding='utf-8'))
                summaries={x['budget']:x for x in tr['results']}
                ablation=ROOT/'frozen_query_ablation_v1'/f'{drug.replace("/","-")}_t{t}_diagnostic_path_gene_{seed}.npz'
                gene=np.load(ablation,allow_pickle=False)
                assert np.array_equal(gene['evaluation_samples'],arr['evaluation_samples'])
                export={'labels':y,'evaluation_samples':arr['evaluation_samples'],'evaluation_clusters':arr['evaluation_clusters'],'baseline':base}
                for kind,candidates in [('decision_diagnostic',arr),('diagnostic_path_gene',gene)]:
                    for b in BUDGETS:
                        p=candidates[f'p_b{b}'];pg=guard(base,p);export[f'{kind}_b{b}']=pg
                        assert np.all(pg[(base>=.5)]>=.5)
                        before=metrics(y,p,base);after=metrics(y,pg,base)
                        assert after['resistant_harmed']==0
                        assert after['false_susceptible']<=int(((y==1)&(base<.5)).sum())
                        rows.append({'drug':drug,'threshold':t,'seed':seed,'strategy':kind,'budget':b,
                                     'total_target_assay_cost':summaries[b]['total_target_assay_cost'],
                                     'blocked_R_to_S_predictions':int(((base>=.5)&(p<.5)).sum()),
                                     'relative_no_new_false_susceptible':True,'raw':before,'guarded':after,
                                     'brier_delta_guard_minus_raw':after['brier']-before['brier'],
                                     'false_resistant_delta':after['false_resistant']-before['false_resistant']})
                path=OUT/f'{drug.replace("/","-")}_t{t}_{seed}.npz';np.savez_compressed(path,**export)
                artifacts.append({'path':str(path),'sha256':sha(path)})
    res={'status':'complete','designation':'post-result exploratory fixed-prediction intervention','rows':rows,'binding':binding,'prediction_trajectories':len(artifacts),'metric_rows':len(rows),'artifacts':artifacts,'limitations':['Previously viewed retrospective cohort; not new confirmation.','A structural non-regression property relative to baseline, not a clinical safety bound.','Baseline false susceptibility remains possible.','Guard sacrifices some false-resistant corrections; Brier trade-offs are retained.','No labels were used by the guard function.']}
    rp=OUT/'results.json';rp.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    mp.write_text(json.dumps({'status':'complete','binding':binding,'result_sha256':sha(rp),'trajectories':len(artifacts),'metric_rows':len(rows)},indent=2),encoding='utf-8')
    selected=[r for r in rows if r['threshold']==.99 and r['budget']==25]
    lines=['# Directional repair guard','','Post-result exploratory experiment; no new blind cohort. The frozen baseline threshold is 0.5. If baseline predicts R and candidate predicts S, return p(R)=0.5 (R), otherwise retain candidate probability. No evaluation labels enter this operation.','','|Drug|Candidate|Brier change guard minus raw|New resistant harms raw / guarded|False-resistant change|','|---|---|---:|---:|---:|']
    for d in DRUGS:
        for kind in ('decision_diagnostic','diagnostic_path_gene'):
            rs=[r for r in selected if r['drug']==d and r['strategy']==kind]
            lines.append(f"|{d}|{kind}|{np.mean([r['brier_delta_guard_minus_raw'] for r in rs]):+.6f}|{np.mean([r['raw']['resistant_harmed'] for r in rs]):.1f} / 0|{np.mean([r['false_resistant_delta'] for r in rs]):+.1f}|")
    lines+=['','All nine drugs, both thresholds, five splits and four budgets are retained (720 metric rows). Baseline errors are not removed by this guarantee. This guard cannot establish clinical safety or general SOTA. Full data: '+str(rp)]
    (HERE/'ECOLI_DIRECTIONAL_GUARD.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'complete','trajectories':len(artifacts),'metric_rows':len(rows),'all_guarded_resistant_harmed_zero':all(r['guarded']['resistant_harmed']==0 for r in rows)}),flush=True)
if __name__=='__main__':main()
