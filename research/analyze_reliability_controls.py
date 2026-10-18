"""Complete, paired descriptive comparisons of real and synthetic controls."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from run_reliability_controls import OUT, HERE, DATA, DRUGS, sha, read


def main():
    validation=read(HERE/'reliability_controls_validation.json');assert validation['status']=='passed'
    result=read(OUT/'results.json');assert sha(OUT/'results.json')==validation['result_sha256']
    real=read(DATA/'ani_annotation_repair_v1/results.json')
    real_validation=read(HERE/'ecoli_ani_annotation_validation.json')
    assert real_validation['status']=='passed'
    real_manifest=read(DATA/'ani_annotation_repair_v1/manifest.json')
    assert real_manifest['result_sha256']==sha(DATA/'ani_annotation_repair_v1/results.json')
    real_predictions={Path(a['path']).stem:Path(a['path']) for a in real['predictions']}
    paired_files=set()
    for artifact in result['predictions']:
        p=Path(artifact['path']);shard=read(p.with_suffix('.json'));r=shard['rows'][0]
        if not (r['condition'].startswith('label_') or r['condition']=='features_within_context'):continue
        stem=f"{r['drug'].replace('/','-')}_ani{r['threshold']}_{r['feature']}_{r['strategy']}_{r['seed']}"
        if stem not in real_predictions:continue
        rp=real_predictions[stem]
        expected=next(a['sha256'] for a in real['predictions'] if Path(a['path'])==rp)
        assert sha(rp)==expected
        with np.load(p,allow_pickle=False) as z,np.load(rp,allow_pickle=False) as rz:
            for role in ('source','query_pool','audit','evaluation'):
                for field in ('samples','clusters'):
                    assert np.array_equal(z[role+'_'+field],rz[role+'_'+field])
            assert np.array_equal(z['labels'],rz['labels']) and np.array_equal(z['audit_labels'],rz['audit_labels'])
        paired_files.add((r['drug'],r['threshold'],r['feature'],r['seed'],r['condition'],r['strategy']))
    real_by={(r['drug'],r['ANI_threshold'],r['feature'],r['seed'],r['strategy'],r['budget']):r for r in real['results']}
    by={(r['drug'],r['threshold'],r['feature'],r['seed'],r['condition'],r['strategy'],r['budget']):r for r in result['rows']}
    contrasts=[]
    for k,r in by.items():
        drug,t,feature,seed,condition,strategy,b=k
        if strategy=='decision_diagnostic':
            control=by[(drug,t,feature,seed,condition,'random_gene',b)]
            contrasts.append({**dict(zip(('drug','threshold','feature','seed','condition','budget'),(drug,t,feature,seed,condition,b))),
                              'comparison':'controller_minus_random','brier_delta':r['brier']-control['brier'],
                              'false_susceptible_delta':r['false_susceptible']-control['false_susceptible'],'n':r['n'],'R':r['resistant']})
        if condition.startswith('label_') or condition=='features_within_context':
            base=real_by.get((drug,t,feature,seed,strategy,b))
            if base:
                assert r['n']==base['evaluation_n']
                assert (drug,t,feature,seed,condition,strategy) in paired_files
                contrasts.append({'drug':drug,'threshold':t,'feature':feature,'seed':seed,'condition':condition,'budget':b,
                                  'comparison':'synthetic_minus_real_'+strategy,'brier_delta':r['brier']-base['brier'],
                                  'false_susceptible_delta':r['false_susceptible']-base['false_susceptible'],'n':r['n'],'R':r['resistant']})
    buckets=defaultdict(list)
    for r in contrasts:
        buckets[(r['drug'],r['threshold'],r['feature'],r['condition'],r['budget'],r['comparison'])].append(r)
    summaries=[]
    for k,rs in sorted(buckets.items()):
        vals=[r['brier_delta'] for r in rs]
        summaries.append({**dict(zip(('drug','threshold','feature','condition','budget','comparison'),k)),
                          'partitions':len(rs),'mean_brier_delta':float(np.mean(vals)),
                          'range_brier_delta':[min(vals),max(vals)],'lower_loss_partitions':sum(v<0 for v in vals),
                          'mean_false_susceptible_delta':float(np.mean([r['false_susceptible_delta'] for r in rs])),
                          'evaluation_n_range':[min(r['n'] for r in rs),max(r['n'] for r in rs)],
                          'evaluation_R_range':[min(r['R'] for r in rs),max(r['R'] for r in rs)]})
    plots=[r for r in summaries if r['threshold']==.99 and r['budget']==25 and r['comparison']=='controller_minus_random']
    cols=('label_global','label_within_component','features_within_context','time_2012','time_2014','EUCAST_to_CLSI','CLSI_to_EUCAST')
    fig,axes=plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
    limit=max([abs(r['mean_brier_delta']) for r in plots]+[.001])
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
    for ax,feature in zip(axes,('homolog','sketch')):
        grid=np.full((9,7),np.nan)
        for i,drug in enumerate(DRUGS):
            for j,condition in enumerate(cols):
                rs=[r for r in plots if r['drug']==drug and r['feature']==feature and r['condition']==condition]
                if rs:grid[i,j]=rs[0]['mean_brier_delta']
        im=ax.imshow(grid,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
        for i in range(9):
            for j in range(7):
                rs=[r for r in plots if r['drug']==DRUGS[i] and r['feature']==feature and r['condition']==cols[j]]
                txt=f"{grid[i,j]:+.3f}\nk={rs[0]['partitions']}" if rs else 'not feasible'
                ax.text(j,i,txt,ha='center',va='center',fontsize=7,color='white' if np.isfinite(grid[i,j]) and abs(grid[i,j])>limit*.65 else '#222222')
        ax.set_xticks(range(7),[c.replace('_','\n') for c in cols],fontsize=8)
        ax.set_yticks(range(9),[d.replace('trimethoprim/sulfamethoxazole','TMP-SMX') for d in DRUGS],fontsize=9)
        ax.set_title(feature.title()+' input')
    fig.colorbar(im,ax=axes,shrink=.7,label='Controller minus random Brier; negative = lower loss')
    fig.suptitle('Negative controls and transport stress tests: all drugs, fixed ANI 0.99 and budget 25')
    png=OUT/'reliability_controls_landscape.png'
    for ext in ('png','pdf','svg'):fig.savefig(png.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    fp=HERE/'reliability_controls_figure_validation.json'
    old=read(fp) if fp.exists() else {}
    digest=sha(png)
    review=old.get('visual_review',{})
    if review.get('figure_sha256')!=digest:review={'status':'pending','figure_sha256':digest}
    figure_record={'status':'generated_pending_visual_review','result_sha256':validation['result_sha256'],
        'figure_sha256':digest,'visual_review':review,
        'export_sha256':{str(png.with_suffix('.'+e)):sha(png.with_suffix('.'+e)) for e in ('png','pdf','svg')},
        'display_threshold':.99,'display_budget':25,'display_metric':'controller_minus_random_brier',
        'limitations':['Export/hash provenance is not an independent pixel-to-data validation.']}
    fp.write_text(json.dumps(figure_record,indent=2),encoding='utf-8')
    analysis={'status':'complete','result_sha256':validation['result_sha256'],'real_results_sha256':sha(DATA/'ani_annotation_repair_v1/results.json'),
              'contrast_rows':len(contrasts),'summaries':summaries,'contrasts':contrasts,'skipped':result['skipped'],
              'limitations':['Negative controls train synthetic labels/features; no clinical interpretation.',
                             'Partitions share public data; seed ranges are not confidence intervals.',
                             'Standard transport confounds assay, lineage, prevalence and year.',
                             'No null p-value: one permutation per fixed seed.']}
    (HERE/'reliability_controls_analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    for name,rows in [('reliability_controls_contrasts',contrasts),('reliability_controls_summary',summaries)]:
        if rows:
            with (OUT/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Negative controls, temporal extrapolation and standard transport','',
           f"Validated trajectories: {validation['prediction_files']}; metric rows: {validation['metric_rows']}; skipped strategy configurations: {validation['skips']}.",'',
           'All thresholds/budgets/conditions are in the JSON/CSV. The table uses existing display convention ANI 0.99, budget 25; it is post-result exploratory and not blind confirmation.','',
           '|Drug|Input|Condition|Comparison|Partitions|Mean Brier delta|Evaluation n/R range|',
           '|---|---|---|---|---:|---:|---|']
    for r in summaries:
        if r['threshold']==.99 and r['budget']==25:
            lines.append(f"|{r['drug']}|{r['feature']}|{r['condition']}|{r['comparison']}|{r['partitions']}|{r['mean_brier_delta']:+.6f}|{r['evaluation_n_range']}/{r['evaluation_R_range']}|")
    lines+=['','## Permutation activity','',json.dumps(validation['changed_records'],indent=2),'',
            'Zero label/vector changes are retained, not treated as informative destruction of genomic associations.',
            '', '## Feasibility', '',
            'Every skipped configuration and exclusion count is retained in results.json. Empty tables are possible under giant ANI components; boundaries will not be changed to improve results.',
            '', 'These results cannot isolate causal assay drift or establish stable SOTA or clinical safety.']
    (HERE/'RELIABILITY_CONTROLS_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'complete','contrasts':len(contrasts),'summaries':len(summaries)}),flush=True)


if __name__=='__main__':main()
