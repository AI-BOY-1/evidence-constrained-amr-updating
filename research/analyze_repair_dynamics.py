"""All-configuration reporting and actual-sample visualization of repair dynamics."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
DRUGS=('meropenem','imipenem','ertapenem','ceftazidime','ciprofloxacin',
       'trimethoprim/sulfamethoxazole','amikacin','gentamicin','cefepime')
STRATEGIES=('decision_diagnostic','random_gene','uncertainty_gene','random_tree')
COLORS=('#007F83','#CA7048','#72538E','#4881B0')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def short(d):return 'TMP-SMX' if '/' in d else d.capitalize()


def main(dataset):
    dest=ROOT/f'repair_dynamics_{dataset}_v1'
    validation=read(dest/'validation.json');raw=read(dest/'results.json')
    assert validation['status']=='passed' and validation['result_sha256']==sha(dest/'results.json')
    adjacent=[r for r in raw['transitions'] if r['adjacent']]
    discordant=[r for r in adjacent if r['brier_delta']<0 and r['FS_after']>r['FS_before']]
    headline={'adjacent_transitions':len(adjacent),'improved_brier_increased_FS':len(discordant),
              'introduced_FS_transitions':sum(r['FS_introduced']>0 for r in adjacent),
              'both_introduced_and_corrected':sum(r['introduced']>0 and r['corrected']>0 for r in adjacent),
              'brier_worsened':sum(r['brier_delta']>0 for r in adjacent)}
    gate_counts={}
    for rule in ('brier_only','brier_and_FS'):
        all_rows=[r for r in raw['gates'] if r['rule']==rule]
        feasible=[r for r in all_rows if r['status']=='complete']
        rejected=[r for r in feasible if r['selected_budget']<r['consumed_budget']]
        gate_counts[rule]={'configured':len(all_rows),'feasible':len(feasible),'infeasible':len(all_rows)-len(feasible),
                          'rollback':len(rejected),'lower_eval_brier':sum(r['brier_delta']<0 for r in rejected),
                          'higher_eval_brier':sum(r['brier_delta']>0 for r in rejected),
                          'equal_eval_brier':sum(r['brier_delta']==0 for r in rejected),
                          'increased_eval_FS':sum(r['FS_delta']>0 for r in rejected),
                          'decreased_eval_FS':sum(r['FS_delta']<0 for r in rejected)}
    # All configurations are reported; the compact display retains the old fixed convention.
    feature='sketch';threshold=.99
    shown=[r for r in adjacent if r['feature']==feature and r['threshold']==threshold]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                         'axes.titlesize':9,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(10,12))
    grid=fig.add_gridspec(3,2,left=.10,right=.96,bottom=.10,top=.855,hspace=.72,wspace=.38,
                          height_ratios=[1,1.45,1])
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[1,:]);d=fig.add_subplot(grid[2,0]);e=fig.add_subplot(grid[2,1])
    fig.text(.10,.975,'What changes when a prediction model is repaired?',fontsize=15,fontweight='bold')
    fig.text(.10,.948,f'{dataset.upper()} cohort | sketch input | partition threshold 0.99 | exploratory frozen-prediction replay',fontsize=8)
    for ax in (a,b,d,e):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=7)
    for strategy,color in zip(STRATEGIES,COLORS):
        rs=[r for r in shown if r['strategy']==strategy and r['R']>0]
        a.scatter([r['brier_delta'] for r in rs],[(r['FS_after']-r['FS_before'])/r['R']*100 for r in rs],
                  s=13,c=color,alpha=.55,linewidths=0,rasterized=True)
        b.scatter([r['corrected'] for r in rs],[r['introduced'] for r in rs],
                  s=13,c=color,alpha=.55,linewidths=0,rasterized=True)
    a.axhline(0,color='#777777',lw=.6);a.axvline(0,color='#777777',lw=.6)
    a.set(xlabel='Change in Brier (new minus previous)',ylabel='Change in false-susceptible rate (pp)',
          title='a  Loss and error direction can disagree')
    maxv=max([r[k] for r in shown for k in ('corrected','introduced')]+[1])
    b.plot([0,maxv],[0,maxv],color='#888888',lw=.6,ls='--')
    b.set(xlabel='Errors corrected',ylabel='New errors introduced',title='b  Net change conceals error replacement')
    handles=[Line2D([],[],marker='o',ls='',color=co,ms=4,label=s.replace('decision_diagnostic','Controller').replace('_',' ')) for s,co in zip(STRATEGIES,COLORS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.900),ncol=4,frameon=False,fontsize=7)
    cmap=ListedColormap(('#DFEBE8','#A6CECC','#669BAA','#68628A','#B54D4E'))
    sample_map=[]
    all_samples=raw['shared_samples']
    for i,drug in enumerate(DRUGS):
        before={r['sample']:r['strategies_FS'] for r in all_samples if r['drug']==drug and r['threshold']==threshold and r['feature']==feature and r['seed']==7 and r['budget']==0}
        rs=[r for r in all_samples if r['drug']==drug and r['threshold']==threshold and r['feature']==feature and r['seed']==7 and r['budget']==25]
        rs=sorted(rs,key=lambda r:(before.get(r['sample'],0),r['component'],r['sample']))
        if rs:
            values=[r['strategies_FS'] for r in rs]
            c.imshow(np.asarray(values)[None,:],extent=(0,1,i+.30,i+.90),cmap=cmap,vmin=-.5,vmax=4.5,
                     aspect='auto',interpolation='nearest',rasterized=True)
            c.text(1.015,i+.60,f"{sum(v==4 for v in values)}/{len(values)}",va='center',fontsize=7)
            sample_map.extend(rs)
        else:c.text(.5,i+.60,'not feasible',ha='center',va='center',fontsize=7)
    c.set(ylim=(9.2,0),xlim=(0,1.18),yticks=np.arange(9)+.6,
          yticklabels=[short(x) for x in DRUGS],xticks=[0,.5,1],
          xticklabels=['0%','50%','100%'],xlabel='Every resistant evaluation genome, within each drug',
          title='c  Shared false susceptibility across four strategies | seed 7, budget 25')
    c.text(1.015,-.10,'All 4 / R',fontsize=7,va='bottom')
    c.spines[['top','right','bottom','left']].set_visible(False);c.tick_params(length=0,labelsize=7)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-.5,4.5),cmap=cmap),ax=c,
                    orientation='horizontal',fraction=.045,pad=.14,aspect=60,ticks=range(5))
    cb.set_label('Number of strategies calling a resistant genome susceptible',fontsize=7)
    for rule,marker in [('brier_only','o'),('brier_and_FS','^')]:
        rs=[r for r in raw['gates'] if r['status']=='complete' and r['rule']==rule
            and r['feature']==feature and r['threshold']==threshold and r['selected_budget']<r['consumed_budget']]
        d.scatter([r['brier_delta'] for r in rs],[r['FS_delta']/r['R']*100 if r['R'] else np.nan for r in rs],
                  s=16,marker=marker,alpha=.55,linewidths=0,label=rule.replace('_',' '),rasterized=True)
    d.axvline(0,color='#777777',lw=.6);d.axhline(0,color='#777777',lw=.6)
    d.set(xlabel='Rollback minus same-cost update: Brier',ylabel='False-susceptible rate difference (pp)',
          title='d  Audit acceptance is not evaluation protection')
    d.legend(frameon=False,fontsize=6,loc='best')
    for rule,color,offset in [('brier_only',COLORS[0],-.08),('brier_and_FS',COLORS[1],.08)]:
        for i,drug in enumerate(DRUGS):
            rows=[r for r in raw['gate_summary'] if r['drug']==drug and r['threshold']==threshold
                  and r['feature']==feature and r['strategy']=='decision_diagnostic' and r['rule']==rule]
            if rows:e.scatter(rows[0]['mean_brier_delta'],i+offset,s=20,color=color)
    e.axvline(0,color='#888888',lw=.6);e.invert_yaxis()
    e.set(yticks=range(9),yticklabels=[short(x) for x in DRUGS],xlabel='Mean rollback minus update: Brier',
          title='e  Controller gate: all drugs, including failures')
    fig.text(.10,.052,'Repeated partitions share the same cohort. Points are configuration observations, not independent patients.',fontsize=7)
    fig.text(.10,.032,'Original sketch groups are not ANI/SNP isolation. Shared misses do not prove missing mechanisms or irreducible error.',fontsize=7)
    stem=dest/'repair_dynamics_atlas'
    fig.canvas.draw()
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    record={'status':'generated_pending_visual_review','result_sha256':validation['result_sha256'],
            'display_threshold':threshold,'display_feature':feature,'sample_display_seed':7,'sample_display_budget':25,
            'sample_display':sample_map,'headline':headline,'gate_counts':gate_counts,
            'plot_counts':{'adjacent_transition_points':len(shown)},
            'exports':{ext:sha(stem.with_suffix('.'+ext)) for ext in ('png','pdf','svg')},
            'code_sha256':sha(__file__)}
    old=read(dest/'figure_validation.json') if (dest/'figure_validation.json').exists() else {}
    record['visual_review']=old.get('visual_review',{}) if old.get('exports',{}).get('png')==record['exports']['png'] else {'status':'pending'}
    write=lambda p,obj:Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    write(dest/'figure_validation.json',record)
    findings={'status':'complete','dataset':dataset,'result_sha256':validation['result_sha256'],
              'headline':headline,'gate_counts':gate_counts,'figure_path':str(stem.with_suffix('.png'))}
    write(dest/'findings.json',findings)
    lines=['# Repair dynamics: evidence, interventions and remaining questions','',
           f"Validated {validation['prediction_files']} prediction files; {validation['transitions']} forward budget contrasts; {validation['gate_replays']} stopping-rule replays; {validation['overlap_configs']} four-strategy overlap configurations.",'',
           '## More measurements do not guarantee monotone improvement','',
           f"Among {len(adjacent)} adjacent configuration transitions, {headline['brier_worsened']} have higher Brier, {headline['improved_brier_increased_FS']} combine lower Brier with more false-susceptible calls, and {headline['both_introduced_and_corrected']} both correct old errors and introduce new ones.",
           'These are repeated-configuration counts, not independent studies, unique events or significance tests.','',
           '## Explicit matched stopping intervention','',
           '|Gate|Feasible|Infeasible|Rollback|Lower evaluation loss|Higher evaluation loss|Equal|More evaluation FS|Fewer evaluation FS|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rule,x in gate_counts.items():
        lines.append(f"|{rule}|{x['feasible']}|{x['infeasible']}|{x['rollback']}|{x['lower_eval_brier']}|{x['higher_eval_brier']}|{x['equal_eval_brier']}|{x['increased_eval_FS']}|{x['decreased_eval_FS']}|")
    lines+=['','Evaluation differences compare rollback to the identical strategy updated to the consumed query budget. Rejected query labels and all audit labels are charged. Audit labels alone decide acceptance; evaluation labels only score outputs. Directional acceptance preserves audit FS at accepted checkpoints, not evaluation FS. Both gates fail in many configurations. No clinical safety guarantee follows.','',
            '## Every controller transition, not only a selected success','',
            '|Drug|Threshold|Input|From|To|Partitions|Mean Brier change|FS corrected|FS introduced|FR corrected|FR introduced|R range|',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in raw['transition_summary']:
        if r['strategy']!='decision_diagnostic':continue
        lines.append(f"|{r['drug']}|{r['threshold']}|{r['feature']}|{r['from_budget']}|{r['to_budget']}|{r['partition_n']}|{r['mean_brier_delta']:+.6f}|{r['mean_FS_corrected']:.2f}|{r['mean_FS_introduced']:.2f}|{r['mean_FR_corrected']:.2f}|{r['mean_FR_introduced']:.2f}|{r['evaluation_R_range']}|")
    lines+=['','## Shared errors, all drugs and budgets','',
            '|Drug|Threshold|Input|Budget|Partitions|Mean R|Mean all-four FS|Mean any-strategy FS|R range|',
            '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in raw['overlap_summary']:
        lines.append(f"|{r['drug']}|{r['threshold']}|{r['feature']}|{r['budget']}|{r['partition_n']}|{r['mean_R']:.2f}|{r['mean_all_four_FS']:.2f}|{r['mean_at_least_one_FS']:.2f}|{r['evaluation_R_range']}|")
    lines+=['','## All gate contrasts','',
            '|Drug|Threshold|Input|Strategy|Rule|Partitions|Mean loss delta|FS delta|FR delta|Consumed queries|Selected checkpoint|Audit R|',
            '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in raw['gate_summary']:
        lines.append(f"|{r['drug']}|{r['threshold']}|{r['feature']}|{r['strategy']}|{r['rule']}|{r['partition_n']}|{r['mean_brier_delta']:+.6f}|{r['mean_FS_delta']:+.2f}|{r['mean_FR_delta']:+.2f}|{r['mean_consumed_budget']:.2f}|{r['mean_selected_budget']:.2f}|{r['mean_audit_R']:.2f}|")
    lines+=['','## Limits and discriminating next experiments','',
            '- Shared misses are correlated model failures, not evidence that resistance is unknowable from sequence. Replicate with acquired-homolog inputs and ANI partitions; mutations and annotation sensitivity remain missing.',
            '- Audit failure may reflect audit sample size, audit/evaluation shift, rare-R counts or sequential reuse. This experiment does not identify their separate causal effects. A new version should vary audit acquisition and budget on frozen protocols, followed by independent confirmation.',
            '- Test whether the selection/action decomposition persists under richer inputs; distinguish a failed action from inadequate evidence for accepting it.',
            '- Preserve MIC/standard uncertainty, closest published AMR baselines and holder-blind confirmation as open evidence gaps.',
            '', '## Output and complete skips','',f'Figure: {stem.with_suffix(".png")}',f'Data: {dest/"results.json"}',
            'All original input skips:',json.dumps(raw['prior_skipped'],indent=2),
            'All gate infeasibilities:',json.dumps([r for r in raw['gates'] if r['status']=='infeasible'],indent=2)]
    report=HERE/f'REPAIR_DYNAMICS_{dataset.upper()}_RESULTS.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(findings))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('full','ani'),default='full')
    main(parser.parse_args().dataset)
