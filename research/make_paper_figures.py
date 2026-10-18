"""Render validated completion results without fitting or selecting any models.

Run with python. Tables are Markdown/HTML
with machine-readable JSON; figures are PNG/PDF/SVG with exact source data.
"""
from __future__ import annotations

import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figures'
FIGS, TABLES = [], []
DRUGS = ['meropenem', 'imipenem', 'ertapenem', 'ceftazidime', 'ciprofloxacin',
         'trimethoprim-sulfamethoxazole', 'amikacin', 'gentamicin', 'cefepime']
SHORT = dict(zip(DRUGS, ['Meropenem', 'Imipenem', 'Ertapenem', 'Ceftazidime',
                         'Ciprofloxacin', 'TMP–SMX', 'Amikacin', 'Gentamicin', 'Cefepime']))
PROTOCOLS = ['forward_2019', 'forward_2017']
PLABEL = {'forward_2019': 'Collection-year split 2019',
          'forward_2017': 'Collection-year split 2017'}
LABEL = {'random_gene': 'Random / gene', 'uncertainty_gene': 'Uncertainty / gene',
         'disagreement_gene': 'Disagreement / gene', 'diversity_gene': 'Diversity / gene',
         'random_diagnostic': 'Random / diagnostic',
         'uncertainty_diagnostic': 'Uncertainty / diagnostic',
         'information_diagnostic': 'Information / diagnostic',
         'decision_diagnostic': 'Decision / diagnostic', 'decision_gene': 'Decision / gene',
         'random_tree': 'Random / tree', 'gene': 'Logistic: genes',
         'gene_auxiliary_AST': 'Logistic: genes + AST', 'tree_gene': 'Trees: genes',
         'tree_gene_auxiliary_AST': 'Trees: genes + AST'}
COLOR = {'random_gene': '#3478A7', 'decision_diagnostic': '#C35A41',
         'random_tree': '#188477', 'random_diagnostic': '#9276AE',
         'decision_gene': '#BA9340', 'gene': '#3478A7', 'gene_auxiliary_AST': '#7EACCF',
         'tree_gene': '#188477', 'tree_gene_auxiliary_AST': '#83BEAF'}
POLICIES = ['random', 'uncertainty', 'information', 'decision']
PCOLOR = dict(zip(POLICIES, ['#3478A7', '#BA9340', '#9276AE', '#C35A41']))
WORLD = {'stable': 'Stable', 'prior_shift': 'Prior shift',
         'conditional_shift': 'Conditional shift', 'omitted_latent': 'Omitted information',
         'laboratory_noise': 'Laboratory noise'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def group_mean(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        obj = dict(zip(keys, key))
        obj['runs'] = len(group)
        for field in fields:
            vals = [float(r[field]) for r in group if r.get(field) is not None]
            obj[field] = float(np.mean(vals)) if vals else None
            obj[field + '_available_runs'] = len(vals)
        output.append(obj)
    return output


def format_value(v):
    if v is None:
        return 'NA'
    if isinstance(v, float):
        return f'{v:.6g}'
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def table(name, title, rows, columns, caption):
    write_json(OUT / 'tables' / (name + '.json'), {'title': title, 'caption': caption, 'rows': rows})
    headers = [label for key, label in columns]
    body = [[format_value(r.get(key)) for key, label in columns] for r in rows]
    md = [f'# {title}', '', caption, '', '| ' + ' | '.join(headers) + ' |',
          '| ' + ' | '.join('---' for _ in headers) + ' |']
    md += ['| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in r) + ' |' for r in body]
    (OUT / 'tables' / (name + '.md')).write_text('\n'.join(md) + '\n', encoding='utf-8')
    th = ''.join(f'<th>{html.escape(v)}</th>' for v in headers)
    tr = ''.join('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in row) + '</tr>' for row in body)
    page = '<!doctype html><meta charset="utf-8"><style>body{font:14px system-ui;margin:32px;color:#243446}table{border-collapse:collapse;width:100%}th,td{padding:9px 12px;border-bottom:1px solid #dce3e9;text-align:left}th{position:sticky;top:0;background:#eaf1f5}tr:nth-child(even){background:#f6f8fa}td{font-variant-numeric:tabular-nums}</style>'
    page += f'<h1>{html.escape(title)}</h1><p>{html.escape(caption)}</p><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'
    (OUT / 'tables' / (name + '.html')).write_text(page, encoding='utf-8')
    TABLES.append({'id': name, 'title': title, 'rows': len(rows), 'caption': caption})


def decorate(fig, title, note):
    fig.suptitle(title, x=.065, y=.985, ha='left', fontsize=15, fontweight='bold', color='#243446')
    fig.text(.065, .018, note, fontsize=8, color='#52616F', va='bottom')


def save(fig, name, title, caption, data, sources, role='main'):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / (name + '.' + ext), dpi=350 if ext == 'png' else 150,
                    facecolor='white', bbox_inches='tight', pad_inches=.12)
    write_json(OUT / 'source_data' / (name + '.json'),
               {'figure': name, 'title': title, 'caption': caption, 'sources': sources, 'data': data})
    FIGS.append({'id': name, 'title': title, 'caption': caption, 'role': role, 'sources': sources})
    plt.close(fig)
    print(f'Completed {name}', flush=True)


def forest(ax, rows, drugs, title, color='#C35A41'):
    index = {r['drug']: r for r in rows}
    for y, drug in enumerate(drugs):
        r = index[drug]
        if r['ci95'] is not None:
            ax.plot(r['ci95'], [y, y], color=color, lw=1.6)
            ax.plot(r['ci95'], [y, y], '|', color=color, ms=6)
        ax.plot(r['mean_difference'], y, 'o', color=color, ms=5)
    ax.axvline(0, color='#647685', lw=.9, ls='--')
    ax.set_yticks(range(len(drugs)), [SHORT[d] for d in drugs])
    ax.invert_yaxis()
    ax.set_title(title, loc='left')
    ax.set_xlabel('Brier difference (left − right)')
    ax.grid(axis='x', alpha=.22)


def heatmap(ax, matrix, rowlabels, collabels, title, cmap='YlGnBu', norm=None):
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm)
    ax.set_yticks(range(len(rowlabels)), rowlabels)
    ax.set_xticks(range(len(collabels)), collabels, rotation=55, ha='right', fontsize=8)
    ax.set_title(title, loc='left')
    ax.tick_params(length=0)
    for (y, x), val in np.ndenumerate(matrix):
        if np.isfinite(val):
            rgba = im.cmap(im.norm(val))
            lum = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
            ax.text(x, y, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if lum < .46 else '#253443')
        else:
            ax.text(x, y, 'NA', ha='center', va='center', fontsize=7)
    return im


def figures_main(internal, external, aux, tree, controlled, analysis, audit):
    # Fig. 1: actual protocol, explicitly separate external evaluation from querying.
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(top=.88, bottom=.13, left=.02, right=.98)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    def box(x, y, w, h, heading, text, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.12',
                                   facecolor=color, edgecolor='#D8E1E8', lw=.8))
        ax.text(x+.15,y+h-.24,heading,fontsize=11,fontweight='bold',va='top',color='#243446')
        ax.text(x+.15,y+h-.7,text,fontsize=9,va='top',linespacing=1.65,color='#344C5F')
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,
                                     lw=1.3,color='#647B8E'))
    box(.3,3.5,3.1,1.85,'Development data','Collection-year source / query splits\nGene annotation: 2026 RefGeneDB\nNine antibiotic prediction tasks','#EAF2F8')
    box(4.2,3.5,3.1,1.85,'Label-feedback loop','Ten query / repair strategies\nBudgets: 0, 10, 25, 50 if eligible\nSeparate query / audit / evaluation','#F5ECE7')
    box(8.1,3.5,3.1,1.85,'Internal evaluation','Five query seeds; same held samples\nPaired project bootstrap at budget 25\nUnadjusted exploratory intervals','#EAF3EF')
    arrow((3.5,4.5),(4.05,4.5)); arrow((7.4,4.5),(7.95,4.5))
    box(.3,.55,3.1,1.85,'Controlled diagnostics','Five simulated worlds; known truth\nFour policies; 20 seeds\nExplicit non-identifiability check','#F0EDF5')
    box(4.2,.55,3.1,1.85,'Information / type tests','Two additional measured AST labels\nMatched cases + strong tree control\nType-only transport; skips retained','#EEF4F5')
    box(8.1,.55,3.1,1.85,'Frozen external test','2,542 distinct BioSamples\n16,687 drug–sample pairs\nBudget-25 source models frozen','#EAF2F8')
    arrow((9.65,3.35),(9.65,2.5))
    title='Figure 1 | Study design and evidence boundaries'
    decorate(fig,title,'Retrospective computational study. BioSamples are not independent patients; external labels never choose queries or repairs.')
    save(fig,'Fig01_study_design',title,
         'Schematic of implemented computational experiments, not a validated biological mechanism. Internal evaluation uses collection-year splits; 2026 annotation knowledge is shared. External acquisition was screened and is not a blind prospective cohort.',
         {'internal_tasks':18,'strategies':internal['strategies'],'budgets':internal['budgets'],
          'external_BioSamples':analysis['unique_external_biosamples'],
          'external_drug_sample_pairs':analysis['external_drug_sample_pairs']},
         ['completion_analysis.json','decision_repair_results.json','COMPLETION_RESULTS_2026-10-08.md'])

    # Fig. 2: all strategies, all drugs, consistent color scale.
    strategies=list(internal['strategies'])
    means=group_mean([r for r in internal['results'] if r['budget']==25],
                     ['protocol','drug','strategy'],['brier'])
    em=group_mean(external['results'],['internal_protocol','drug','strategy'],['brier'])
    idx={(r['protocol'],r['drug'],r['strategy']):r['brier'] for r in means}
    eidx={(r['internal_protocol'],r['drug'],r['strategy']):r['brier'] for r in em}
    norm=Normalize(0,max(r['brier'] for r in means+em))
    fig,axes=plt.subplots(2,2,figsize=(15,11.4))
    fig.subplots_adjust(top=.91,bottom=.19,left=.1,right=.9,hspace=.65,wspace=.31)
    for domain,row in [('Internal',0),('External',1)]:
        for col,p in enumerate(PROTOCOLS):
            lookup=idx if row==0 else eidx
            matrix=np.array([[lookup[(p,d,s)] for s in strategies] for d in DRUGS])
            im=heatmap(axes[row,col],matrix,[SHORT[d] for d in DRUGS],
                       [LABEL[s] for s in strategies],f'{chr(97+row*2+col)}  {domain} | {PLABEL[p]}',norm=norm)
    cax=fig.add_axes([.93,.3,.015,.5]); fig.colorbar(im,cax=cax,label='Brier score (lower is better)')
    title='Figure 2 | Complete strategy comparison at query budget 25'
    decorate(fig,title,'Each cell averages five seeds on the same drug cohort. Shared samples and overlapping protocols are not independent replications.')
    save(fig,'Fig02_all_strategy_brier',title,'All ten implemented strategies and nine drugs. No strategy or drug is excluded according to performance. Absolute internal/external values reflect different cohorts and prevalences; they do not establish that external prediction is easier.',
         {'internal':means,'external':em},['decision_repair_results.json','external_transport_results.json'])

    contrasts=[r for r in analysis['contrasts'] if r['domain'] in ('internal','external') and
               (r['left'],r['right'])==('decision_diagnostic','random_gene')]
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    fig.subplots_adjust(top=.9,bottom=.13,left=.14,right=.96,hspace=.5,wspace=.48)
    low=min(r['ci95'][0] if r['ci95'] else r['mean_difference'] for r in contrasts)
    high=max(r['ci95'][1] if r['ci95'] else r['mean_difference'] for r in contrasts)
    for i,domain in enumerate(['internal','external']):
        for j,p in enumerate(PROTOCOLS):
            rows=[r for r in contrasts if r['domain']==domain and r['protocol']==p]
            forest(axes[i,j],rows,DRUGS,f'{chr(97+i*2+j)}  {domain.capitalize()} | {PLABEL[p]}')
            axes[i,j].set_xlim(low-.015,high+.015)
    title='Figure 3 | Diagnostic loop versus matched random queries'
    decorate(fig,title,'Left of zero favors the diagnostic loop. Project bootstrap: 1,000 replicates; exploratory 95% intervals, without multiplicity adjustment.')
    save(fig,'Fig03_paired_project_intervals',title,'Seeds are averaged within samples before project-level resampling. Intervals are conditional on frozen models/cohorts and do not account for development selection. All nine drugs and both protocols are shown. No CI is supplied when cluster information is inadequate.',contrasts,['completion_analysis.json'])

    # Fig. 4: matched-case information value, both estimator families, paired uncertainty.
    modes=['gene','gene_auxiliary_AST','tree_gene','tree_gene_auxiliary_AST']
    am=group_mean([r for r in aux['results']+tree['results'] if r['budget']==25],
                  ['protocol','drug','mode'],['brier','extra_training_assay_labels','extra_evaluation_assay_labels'])
    aix={(r['protocol'],r['drug'],r['mode']):r['brier'] for r in am}
    ac=[r for r in analysis['contrasts'] if r['domain']=='auxiliary_information' and
        (r['left'],r['right']) in [('gene_auxiliary_AST','gene'),('tree_gene_auxiliary_AST','tree_gene')]]
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.5))
    fig.subplots_adjust(top=.88,bottom=.13,left=.12,right=.97,hspace=.48,wspace=.4)
    for j,p in enumerate(PROTOCOLS):
        ax=axes[0,j]; x=np.arange(2)
        for k,(baseline,enriched) in enumerate([('gene','gene_auxiliary_AST'),('tree_gene','tree_gene_auxiliary_AST')]):
            values=[np.mean([aix[(p,d,m)] for d in DRUGS]) for m in [baseline,enriched]]
            ax.plot([k-.15,k+.15],values,'-',color=COLOR[baseline],lw=2)
            for off,m,v in zip([-.15,.15],[baseline,enriched],values):
                ax.scatter(k+off,v,s=80,color=COLOR[m],zorder=3)
                ax.text(k+off,v+.006,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_xticks(x,['Logistic','ExtraTrees']); ax.set_xlim(-.5,1.5)
        ax.set_ylim(.17,.38); ax.set_ylabel('Drug-macro Brier')
        ax.set_title(f'{chr(97+j)}  {PLABEL[p]} | matched cases',loc='left')
        ax.text(.02,.03,'Left point: genes\nRight point: genes + two measured AST',transform=ax.transAxes,fontsize=8)
        ax.grid(axis='y',alpha=.22)
        ax=axes[1,j]
        for offset, pair, color in [(-.13,('gene_auxiliary_AST','gene'),COLOR['gene']),
                                    (.13,('tree_gene_auxiliary_AST','tree_gene'),COLOR['tree_gene'])]:
            rows=[r for r in ac if r['protocol']==p and (r['left'],r['right'])==pair]
            by={r['drug']:r for r in rows}
            for y,d in enumerate(DRUGS):
                r=by[d]
                if r['ci95']:
                    ax.plot(r['ci95'],[y+offset]*2,color=color,lw=1)
                ax.plot(r['mean_difference'],y+offset,'o',color=color,ms=4)
        ax.set_yticks(range(9),[SHORT[d] for d in DRUGS]); ax.invert_yaxis()
        ax.axvline(0,color='#647685',ls='--',lw=.8)
        ax.set_xlabel('Brier difference: +AST − genes')
        ax.set_title(f'{chr(99+j)}  Per-drug information value',loc='left'); ax.grid(axis='x',alpha=.2)
    title='Figure 4 | Additional measured information improves matched-case prediction'
    decorate(fig,title,'Two other-drug assays are charged for every enriched training and evaluation sample. Tree controls were added after logistic results.')
    save(fig,'Fig04_auxiliary_information_value',title,'Descriptive nine-drug means plus unadjusted project-bootstrap contrasts. Complete cases and query sequences are matched within each estimator comparison. Auxiliary phenotypes are measured information, not free features or proof of an optimal clinical panel. Blue: logistic; green: ExtraTrees.',
         {'means':am,'contrasts':ac},['auxiliary_information_results.json','auxiliary_tree_results.json','completion_analysis.json'])

    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    fig.subplots_adjust(top=.88,bottom=.16,left=.075,right=.97,hspace=.48,wspace=.3)
    for ax,(world,name) in zip(axes.flat,list(WORLD.items())):
        for policy in POLICIES:
            rows=sorted([r for r in controlled['summary'] if r['world']==world and r['policy']==policy],key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in rows],[r['expected_excess_brier_vs_known_truth_mean'] for r in rows],
                    '-o',color=PCOLOR[policy],label=policy.capitalize(),ms=4)
        ax.set_title(name,loc='left'); ax.set_xlabel('Revealed labels'); ax.set_ylabel('Excess expected Brier')
        ax.set_ylim(-.006,.135); ax.grid(alpha=.2); ax.set_xticks([0,10,25,50])
    ax=axes.flat[-1]; ax.axis('off')
    ax.text(.02,.95,'Identifiability boundary',fontsize=13,fontweight='bold',va='top')
    ax.text(.02,.78,'Omitted information and laboratory noise\nhave identical observable distributions.\n\nTheir query sequences, weights and\npredictions match exactly.\n\nTruth is included in the candidate library;\nthis is a favorable algorithm benchmark,\nnot evidence of real AMR mechanisms.',fontsize=10,va='top',linespacing=1.5)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False)
    title='Figure 5 | Controlled success and non-identifiability boundaries'
    decorate(fig,title,'Means of 20 simulation seeds per world/policy. No inferential confidence claim is attached to these descriptive curves.')
    save(fig,'Fig05_controlled_boundaries',title,'Expected squared probability error relative to known simulated truth. All five worlds and four policies are shown. Diagnostic querying is not universally superior; identical observable worlds cannot be uniquely assigned a causal mechanism.',
         controlled['summary'],['controlled_diagnosis.json'])

    # Fig. 6: evaluation risk; older audit does not certify future safety.
    selected=['random_gene','decision_diagnostic','random_tree']
    rr=[r for r in internal['risk_coverage'] if r['budget']==25 and r['strategy'] in selected]
    rm=group_mean(rr,['protocol','drug','strategy','requested_coverage'],['brier','coverage'])
    macro=group_mean(rm,['protocol','strategy','requested_coverage'],['brier','coverage'])
    ar=[r for r in audit['rows'] if r['budget']==25 and r['seed']==7 and
        r['strategy']=='decision_diagnostic' and r['requested_evaluation_coverage']==.9]
    fig,axes=plt.subplots(2,2,figsize=(12,8.6))
    fig.subplots_adjust(top=.89,bottom=.16,left=.08,right=.97,hspace=.52,wspace=.3)
    for j,p in enumerate(PROTOCOLS):
        ax=axes[0,j]
        for s in selected:
            rows=sorted([r for r in macro if r['protocol']==p and r['strategy']==s],key=lambda r:r['requested_coverage'])
            ax.plot([r['coverage'] for r in rows],[r['brier'] for r in rows],'-o',label=LABEL[s],color=COLOR[s],ms=4)
        ax.set_title(f'{chr(97+j)}  Evaluation | {PLABEL[p]}',loc='left')
        ax.set_xlabel('Actual accepted fraction (drug mean)'); ax.set_ylabel('Accepted-sample Brier (drug mean)')
        ax.set_ylim(0,.65); ax.set_xlim(.2,1.03); ax.grid(alpha=.2)
        ax.legend(loc='upper right',frameon=False,fontsize=8)
        ax=axes[1,j]; by={r['drug']:r for r in ar if r['protocol']==p}
        for k,d in enumerate(DRUGS):
            r=by[d]; contamination=r['audit_sensitive_contamination']
            ax.plot(k,r['iid_binomial_upper95'],'_',color='#C35A41',ms=11)
            if contamination is not None:
                ax.plot([k,k],[contamination,r['iid_binomial_upper95']],color='#BDC6CD',lw=1)
                ax.plot(k,contamination,'o',color='#3478A7',ms=5)
            else:
                ax.plot(k,0,'x',color='#647685',ms=6)
            ax.text(k,1.055,str(r['audit_accepted_sensitive_n']),ha='center',fontsize=8)
        ax.set_xticks(range(9),[SHORT[d] for d in DRUGS],rotation=45,ha='right',fontsize=8)
        ax.set_ylim(-.03,1.12); ax.set_ylabel('R among audit-predicted S')
        ax.set_title(f'{chr(99+j)}  Older audit | seed 7 | requested 90%',loc='left'); ax.grid(axis='y',alpha=.2)
    title='Figure 6 | Coverage trade-offs and limits of audit evidence'
    decorate(fig,title,'Audit: blue = observed fraction; red = IID reference upper95; × = undefined (no predicted S). Top numbers are accepted-S counts.\nAudit bounds assume IID observations and do not certify future risk under shift. Evaluation curves are descriptive nine-drug means.')
    save(fig,'Fig06_risk_coverage_and_audit',title,'Budget 25. Evaluation Brier is averaged over seeds within drug, then equally over nine drugs at each requested coverage. Actual accepted fractions are plotted. Older audit uses a fixed illustrative seed 7 and diagnostic strategy, without seed selection. Binomial reference bounds ignore project dependence.',
         {'evaluation_drug_means':rm,'evaluation_macro':macro,'illustrative_audit':ar},
         ['decision_repair_results.json','completion_audit_risk.json'])


def figures_supp(internal, external, aux, tree, typed, analysis, benchmark):
    selected=['random_gene','decision_diagnostic','random_tree','decision_gene','random_diagnostic']
    # Each drug uses its own eligible budgets, avoiding composition-shifted macro curves.
    for number,p in enumerate(PROTOCOLS,1):
        rows=[r for r in internal['results'] if r['protocol']==p and r['strategy'] in selected]
        means=group_mean(rows,['drug','strategy','budget'],['brier'])
        fig,axes=plt.subplots(3,3,figsize=(12,9))
        fig.subplots_adjust(top=.89,bottom=.15,left=.07,right=.97,hspace=.53,wspace=.32)
        for ax,d in zip(axes.flat,DRUGS):
            for s in selected:
                rr=sorted([r for r in means if r['drug']==d and r['strategy']==s],key=lambda r:r['budget'])
                ax.plot([r['budget'] for r in rr],[r['brier'] for r in rr],'-o',color=COLOR[s],label=LABEL[s],ms=3)
            ax.set_title(SHORT[d],loc='left'); ax.set_xlabel('Revealed query labels'); ax.set_ylabel('Brier')
            ax.set_ylim(0,.65); ax.grid(alpha=.2); ax.set_xticks([0,10,25,50])
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False,fontsize=8)
        title=f'Figure S{number} | Per-drug learning curves: {PLABEL[p]}'
        decorate(fig,title,'Five-seed means, without seed-based inference. Unavailable budgets are omitted; no extrapolation. Audit labels add a separate fixed cost.')
        save(fig,f'S{number:02d}_learning_{p}',title,'Five fixed comparisons expose query-policy and repair-action differences. All nine drugs shown; per-drug budget eligibility preserved. Curves are descriptive and do not treat seeds as independent samples.',means,['decision_repair_results.json'],'supplement')

    em=group_mean(external['results'],['internal_protocol','drug','strategy'],['auroc','auprc','resistant','n'])
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    fig.subplots_adjust(top=.89,bottom=.22,left=.08,right=.97,hspace=.5,wspace=.3)
    for i,metric in enumerate(['auroc','auprc']):
        for j,p in enumerate(PROTOCOLS):
            ax=axes[i,j]
            for k,s in enumerate(['random_gene','decision_diagnostic','random_tree']):
                by={r['drug']:r for r in em if r['internal_protocol']==p and r['strategy']==s}
                ax.plot(range(9),[by[d][metric] for d in DRUGS],'o-',color=COLOR[s],label=LABEL[s],ms=4)
            if metric=='auprc':
                by={r['drug']:r for r in em if r['internal_protocol']==p and r['strategy']=='random_gene'}
                ax.plot(range(9),[by[d]['resistant']/by[d]['n'] for d in DRUGS],':',color='#7E8C98',label='R prevalence')
            ax.set_xticks(range(9),[SHORT[d] for d in DRUGS],rotation=45,ha='right',fontsize=8)
            ax.set_ylim(0,1.05); ax.set_ylabel(metric.upper()); ax.grid(axis='y',alpha=.2)
            ax.set_title(f'{metric.upper()} | {PLABEL[p]}',loc='left')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.045),frameon=False)
    title='Figure S3 | External discrimination and prevalence reference'
    decorate(fig,title,'Five-seed means from internally frozen models. Drug prevalences differ; ranking scores are not clinical-benefit estimates.')
    save(fig,'S03_external_discrimination',title,'Nine drugs; fixed random-gene, diagnostic and tree comparisons. AUPRC includes drug-specific resistant prevalence as a descriptive no-skill reference. No external labels were used for fitting.',em,['external_transport_results.json'],'supplement')

    contrasts=[r for r in analysis['contrasts'] if r['domain']=='typed_transport']
    fig,axes=plt.subplots(1,2,figsize=(11,5.8),gridspec_kw={'width_ratios':[2.3,1]})
    fig.subplots_adjust(top=.86,bottom=.2,left=.15,right=.97,wspace=.6)
    for ax,p in zip(axes,['mlst','cgmlst_hc100']):
        rows=[r for r in contrasts if r['protocol']==p and r['left']=='decision_diagnostic']
        drugs=[d for d in DRUGS if d in {r['drug'] for r in rows}]
        forest(ax,rows,drugs,f'{p} | {len(drugs)} drug task(s)')
        ax.set_xlabel('Brier: diagnostic − random gene')
    title='Figure S4 | Type-only transport with skipped tasks preserved'
    decorate(fig,title,'Source excludes target types; adaptation queries come from target types. No simultaneous forward-time isolation.\nIntervals resample evaluation types; absent intervals indicate inadequate clusters. Eight amended tasks were skipped.')
    save(fig,'S04_typed_transport',title,'Do not interpret cgMLST results as nine-drug replication: only one drug is eligible. Source and target types are separated before adaptation, not between adapted training and evaluation.',
         {'contrasts':contrasts,'skipped':typed['skipped'],'cohorts':typed['cohort_audits']},
         ['completion_analysis.json','typed_transport_results.json'],'supplement')

    audits=aux['cohort_audits']
    costmeans=group_mean([r for r in aux['results'] if r['budget']==25 and r['mode']=='gene_auxiliary_AST'],
                        ['protocol','drug'],['extra_training_assay_labels','extra_evaluation_assay_labels','target_query_assay_labels'])
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(top=.84,bottom=.24,left=.09,right=.97,wspace=.35)
    ax=axes[0]; by={r['drug']:r for r in audits if r['protocol']=='forward_2019'}
    ax.bar(range(9),[by[d]['complete_records']/by[d]['all_records'] for d in DRUGS],color='#83BEAF')
    ax.set_ylim(0,1); ax.set_ylabel('Fraction with both auxiliary AST labels')
    ax.set_title('a  Complete-case selection',loc='left')
    ax=axes[1]; x=np.arange(9)
    for off,p,color in [(-.18,'forward_2019','#3478A7'),(.18,'forward_2017','#9276AE')]:
        by={r['drug']:r for r in costmeans if r['protocol']==p}
        ax.bar(x+off,[by[d]['extra_training_assay_labels']+by[d]['extra_evaluation_assay_labels'] for d in DRUGS],width=.34,color=color,label=p[-4:])
    ax.set_ylabel('Extra auxiliary labels: training + evaluation'); ax.set_title('b  Additional assay burden at budget 25',loc='left'); ax.legend(frameon=False)
    for ax in axes:
        ax.set_xticks(range(9),[SHORT[d] for d in DRUGS],rotation=45,ha='right',fontsize=8); ax.grid(axis='y',alpha=.2)
    title='Figure S5 | Selection and costs behind auxiliary-information gains'
    decorate(fig,title,'Costs count retrospective label reveals, not measured money or turnaround time. Two extra assays per enriched sample; target queries cost 25 more.')
    save(fig,'S05_auxiliary_selection_cost',title,'Matched complete cases are a selected subset of the development cohort. Training and evaluation auxiliary labels are both charged, including source training labels. The two protocols reuse some samples.',
         {'cohorts':audits,'costs':costmeans},['auxiliary_information_results.json'],'supplement')

    means=group_mean(aux['results']+tree['results'],['protocol','drug','mode','budget'],['brier'])
    fig,axes=plt.subplots(1,2,figsize=(11,5.8))
    fig.subplots_adjust(top=.84,bottom=.26,left=.08,right=.97,wspace=.32)
    for ax,p,budgets in zip(axes,PROTOCOLS,[[0,10,25,50],[0,10,25]]):
        for mode in ['gene','gene_auxiliary_AST','tree_gene','tree_gene_auxiliary_AST']:
            yy=[]
            for b in budgets:
                rr=[r for r in means if r['protocol']==p and r['mode']==mode and r['budget']==b]
                assert {r['drug'] for r in rr}==set(DRUGS)
                yy.append(np.mean([r['brier'] for r in rr]))
            ax.plot(budgets,yy,'o-',label=LABEL[mode],color=COLOR[mode],ms=4)
        ax.set_title(PLABEL[p],loc='left'); ax.set_xlabel('Target labels queried'); ax.set_ylabel('Drug-macro Brier')
        ax.set_ylim(.15,.45); ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.065),frameon=False)
    title='Figure S6 | Auxiliary-information benefit across eligible budgets'
    decorate(fig,title,'Exactly nine drugs at every displayed budget. Same complete cases and query orders; enriched models require additional measured labels.')
    save(fig,'S06_auxiliary_budget_curves',title,'The budget-100 data are retained in tables/source results but excluded from this nine-drug macro curve because only five drugs qualify. No drug composition change across plotted budgets.',means,
         ['auxiliary_information_results.json','auxiliary_tree_results.json'],'supplement')

    strata=group_mean([r for r in external['stratified_results'] if 'testing_standard' in r],
                     ['internal_protocol','drug','strategy','testing_standard'],['brier','n','resistant'])
    standards=sorted({r['testing_standard'] for r in strata})
    standard_label={s:('BSAC' if s=='British Society for Antimicrobial Chemotherapy (BSAC)'
                       else s.replace(', ', '+')) for s in standards}
    columns=[(s,t) for s in ['random_gene','decision_diagnostic','random_tree'] for t in standards]
    strategy_label={'random_gene':'Random gene','decision_diagnostic':'Decision diagnostic','random_tree':'Random tree'}
    fig,axes=plt.subplots(1,2,figsize=(13,6.7))
    fig.subplots_adjust(top=.85,bottom=.3,left=.1,right=.89,wspace=.35)
    idx={(r['internal_protocol'],r['drug'],r['strategy'],r['testing_standard']):r['brier'] for r in strata}
    for ax,p in zip(axes,PROTOCOLS):
        matrix=np.array([[idx.get((p,d,s,t),np.nan) for s,t in columns] for d in DRUGS])
        im=heatmap(ax,matrix,[SHORT[d] for d in DRUGS],
                   [strategy_label[s]+' / '+standard_label[t] for s,t in columns],
                   PLABEL[p],norm=Normalize(0,max(r['brier'] for r in strata)))
    cax=fig.add_axes([.93,.27,.015,.53]); fig.colorbar(im,cax=cax,label='Brier (lower is better)')
    title='Figure S7 | External labels stratified by reported testing standard'
    decorate(fig,title,'Reported standard groups are different cohorts. MIC breakpoints and standard versions have not been harmonized; NA is not zero.')
    save(fig,'S07_external_testing_standards',title,'All reported standard categories shown for fixed three-strategy comparison. BSAC abbreviates British Society for Antimicrobial Chemotherapy; CLSI+EUCAST is the literal jointly reported category. Full names and sample counts are in source data. Standard-stratified performance does not identify a causal effect of laboratory protocol or remove institutional dependence.',strata,
         ['external_transport_results.json'],'supplement')

    # Fixed source protocol/seed for calibration; verify hashes before reading NPZ.
    expected={r['file']:r['sha256'] for r in read('completion_validation.json')['artifact_hashes']}
    entries=[r for r in external['prediction_files'] if r['internal_protocol']=='forward_2019' and
             r['seed']==7 and r['strategy'] in ('random_gene','decision_diagnostic')]
    calibration=[]
    fig,axes=plt.subplots(3,3,figsize=(10.5,10))
    fig.subplots_adjust(top=.88,bottom=.14,left=.08,right=.97,hspace=.5,wspace=.37)
    for ax,d in zip(axes.flat,DRUGS):
        ax.plot([0,1],[0,1],'--',color='#A1AFB9',lw=1)
        for s in ['random_gene','decision_diagnostic']:
            entry=next(r for r in entries if r['drug']==d and r['strategy']==s)
            assert sha(entry['file'])==expected[entry['file']]
            z=np.load(entry['file'],allow_pickle=False); y,p=z['labels'],z['probabilities']
            bins=np.minimum((p*10).astype(int),9); rows=[]
            for k in range(10):
                mask=bins==k
                if not mask.any(): continue
                rows.append({'drug':d,'strategy':s,'bin':k,'n':int(mask.sum()),
                             'predicted_mean':float(p[mask].mean()),'observed_R':float(y[mask].mean())})
            ax.plot([r['predicted_mean'] for r in rows],[r['observed_R'] for r in rows],color=COLOR[s],lw=1,label=LABEL[s])
            ax.scatter([r['predicted_mean'] for r in rows],[r['observed_R'] for r in rows],
                       s=[8+np.sqrt(r['n'])*1.5 for r in rows],color=COLOR[s],alpha=.85)
            calibration.extend(rows)
        ax.set_title(SHORT[d],loc='left'); ax.set_xlim(-.02,1.02); ax.set_ylim(-.02,1.02)
        ax.set_xlabel('Mean predicted resistance'); ax.set_ylabel('Observed R fraction'); ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False)
    title='Figure S8 | External calibration with per-bin sample counts'
    decorate(fig,title,'Fixed illustrative protocol 2019 and seed 7, chosen without score ranking. Ten equal-width probability bins; marker area grows with √n.\nSparse bins are noisy; no smoothing or external recalibration. Exact sample counts are supplied in source data.')
    save(fig,'S08_external_calibration',title,'Nine drugs shown separately to avoid pooling distinct resistance prevalences. This is a descriptive reliability display for a fixed seed, not a confidence interval or model-selection analysis.',calibration,
         ['external_transport_results.json','completion_validation.json'],'supplement')

    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    fig.subplots_adjust(top=.83,bottom=.23,left=.09,right=.97,wspace=.3)
    data=benchmark['rows']
    for h,color in [(5,'#3478A7'),(10,'#188477'),(25,'#C35A41')]:
        for objective,style in [('information','--'),('decision','-')]:
            rr=sorted([r for r in data if r['hypotheses']==h and r['objective']==objective],key=lambda r:r['pool_n'])
            for ax,field,scale in [(axes[0],'median_seconds',1000),(axes[1],'traced_peak_bytes',1/2**20)]:
                ax.plot([r['pool_n'] for r in rr],[r[field]*scale for r in rr],style+'o',color=color,ms=4,label=f'{objective}, H={h}')
                ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Median query-scoring time (ms)'); axes[1].set_ylabel('Traced peak allocation (MiB)')
    for ax in axes: ax.set_xlabel('Query pool samples')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.05),frameon=False,fontsize=8)
    title='Figure S9 | Scoped query-scoring time and memory'
    decorate(fig,title,'Three timings after warm-up per case. Concurrent workloads; traced allocation is not process RSS. No end-to-end performance ranking.')
    save(fig,'S09_computational_scaling',title,benchmark['scope']+'. Twenty-four fixed cases; direct enumeration and invariance checks passed.',data,['computational_benchmark.json'],'supplement')

    actions=group_mean([dict(r,indicator=1) for r in internal['results'] if r['budget']==25 and
                        r['strategy']=='decision_diagnostic'],['protocol','action'],['indicator'])
    names=sorted({r['action'] for r in actions})
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    fig.subplots_adjust(top=.84,bottom=.27,left=.09,right=.97,wspace=.35)
    for ax,p in zip(axes,PROTOCOLS):
        counts=Counter(r['action'] for r in internal['results'] if r['protocol']==p and r['budget']==25 and r['strategy']=='decision_diagnostic')
        ax.bar(range(len(names)),[counts[n] for n in names],color='#C38E7F')
        ax.set_ylim(0,35)
        ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names],fontsize=8)
        ax.set_ylabel('Drug × seed runs'); ax.set_title(PLABEL[p],loc='left'); ax.grid(axis='y',alpha=.2)
    title='Figure S10 | Repair actions selected by the diagnostic loop'
    decorate(fig,title,'Each protocol has 9 drugs × 5 seeds. These are selected prediction rules, not identified causal mechanisms or independent patients.')
    save(fig,'S10_repair_actions',title,'All budget-25 selected actions shown without filtering by outcome. Frequencies describe algorithm behavior only.',actions,['decision_repair_results.json'],'supplement')


def make_tables(internal, external, aux, tree, controlled, typed, analysis, benchmark):
    ea=read('external_features_v2.json')
    rows=[]
    for d in DRUGS:
        c=ea['drug_counts'][d]; e=next(r for r in external['cohort_audits'] if r['drug']==d)
        rows.append({'drug':d,'n':c['paired_genotype_RS'],'S':c['paired_S'],'R':c['paired_R'],
                     'projects':len(e['projects']),'standards':e['standards']})
    table('Table01_external_cohort','外部队列：九药物样本构成',rows,
          [('drug','Drug'),('n','Drug–sample pairs'),('S','S'),('R','R'),('projects','Projects'),('standards','Reported standards')],
          '共有2,542个不同BioSample，16,687个药物—样本配对；均非独立患者数。accession隔离不能替代序列、患者或实验室独立性。')
    table('Table02_budget25_comparison','预算25：所有策略的描述性药物平均',analysis['budget25_descriptive_means'],
          [('domain','Domain'),('protocol','Protocol'),('strategy','Strategy'),('drugs','Drugs'),('drug_macro_brier','Drug-macro Brier')],
          '先在每种药物内平均种子，再平均药物。不同domain可能为不同病例，不能直接作为同病例效应比较。cgMLST仅一药物。')
    am=group_mean([r for r in aux['results']+tree['results'] if r['budget']==25],
                  ['protocol','drug','mode'],['brier','auroc','auprc','extra_training_assay_labels','extra_evaluation_assay_labels','target_query_assay_labels'])
    table('Table03_auxiliary_information','辅助药敏信息：匹配完整病例及额外成本',am,
          [('protocol','Protocol'),('drug','Drug'),('mode','Estimator / features'),('brier','Brier'),('auroc','AUROC'),('auprc','AUPRC'),('extra_training_assay_labels','Extra training AST'),('extra_evaluation_assay_labels','Extra evaluation AST'),('target_query_assay_labels','Target labels')],
          '预算25。两种辅助药物对全部富集训练/评价样本计费；树模型属于看到Logistic结果后追加的探索性强对照。')
    table('Table04_controlled','受控任务：全部世界、策略和预算',controlled['summary'],
          [('world','World'),('policy','Policy'),('budget','Budget'),('runs','Runs'),('expected_excess_brier_vs_known_truth_mean','Excess expected Brier'),('correct_declaration_mean','Correct declaration fraction'),('unknown_mean','Unknown fraction')],
          '20次模拟种子的描述性均值。真值在候选库中；不可识别世界中的声明不能解释为真实生物机制诊断。')
    inventory=[]
    for name,obj,field in [('Diagnostic loop',internal,'trajectory_audits'),('Controlled benchmark',controlled,'queries'),
                            ('Auxiliary AST',aux,'prediction_files'),('Typed transport',typed,'prediction_files'),
                            ('External transport',external,'prediction_files'),('Auxiliary tree control',tree,'prediction_files')]:
        inventory.append({'experiment':name,'budget_rows':len(obj['results']),'prediction_files':len(obj[field]),'skipped':len(obj.get('skipped',[]))})
    table('Table05_experiment_inventory','本批实验量与验证范围',inventory,
          [('experiment','Experiment'),('budget_rows','Budget metric rows'),('prediction_files','Prediction files'),('skipped','Skipped tasks')],
          '合计7,750条预算指标和2,580个预测文件。另有分层、风险记录和24个计算规模案例；实验数量不是期刊录用标准。')
    table('TableS01_all_paired_intervals','全部预算25配对比较与区间',analysis['contrasts'],
          [('domain','Domain'),('protocol','Protocol'),('drug','Drug'),('left','Left'),('right','Right'),('mean_difference','Brier left − right'),('n','Evaluation samples'),('clusters','Clusters'),('ci95','Exploratory 95% CI')],
          '每样本先平均种子差值，再对项目/类型进行1000次重采样。未多重校正，条件于固定模型与队列；缺少区间不等于零不确定性。')
    table('TableS02_strategy_definitions','十种实现策略',
          [{'strategy':s,'query':q,'repair':r} for s,(q,r) in internal['strategies'].items()],
          [('strategy','Strategy'),('query','Query policy'),('repair','Repair family')],
          '通用实现的对照策略；不是已发表主动测试/主动域适应方法的忠实复现。')
    cohorts=[r for r in internal['trajectory_audits'] if r['seed']==7 and r['strategy']=='random_gene']
    table('TableS03_internal_splits','内部四组划分与预算可用性',cohorts,
          [('protocol','Protocol'),('drug','Drug'),('source_n','Source'),('query_pool_n','Query pool'),('audit_n','Held audit'),('evaluation_n','Evaluation'),('budgets','Eligible budgets')],
          '病例按采集年份划分；基因注释来自2026 RefGeneDB，没有复原历史时点可用注释。四组BioSample互斥。')
    table('TableS04_auxiliary_selection','辅助信息完整病例筛选',aux['cohort_audits'],
          [('protocol','Protocol'),('drug','Drug'),('auxiliary_drugs','Other-drug assays'),('all_records','All records'),('complete_records','Complete records'),('source_n','Source'),('pool_n','Query pool'),('evaluation_n','Evaluation'),('budgets','Eligible budgets')],
          '同一药物两协议的all_records/complete_records对应同一全体记录，不应相加作为独立病例。')
    table('TableS05_typed_skips','分型迁移：未通过可用性门槛的任务',typed['skipped'],
          [('drug','Drug'),('lineage_field','Typing'),('source_n','Source'),('query_pool_n','Query pool'),('evaluation_n','Evaluation'),('reason','Reason')],
          '这里保存修订后的type-only任务8个跳过；原同时前向时间与分型隔离的18个任务均失败，原记录在typed_lineage_results.json。跳过任务不能记为成功。')
    strata=[]
    for row in external['stratified_results']:
        strata.append({**row,'stratum_kind':'testing_standard' if 'testing_standard' in row else 'lineage_screen',
                       'stratum':row.get('testing_standard',row.get('lineage_screen'))})
    sm=group_mean(strata,['internal_protocol','drug','strategy','stratum_kind','stratum'],['n','resistant','brier','auroc','auprc'])
    table('TableS06_external_strata','外部标准及已知源分型筛查：所有策略',sm,
          [('internal_protocol','Source protocol'),('drug','Drug'),('strategy','Strategy'),('stratum_kind','Stratum kind'),('stratum','Stratum'),('n','N'),('resistant','R'),('brier','Brier'),('auroc','AUROC'),('auprc','AUPRC')],
          '五种子均值，NA表示指标不可计算。检测标准版本/MIC未统一，分型缺失时不能证明完全谱系独立；表格不代表独立人群复现。')
    table('TableS07_computational_benchmark','查询打分规模、时长与内存',benchmark['rows'],
          [('hypotheses','Hypotheses'),('pool_n','Pool samples'),('objective','Objective'),('median_seconds','Median seconds'),('durations_seconds','Three timings'),('traced_peak_bytes','Traced peak bytes')],
          benchmark['scope']+'; 不可解释为端到端临床速度、整进程峰值内存或严格硬件公平排行。')


def package(inputs):
    # Lightweight contact sheet for review; full-size exports remain independent.
    main=[r for r in FIGS if r['role']=='main']
    sheet=Image.new('RGB',(1440,3*465),'#EDF1F5'); draw=ImageDraw.Draw(sheet)
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),18) if fontpath.exists() else ImageFont.load_default()
    for i,entry in enumerate(main):
        im=Image.open(OUT/(entry['id']+'.png')).convert('RGB'); im.thumbnail((696,420))
        x=12+(i%2)*720; y=12+(i//2)*465
        sheet.paste(im,(x+(696-im.width)//2,y))
        draw.text((x+8,y+428),entry['id'],fill='#243446',font=font)
    sheet.save(OUT/'main_figures_preview.jpg',quality=94)
    cards=[]
    for r in FIGS:
        name=r['id']; cards.append(f'<article class="{r["role"]}"><h2>{html.escape(r["title"])}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png"></a><p>{html.escape(r["caption"])}</p><nav><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a> · <a href="source_data/{name}.json">Source data</a></nav></article>')
    tables=''.join(f'<li><a href="tables/{r["id"]}.html">{html.escape(r["title"])}</a> ({r["rows"]} rows) · <a href="tables/{r["id"]}.md">Markdown</a> · <a href="tables/{r["id"]}.json">JSON</a></li>' for r in TABLES)
    gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AMR experiment figures</title><style>body{font:16px system-ui;max-width:1350px;margin:40px auto;padding:0 24px;color:#243446;background:#f2f5f7}header{padding:30px;background:#fff;border-radius:14px}article{background:#fff;border-radius:14px;padding:28px;margin:28px 0}img{width:100%;height:auto}h2{font-size:20px}p{line-height:1.7}a{color:#26739d}li{margin:14px 0}</style><header><h1>实验图表 · 2026-10-08</h1><p>6张主图、10张补充图、12张表格。已验证结果的探索性展示，不是投稿就绪证明。全部导出包含原始来源及说明。</p><p>诊断闭环未取得药物平均优势。辅助药敏改善需要额外实测信息。保留负结果、跳过任务与不确定性。</p><a href="#tables">查看表格</a></header>'+''.join(cards)+'<article id="tables"><h2>Tables</h2><ul>'+tables+'</ul></article></html>'
    (OUT/'index.html').write_text(gallery,encoding='utf-8')
    lines=['# 实验图表包：2026-10-08','','已完成 **6张主图、10张补充图、12张表格**。每图包含350 dpi PNG、PDF、SVG及独立JSON源数据；每表包含Markdown、HTML及JSON。',
           '', '**当前为探索性研究图表，不代表Nature/NCS投稿证据已完备。**', '',
           '![主图预览](main_figures_preview.jpg)', '', '## 图表目录','']
    for r in FIGS:
        n=r['id']; lines += [f'### {r["title"]}', '', f'[PNG]({n}.png) · [PDF]({n}.pdf) · [SVG]({n}.svg) · [源数据](source_data/{n}.json)', '',r['caption'],'']
    lines += ['## 表格','']+[f'- [{r["title"]}](tables/{r["id"]}.md)：{r["rows"]}行。' for r in TABLES]
    lines += ['', '## 复现与解释', '',
              '运行 `python research/make_paper_figures.py` 仅重绘，不重新训练或选择模型。',
              '', '主图全十策略、九药物和两个协议均保留。种子是计算重复，不作为独立患者；药物与协议共享样本。项目重采样区间未经多重校正。',
              '', '辅助药敏的完整病例、训练/评价额外标签成本、分型任务失败及外部独立性边界均披露。校准与审计示例固定seed 7，不按成绩挑选。',
              '', '所有表格为描述性或既有配对分析，未添加显著性星号、选择有利预算或新拟合结果。正式投稿仍需检查期刊尺寸要求并根据最终主张调整图序。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('figure_manifest.json','figure_validation.json')]
    write_json(OUT/'figure_manifest.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
              'main_figures':len(main),'supplement_figures':len(FIGS)-len(main),'tables':len(TABLES),
              'matplotlib_version':version('matplotlib'),'png_dpi':350,
              'input_hashes':{n:sha(ROOT/n) for n in inputs},'script_sha256':sha(__file__),
              'figures':FIGS,'table_inventory':TABLES,
              'artifact_hashes':{p.relative_to(OUT).as_posix():sha(p) for p in files},
              'limitations':['exploratory presentation; not a journal-readiness certificate',
                             'No new fitting, performance-based selection, or significance tests performed.']})


def main():
    inputs=['decision_repair_results.json','external_transport_results.json',
            'auxiliary_information_results.json','auxiliary_tree_results.json',
            'controlled_diagnosis.json','typed_transport_results.json','completion_analysis.json',
            'completion_audit_risk.json','computational_benchmark.json','external_features_v2.json',
            'completion_validation.json','COMPLETION_RESULTS_2026-10-08.md']
    manifest=read('completion_manifest.json')
    assert manifest['validation_status']=='passed'
    for n in inputs: assert sha(ROOT/n)==manifest['hashes'][n], n
    for path in [OUT,OUT/'tables',OUT/'source_data']: path.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.titlepad':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#B3C0CB','axes.labelcolor':'#344C5F','xtick.color':'#52616F',
        'ytick.color':'#52616F','grid.color':'#C1CDD6','legend.fontsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    internal,external,aux,tree,controlled,typed,analysis,audit,benchmark=[read(n) for n in inputs[:9]]
    figures_main(internal,external,aux,tree,controlled,analysis,audit)
    figures_supp(internal,external,aux,tree,typed,analysis,benchmark)
    make_tables(internal,external,aux,tree,controlled,typed,analysis,benchmark)
    package(inputs)
    print(json.dumps({'figures':len(FIGS),'tables':len(TABLES),'output':str(OUT)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
