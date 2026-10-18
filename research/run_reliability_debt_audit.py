"""Stratified reliability-debt audit of the frozen E. coli trajectories.

This audit does not alter predictions or labels. It tests whether apparent
repair benefit changes with related-component size, assay standard, or time.
All strata and empty/insufficient cases are retained.
"""
from __future__ import annotations
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv, hashlib, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from counterexample_guided_repair import parse_year

HERE = Path(__file__).resolve().parent
ROOT = ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT = ROOT / 'reliability_debt_audit_v2'
DRUGS = ('meropenem','imipenem','ertapenem','ceftazidime','ciprofloxacin',
         'trimethoprim/sulfamethoxazole','amikacin','gentamicin','cefepime')
STRATEGIES = ('decision_diagnostic','random_gene','uncertainty_gene','random_tree')
THRESHOLDS = (.99,.95)
SEEDS = (7,17,27,37,47)
BUDGETS = (0,10,25,50)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def metric(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    resistant=(y==1); pred_s=(p<.5)
    return {'n':int(len(y)),'R':int(resistant.sum()),
            'brier':float(np.mean((p-y)**2)) if len(y) else None,
            'mean_probability':float(np.mean(p)) if len(p) else None,
            'calibration_in_large':float(np.mean(p)-np.mean(y)) if len(y) else None,
            'ece10':sum(float(np.sum(ix)/len(y)*abs(np.mean(p[ix])-np.mean(y[ix]))) for b in range(10)
                        if np.any(ix:=(np.minimum((p*10).astype(int),9)==b))) if len(y) else None,
            'false_susceptible':int((resistant & pred_s).sum()),
            'false_susceptible_rate':float((resistant & pred_s).sum()/resistant.sum()) if resistant.any() else None,
            'false_resistant':int((~resistant & ~pred_s).sum()),
            'false_resistant_rate':float((~resistant & ~pred_s).sum()/(~resistant).sum()) if (~resistant).any() else None}


def standard(v):
    s=(v or '').upper().replace(';',',')
    vals={x.strip() for x in s.split(',') if x.strip()}
    if vals=={'EUCAST'}: return 'EUCAST'
    if vals=={'CLSI'}: return 'CLSI'
    if vals=={'EUCAST','CLSI'}: return 'mixed'
    return 'unknown'


def year_bin(v):
    years={parse_year(s) for s in (v or '').split(';')}
    y=next(iter(years)) if len(years)==1 else None
    if y is None:return 'unknown'
    if y<2015:return '<2015'
    if y<2020:return '2015-2019'
    return '>=2020'


def main():
    OUT.mkdir(exist_ok=True)
    pairs=defaultdict(list)
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            pairs[(r['sequence_id'] or r['assembly_accession'],r['antibiotic'])].append(r)
    manifest={'status':'running','designation':'frozen post-result stratified reliability audit',
              'inputs':{'pair_sha256':sha(ROOT/'ecoli_human_sr_pairs.tsv'),
                        'full_manifest_sha256':sha(ROOT/'repair_full_scaled5000/manifest.json')},
              'code_sha256':sha(Path(__file__)),
              'strata':{'component_size':['singleton','2-9','>=10'],
                        'standard':['EUCAST','CLSI','mixed','unknown'],
                        'collection_period':['<2015','2015-2019','>=2020','unknown']}}
    rows=[]; contrasts=[]; artifacts=[]
    for drug in DRUGS:
      for threshold in THRESHOLDS:
       ddir=ROOT/'repair_full_scaled5000'/drug.replace('/','-')
       for seed in SEEDS:
        for strategy in STRATEGIES:
         stem=f'{drug.replace("/","-")}_t{threshold}_{strategy}_{seed}'
         npz=ddir/(stem+'.npz'); jp=ddir/(stem+'.json')
         if not npz.exists() or not jp.exists():
             raise RuntimeError('Frozen prediction is missing: '+str(npz))
         with np.load(npz,allow_pickle=False) as z:
             samples=z['evaluation_samples'].astype(str).tolist(); clusters=z['evaluation_clusters'].astype(str).tolist(); y=z['labels']
             all_clusters=[]
             for role in ('source','query_pool','audit','evaluation'):
                 all_clusters.extend(z[role+'_clusters'].astype(str).tolist())
             sizes=Counter(all_clusters)
             metadata=[]
             for n,c in zip(samples,clusters):
                 rs=pairs.get((n,drug),[])
                 std=standard(';'.join(sorted({r['standards'] for r in rs})))
                 yr=year_bin(';'.join(sorted({r['collection_date'] for r in rs})))
                 sz=sizes[c]; sb='singleton' if sz==1 else '2-9' if sz<10 else '>=10'
                 metadata.append({'sample':n,'component':c,'component_size':sb,'standard':std,'collection_period':yr})
             for budget in BUDGETS:
                 key=f'p_b{budget}'
                 if key not in z.files: continue
                 m_all=metric(y,z[key])
                 base={'drug':drug,'threshold':threshold,'seed':seed,'strategy':strategy,'budget':budget,
                       'stratum':'all','level':'all',**m_all}
                 rows.append(base)
                 for field in ('component_size','standard','collection_period'):
                     vals=defaultdict(list)
                     for i,meta in enumerate(metadata): vals[meta[field]].append(i)
                     for level in manifest['strata'][field]:
                         ix=np.asarray(vals.get(level,[]),dtype=int)
                         rows.append({'drug':drug,'threshold':threshold,'seed':seed,'strategy':strategy,'budget':budget,
                                      'stratum':field,'level':level,**metric(y[ix],z[key][ix])})
             artifacts.append({'path':str(npz),'sha256':sha(npz)})
    # Matched controller-minus-random contrasts from exactly the same files/strata.
    lookup={(r['drug'],r['threshold'],r['seed'],r['strategy'],r['budget'],r['stratum'],r['level']):r for r in rows}
    omitted=[]
    for drug in DRUGS:
      for threshold in THRESHOLDS:
       for seed in SEEDS:
        for budget in (25,50):
         for strata in ('all','component_size','standard','collection_period'):
          levels={'all'} if strata=='all' else manifest['strata'][strata]
          for level in sorted(levels):
           left=lookup.get((drug,threshold,seed,'decision_diagnostic',budget,strata,level)); right=lookup.get((drug,threshold,seed,'random_gene',budget,strata,level))
           if not left or not right or left['n']<10 or right['n']<10:
               omitted.append({'drug':drug,'threshold':threshold,'seed':seed,'budget':budget,'stratum':strata,'level':level,'n':left['n'] if left else 0,'reason':'fewer_than_10_evaluation_records'});continue
           contrasts.append({'drug':drug,'threshold':threshold,'seed':seed,'budget':budget,'stratum':strata,'level':level,
             'n':left['n'],'R':left['R'],'brier_delta':left['brier']-right['brier'],
             'false_susceptible_delta':left['false_susceptible']-right['false_susceptible'],
             'false_susceptible_rate_delta':(left['false_susceptible_rate']-right['false_susceptible_rate']) if left['false_susceptible_rate'] is not None and right['false_susceptible_rate'] is not None else None})
    grouped=defaultdict(list)
    for r in contrasts: grouped[(r['drug'],r['threshold'],r['budget'],r['stratum'],r['level'])].append(r)
    summaries=[]
    for key,rs in sorted(grouped.items()):
        vals=[r['brier_delta'] for r in rs]
        summaries.append(dict(zip(('drug','threshold','budget','stratum','level'),key),partition_n=len(rs),
          mean_brier_delta=float(np.mean(vals)),median_brier_delta=float(np.median(vals)),
          lower_loss_partitions=int(sum(v<0 for v in vals)),brier_range=[float(min(vals)),float(max(vals))],
          mean_false_susceptible_delta=float(np.mean([r['false_susceptible_delta'] for r in rs])),
          mean_n=float(np.mean([r['n'] for r in rs])),mean_R=float(np.mean([r['R'] for r in rs]))))
    out={'status':'complete','manifest':manifest,'stratified_rows':rows,'contrasts':contrasts,'summaries':summaries,'omitted_contrasts':omitted,
         'prediction_artifacts':artifacts,'limitations':['Retrospective public labels and old sketch components; this is not strict SNP or blind validation.',
          'Strata are descriptive and no threshold or drug is selected after seeing outcomes.','Small-R strata retain rates and denominators; no clinical safety inference.']}
    p=OUT/'results.json';p.write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
    manifest.update(status='complete',prediction_files=len(artifacts),metric_rows=len(rows),contrast_rows=len(contrasts),
                    summary_rows=len(summaries),result_sha256=sha(p));(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Fixed grid exposes missing strata and avoids long rotated drug labels.
    plot=[r for r in summaries if r['threshold']==.99 and r['budget']==25 and r['stratum'] in ('component_size','standard')]
    if plot:
        fig,axs=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
        limit=max(abs(r['mean_brier_delta']) for r in plot)
        for ax,field in zip(axs,('component_size','standard')):
            levels=manifest['strata'][field];table=np.full((len(DRUGS),len(levels)),np.nan)
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
            for i,drug in enumerate(DRUGS):
                for j,level in enumerate(levels):
                    rs=[r for r in plot if r['drug']==drug and r['stratum']==field and r['level']==level]
                    if rs:table[i,j]=rs[0]['mean_brier_delta']
            im=ax.imshow(table,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
            for i in range(len(DRUGS)):
                for j in range(len(levels)):
                    rs=[r for r in plot if r['drug']==DRUGS[i] and r['stratum']==field and r['level']==levels[j]]
                    text=f"{table[i,j]:+.4f}\nn/R {rs[0]['mean_n']:.0f}/{rs[0]['mean_R']:.1f}\nk={rs[0]['partition_n']}" if rs else 'n<10'
                    ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if np.isfinite(table[i,j]) and abs(table[i,j])>limit*.65 else '#222222')
            ax.set_xticks(range(len(levels)),levels);ax.set_yticks(range(len(DRUGS)),[d.replace('trimethoprim/sulfamethoxazole','TMP-SMX') for d in DRUGS],fontsize=9)
            ax.set_title(field.replace('_',' ').title())
        fig.colorbar(im,ax=axs,shrink=.7,label='Controller minus random Brier; negative = lower loss')
        fig.suptitle('Frozen prediction strata: descriptive effects, not causal assay drift',fontsize=13)
        fig.savefig(OUT/'reliability_debt_strata.png',dpi=220);fig.savefig(OUT/'reliability_debt_strata.pdf');plt.close(fig)
    md=['# Reliability debt audit','',f"Stratified rows: {len(rows):,}; matched contrasts: {len(contrasts):,}; summaries: {len(summaries):,}.",'',
        'Brier difference is decision controller minus random_gene on the same frozen evaluation samples; negative is better. Threshold 0.99 and budget 25 follow the existing descriptive display convention; this is exploratory analysis of previously inspected results. All strata including empty cases are retained. Ambiguous/range dates are unknown. Calibration-in-the-large and fixed ten-bin ECE are descriptive.','',
        '|Drug|Stratum|Level|Partitions|Mean Brier delta|Lower-loss|Mean false-susceptible delta|Mean n/R|','|---|---|---|---:|---:|---:|---:|---|']
    for r in summaries:
        if r['threshold']==.99 and r['budget']==25:
            md.append(f"|{r['drug']}|{r['stratum']}|{r['level']}|{r['partition_n']}|{r['mean_brier_delta']:+.6f}|{r['lower_loss_partitions']}/{r['partition_n']}|{r['mean_false_susceptible_delta']:+.2f}|{r['mean_n']:.1f}/{r['mean_R']:.1f}|")
    md += ['', 'The audit tests a reliability hypothesis; it does not prove clinical safety, patient independence, causality or blind external validity.', '', f"Figure: {OUT/'reliability_debt_strata.png'}", f"Full machine-readable output: {p}"]
    (HERE/'RELIABILITY_DEBT_AUDIT.md').write_text('\n'.join(md),encoding='utf-8')
    (HERE/'reliability_debt_audit.json').write_text(json.dumps({'status':'complete','result_path':str(p),'manifest_path':str(OUT/'manifest.json'),'manifest_sha256':sha(OUT/'manifest.json'),'result_sha256':sha(p),'summary_rows':len(summaries),'contrast_rows':len(contrasts),'version':2},indent=2),encoding='utf-8')
    print(json.dumps({'status':'complete','prediction_files':len(artifacts),'stratified_rows':len(rows),'contrast_rows':len(contrasts),'summary_rows':len(summaries)}),flush=True)


if __name__=='__main__': main()
