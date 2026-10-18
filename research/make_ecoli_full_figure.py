"""Sequence-space views and actual held-out repair effects, no synthetic images."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import TruncatedSVD
from ecoli_clusters import ROOT

HERE=Path(__file__).resolve().parent;OUT=HERE/'advanced_figures';OUT.mkdir(exist_ok=True)
def main():
    matrix=ROOT/'ecoli_assemblies_jaccard.csv';matrix_hash=hashlib.sha256(matrix.read_bytes()).hexdigest()
    cache=ROOT/'ecoli_jaccard_profile_embedding.npz'
    with matrix.open(encoding='utf-8') as f:rows=list(csv.reader(f))
    names=np.array([Path(p).stem for p in rows[0]]);J=np.asarray(rows[1:],np.float32)
    if cache.exists() and str(np.load(cache)['matrix_sha256'])==matrix_hash:
        coords=np.load(cache)['coordinates']
    else:
        J-=J.mean(axis=0,keepdims=True)
        coords=TruncatedSVD(n_components=2,random_state=20261009,n_iter=7).fit_transform(J)
        np.savez_compressed(cache,sequence_ids=names,coordinates=coords,matrix_sha256=matrix_hash)
    ids={a:i for i,a in enumerate(names)}
    d='ciprofloxacin';directory=ROOT/'repair_full_scaled5000'/d
    p=directory/f'{d}_t0.99_decision_diagnostic_7.npz';a=np.load(p,allow_pickle=False)
    trace=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    snap=next(r for r in trace['results'] if r['budget']==25)
    eval_ix=np.array([ids[s] for s in a['evaluation_samples'].astype(str)])
    queried=snap['queried_samples'];query_ix=np.array([ids[s] for s in queried])
    ev=a['labels'];delta=(a['p_b0']-ev)**2-(a['p_b25']-ev)**2
    integrity=json.loads((ROOT/'ecoli_fasta_integrity.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(2,2,left=.07,right=.96,bottom=.12,top=.89,wspace=.27,hspace=.32)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    colors={'S':'#3d86a6','R':'#d35c51','query':'#d09c16'}
    for ax in axes[:3]:
        ax.scatter(coords[:,0],coords[:,1],s=6,color='#a9afb4',alpha=.17,rasterized=True,linewidths=0)
        ax.set_xlabel('Jaccard-profile component 1');ax.set_ylabel('Jaccard-profile component 2')
    ax=axes[0]
    duplicates={x for group in integrity['duplicate_sequence_groups'] for x in group}
    ix=np.array([ids[x] for x in duplicates if x in ids],dtype=int)
    if len(ix):ax.scatter(coords[ix,0],coords[ix,1],s=20,facecolors='none',edgecolors='#8b5a9b',linewidths=.7,label='Identical normalized contigs')
    ax.set_title(f'A  Genome sequence space: n={len(names):,}',loc='left',fontweight='bold')
    if len(ix):ax.legend(frameon=False,fontsize=8,loc='best')
    ax=axes[1]
    for label,color,name in [(0,colors['S'],'Held-out susceptible'),(1,colors['R'],'Held-out resistant')]:
        ix=eval_ix[ev==label];ax.scatter(coords[ix,0],coords[ix,1],s=17,color=color,alpha=.7,label=f'{name} (n={len(ix)})',linewidths=0,rasterized=True)
    ax.scatter(coords[query_ix,0],coords[query_ix,1],s=48,facecolors='none',edgecolors=colors['query'],marker='s',linewidths=1,label='25 acquired AST labels')
    ax.set_title('B  Where the controller requests a measurement',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='best')
    ax=axes[2];scale=max(float(np.max(np.abs(delta))),.01)
    image=ax.scatter(coords[eval_ix,0],coords[eval_ix,1],s=25,c=delta,cmap='RdBu',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),alpha=.85,linewidths=0,rasterized=True)
    ax.set_title('C  Actual held-out loss change after 25 queries',loc='left',fontweight='bold')
    cb=fig.colorbar(image,ax=ax,fraction=.045,pad=.02);cb.set_label('Squared loss at b0 minus b25\nPositive = improvement',fontsize=9)
    ax=axes[3]
    analysis=json.loads((HERE/'ecoli_full_analysis.json').read_text(encoding='utf-8'))
    contrasts=[r for r in analysis['paired_contrasts'] if r['threshold']==.99 and r['seed']==7 and r['budget']==25]
    drugs=sorted({r['drug'] for r in contrasts});controls=['random_gene','uncertainty_gene','random_tree']
    palette=['#538bb0','#b38a38','#855c91'];offsets=[-.19,0,.19]
    for k,(control,color) in enumerate(zip(controls,palette)):
        for i,drug in enumerate(drugs):
            matches=[r for r in contrasts if r['drug']==drug and r['right']==control]
            if not matches:continue
            r=matches[0];y=i+offsets[k];ax.plot(r['paired_brier_delta'],y,'o',color=color,ms=4,label=control.replace('_',' ') if i==0 else None)
            if r['ci95']:ax.plot(r['ci95'],[y,y],color=color,lw=1)
    ax.axvline(0,color='#666',ls='--',lw=1)
    ax.set_yticks(range(len(drugs)),[d.replace('trimethoprim/sulfamethoxazole','TMP–SMX') for d in drugs],fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('Brier: controller minus control (negative = better)')
    ax.set_title('D  Paired effects on a fixed cluster partition',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Auditing and repair across the E. coli genome landscape',fontsize=17,fontweight='bold',y=.975)
    fig.text(.07,.92,'All genomes in A; ciprofloxacin AST outcomes in B–C, seed 7, threshold 0.99, budget 25.',fontsize=10,color='#444')
    fig.text(.07,.035,'Embedding uses only genome similarities; geometry does not prove a biological mechanism. Whole sketch/BioSample/identical-contig components are held apart.\nIntervals resample evaluation components conditional on one fitted model/partition; retrospective, exploratory, no multiplicity adjustment or clinical guarantee.',fontsize=8,color='#555')
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'ecoli_genome_repair_landscape.{ext}',dpi=250)
    plt.close(fig)
    source={'matrix_sha256':matrix_hash,'embedding':'TruncatedSVD of column-centred Jaccard profiles, seed20261009, 2 components; no label input','sequence_ids':names.tolist(),'coordinates':coords.tolist(),'example_drug':d,'example_seed':7,'example_threshold':.99,'example_budget':25,'evaluation_samples':a['evaluation_samples'].tolist(),'evaluation_labels':ev.tolist(),'heldout_squared_loss_improvement':delta.tolist(),'queried_samples':queried,'contrasts':contrasts,'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (OUT/'ecoli_genome_repair_landscape_source.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    from validate_ecoli_full_figure import main as validate_figure
    validate_figure()
    print(json.dumps({'status':'complete','sequence_n':len(names),'example_evaluation_n':len(ev),'example_R':int(ev.sum()),'figure':str(OUT/'ecoli_genome_repair_landscape.png')}),flush=True)
if __name__=='__main__':main()
