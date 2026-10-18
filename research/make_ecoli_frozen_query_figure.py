"""Full-coverage diagnostic figure, no favorable trajectory selection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from run_ecoli_frozen_query_ablation import HERE, OUT, DRUGS, THRESHOLDS, SEEDS, read, write, sha

FIG = HERE / 'advanced_figures' / 'ecoli_query_action_decomposition'
LABELS = ['Meropenem','Imipenem','Ertapenem','Ceftazidime','Ciprofloxacin','TMP-SMX','Amikacin','Gentamicin','Cefepime']


def main():
    analysis_path = HERE / 'ecoli_frozen_query_analysis.json'
    validation_path = HERE / 'ecoli_frozen_query_validation.json'
    validation = read(validation_path)
    assert validation['status'] == 'passed' and validation['results_sha256'] == sha(OUT / 'results.json')
    analysis = read(analysis_path)
    results = read(OUT / 'results.json')
    summaries = {(r['drug'], r['threshold'], r['budget']): r for r in analysis['summaries']}
    assert len(summaries) == 72
    columns = [(t,b) for t in THRESHOLDS for b in (10,25,50)]
    query = np.array([[summaries[(d,t,b)]['query_delta'] for t,b in columns] for d in DRUGS])
    action = np.array([[summaries[(d,t,b)]['action_delta'] for t,b in columns] for d in DRUGS])
    limit = max(np.abs(query).max(), np.abs(action).max(), 1e-8)
    transitions = []
    for d in DRUGS:
        m = summaries[(d,.99,25)]['metrics']
        transitions.append([value for name in ('diagnostic','fixed_path_gene','random_gene')
                            for value in (-m[name]['resistant_fixed'],m[name]['resistant_harmed'])])
    transitions = np.asarray(transitions)
    scatter = [c for c in results['contrasts'] if c['budget'] > 0]
    assert len(scatter) == 270
    # Independently check each plotted mean against all five original contrasts.
    for i,d in enumerate(DRUGS):
        for j,(t,b) in enumerate(columns):
            group = [c for c in scatter if c['drug']==d and c['threshold']==t and c['budget']==b]
            assert len(group) == len(SEEDS)
            assert abs(query[i,j] - np.mean([c['query_delta'] for c in group])) < 1e-12
            assert abs(action[i,j] - np.mean([c['action_delta'] for c in group])) < 1e-12
        group = [c for c in scatter if c['drug']==d and c['threshold']==.99 and c['budget']==25]
        expected = [value for name in ('diagnostic','fixed_path_gene','random_gene')
                    for value in (-np.mean([c['metrics'][name]['resistant_fixed'] for c in group]),
                                  np.mean([c['metrics'][name]['resistant_harmed'] for c in group]))]
        assert np.allclose(transitions[i],expected,atol=1e-12,rtol=0)
    source = dict(query_heatmap=query.tolist(), action_heatmap=action.tolist(),
                  heatmap_columns=columns, drugs=DRUGS, resistant_transitions=transitions.tolist(),
                  scatter=[{k:c[k] for k in ('drug','threshold','seed','budget','query_delta','action_delta','total_delta')} for c in scatter],
                  analysis_sha256=sha(analysis_path), results_sha256=sha(OUT / 'results.json'))
    source_path = FIG.with_name(FIG.name + '_source.json')
    write(source_path,source)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig = plt.figure(figsize=(19,14),facecolor='white')
    gs = fig.add_gridspec(2,2,left=.085,right=.95,top=.87,bottom=.145,hspace=.38,wspace=.36)
    axes = [fig.add_subplot(gs[i,j]) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    for ax,data,title in zip(axes[:2],(query,action),('A  Query-path contribution: G minus R','B  Action contribution: D minus G')):
        im = ax.imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_title(title,loc='left',fontweight='bold',pad=16)
        ax.set_yticks(range(9),LABELS)
        ax.set_xticks(range(6),[f'{t:.2f}\nb={b}' for t,b in columns])
        ax.set_xlabel('Component threshold / queried-label budget',labelpad=8)
        ax.axvline(2.5,color='white',lw=3)
        for i in range(9):
            for j in range(6):
                ax.text(j,i,f'{data[i,j]:+.3f}',ha='center',va='center',fontsize=10,
                        color='white' if abs(data[i,j])>.6*limit else '#17212b')
        cb = fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
        cb.set_label('Brier difference (negative = lower loss)',fontsize=10)
    ax=axes[2]
    colors=plt.get_cmap('tab10').colors[:9]
    markers={10:'o',25:'s',50:'^'}
    for i,d in enumerate(DRUGS):
        for t in THRESHOLDS:
            for b in markers:
                group=[c for c in scatter if c['drug']==d and c['threshold']==t and c['budget']==b]
                ax.scatter([c['query_delta'] for c in group],[c['action_delta'] for c in group],
                           s=35,marker=markers[b],edgecolors=[colors[i]],
                           facecolors=[colors[i]] if t==.99 else 'none',alpha=.7,lw=.9)
    extent=max(abs(c[k]) for c in scatter for k in ('query_delta','action_delta'))*1.12
    ax.plot([-extent,extent],[extent,-extent],ls='--',color='#777777',lw=1,label='Total change = 0')
    ax.axhline(0,color='#b0b0b0',lw=.8);ax.axvline(0,color='#b0b0b0',lw=.8)
    ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent)
    ax.set_title('C  All 270 nonzero-budget configuration contrasts',loc='left',fontweight='bold',pad=16)
    ax.set_xlabel('Query-path Brier change (G minus R)')
    ax.set_ylabel('Action Brier change (D minus G)')
    ax.text(.025,.97,'Below dashed line: lower total loss\nFilled: 0.99; hollow: 0.95\no: b=10; square: b=25; triangle: b=50',
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.88))
    handles=[Line2D([0],[0],marker='o',linestyle='',color=colors[i],label=LABELS[i],markersize=6) for i in range(9)]
    ax.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=9,columnspacing=1)
    ax=axes[3]
    count_limit=max(np.abs(transitions).max(),1)
    im=ax.imshow(transitions,cmap='RdBu_r',vmin=-count_limit,vmax=count_limit,aspect='auto')
    ax.set_title('D  Resistant error transitions versus budget zero',loc='left',fontweight='bold',pad=16)
    ax.set_yticks(range(9),LABELS)
    ax.set_xticks(range(6),['D\nfixed','D\nnew FS','G\nfixed','G\nnew FS','R\nfixed','R\nnew FS'])
    ax.set_xlabel('Threshold 0.99 / budget 25; mean counts across five splits',labelpad=8)
    for x in (1.5,3.5):ax.axvline(x,color='white',lw=3)
    for i in range(9):
        for j in range(6):
            value=transitions[i,j]
            ax.text(j,i,f'{abs(value):.1f}',ha='center',va='center',fontsize=10,
                    color='white' if abs(value)>.6*count_limit else '#17212b')
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    cb.set_label('Blue: false-sensitive fixed; red: newly false-sensitive',fontsize=10)
    fig.text(.085,.955,'What contributes to AMR prediction repair?',fontsize=23,fontweight='bold')
    fig.text(.085,.918,'Fixed adaptive paths separate querying from action choice; improvements and failures are both retained.',fontsize=13,color='#465568')
    fig.text(.54,.096,'D: original diagnostic action; G: same-path sequence model; R: random-path model.\nFS: resistant sample predicted susceptible at the fixed 0.5 cutoff.\nCounts are split averages, not independent patient events.',fontsize=10,va='top',color='#465568')
    fig.text(.085,.035,'Post-result exploratory ablation | Sketch features, not AMR annotations | Components are not strict ANI/SNP | No clinical-safety claim',fontsize=10,color='#465568')
    exports={}
    for suffix in ('.png','.pdf','.svg'):
        path=FIG.with_suffix(suffix)
        fig.savefig(path,dpi=240,facecolor='white')
        exports[str(path)]=sha(path)
    plt.close(fig)
    with Image.open(FIG.with_suffix('.png')) as im:
        dimensions=list(im.size)
        assert dimensions==[4560,3360]
    write(HERE/'ecoli_frozen_query_figure_validation.json',dict(
        status='passed',source_correspondence=True,drug_n=9,heatmap_cells=108,
        scatter_contrasts=270,resistant_transition_cells=54,
        pixel_dimensions=dimensions,source_sha256=sha(source_path),
        analysis_sha256=sha(analysis_path),result_validation_sha256=sha(validation_path),
        figure_code_sha256=sha(Path(__file__)),export_sha256=exports,visual_review='pending'))
    print('Source checks passed; manual visual review is pending:',FIG.with_suffix('.png'))


if __name__=='__main__':main()
