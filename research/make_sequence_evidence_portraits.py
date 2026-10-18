"""Actual contig coordinates, sequence alignments and raw AST records."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from pathlib import Path

from Bio import SeqIO, Align
from Bio.Seq import Seq
import Bio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Patch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=HERE/'advanced_figures/sequence_evidence_v1'
REF=ncs_paths.data_path('reference/resfinder_db/genomicepidemiology-resfinder_db-eecf0aa20759/all.fsa')
FAMILIES=(('blaNDM','meropenem'),('blaCTX-M','ceftazidime'),('qnrS','ciprofloxacin'))
BASES='ACGTN-'
COLORS=('#368A7E','#4E83BE','#D1AA37','#C96864','#9B9B9B','#FFFFFF')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def align_strings(a,b):
    engine=Align.PairwiseAligner(mode='global',match_score=2,mismatch_score=-2,
                              open_gap_score=-5,extend_gap_score=-.5)
    al=engine.align(a,b)[0]
    target,query=al.indices
    strings=[''.join('-' if i<0 else seq[i] for i in indices)
             for seq,indices in ((a,target),(b,query))]
    return strings,al.coordinates.tolist(),float(al.score)


def main():
    OUT.mkdir(exist_ok=True)
    vp=HERE/'ecoli_acquired_amr_validation.json';v=read(vp);manifest=read(ROOT/'acquired_amr_v1/manifest.json')
    assert v['status']=='passed' and v['manifest_sha256']==sha(ROOT/'acquired_amr_v1/manifest.json')
    assert manifest['hashes']['reference_sha256']==sha(REF)
    by={}
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):by.setdefault(r['sequence_id'],[]).append(r)
    reference={r.id:str(r.seq).upper() for r in SeqIO.parse(REF,'fasta')}
    selected=[];binding={str(vp):sha(vp),str(REF):sha(REF),
        str(ROOT/'ecoli_human_sr_pairs.tsv'):sha(ROOT/'ecoli_human_sr_pairs.tsv'),
        str(ROOT/'ecoli_lab_amr.tsv'):sha(ROOT/'ecoli_lab_amr.tsv')}
    for family,drug in FAMILIES:
        cases=[]
        for n in sorted(by):
            pairs=[r for r in by[n] if r['antibiotic']==drug]
            p=ROOT/'acquired_amr_v1'/(n+'.json')
            if not pairs or not p.exists():continue
            ann=read(p)
            hits=[h for h in ann['qualifying_hits'] if h['reference'].startswith(family)]
            if not hits:continue
            best=sorted(hits,key=lambda h:(-h['bitscore'],h['reference'],h['contig'],h['qstart']))[0]
            fasta=ROOT/'fasta'/(n+'.fna');raw=ROOT/'acquired_amr_v1'/(n+'.blast.tsv')
            assert sha(fasta)==ann['fasta_sha256'] and sha(raw)==ann['raw_sha256']
            for path in (p,fasta,raw):binding[str(path)]=sha(path)
            contigs=SeqIO.to_dict(SeqIO.parse(fasta,'fasta'));seq=str(contigs[best['contig']].seq).upper()
            lo,hi=sorted((best['qstart'],best['qend']))
            fragment=seq[lo-1:hi]
            reverse=(best['qend']-best['qstart'])*(best['send']-best['sstart'])<0
            if reverse:fragment=str(Seq(fragment).reverse_complement())
            flank_start=max(1,lo-1000);flank_end=min(len(seq),hi+1000)
            local=[]
            for locus in ann['loci']:
                if locus['contig']!=best['contig'] or locus['end']<flank_start or locus['start']>flank_end:continue
                alternatives=[h for h in ann['qualifying_hits'] if h['reference'] in locus['references']
                              and h['contig']==locus['contig'] and min(h['qstart'],h['qend'])>=locus['start']
                              and max(h['qstart'],h['qend'])<=locus['end']]
                representative=sorted(alternatives,key=lambda h:(-h['bitscore'],h['reference']))[0]
                local.append({'start':locus['start'],'end':locus['end'],'reference':representative['reference'],
                              'reverse':(representative['qend']-representative['qstart'])*(representative['send']-representative['sstart'])<0,
                              'allele_candidates':len(locus['references'])})
            cases.append({'sequence_id':n,'pair_records':pairs,'best_hit':best,'contig_length':len(seq),
                'flank_start':flank_start,'flank_end':flank_end,'contig_window':seq[flank_start-1:flank_end],
                'fragment':fragment,'reverse_complemented':reverse,'local_candidates':local})
            if len(cases)==2:break
        assert len(cases)==2
        refname=cases[0]['best_hit']['reference'];refseq=reference[refname]
        pair,coords,score=align_strings(cases[0]['fragment'],cases[1]['fragment'])
        for case in cases:
            aligned,rc,sc=align_strings(refseq,case['fragment'])
            refpos=[i for i,c in enumerate(aligned[0]) if c!='-']
            center=len(refseq)//2;start=max(0,center-32);end=min(len(refseq),start+64)
            case['reference_zoom_start']=start+1
            case['zoom_reference']=refseq[start:end]
            case['zoom_fragment']=''.join(aligned[1][refpos[i]] for i in range(start,end))
            case['reference_alignment']={'strings':aligned,'coordinates':rc,'score':sc}
        selected.append({'family':family,'drug':drug,'cases':cases,'reference_id':refname,
                         'pair_alignment':{'strings':pair,'coordinates':coords,'score':score}})
    genomes={r['genome_id'] for group in selected for case in group['cases'] for r in case['pair_records']}
    lab=[]
    with (ROOT/'ecoli_lab_amr.tsv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['genome_id'] in genomes:lab.append(r)
    for group in selected:
        for case in group['cases']:
            ids={r['genome_id'] for r in case['pair_records']}
            case['raw_AST_records']=[r for r in lab if r['genome_id'] in ids and r['antibiotic']==group['drug']]
            assert case['raw_AST_records']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                        'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,11))
    grid=fig.add_gridspec(3,3,left=.05,right=.96,bottom=.13,top=.87,hspace=.66,wspace=.36,
                          height_ratios=[1.35,1,1])
    fig.text(.05,.97,'Inside the actual genome records',fontsize=20,fontweight='bold')
    fig.text(.05,.936,'Contig coordinates, nucleotide alignments and linked laboratory AST | six real E. coli genomes',fontsize=10)
    for j,group in enumerate(selected):
        ax=fig.add_subplot(grid[0,j]);ax.set_title(group['family']+' candidates | '+group['drug'],loc='left',fontsize=10,pad=12)
        lengths=[]
        for i,case in enumerate(group['cases']):
            hit=case['best_hit'];center=(hit['qstart']+hit['qend'])/2
            x0=case['flank_start']-center;x1=case['flank_end']-center;lengths.extend([x0,x1])
            y=1-i
            ax.plot([x0,x1],[y,y],color='#657079',lw=1)
            for k,locus in enumerate(case['local_candidates']):
                lo,hi=locus['start']-center,locus['end']-center
                start=hi if locus['reverse'] else lo;dx=lo-hi if locus['reverse'] else hi-lo
                color='#007F83' if locus['reference'].startswith(group['family']) else '#8995A2'
                ax.add_patch(FancyArrow(start,y-.05,dx,0,width=.13,head_width=.24,
                    head_length=min(90,abs(dx)*.2),length_includes_head=True,color=color))
                ax.text((lo+hi)/2,y+.14,locus['reference'].split('_')[0],ha='center',fontsize=7)
            pair=case['pair_records'][0];raw=case['raw_AST_records']
            labels='/'.join(sorted({r['label'] for r in case['pair_records']}))
            label=('R' if labels=='Resistant' else 'S' if labels=='Susceptible' else labels)
            values=', '.join(sorted({(r['measurement']+' '+r['measurement_unit']).strip() for r in raw})).strip()
            values=values or 'measurement not recorded'
            text=case['sequence_id']+' | AST '+label+' | '+(pair['standards'] or 'standard unknown')
            ax.text(0,y+.37,text,ha='center',fontsize=7,fontweight='bold')
            ax.text(0,y-.25,hit['contig']+' : '+str(min(hit['qstart'],hit['qend']))+'-'+str(max(hit['qstart'],hit['qend'])),ha='center',fontsize=6)
            ax.text(0,y-.43,values+' | '+pair['methods'],ha='center',fontsize=6)
        ax.set(ylim=(-.65,1.65),xlim=(min(lengths)-50,max(lengths)+50),yticks=[],
               xlabel='Bases relative to selected HSP midpoint')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(labelsize=7)
        ax=fig.add_subplot(grid[1,j]);alignment=group['pair_alignment'];coords=np.array(alignment['coordinates'])
        lines=[]
        for k in range(coords.shape[1]-1):
            if np.all(coords[:,k+1]-coords[:,k]>0):lines.append([(coords[0,k],coords[1,k]),(coords[0,k+1],coords[1,k+1])])
        ax.add_collection(LineCollection(lines,colors='#007F83',linewidths=2))
        ax.set(xlim=(0,len(group['cases'][0]['fragment'])),ylim=(0,len(group['cases'][1]['fragment'])),
               xlabel='Genome 1 extracted HSP bases',ylabel='Genome 2 extracted HSP bases')
        ax.set_aspect('equal',adjustable='box');ax.spines[['top','right']].set_visible(False)
        ax.set_title('Actual locus alignment',loc='left',fontsize=9)
        st=alignment['strings'];matches=sum(a==b and a!='-' for a,b in zip(*st));cols=len(st[0])
        ax.text(.02,.96,f'{matches}/{cols} matching columns',transform=ax.transAxes,va='top',fontsize=7)
        ax=fig.add_subplot(grid[2,j]);c0,c1=group['cases'];strings=[c0['zoom_reference'],c0['zoom_fragment'],c1['zoom_fragment']]
        data=np.array([[BASES.index(c) if c in BASES else 4 for c in s] for s in strings])
        # Two 32-base blocks retain the actual letters at readable print size.
        stack=np.concatenate([data[:,:32],data[:,32:64]],axis=0)
        ax.imshow(stack,cmap=ListedColormap(COLORS),vmin=-.5,vmax=5.5,interpolation='nearest',aspect='auto')
        for row in range(6):
            for col in range(32):ax.text(col,row,BASES[stack[row,col]],ha='center',va='center',fontsize=6,color='#FFFFFF' if stack[row,col] in (0,1,3) else '#222222')
        ax.axhline(2.5,color='white',lw=5)
        ax.set(yticks=range(6),yticklabels=['Ref','G1','G2','Ref','G1','G2'],xticks=[],
               title='Fixed central 64 bases | reference orientation')
        ax.tick_params(length=0,labelsize=7);ax.spines[:].set_visible(False)
        ax.text(0,-.17,group['reference_id'],transform=ax.transAxes,fontsize=6)
        ax.text(0,-.28,f'Reference bases {c0["reference_zoom_start"]}-{c0["reference_zoom_start"]+63}',transform=ax.transAxes,fontsize=6)
    fig.legend(handles=[Patch(color=c,label=b) for b,c in zip(BASES[:5],COLORS[:5])],loc='lower center',
               bbox_to_anchor=(.51,.057),ncol=5,frameon=False,fontsize=8)
    fig.text(.05,.028,'Homology candidates are not functional proof. Public AST is retained as recorded; MIC revisions and mechanisms remain unresolved.',fontsize=8)
    fig.text(.05,.010,'Selection: first two lexicographic sequence IDs with each fixed family and drug record. Examples are illustrative, not representative.',fontsize=8)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/('sequence_evidence_portraits.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    source={'status':'complete','selection_rule':'Fixed blaNDM/meropenem, blaCTX-M/ceftazidime, qnrS/ciprofloxacin; first two lexicographic eligible IDs, independent of AST outcome and model performance.',
            'inputs':binding,'groups':selected,'biopython_version':Bio.__version__,
            'alignment_parameters':{'mode':'global','match':2,'mismatch':-2,'open_gap':-5,'extend_gap':-.5},
            'code_sha256':sha(__file__),'exports':{e:sha(OUT/('sequence_evidence_portraits.'+e)) for e in ('png','pdf','svg')}}
    write(OUT/'source.json',source)
    records=[]
    for g in selected:
        for c in g['cases']:
            for r in c['raw_AST_records']:records.append({'sequence_id':c['sequence_id'],'family_example':g['family'],**r})
    with (OUT/'raw_AST_examples.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    text=['# Actual sequence evidence portraits','',
          'Three fixed family/drug examples; first two eligible sequence IDs in lexicographic order. No prediction or AST direction was used to choose examples. They are not representative cases or a new mechanism discovery.','',
          'Top: actual FASTA contig coordinates and qualifying BLAST homolog loci, aligned by selected HSP midpoint. Unannotated lines are assembly sequence, not unknown genes inferred by drawing. Allele ambiguity is retained in source.json.','',
          'Middle: global PairwiseAligner alignment of the actual extracted loci after orienting to the pinned reference. Coordinates are locus-level, not whole-genome ANI/SNP phylogeny.','',
          'Bottom: fixed central 64 reference positions with actual nucleotide letters. Insertions relative to the reference are retained in full source alignments, not drawn in this compact reference-coordinate zoom.','',
          '|Sequence ID|Family|Drug|Recorded AST|Standard|Raw measurement|Contig|HSP start/end|',
          '|---|---|---|---|---|---|---|---|']
    for g in selected:
        for c in g['cases']:
            p=c['pair_records'][0];h=c['best_hit'];rr=c['raw_AST_records']
            text.append(f"|{c['sequence_id']}|{g['family']}|{g['drug']}|{p['label']}|{p['standards'] or 'unknown'}|{'; '.join(r['measurement']+' '+r['measurement_unit'] for r in rr)}|{h['contig']}|{h['qstart']}/{h['qend']}|")
    text+=['','The susceptible record paired with an NDM homolog is a record audit flag, not proof that NDM has no effect. Label provenance, linkage, breakpoint context, expression and functionality require separate investigation.','',
           'These data contain assembly FASTA and laboratory records, not specimen photographs, microscopy, gels or FASTQ pileups. No synthetic biological photos are substituted for missing experiments.']
    (OUT/'README.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'status':'complete','real_genomes':6,'raw_AST_records':len(records),'output':str(OUT)}))


if __name__=='__main__':main()
