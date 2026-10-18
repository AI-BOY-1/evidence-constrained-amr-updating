"""Verify that sequence portraits show recorded bases, HSPs and AST rows."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv,hashlib,json
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=HERE/'advanced_figures/sequence_evidence_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    s=read(OUT/'source.json')
    for p,h in s['inputs'].items():assert sha(p)==h
    for ext,h in s['exports'].items():assert sha(OUT/('sequence_evidence_portraits.'+ext))==h
    assert s['code_sha256']==sha(HERE/'make_sequence_evidence_portraits.py')
    refs={r.id:str(r.seq).upper() for r in SeqIO.parse(ncs_paths.data_path('reference/resfinder_db/genomicepidemiology-resfinder_db-eecf0aa20759/all.fsa'),'fasta')}
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:pairs=list(csv.DictReader(f,delimiter='\t'))
    by={}
    for r in pairs:by.setdefault(r['sequence_id'],[]).append(r)
    with (ROOT/'ecoli_lab_amr.tsv').open(encoding='utf-8') as f:lab=list(csv.DictReader(f,delimiter='\t'))
    checked=0
    for group in s['groups']:
        actual=[]
        for n in sorted(by):
            if not any(r['antibiotic']==group['drug'] for r in by[n]):continue
            p=ROOT/'acquired_amr_v1'/(n+'.json')
            if p.exists() and any(h['reference'].startswith(group['family']) for h in read(p)['qualifying_hits']):actual.append(n)
            if len(actual)==2:break
        assert actual==[c['sequence_id'] for c in group['cases']]
        for case in group['cases']:
            n=case['sequence_id'];h=case['best_hit'];a=read(ROOT/'acquired_amr_v1'/(n+'.json'))
            expected=sorted([h for h in a['qualifying_hits'] if h['reference'].startswith(group['family'])],
                            key=lambda h:(-h['bitscore'],h['reference'],h['contig'],h['qstart']))[0]
            assert h==expected
            seqs=SeqIO.to_dict(SeqIO.parse(ROOT/'fasta'/(n+'.fna'),'fasta'))
            seq=str(seqs[h['contig']].seq).upper();lo,hi=sorted((h['qstart'],h['qend']))
            fragment=seq[lo-1:hi]
            reverse=(h['qend']-h['qstart'])*(h['send']-h['sstart'])<0
            if reverse:fragment=str(Seq(fragment).reverse_complement())
            assert fragment==case['fragment'] and reverse==case['reverse_complemented']
            assert case['contig_window']==seq[case['flank_start']-1:case['flank_end']]
            assert case['contig_length']==len(seq)
            for locus in case['local_candidates']:
                assert any(l['contig']==h['contig'] and l['start']==locus['start'] and l['end']==locus['end']
                           and locus['reference'] in l['references'] and len(l['references'])==locus['allele_candidates'] for l in a['loci'])
            strings=case['reference_alignment']['strings']
            ref=refs[group['reference_id']]
            assert strings[0].replace('-','')==ref and strings[1].replace('-','')==fragment
            refpos=[i for i,c in enumerate(strings[0]) if c!='-'];start=case['reference_zoom_start']-1
            assert case['zoom_reference']==ref[start:start+64]
            assert case['zoom_fragment']==''.join(strings[1][refpos[i]] for i in range(start,start+64))
            matching=[r for r in pairs if r['sequence_id']==n and r['antibiotic']==group['drug']]
            assert matching==case['pair_records']
            ids={r['genome_id'] for r in matching}
            assert case['raw_AST_records']==[r for r in lab if r['genome_id'] in ids and r['antibiotic']==group['drug']]
            checked+=1
        alignment=group['pair_alignment'];p=alignment['strings'];assert len(p[0])==len(p[1])
        for i,c in enumerate(group['cases']):assert p[i].replace('-','')==c['fragment']
        coordinates=alignment['coordinates']
        assert coordinates[0][0]==coordinates[1][0]==0
        assert coordinates[0][-1]==len(group['cases'][0]['fragment'])
        assert coordinates[1][-1]==len(group['cases'][1]['fragment'])
    with Image.open(OUT/'sequence_evidence_portraits.png') as im:
        assert im.size==(6000,4400) and all(a<b for a,b in im.convert('RGB').getextrema())
    validation={'status':'source_checks_passed','genomes_checked':checked,
                'source_sha256':sha(OUT/'source.json'),'figure_sha256':s['exports']['png'],
                'validator_sha256':sha(__file__),
                'checks':{'deterministic_selection':True,'input_hashes':True,'actual_FASTA_bases':True,
                          'HSP_coordinates':True,'raw_AST_records':True,'reference_zoom':True,'export_hashes':True},
                'visual_review':{'status':'pending'},
                'limitations':['Source checks are not pixel reconstruction or biological functionality validation.']}
    old=read(OUT/'visual_review.json') if (OUT/'visual_review.json').exists() else {}
    if old.get('status')=='passed' and old.get('figure_sha256')==validation['figure_sha256']:
        validation['status']='passed';validation['visual_review']=old
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps({'status':validation['status'],'genomes_checked':checked,'figure_sha256':validation['figure_sha256']}))


if __name__=='__main__':main()
