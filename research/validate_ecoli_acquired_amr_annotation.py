"""Audit frozen acquired homolog reports against FASTA and raw BLAST HSPs."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = ncs_paths.data_path('experiments/ecoli_bvbrc')
ROOT = DATA / 'acquired_amr_v1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    manifest = read(ROOT/'manifest.json')
    assert manifest['status']=='complete' and manifest['completed']==3516
    hashes = manifest['hashes']
    reference = ncs_paths.data_path('reference/resfinder_db/genomicepidemiology-resfinder_db-eecf0aa20759/all.fsa')
    assert sha(reference)==hashes['reference_sha256']
    assert sha(HERE/'run_ecoli_acquired_amr_annotation.py')==hashes['code_sha256']
    assert sha(DATA/'ecoli_fasta_integrity.json')==hashes['integrity_sha256']
    assert sha(ncs_paths.data_path('tools/blast/ncbi-blast-2.17.0+/bin/blastn.exe'))==hashes['blast_sha256']
    ref_lengths={};name=None
    for line in reference.read_text(encoding='utf-8').splitlines():
        if line.startswith('>'):
            name=line[1:].split()[0];ref_lengths[name]=0
        else:
            ref_lengths[name]+=len(line.strip())
    names=sorted(r['sequence_id'] for r in read(DATA/'ecoli_fasta_integrity.json')['inventory'] if r['status']=='valid')
    total_hits=total_loci=with_hits=0
    for n in names:
        report=read(ROOT/(n+'.json'));raw=ROOT/(n+'.blast.tsv')
        assert report['status']=='complete' and report['sequence_id']==n and report['hashes']==hashes
        assert report['fasta_sha256']==sha(DATA/'fasta'/(n+'.fna')) and report['raw_sha256']==sha(raw)
        hits=[]
        for line in raw.read_text(encoding='utf-8').splitlines():
            q,s,identity,length,slen,qs,qe,ss,se,bits,evalue=line.split('\t')
            assert s in ref_lengths and int(slen)==ref_lengths[s]
            coverage=(abs(int(se)-int(ss))+1)/int(slen)
            assert float(evalue)<=1e-10
            if float(identity)>=90 and coverage>=.8:
                hits.append({'contig':q,'reference':s,'identity_pct':float(identity),'reference_coverage':coverage,
                    'aligned_columns':int(length),'reference_length':int(slen),'qstart':int(qs),'qend':int(qe),
                    'sstart':int(ss),'send':int(se),'bitscore':float(bits),'evalue':float(evalue)})
        hits.sort(key=lambda h:(h['contig'],min(h['qstart'],h['qend']),-h['bitscore'],h['reference']))
        assert hits==report['qualifying_hits']
        assert sorted({h['reference'] for h in hits})==report['qualifying_reference_tokens']
        loci=[]
        for h in sorted(hits,key=lambda h:(-h['bitscore'],h['reference'])):
            lo,hi=sorted((h['qstart'],h['qend']))
            candidates=[l for l in loci if l['contig']==h['contig'] and
                max(0,min(hi,l['end'])-max(lo,l['start'])+1)/min(hi-lo+1,l['end']-l['start']+1)>=.8]
            if candidates:
                locus=candidates[0]
            else:
                locus={'contig':h['contig'],'start':lo,'end':hi,'references':[]};loci.append(locus)
            locus['references'].append(h['reference'])
        assert loci==report['loci']
        total_hits+=len(hits);total_loci+=len(loci);with_hits+=bool(hits)
    assert total_hits==manifest['qualifying_hits'] and total_loci==manifest['loci']
    out={'status':'passed','genomes':len(names),'qualifying_hits':total_hits,'descriptive_loci':total_loci,
         'genomes_with_qualifying_hits':with_hits,'manifest_sha256':sha(ROOT/'manifest.json'),
         'validator_sha256':sha(Path(__file__)),'checks':{'reference_and_inputs':True,'raw_HSP_replay':True,
         'allele_ambiguity_retained':True,'locus_replay':True},
         'limitations':['Raw HSP replay does not establish BLAST sensitivity or resistance functionality.',
                        'Acquired homolog candidates only; no mutations, expression or phenotype proof.']}
    (HERE/'ecoli_acquired_amr_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    main()
