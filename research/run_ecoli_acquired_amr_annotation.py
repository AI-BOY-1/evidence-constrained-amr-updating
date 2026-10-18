"""Frozen acquired-gene homology annotation; separate from sketch predictions.

This is BLAST against a pinned ResFinder reference snapshot, not a full
ResFinder/AMRFinder pipeline, and does not identify resistance mutations.
"""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv, hashlib, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT=ROOT/'acquired_amr_v1'; OUT.mkdir(exist_ok=True)
REF=ncs_paths.data_path('reference/resfinder_db/genomicepidemiology-resfinder_db-eecf0aa20759')
BIN=ncs_paths.data_path('tools/blast/ncbi-blast-2.17.0+/bin')
DB=OUT/'reference'
CODE=Path(__file__)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def one(p, hashes):
    report=OUT/(p.stem+'.json');raw=OUT/(p.stem+'.blast.tsv')
    if report.exists():
        x=json.loads(report.read_text(encoding='utf-8'))
        if x.get('status')=='complete' and x.get('hashes')==hashes and x.get('fasta_sha256')==sha(p):return x
        raise RuntimeError('Existing annotation differs from frozen inputs; refusing overwrite: '+str(report))
    cmd=[str(BIN/'blastn.exe'),'-task','blastn','-query',str(p),'-db',str(DB),'-perc_identity','90','-evalue','1e-10','-word_size','11','-dust','no','-max_target_seqs','10000','-max_hsps','1','-num_threads','1','-outfmt','6 qseqid sseqid pident length slen qstart qend sstart send bitscore evalue','-out',str(raw)]
    subprocess.run(cmd,check=True,capture_output=True)
    hits=[]
    with raw.open(encoding='utf-8') as f:
        for line in f:
            q,s,ident,length,slen,qs,qe,ss,se,bit,e=line.rstrip().split('\t')
            span=abs(int(se)-int(ss))+1;coverage=span/int(slen)
            if float(ident)>=90 and coverage>=.8:
                hits.append({'contig':q,'reference':s,'identity_pct':float(ident),'reference_coverage':coverage,'aligned_columns':int(length),'reference_length':int(slen),'qstart':int(qs),'qend':int(qe),'sstart':int(ss),'send':int(se),'bitscore':float(bit),'evalue':float(e)})
    hits.sort(key=lambda x:(x['contig'],min(x['qstart'],x['qend']),-x['bitscore'],x['reference']))
    # Keep all allele ambiguity; a physical locus count is descriptive only.
    loci=[]
    for h in sorted(hits,key=lambda x:(-x['bitscore'],x['reference'])):
        lo,hi=sorted((h['qstart'],h['qend']))
        found=None
        for locus in loci:
            overlap=max(0,min(hi,locus['end'])-max(lo,locus['start'])+1)
            if h['contig']==locus['contig'] and overlap/min(hi-lo+1,locus['end']-locus['start']+1)>=.8:
                found=locus;break
        if found is None:
            found={'contig':h['contig'],'start':lo,'end':hi,'references':[]};loci.append(found)
        found['references'].append(h['reference'])
    x={'status':'complete','sequence_id':p.stem,'fasta_sha256':sha(p),'hashes':hashes,'raw_sha256':sha(raw),'qualifying_hits':hits,'loci':loci,'qualifying_reference_tokens':sorted({h['reference'] for h in hits}),'limitations':['Acquired-gene homology candidates only; no mutation calling, expression, gene functionality or phenotype inference.','All matching alleles retained; ambiguous homologs are not resolved by phenotype.']}
    write(report,x);return x
def main():
    hashes={'code_sha256':sha(CODE),'reference_sha256':sha(REF/'all.fsa'),'integrity_sha256':sha(ROOT/'ecoli_fasta_integrity.json'),'blast_sha256':sha(BIN/'blastn.exe')}
    manifest={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hashes':hashes,'reference_version':(REF/'VERSION').read_text().strip(),'reference_commit':'eecf0aa20759','method':'BLASTn homology: identity>=90%, one-HSP reference span coverage>=80%, evalue<=1e-10; all qualifying alleles retained','mutation_calls':False,'workers':2,'original_prediction_files_modified':False}
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf-8'))['hashes']==hashes, 'Annotation version changed; use new directory'
    write(mp,manifest)
    if not DB.with_suffix('.nhr').exists() and not DB.with_suffix('.ndb').exists():
        subprocess.run([str(BIN/'makeblastdb.exe'),'-in',str(REF/'all.fsa'),'-dbtype','nucl','-out',str(DB)],check=True,capture_output=True)
    inv=json.loads((ROOT/'ecoli_fasta_integrity.json').read_text(encoding='utf-8'))
    paths=sorted(ROOT/'fasta'/f"{r['sequence_id']}.fna" for r in inv['inventory'] if r['status']=='valid')
    completed=0;tot_hits=0;tot_loci=0
    with ThreadPoolExecutor(2) as ex:
        for f in as_completed([ex.submit(one,p,hashes) for p in paths]):
            x=f.result();completed+=1;tot_hits+=len(x['qualifying_hits']);tot_loci+=len(x['loci'])
            if completed%10==0:
                manifest.update(completed=completed,total=len(paths),qualifying_hits=tot_hits,loci=tot_loci);write(mp,manifest);print(json.dumps({'done':completed,'total':len(paths),'loci':tot_loci}),flush=True)
    manifest.update(status='complete',completed=completed,total=len(paths),qualifying_hits=tot_hits,loci=tot_loci,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(mp,manifest)
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
