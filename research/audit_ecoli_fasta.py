"""Validate FASTA alphabet, lengths, hashes, duplicates and provenance inventory."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib,json,collections,csv
from pathlib import Path
ROOT=ncs_paths.data_path('experiments/ecoli_bvbrc')
COMPLEMENT=str.maketrans('ACGTRYSWKMBDHVN','TGCAYRSWMKVHDBN')
def main():
    inventory=[];duplicates=collections.defaultdict(list)
    provenance={}
    for p in (ROOT/'bvbrc_sequence_download.jsonl',ROOT/'fasta/download_manifest.jsonl'):
        if not p.exists():continue
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);a=r.get('sequence_id') or r.get('accession')
            if a and (r.get('url') or a not in provenance):provenance[a]=r
    for p in sorted((ROOT/'fasta').glob('*.fna')):
        blob=p.read_bytes();sequences=[];seq=[];headers=[];error=None
        try:
            for line in blob.decode('ascii').splitlines():
                if line.startswith('>'):
                    if seq:sequences.append(''.join(seq));seq=[]
                    headers.append(line[1:].split()[0])
                elif line.strip():
                    assert headers,'sequence before FASTA header';seq.append(line.strip().upper())
            if seq:sequences.append(''.join(seq))
            assert len(sequences)==len(headers)>0 and len(set(headers))==len(headers),'empty/duplicate headers'
            invalid=set(''.join(sequences))-set('ACGTRYSWKMBDHVN');assert not invalid,invalid
            total=sum(map(len,sequences));assert 1000000<=total<=15000000,total
            lengths=sorted(map(len,sequences),reverse=True);cumulative=0;n50=0
            for length in lengths:
                cumulative+=length
                if cumulative>=total/2:n50=length;break
            canonical=hashlib.sha256('\n'.join(sorted(min(s,s.translate(COMPLEMENT)[::-1]) for s in sequences)).encode('ascii')).hexdigest()
            duplicates[canonical].append(p.stem)
            inventory.append({'sequence_id':p.stem,'status':'valid','bases':total,'contigs':len(headers),'N50':n50,
                'ambiguous_fraction':sum(sum(s.count(b) for b in 'RYSWKMBDHVN') for s in sequences)/total,
                'file_sha256':hashlib.sha256(blob).hexdigest(),'canonical_contig_sha256':canonical,
                'provenance':provenance.get(p.stem,{'warning':'provenance not in download logs'})})
        except Exception as e:inventory.append({'sequence_id':p.stem,'status':'invalid','error':repr(e)})
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8',newline='') as f:
        expected={r['sequence_id'] for r in csv.DictReader(f,delimiter='\t')}
    valid={r['sequence_id'] for r in inventory if r['status']=='valid'}
    result={'status':'complete','cohort_status':'complete' if expected<=valid else 'partial','valid_n':len(valid),'invalid_n':len(inventory)-len(valid),'expected_n':len(expected),
        'missing':sorted(expected-valid),'duplicate_sequence_groups':[v for v in duplicates.values() if len(v)>1],
        'inventory':inventory,'canonical_digest_rule':'Sort contigs after choosing lexicographically smaller sequence/reverse complement; preserves contig boundaries.',
        'limitations':['FASTA/length validation does not establish taxonomic purity or clinical isolate independence.','Canonical digest normalizes contig order and orientation; alternative fragmentation still requires relatedness analysis.']}
    (ROOT/'ecoli_fasta_integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'ecoli_valid_fasta_list.txt').write_text('\n'.join(str(ROOT/'fasta'/f'{a}.fna') for a in sorted(valid)),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','duplicate_sequence_groups','missing')}),flush=True)
if __name__=='__main__':main()
