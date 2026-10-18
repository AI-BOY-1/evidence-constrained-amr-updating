"""New version: same controllers, ANI-screen partitions, homolog/sketch inputs."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np,sourmash
from counterexample_guided_repair import Record,parse_year
from run_decision_repair import trajectory,mask,write_json
from run_ecoli_repair import DRUGS,STRATEGIES,SEEDS,measured_metrics

HERE=Path(__file__).resolve().parent;DATA=ncs_paths.data_path('experiments/ecoli_bvbrc');DEST=DATA/'ani_annotation_repair_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def components(names,screen,threshold,pairs,integrity):
 parent={n:n for n in names}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  x,y=find(a),find(b)
  if x!=y:parent[max(x,y)]=min(x,y)
 for r in screen['pairs']:
  if any(d and d['identity']>=threshold and min(d['query_fraction'],d['reference_fraction'])>=.8 for d in (r.get('a_to_b'),r.get('b_to_a'))):union(r['a'],r['b'])
 for g in integrity['duplicate_sequence_groups']:
  available=[n for n in g if n in parent]
  for n in available[1:]:union(available[0],n)
 bs={}
 for r in pairs:
  n=r['sequence_id'] or r['assembly_accession'];b=r['biosample_accession']
  if n in parent and b:
   if b in bs:union(n,bs[b])
   bs[b]=n
 return {n:find(n) for n in names}
def main():
 DEST.mkdir(exist_ok=True)
 sp=HERE/'ecoli_full_skani_screen.json';screen=read(sp);assert screen['status']=='complete' and screen['queries_completed']==3516
 annroot=DATA/'acquired_amr_v1';annmanifest=read(annroot/'manifest.json');assert annmanifest['status']=='complete' and annmanifest['completed']==3516
 inv=read(DATA/'ecoli_fasta_integrity.json');names=sorted(x['sequence_id'] for x in inv['inventory'] if x['status']=='valid')
 homologs={};annhash={}
 for name in names:
  p=annroot/(name+'.json');x=read(p);assert x['status']=='complete' and x['hashes']==annmanifest['hashes']
  homologs[name]=tuple(sorted({'homolog::'+token.split('_')[0] for token in x['qualifying_reference_tokens']}));annhash[name]=sha(p)
 sketch={}
 for sig in sourmash.load_file_as_signatures(str(DATA/'ecoli_assemblies_k31_scaled1000.sig')):
  sketch[Path(sig.filename or sig.name).stem]=tuple('sketch31::'+str(h) for h in sorted(sig.minhash.downsample(scaled=5000).hashes))
 with (DATA/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8') as f:pairs=list(csv.DictReader(f,delimiter='\t'))
 by=defaultdict(list);bslabels=defaultdict(set);exactlabels=defaultdict(set)
 exact={n:g[0] for g in inv['duplicate_sequence_groups'] for n in g}
 for r in pairs:
  n=r['sequence_id'] or r['assembly_accession'];d=r['antibiotic']
  if n in homologs:by[(n,d)].append(r);exactlabels[(exact.get(n,n),d)].add(r['label'])
  if r['biosample_accession']:bslabels[(r['biosample_accession'],d)].add(r['label'])
 code_names=('run_ecoli_ani_annotation_repair.py','run_decision_repair.py','decision_repair_core.py','counterexample_guided_repair.py','run_ecoli_repair.py')
 binding={'code':{n:sha(HERE/n) for n in code_names},'screen_sha256':sha(sp),'annotation_manifest_sha256':sha(annroot/'manifest.json'),'annotation_sha256':annhash,'pair_sha256':sha(DATA/'ecoli_human_sr_pairs.tsv'),'protocol_sha256':sha(HERE/'SCIENCE_EXTENSION_PROTOCOL.md'),'integrity_sha256':sha(DATA/'ecoli_fasta_integrity.json'),'signature_sha256':sha(DATA/'ecoli_assemblies_k31_scaled1000.sig')}
 mp=DEST/'manifest.json'
 if mp.exists():assert read(mp)['binding']==binding,'Refuse mixed versions'
 bp=DEST/'binding.json'
 if bp.exists():assert read(bp)==binding,'Refuse mixed binding versions'
 else:write_json(bp,binding)
 manifest={'status':'running','binding':binding,'designation':'post-result exploratory ANI-screen feature sensitivity','features':['homolog','sketch'],'thresholds':[.99,.999],'expected_trajectories':720}
 write_json(mp,manifest);results=[];skips=[];audits=[];files=[]
 for th in (.99,.999):
  clusters=components(names,screen,th,pairs,inv)
  for drug in DRUGS:
   eligible=[];rejections=defaultdict(int)
   for (n,d),rs in sorted(by.items()):
    if d!=drug:continue
    if len({r['label'] for r in rs})!=1:rejections['discordant_sequence_id']+=1;continue
    if len(exactlabels[(exact.get(n,n),d)])>1:rejections['discordant_exact']+=1;continue
    if any(len(bslabels[(r['biosample_accession'],d)])>1 for r in rs if r['biosample_accession']):rejections['discordant_biosample']+=1;continue
    eligible.append((n,rs))
   audits.append({'drug':drug,'ANI_threshold':th,'n':len(eligible),'components':len({clusters[n] for n,rs in eligible}),'rejections':dict(rejections)})
   for seed in SEEDS:
    role=lambda c:int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10
    for feature,tokens in [('homolog',homologs),('sketch',sketch)]:
     records=[]
     for n,rs in eligible:
      years={parse_year(r['collection_date']) for r in rs};year=next(iter(years)) if len(years)==1 else 0
      standards=';'.join(sorted({s for r in rs for s in r['standards'].split(';') if s}))
      records.append(Record(n,clusters[n],year or 0,standards,tokens[n],('__HOMOLOGY_CANDIDATES__' if feature=='homolog' else '__SEQUENCE_SKETCH__',),int(rs[0]['label']=='Resistant')))
     groups=[[r for r in records if role(r.project)<4],[r for r in records if 4<=role(r.project)<7],[r for r in records if role(r.project)==7],[r for r in records if role(r.project)>=8]]
     source,pool,audit,evaluation=groups
     if len({r.label for r in source})<2 or len(pool)<10 or len(evaluation)<10 or not any(r.features for r in source):
      skips.append({'drug':drug,'threshold':th,'seed':seed,'feature':feature,'sizes':list(map(len,groups)),'reason':'fixed_ANI_split_insufficient_source_classes_query_eval_or_features'});continue
     sets=[{r.project for r in g} for g in groups];assert all(not(sets[i]&sets[j]) for i in range(4) for j in range(i))
     budgets=[b for b in (0,10,25,50) if b<=len(pool)];oracle={r.sample:r for r in pool}
     for strategy in STRATEGIES:
      stem=f'{drug.replace("/","-")}_ani{th}_{feature}_{strategy}_{seed}';jp=DEST/(stem+'.json');npz=DEST/(stem+'.npz')
      if jp.exists():
       old=read(jp);assert old['binding_sha256']==sha(mp.with_name('binding.json')) and sha(npz)==old['prediction_sha256'];results.extend(old['rows']);files.append({'path':str(npz),'sha256':sha(npz)});continue
      queried=[]
      def reveal(n):assert n in oracle and n not in queried;queried.append(n);return oracle[n]
      snapshots,events=trajectory(source,mask(pool),mask(evaluation),mask(audit),reveal,strategy,seed,budgets)
      arrays={'labels':np.array([r.label for r in evaluation]),'audit_labels':np.array([r.label for r in audit])}
      for label,g in zip(('source','query_pool','audit','evaluation'),groups):arrays[label+'_samples']=np.array([r.sample for r in g]);arrays[label+'_clusters']=np.array([r.project for r in g])
      rows=[]
      for snap in snapshots:
       b=snap['budget'];p=snap['evaluation_p'];arrays[f'p_b{b}']=p;arrays[f'audit_p_b{b}']=snap['audit_p']
       rows.append({'drug':drug,'ANI_threshold':th,'feature':feature,'strategy':strategy,'seed':seed,'budget':b,'source_n':len(source),'query_n':len(pool),'audit_n':len(audit),'evaluation_n':len(evaluation),'total_target_assay_cost':b+len(audit),'action':snap['action'],'evidence_weights':snap['evidence_weights'],'queried_samples':snap['queried_samples'],**measured_metrics(arrays['labels'],p)})
      bp=DEST/'binding.json'
      if not bp.exists():write_json(bp,binding)
      np.savez_compressed(npz,**arrays);write_json(jp,{'binding_sha256':sha(bp),'prediction_sha256':sha(npz),'rows':rows,'events':events})
      results.extend(rows);files.append({'path':str(npz),'sha256':sha(npz)})
    manifest.update(trajectories=len(files),metric_rows=len(results),skipped=len(skips));write_json(mp,manifest)
    print(json.dumps({'drug':drug,'threshold':th,'seed':seed,'trajectories':len(files),'skips':len(skips)}),flush=True)
 write_json(DEST/'results.json',{'status':'complete','results':results,'skipped':skips,'cohort_audits':audits,'predictions':files,'limitations':['Approximate ANI-screen components, not SNP or patient independence.','Post-result exploratory comparison, not blind confirmation.','Homology candidates do not include resistance mutations.']})
 manifest.update(status='complete',result_sha256=sha(DEST/'results.json'));write_json(mp,manifest)
if __name__=='__main__':main()
