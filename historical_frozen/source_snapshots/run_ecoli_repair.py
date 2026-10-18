"""Second-species closed-loop replication using sequence-sketch features.

Same trajectory/controller code as K. pneumoniae, different sequence tokens.
Retrospective cluster-held-out study, not cross-species model transfer or blind
confirmation. Explicitly does not assert AMR gene/mechanism equivalence.
"""
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
import sourmash
from counterexample_guided_repair import Record,metrics,parse_year
from run_decision_repair import trajectory,mask,write_json,selective_metrics
from ecoli_clusters import cohort,ROOT

HERE=Path(__file__).resolve().parent
OUT=ROOT/'repair_scaled5000';OUT.mkdir(exist_ok=True)
DRUGS=('meropenem','imipenem','ertapenem','ceftazidime','ciprofloxacin','trimethoprim/sulfamethoxazole','amikacin','gentamicin','cefepime')
STRATEGIES=('decision_diagnostic','random_gene','uncertainty_gene','random_tree')
SEEDS=(7,17,27,37,47)
THRESHOLDS=(.99,.95)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def measured_metrics(y,p):
    values=metrics(y,p)
    if not np.any(y==1):values['false_susceptible_rate']=None
    if not np.any(y==0):values['false_resistant_rate']=None
    return values

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--drug',choices=DRUGS)
    args=ap.parse_args()
    drugs=(args.drug,) if args.drug else DRUGS
    requested_budgets=(0,10,25,50) if args.full else (0,10,25)
    if args.full:
        OUT=ROOT/'repair_full_scaled5000'
        if args.drug:OUT=OUT/args.drug.replace('/','-')
        OUT.mkdir(parents=True,exist_ok=True)
    result_file=HERE/('ecoli_repair_full_results'+('_'+args.drug.replace('/','-') if args.drug else '')+'.json') if args.full else HERE/'ecoli_repair_results.json'
    tokens={}
    for sig in sourmash.load_file_as_signatures(str(ROOT/'ecoli_assemblies_k31_scaled1000.sig')):
        tokens[Path(sig.filename or sig.name).stem]=tuple('sketch31::'+str(h) for h in sorted(sig.minhash.downsample(scaled=5000).hashes))
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'designation':'retrospective second-species algorithm replication',
        'features':'sourmash k31 scaled5000 hash presence, NOT annotated AMR genes; clustering uses scaled1000','strategies':STRATEGIES,'seeds':SEEDS,'thresholds':THRESHOLDS,
        'budgets':requested_budgets,'drugs':drugs,'sequence_n':len(tokens),'result_file':str(result_file),'code_sha256':{p.name:sha(p) for p in (Path(__file__),HERE/'ecoli_clusters.py',HERE/'run_decision_repair.py',HERE/'decision_repair_core.py',HERE/'counterexample_guided_repair.py')},
        'input_sha256':{p.name:sha(p) for p in (ROOT/'ecoli_human_sr_pairs.tsv',ROOT/'ecoli_assemblies_jaccard.csv',ROOT/'ecoli_assemblies_k31_scaled1000.sig',ROOT/'ecoli_fasta_integrity.json')}}
    write_json(OUT/'manifest.json',manifest)
    summaries=[];skipped=[];completed_trajectories=0;cohort_audit=[]
    for threshold in THRESHOLDS:
        names,J,clusters,pairs=cohort(threshold)
        by=defaultdict(list)
        for r in pairs:
            a=r.get('sequence_id') or r['assembly_accession']
            if a in tokens:by[(a,r['antibiotic'])].append(r)
        # Reject discordant labels across same BioSample or exact accession;
        # related but different isolates may legitimately have different ASTs.
        bs_labels=defaultdict(set)
        for r in pairs:
            if r['biosample_accession']:bs_labels[(r['biosample_accession'],r['antibiotic'])].add(r['label'])
        exact_group={a:group[0] for group in json.loads((ROOT/'ecoli_fasta_integrity.json').read_text(encoding='utf-8'))['duplicate_sequence_groups'] for a in group}
        exact_labels=defaultdict(set)
        for (a,d),rs in by.items():
            for r in rs:exact_labels[(exact_group.get(a,a),d)].add(r['label'])
        for drug in drugs:
            records=[];rejected={'discordant_same_sequence_id':0,'discordant_exact_sequence_group':0,'discordant_shared_biosample':0}
            for (a,d),rs in sorted(by.items()):
                if d!=drug:continue
                if len({r['label'] for r in rs})!=1:rejected['discordant_same_sequence_id']+=1;continue
                if len(exact_labels[(exact_group.get(a,a),d)])>1:rejected['discordant_exact_sequence_group']+=1;continue
                if any(len(bs_labels[(r['biosample_accession'],d)])>1 for r in rs if r['biosample_accession']):rejected['discordant_shared_biosample']+=1;continue
                years={parse_year(r['collection_date']) for r in rs};year=next(iter(years)) if len(years)==1 else 0
                standards=';'.join(sorted({s for r in rs for s in r['standards'].split(';') if s}))
                records.append(Record(a,clusters[a],year or 0,standards,tokens[a],('__SEQUENCE_SKETCH__',),int(rs[0]['label']=='Resistant')))
            cohort_audit.append({'drug':drug,'threshold':threshold,'labelled_sequences':len(records),'resistant':sum(r.label for r in records),
                'components':len({r.project for r in records}),'rejected_sequence_groups':rejected,'rejection_order':list(rejected),
                'sequence_available_drug_groups':sum(d==drug for a,d in by),'input_drug_pairs':sum(r['antibiotic']==drug for r in pairs)})
            if len(records)<50:
                skipped.append({'drug':drug,'threshold':threshold,'reason':'fewer_than_50_labelled_assemblies','n':len(records)});continue
            for seed in SEEDS:
                def role(c):return int(hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest()[:8],16)%10
                source=[r for r in records if role(r.project)<4]
                pool=[r for r in records if 4<=role(r.project)<7]
                audit=[r for r in records if role(r.project)==7]
                evaluation=[r for r in records if role(r.project)>=8]
                roles=[source,pool,audit,evaluation]
                sets=[{r.project for r in x} for x in roles]
                assert all(not(sets[i]&sets[j]) for i in range(4) for j in range(i))
                if len({r.label for r in source})<2 or len(pool)<10 or len(evaluation)<10:
                    skipped.append({'drug':drug,'threshold':threshold,'seed':seed,'reason':'insufficient_source_classes_or_query_eval','sizes':list(map(len,roles))});continue
                budgets=[b for b in requested_budgets if b<=len(pool)]
                oracle={r.sample:r for r in pool}
                for strategy in STRATEGIES:
                    stem=f'{drug.replace("/","-")}_t{threshold}_{strategy}_{seed}'
                    result_path=OUT/(stem+'.json')
                    if result_path.exists():
                        previous=json.loads(result_path.read_text(encoding='utf-8'))
                        if previous.get('code_hashes')==manifest['code_sha256'] and previous.get('input_hashes')==manifest['input_sha256']:
                            summaries.extend(previous['results']);completed_trajectories+=1;continue
                    queried=[]
                    def reveal(a):
                        assert a in oracle and a not in queried
                        queried.append(a);return oracle[a]
                    snapshots,events=trajectory(source,mask(pool),mask(evaluation),mask(audit),reveal,strategy,seed,budgets)
                    arrays={'labels':np.array([r.label for r in evaluation]),'audit_labels':np.array([r.label for r in audit]),
                        'evaluation_samples':np.array([r.sample for r in evaluation]),'evaluation_clusters':np.array([r.project for r in evaluation]),
                        'source_samples':np.array([r.sample for r in source]),'query_pool_samples':np.array([r.sample for r in pool]),'audit_samples':np.array([r.sample for r in audit]),
                        'source_clusters':np.array([r.project for r in source]),'query_pool_clusters':np.array([r.project for r in pool]),'audit_clusters':np.array([r.project for r in audit])}
                    results=[]
                    for snap in snapshots:
                        b,p=snap['budget'],snap['evaluation_p'];arrays[f'p_b{b}']=p;arrays[f'audit_p_b{b}']=snap['audit_p']
                        results.append({'drug':drug,'threshold':threshold,'seed':seed,'strategy':strategy,'budget':b,
                            'source_n':len(source),'query_n':len(pool),'audit_n':len(audit),'evaluation_n':len(evaluation),
                            'source_R':sum(r.label for r in source),'query_pool_R':sum(r.label for r in pool),'audit_R':sum(r.label for r in audit),
                            'evaluation_clusters':len(sets[3]),'evaluation_R':int(arrays['labels'].sum()),'total_target_assay_cost':b+len(audit),
                            'action':snap['action'],'evidence_weights':snap['evidence_weights'],'queried_samples':snap['queried_samples'],**measured_metrics(arrays['labels'],p),
                            'risk_coverage':[selective_metrics(arrays['labels'],p,c) for c in (.25,.5,.75,1)]})
                    assert len(queried)==len(set(queried))==max(budgets)
                    np.savez_compressed(OUT/(stem+'.npz'),**arrays)
                    write_json(result_path,{'code_hashes':manifest['code_sha256'],'input_hashes':manifest['input_sha256'],'results':results,'events':events,'cluster_disjoint':True,'labels_masked':True})
                    summaries.extend(results)
                    completed_trajectories+=1
                    manifest.update(completed_trajectories=completed_trajectories,result_rows=len(summaries))
                    write_json(OUT/'manifest.json',manifest)
                print(json.dumps({'threshold':threshold,'drug':drug,'seed':seed,'result_rows':len(summaries),'sizes':list(map(len,roles))}),flush=True)
    write_json(result_file,{'results':summaries,'skipped':skipped,'limitations':['Sequence-available human S/R subset; availability selection bias.','Sketch-connected components, identical-contig hashes and BioSample exclusion are not strict ANI/SNP independence.','Sequence-sketch features differ from K. pneumoniae AMR annotation; this replicates the controller, not a cross-species frozen model.','Country field of Record is explicitly repurposed as source-reported testing-standard context; no country information is available.','No clinical outcomes or prospective/blinded data.']})
    write_json(OUT/'cohort_audit.json',cohort_audit)
    manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),result_rows=len(summaries),completed_trajectories=completed_trajectories,skipped=len(skipped),cohort_audit_sha256=sha(OUT/'cohort_audit.json'),output_sha256=sha(result_file))
    write_json(OUT/'manifest.json',manifest);print(json.dumps({'status':'complete','rows':len(summaries),'skipped':len(skipped)}),flush=True)

if __name__=='__main__':main()
