"""Validate closed-loop artifacts and report paired, cluster-aware comparisons."""
import json,hashlib,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from counterexample_guided_repair import metrics
from decision_repair_core import update_weights
from run_decision_repair import write_json
from run_ecoli_repair import DRUGS,SEEDS,STRATEGIES,THRESHOLDS,ROOT

HERE=Path(__file__).resolve().parent;OUT=ROOT/'repair_full_scaled5000'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stem(r):return f"{r['drug'].replace('/','-')}_t{r['threshold']}_{r['strategy']}_{r['seed']}"
def main():
    rows=[];skipped=[];manifests=[];cohort_audit=[]
    for d in DRUGS:
        directory=OUT/d.replace('/','-');m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        assert m['status']=='complete',d
        p=Path(m['result_file']);assert sha(p)==m['output_sha256']
        value=json.loads(p.read_text(encoding='utf-8'));rows+=value['results'];skipped+=value['skipped'];manifests.append(m)
        c=directory/'cohort_audit.json';assert sha(c)==m['cohort_audit_sha256'];cohort_audit+=json.loads(c.read_text(encoding='utf-8'))
    assert all(m['input_sha256']==manifests[0]['input_sha256'] and m['code_sha256']==manifests[0]['code_sha256'] for m in manifests)
    keys=[(r['drug'],r['threshold'],r['seed'],r['strategy'],r['budget']) for r in rows]
    assert len(keys)==len(set(keys)), 'Duplicate configuration rows'
    coverage=[]
    for d in DRUGS:
        for t in THRESHOLDS:
            for s in SEEDS:
                group=[r for r in rows if (r['drug'],r['threshold'],r['seed'])==(d,t,s)]
                omissions=[r for r in skipped if (r['drug'],r['threshold'])==(d,t) and r.get('seed',s)==s]
                if omissions:
                    assert not group, (d,t,s,'Skipped configuration also has results')
                    coverage.append({'drug':d,'threshold':t,'seed':s,'status':'skipped','reason':omissions[0]['reason']})
                    continue
                assert group, (d,t,s,'Missing unreported configuration')
                sizes={(r['source_n'],r['query_n'],r['audit_n'],r['evaluation_n']) for r in group}
                assert len(sizes)==1
                query_n=group[0]['query_n'];budgets=[b for b in (0,10,25,50) if b<=query_n]
                assert {(r['strategy'],r['budget']) for r in group}=={(a,b) for a in STRATEGIES for b in budgets}
                assert all(r['total_target_assay_cost']==r['budget']+r['audit_n'] for r in group)
                coverage.append({'drug':d,'threshold':t,'seed':s,'status':'complete','budgets':budgets,'metric_rows':len(group)})
    for n,h in manifests[0]['input_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in manifests[0]['code_sha256'].items():assert sha(HERE/n)==h,n
    reference=defaultdict(set)
    with (ROOT/'ecoli_human_sr_pairs.tsv').open(encoding='utf-8',newline='') as f:
        for record in csv.DictReader(f,delimiter='\t'):
            assert record['label'] in ('Susceptible','Resistant')
            reference[(record['sequence_id'],record['antibiotic'])].add(int(record['label']=='Resistant'))
    def true_labels(samples,drug):
        values=[reference[(str(sample),drug)] for sample in samples]
        assert all(len(v)==1 for v in values)
        return np.array([next(iter(v)) for v in values])
    checked=set();maximum_error=0.;index={};prediction_files=[]
    for r in rows:
        p=OUT/r['drug'].replace('/','-')/(stem(r)+'.npz');a=np.load(p,allow_pickle=False)
        y=a['labels'];prob=a[f"p_b{r['budget']}"];assert len(y)==len(prob)==r['evaluation_n'] and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        for k,v in metrics(y,prob).items():
            if r[k] is not None and np.isfinite(v):maximum_error=max(maximum_error,abs(float(r[k])-v))
        index[(r['drug'],r['threshold'],r['seed'],r['budget'],r['strategy'])]=(r,p)
        if p in checked:continue
        checked.add(p);prediction_files.append({'path':str(p),'sha256':sha(p)})
        groups=[set(a[n].astype(str)) for n in ('source_clusters','query_pool_clusters','audit_clusters','evaluation_clusters')]
        assert all(not(groups[i]&groups[j]) for i in range(4) for j in range(i))
        samples=[set(a[n].astype(str)) for n in ('source_samples','query_pool_samples','audit_samples','evaluation_samples')]
        assert all(not(samples[i]&samples[j]) for i in range(4) for j in range(i))
        trace=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));queried=[];w=np.ones(5)/5
        assert trace['code_hashes']==manifests[0]['code_sha256'] and trace['input_hashes']==manifests[0]['input_sha256']
        assert trace['results']==[x for x in rows if stem(x)==stem(r)]
        assert all(len(a[n])==r[k] for n,k in [('source_samples','source_n'),('query_pool_samples','query_n'),('audit_samples','audit_n'),('evaluation_samples','evaluation_n')])
        assert np.array_equal(a['labels'],true_labels(a['evaluation_samples'],r['drug']))
        assert np.array_equal(a['audit_labels'],true_labels(a['audit_samples'],r['drug']))
        for name,count in [('source_samples','source_R'),('query_pool_samples','query_pool_R'),('audit_samples','audit_R'),('evaluation_samples','evaluation_R')]:
            assert int(true_labels(a[name],r['drug']).sum())==r[count]
        for event in trace['events']:
            assert event['query_start']==len(queried)
            ix=event['samples'];assert len(ix)==event['query_n'] and len(set(ix))==len(ix)
            assert set(ix)<=samples[1] and not(set(ix)&set(queried))
            predictions=np.array(event['pre_reveal_predictions']);labels=np.array(event['revealed_labels'])
            assert predictions.shape==(5,len(ix)) and set(labels)<={0,1}
            assert np.array_equal(labels,true_labels(ix,r['drug']))
            w=update_weights(w,predictions,labels);assert np.allclose(w,event['post_reveal_evidence_weights'],atol=1e-12)
            queried+=ix
        assert len(queried)==max(x['budget'] for x in trace['results'])
        for x in trace['results']:assert x['queried_samples']==queried[:x['budget']]
    assert maximum_error<1e-10,maximum_error
    contrasts=[]
    for (d,t,s,b,strategy),(r,p) in index.items():
        if strategy!='decision_diagnostic' or b not in (25,50):continue
        a=np.load(p,allow_pickle=False)
        for control in ('random_gene','uncertainty_gene','random_tree'):
            if (d,t,s,b,control) not in index:continue
            q=np.load(index[(d,t,s,b,control)][1],allow_pickle=False)
            assert np.array_equal(a['evaluation_samples'],q['evaluation_samples']) and np.array_equal(a['labels'],q['labels'])
            delta=(a[f'p_b{b}']-a['labels'])**2-(q[f'p_b{b}']-q['labels'])**2
            clusters=a['evaluation_clusters'].astype(str);unique=np.unique(clusters)
            sums=np.array([delta[clusters==c].sum() for c in unique]);counts=np.array([(clusters==c).sum() for c in unique]);ci=None
            if len(unique)>=2:
                selected=np.random.default_rng(20261009).integers(0,len(unique),(1000,len(unique)))
                draws=sums[selected].sum(axis=1)/counts[selected].sum(axis=1);ci=np.quantile(draws,[.025,.975]).tolist()
            contrasts.append({'drug':d,'threshold':t,'seed':s,'budget':b,'left':strategy,'right':control,'n':len(delta),'R':int(a['labels'].sum()),'clusters':len(unique),'paired_brier_delta':float(delta.mean()),'ci95':ci})
    summary=[]
    for d in DRUGS:
        for t in (.99,.95):
            for b in (0,10,25,50):
                for strategy in STRATEGIES:
                    group=[r for r in rows if (r['drug'],r['threshold'],r['budget'],r['strategy'])==(d,t,b,strategy)]
                    if not group:continue
                    summary.append({'drug':d,'threshold':t,'budget':b,'strategy':strategy,'seeds':len(group),'n_min':min(r['evaluation_n'] for r in group),'n_max':max(r['evaluation_n'] for r in group),'R_min':min(r['evaluation_R'] for r in group),'R_max':max(r['evaluation_R'] for r in group),
                        'brier_seed_mean':float(np.mean([r['brier'] for r in group])),'brier_seed_range':[min(r['brier'] for r in group),max(r['brier'] for r in group)]})
    write_json(HERE/'ecoli_repair_full_results.json',{'results':rows,'skipped':skipped,'cohort_audit':cohort_audit,'limitations':value['limitations']})
    assert len(checked)==sum(m['completed_trajectories'] for m in manifests)
    validation={'status':'passed','prediction_files':len(checked),'metric_rows':len(rows),'max_metric_error':maximum_error,'cluster_and_sample_isolation':True,'query_weight_replay':True,'raw_AST_labels_match':True,'configuration_coverage':coverage,'cost_accounting_checked':True,'input_hashes':manifests[0]['input_sha256'],'code_hashes':manifests[0]['code_sha256'],'predictions':prediction_files}
    write_json(HERE/'ecoli_full_validation.json',validation)
    write_json(HERE/'ecoli_full_analysis.json',{'status':'complete','summary':summary,'paired_contrasts':contrasts,'limitations':['Bootstrap intervals condition on a fitted model and one fixed partition, and resample whole sketch/BioSample components.','Seeds change partitions but share the same finite cohort; they are not independent clinical studies.','Intervals are exploratory, not multiplicity-adjusted, and do not include model-development uncertainty.','The second-species features are sequence sketches, not AMR genes or cross-species transfer.']})
    with (HERE/'ecoli_full_summary.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),delimiter='\t');w.writeheader();w.writerows(summary)
    text=['# E. coli 全序列队列闭环复现','',f"有效序列 {manifests[0]['sequence_n']:,}；完成 {len(checked)} 条轨迹、{len(rows):,} 条预算指标；跳过 {len(skipped)} 个不足配置。已逐条复算指标、核对四分区样本/簇隔离、重放查询权重更新。",'',
        '主要展示固定 Jaccard 0.99＋相同 BioSample＋相同规范化序列的连通分量划分，预算25。0.95为敏感性分析。源、查询、独立审计、评估四种角色按簇哈希分配。下面是五个划分种子的描述均值，同一队列的重复划分不形成五份独立研究。','',
        '|药物|评估 n/R 范围|闭环 Brier|随机重训练|不确定性重训练|ExtraTrees|','|---|---|---:|---:|---:|---:|']
    for d in DRUGS:
        group={r['strategy']:r for r in summary if r['drug']==d and r['threshold']==.99 and r['budget']==25}
        if not group:continue
        first=next(iter(group.values()));values=[f"{group[s]['brier_seed_mean']:.4f}" if s in group else '缺失' for s in STRATEGIES]
        text.append(f"|{d}|{first['n_min']}–{first['n_max']} / {first['R_min']}–{first['R_max']}|"+'|'.join(values)+'|')
    text+=['','Brier越低越好。簇 bootstrap 对比在 ecoli_full_analysis.json 中逐药物、种子和预算保留；不将药物或重复种子当独立患者。某些药物耐药病例稀少，须结合分母判断。', '', '## 队列与标签排除分母','', '|药物|有效标签序列|耐药序列|独立划分组件|排除冲突序列组|','|---|---:|---:|---:|---:|']
    for c in cohort_audit:
        if c['threshold']==.99:text.append(f"|{c['drug']}|{c['labelled_sequences']}|{c['resistant']}|{c['components']}|{sum(c['rejected_sequence_groups'].values())}|")
    text+=['', '组件包括sketch连通关系、共享BioSample和相同规范化contig。各来源/查询/审计/评估分区的n/R保留在原始指标，输入组数、按固定顺序排除的冲突原因保留在cohort_audit。', '', '该队列是公开人源药敏数据的回顾性算法复现；没有统一 MIC 断点、临床结局或盲态新数据。序列 sketch 表征改变了输入，不能称肺炎克雷伯菌模型跨物种迁移。近同序列连通分量也不等于严格 FastANI/SNP 聚类。']
    (HERE/'ECOLI_FULL_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    write_json(OUT/'manifest.json',{'status':'complete','designation':'all nine drug shards completed and validated','sequence_n':manifests[0]['sequence_n'],'metric_rows':len(rows),'prediction_files':len(checked),'results':str(HERE/'ecoli_repair_full_results.json'),'results_sha256':sha(HERE/'ecoli_repair_full_results.json'),'validation':str(HERE/'ecoli_full_validation.json'),'validation_sha256':sha(HERE/'ecoli_full_validation.json'),'shards':manifests})
    from analyze_ecoli_quality_sensitivity import main as quality_sensitivity
    quality_sensitivity()
    print(json.dumps({'status':'complete','trajectories':len(checked),'rows':len(rows),'max_error':maximum_error}),flush=True)
if __name__=='__main__':main()
