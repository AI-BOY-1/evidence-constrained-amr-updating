"""Fixed quality screen of saved held-out predictions, without model retuning."""
import json,hashlib
from pathlib import Path
import numpy as np
from run_ecoli_repair import ROOT,STRATEGIES,DRUGS,measured_metrics
from run_decision_repair import write_json

HERE=Path(__file__).resolve().parent
RULE={'minimum_bases':3000000,'maximum_bases':7000000,'minimum_N50':10000,'maximum_ambiguous_fraction':.01}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    input_path=HERE/'ecoli_repair_full_results.json';integrity_path=ROOT/'ecoli_fasta_integrity.json'
    result=json.loads(input_path.read_text(encoding='utf-8'))
    integrity=json.loads(integrity_path.read_text(encoding='utf-8'))
    inventory={r['sequence_id']:r for r in integrity['inventory'] if r['status']=='valid'}
    eligible={a for a,r in inventory.items() if RULE['minimum_bases']<=r['bases']<=RULE['maximum_bases'] and r['N50']>=RULE['minimum_N50'] and r['ambiguous_fraction']<=RULE['maximum_ambiguous_fraction']}
    rows=[];index={}
    for r in result['results']:
        stem=f"{r['drug'].replace('/','-')}_t{r['threshold']}_{r['strategy']}_{r['seed']}"
        a=np.load(ROOT/'repair_full_scaled5000'/r['drug'].replace('/','-')/(stem+'.npz'),allow_pickle=False)
        selected=np.array([s in eligible for s in a['evaluation_samples'].astype(str)])
        assert len(selected)==r['evaluation_n']
        y=a['labels'][selected];p=a[f"p_b{r['budget']}"][selected]
        v={'drug':r['drug'],'threshold':r['threshold'],'seed':r['seed'],'strategy':r['strategy'],'budget':r['budget'],
            'original_evaluation_n':len(selected),'quality_evaluation_n':len(y),'quality_evaluation_R':int(y.sum()),
            'quality_evaluation_components':len(np.unique(a['evaluation_clusters'][selected])),**(measured_metrics(y,p) if len(y) else {})}
        rows.append(v);index[(r['drug'],r['threshold'],r['seed'],r['strategy'],r['budget'])]=v
    contrasts=[]
    for r in rows:
        if r['strategy']!='decision_diagnostic' or r['budget'] not in (25,50) or not r['quality_evaluation_n']:continue
        for control in ('random_gene','uncertainty_gene','random_tree'):
            c=index.get((r['drug'],r['threshold'],r['seed'],control,r['budget']))
            if c and c['quality_evaluation_n']:
                assert c['quality_evaluation_n']==r['quality_evaluation_n'] and c['quality_evaluation_R']==r['quality_evaluation_R']
                contrasts.append({k:r[k] for k in ('drug','threshold','seed','budget','quality_evaluation_n','quality_evaluation_R')}|{'control':control,'paired_brier_delta':r['brier']-c['brier']})
    report={'status':'complete','rule':RULE,'valid_sequences':len(inventory),'eligible_sequences':len(eligible),'excluded_sequence_ids':sorted(set(inventory)-eligible),
        'rows':rows,'contrasts':contrasts,'input_sha256':{str(p):sha(p) for p in (input_path,integrity_path,HERE/'FULL_ECOLI_ANALYSIS_PROTOCOL.md')},
        'limitations':['Evaluation-only screening of fixed predictions; source/query/audit models retain the original cohort.','Sequence length, N50 and ambiguity do not establish species purity or assembly completeness.','Quality selection changes evaluation composition; this is descriptive sensitivity, not independent confirmation.']}
    write_json(HERE/'ecoli_quality_sensitivity.json',report)
    text=['# E. coli 评估序列质量敏感性','',f"固定条件：3–7 Mb、N50≥10,000 bp、模糊碱基≤1%。{len(inventory)} 份有效FASTA中 {len(eligible)} 份符合该较严格条件。仅筛选已保存的评估预测；没有按结果改阈值、重训或改变查询。方案见 FULL_ECOLI_ANALYSIS_PROTOCOL.md。",'',
        '|药物|筛查后评估n范围|闭环Brier|随机重训|不确定性重训|ExtraTrees|','|---|---:|---:|---:|---:|---:|']
    for drug in DRUGS:
        group=[r for r in rows if r['drug']==drug and r['threshold']==.99 and r['budget']==25 and r['quality_evaluation_n']]
        if not group:continue
        values=[]
        for strategy in STRATEGIES:
            vals=[r['brier'] for r in group if r['strategy']==strategy]
            values.append(f'{np.mean(vals):.4f}' if vals else '缺失')
        text.append(f"|{drug}|{min(r['quality_evaluation_n'] for r in group)}–{max(r['quality_evaluation_n'] for r in group)}|"+'|'.join(values)+'|')
    text+=['','表为同一有限队列五种划分的描述均值；不可当独立临床重复。筛选评估样本无法解决训练集污染或完整性问题，序列选择也可能改变结论。各配置n/R、所有排除ID和固定预测上的成对差异均保留在JSON。']
    (HERE/'ECOLI_QUALITY_SENSITIVITY.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'status':'complete','valid_sequences':len(inventory),'eligible':len(eligible),'metric_rows':len(rows)}))

if __name__=='__main__':main()
