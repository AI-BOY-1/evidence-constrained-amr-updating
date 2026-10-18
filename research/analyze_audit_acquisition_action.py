"""All strata and same-cost comparisons; no independent-patient inference."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ncs_paths.data_path('experiments/ecoli_bvbrc/audit_acquisition_action_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        out.append({**dict(zip(keys,key)),'configurations':len(rs),'eligible':sum(r['eligible'] for r in rs),
                    'accepted':sum(r['accepted'] for r in rs),'audit_R_zero':sum(r['audit_R']==0 for r in rs),
                    'mean_brier_delta_from_frozen':float(np.mean([r['delta'] for r in rs])),
                    'eval_loss_higher':sum(r['delta']>0 for r in rs),
                    'eval_FS_increased':sum(r['FS_delta']>0 for r in rs),
                    'any_new_FS':sum(r['FS_introduced']>0 for r in rs),
                    'lower_loss_more_FS':sum(r['delta']<0 and r['FS_delta']>0 for r in rs),
                    'mean_oracle_regret':float(np.mean([r['regret'] for r in rs])),
                    'audit_R_min':min(r['audit_R'] for r in rs),'audit_R_max':max(r['audit_R'] for r in rs),
                    'eval_R_min':min(r['evaluation_R'] for r in rs),'eval_R_max':max(r['evaluation_R'] for r in rs),
                    'mean_audit_components':float(np.mean([r['audit_components'] for r in rs]))})
    return out
def pair_summary(rows,keys):
    g=defaultdict(list)
    for r in rows:g[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in sorted(g.items()):
        out.append({**dict(zip(keys,key)),'paired_configurations':len(rs),
                    'mean_brier_delta':float(np.mean([r['delta'] for r in rs])),
                    'lower_loss':sum(r['delta']<0 for r in rs),'higher_loss':sum(r['delta']>0 for r in rs),
                    'equal_loss':sum(r['delta']==0 for r in rs),
                    'FS_increased':sum(r['FS_delta']>0 for r in rs),
                    'FS_decreased':sum(r['FS_delta']<0 for r in rs),
                    'lower_loss_more_FS':sum(r['delta']<0 and r['FS_delta']>0 for r in rs)})
    return out
def main():
    v=read(OUT/'validation.json');assert v['status']=='passed' and v['result_sha256']==sha(OUT/'results.json')
    raw=read(OUT/'results.json');rows=[];rulepairs=[];samplingpairs=[];skips=defaultdict(int)
    for art in raw['artifacts']:
        assert sha(art['path'])==art['sha256'];r=read(art['path']);lookup={}
        for c in r['conditions']:
            common={**r['identity'],'query_budget':c['query_budget'],'audit_size':c['audit_size'],'acquisition':c['acquisition'],
                    'audit_R':c['audit_evidence']['old']['R'],'audit_components':c['audit_evidence']['components'],
                    'evaluation_R':r['evaluation_R']}
            actions={a['rule']:a for a in c['actions']}
            for rule,a in actions.items():
                row={**common,'rule':rule,'accepted':a['accepted'],
                     'eligible':not a['decision_status'].startswith('infeasible'),
                     'delta':a['evaluation_brier_delta_from_frozen'],
                     'FS_delta':a['evaluation_FS_delta_from_frozen'],
                     'FS_introduced':a['change_from_frozen']['FS_introduced'],
                     'regret':a['oracle_brier_regret'],'cost':a['total_new_label_cost']}
                rows.append(row);lookup[(c['query_budget'],c['audit_size'],c['acquisition'],rule)]=row
                if rule not in ('retain','update'):
                    for control in ('retain','update'):
                        b=actions[control]
                        rulepairs.append({**common,'rule':rule,'control':control,
                                          'delta':a['evaluation']['brier']-b['evaluation']['brier'],
                                          'FS_delta':a['evaluation']['FS']-b['evaluation']['FS']})
        for key,a in lookup.items():
            b,n,acq,rule=key
            if acq=='random':continue
            control=lookup[(b,n,'random',rule)]
            samplingpairs.append({**r['identity'],'query_budget':b,'audit_size':n,'acquisition':acq,
                                  'rule':rule,'control':'random_acquisition','delta':a['delta']-control['delta'],
                                  'FS_delta':a['FS_delta']-control['FS_delta']})
        for s in r['skipped_conditions']:skips[(s['query_budget'],s['audit_size'],s['reason'])]+=1
    summaries=summarize(rows,('query_budget','audit_size','acquisition','rule'))
    detailed=summarize(rows,('drug','ANI_threshold','feature','strategy','query_budget','audit_size','acquisition','rule'))
    rp=pair_summary(rulepairs,('query_budget','audit_size','acquisition','rule','control'))
    sp=pair_summary(samplingpairs,('query_budget','audit_size','acquisition','rule','control'))
    globalrules=summarize(rows,('rule',))
    globalsampling=pair_summary(samplingpairs,('acquisition','rule','control'))
    drug_sampling=pair_summary(samplingpairs,('drug','acquisition','rule','control'))
    outputs={'summary.csv':summaries,'detailed.csv':detailed,'rule_pairs.csv':rulepairs,
            'rule_pair_summary.csv':rp,'sampling_pairs.csv':samplingpairs,'sampling_pair_summary.csv':sp,
            'drug_sampling_summary.csv':drug_sampling}
    for name,rs in outputs.items():save_csv(OUT/name,rs)
    result={'status':'complete_posthoc_analysis','input_sha256':{str(OUT/n):sha(OUT/n) for n in ('results.json','validation.json')},
            'code_sha256':sha(Path(__file__)),'source_files':len(raw['artifacts']),'action_rows':len(rows),
            'conditions':raw['conditions'],'skipped_conditions':raw['skipped_conditions'],
            'summary_rows':len(summaries),'detailed_rows':len(detailed),'rule_pair_rows':len(rulepairs),
            'sampling_pair_rows':len(samplingpairs),'global_rule_summary':globalrules,
            'global_sampling_summary':globalsampling,
            'drug_sampling_summary':drug_sampling,
            'skips':[{'query_budget':k[0],'audit_size':k[1],'reason':k[2],'conditions':n} for k,n in sorted(skips.items())],
            'outputs_sha256':{str(OUT/n):sha(OUT/n) for n in outputs}}
    (HERE/'audit_acquisition_action_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 审计采集与更新选择：完整冻结预测实验','',
           f"680条轨迹；{raw['conditions']}可行采集条件；{raw['skipped_conditions']}不可行条件；{len(rows)}动作评分；{len(rulepairs)}同采集规则配对；{len(samplingpairs)}相对随机采集配对。",'',
           '固定原query路径和候选预测，非新训练；以隔离审计角色的无标签信息采集，审计标签决定动作，evaluation只评分。每条件所有规则付出同样query+audit成本，拒绝更新不退费。', '',
           '下列为共享队列的配置描述，采集/预算重复计数，不是独立事件率、显著性或患者统计。总体汇总不能替代完整药物/阈值/输入/策略/预算/数量分层。','',
           '|规则|配置|可执行判断|更新|评估损失增加|假敏感净增加|引入新假敏感|损失下降且假敏感增加|平均Brier差|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in globalrules:
        lines.append(f"|{r['rule']}|{r['configurations']}|{r['eligible']}|{r['accepted']}|{r['eval_loss_higher']}|{r['eval_FS_increased']}|{r['any_new_FS']}|{r['lower_loss_more_FS']}|{r['mean_brier_delta_from_frozen']:+.6f}|")
    lines+=['','平均Brier差相对相同策略的预算0候选。方向规则无审计R时标记不可行并回退保留；上述配置列仍包括这些回退，不能称有效保护。','',
            '## 全部采集方式对随机的同成本配对','',
            '|采集|规则|配对|损失更低|更高|相同|FS增加|FS减少|平均Brier差|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in globalsampling:
        lines.append(f"|{r['acquisition']}|{r['rule']}|{r['paired_configurations']}|{r['lower_loss']}|{r['higher_loss']}|{r['equal_loss']}|{r['FS_increased']}|{r['FS_decreased']}|{r['mean_brier_delta']:+.6f}|")
    lines+=['','## 方向规则下全部药物采集对照','',
            '|药物|采集相对random|配对数|Brier更低|更高|相同|FS增加|FS减少|平均Brier差|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in drug_sampling:
        if r['rule']=='audit_direction':
            lines.append(f"|{r['drug']}|{r['acquisition']}|{r['paired_configurations']}|{r['lower_loss']}|{r['higher_loss']}|{r['equal_loss']}|{r['FS_increased']}|{r['FS_decreased']}|{r['mean_brier_delta']:+.6f}|")
    lines+=['','在audit_direction下，预测耐药优先的药物内均值7/9为负，meropenem及TMP–SMX为正；组件覆盖5/9为负，其余4药物为正。不同药物可行配置数不等，这些符号是描述，不是9项独立确认或显著性。', '',
            'random、predicted_R、component_coverage各5808个条件中，审计R=0分别2904、1893、2400；方向规则分别批准1725、2268、1999次更新。更多R支持使规则更常可执行，不自动构成保护：评估损失上升分别463、602、552次，FS净增加分别230、264、278次。分母为获准更新，采集条件共享队列。', '',
            'component_direction相对random采集的平均Brier差为+0.000034，不能报告组件覆盖普遍获益。audit_direction相对固定update减少FS净增加配置，但平均Brier改善也较小；这些是规则间权衡，不能将一个终点改善升级为临床安全。', '',
            '数值边界：严格接受规则按冻结NumPy mean重算；一个已观察相等例子用不同求和顺序会产生−5.55e−17。未改变阈值、候选预测或原输出；近零符号不作实质收益解释。', '',
            '## 完整不可行条件','', '|查询预算|审计数量|原因|条件数|','|---:|---:|---|---:|']
    for r in result['skips']:lines.append(f"|{r['query_budget']}|{r['audit_size']}|{r['reason']}|{r['conditions']}|")
    lines+=['','原ANI输入/划分的10个跳过（对应40策略轨迹）另保留于results.json inherited_skips。未把缺失审计标签或真实R作为免费抽样信息。', '',
            '所有审计量5/10/25/50、查询量10/25/50、两个输入和阈值、全部药物/seed/策略均保留在detailed.csv；小R范围见summary.csv/detailed.csv。', '',
            '固定retain/update在三种采集方式上相同是采集不影响候选模型的校验；不是三次独立复现。component_direction不证明组件等权策略有理论风险保护。oracle只知道两个候选评估Brier的最低值，不能部署。', '',
            '此阶段未改变query路径或把减少审计的预算重新分配给训练；没有新输入检测、来源匹配因果干预或新盲态数据。若某采集方式均值改善，也需同时检查FS方向、完整异质性和同成本对照。', '',
            f'数据目录：{OUT}。独立校验：validation.json。协议：AUDIT_ACQUISITION_ACTION_PROTOCOL_V1.md。']
    (HERE/'AUDIT_ACQUISITION_ACTION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','action_rows','summary_rows','detailed_rows','rule_pair_rows','sampling_pair_rows')}))
if __name__=='__main__':main()
