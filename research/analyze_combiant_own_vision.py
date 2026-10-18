"""Report our held-out visual predictions, full controls, and genuine source photos."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=ncs_paths.data_path('experiments/combiant_public_v1')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('pilot','full'),default='full');args=p.parse_args()
    out=BASE/('own_vision_pilot_v1' if args.scope=='pilot' else 'own_vision_v1')
    result=json.loads((out/'results.json').read_text(encoding='utf-8'))
    validation=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    assert validation['status']=='passed' and validation['result_sha256']==sha(out/'results.json')
    with (out/'predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
    with (out/'review_controls.csv').open(newline='') as f:reviews=list(csv.DictReader(f))
    budgets=[]
    for fraction in (0.,.1,.25,.5):
        for strategy in ('uncertainty','image_quality','random'):
            seed_values=[];costs=[];captures=[]
            for seed in range(100) if strategy=='random' else [-1]:
                rows=[r for r in reviews if float(r['budget_fraction'])==fraction and r['strategy']==strategy and int(r['seed'])==seed]
                denominator=sum(int(r['original_distance_n']) for r in rows)
                unresolved=sum(float(r['unresolved_mae_mm'])*int(r['original_distance_n']) for r in rows)/denominator
                cost=sum(int(r['reviewed_plates']) for r in rows)
                capture=sum(int(r['original_over_2mm'])-int(r['unreviewed_over_2mm']) for r in rows)
                seed_values.append(unresolved);costs.append(cost);captures.append(capture)
            assert len(set(costs))==1
            budgets.append({'nominal_budget_fraction':fraction,'strategy':strategy,'reviewed_plates':costs[0],
                'total_heldout_plates':result['plates'],'random_replicates':len(seed_values),
                'unresolved_mae_mean_mm':float(np.mean(seed_values)),
                'random_p05_mm':float(np.quantile(seed_values,.05)) if strategy=='random' else None,
                'random_p95_mm':float(np.quantile(seed_values,.95)) if strategy=='random' else None,
                'captured_over_2mm_mean':float(np.mean(captures))})
    with (out/'review_budget_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(budgets[0]));writer.writeheader();writer.writerows(budgets)
    pooled=[r for r in result['metrics'] if r['fold']=='pooled']
    lines=['# 本项目独立图像测量实验'+('：冻结试跑' if args.scope=='pilot' else '：全公开队列'),'',
        f"实际自主训练：{result['rows']}幅图像，{result['plates']}个培养皿，5折培养皿隔离。操作者分布：{result['available_user_counts']}。",
        '目标为六项连续距离，原作者软件输出不作训练特征。独立验证逐像素重建输入，重训练回放所有模型，重算指标及等预算复核记录。','',
        '|方法|有效完整照片n|六距离宏平均MAE mm|95%照片宏误差 mm|宏误差>2mm照片|',
        '|---|---:|---:|---:|---:|']
    for r in pooled:
        if r['distance']=='macro':lines.append(f"|{r['method']}|{r['finite_n']}|{r['mae_mm']:.4f}|{r['p95_absolute_mm']:.4f}|{r['over_2mm']}|")
    lines += ['','## 等复核预算完整对照','',
        '复核一个培养皿包含其全部照片，仿真替换为人工参考值；未实施真实人工干预。随机100次排序的区间是描述性分布，不是置信区间或置换检验p值。', '',
        '|名义预算|实际复核培养皿|策略|未解决误差/原始有效距离n mm|随机5%-95%|',
        '|---|---:|---|---:|---|']
    for r in budgets:
        spread=f"{r['random_p05_mm']:.4f}–{r['random_p95_mm']:.4f}" if r['strategy']=='random' else ''
        lines.append(f"|{r['nominal_budget_fraction']:.0%}|{r['reviewed_plates']}/{r['total_heldout_plates']}|{r['strategy']}|{r['unresolved_mae_mean_mm']:.4f}|{spread}|")
    lines += ['','完整指标：metrics.csv。逐照片预测：predictions.csv。全部预算复核ID和随机种子：review_controls.csv。', '',
        '1条A人工距离缺失，训练多输出时只排除该不完整训练行；测试保留其他五距离。宏平均与逐距离的有效分母不同。', '',
        '## 限制','', '这是本项目独立训练运行的标准机器学习基线，不是新发明的网络架构。培养皿隔离不证明菌株/患者独立。图像是公开旧数据，非新的盲态确认。距离预测不等同S/I/R、抗生素协同或临床收益。',
        '试跑使用下载时冻结快照，主要为user1，不能声称完整队列或跨操作者泛化。参数不因试跑负结果改变。' if args.scope=='pilot' else
        '完整队列跨操作者压力测试单列operator_transfer结果，仍非新机构外部确认。']
    report=ROOT/('COMBIANT_OWN_VISION_PILOT_RESULTS.md' if args.scope=='pilot' else 'COMBIANT_OWN_VISION_RESULTS.md')
    report.write_text('\n'.join(lines),encoding='utf-8')
    figdir=ROOT/'advanced_figures'/('combiant_own_pilot_v1' if args.scope=='pilot' else 'combiant_own_v1')
    figdir.mkdir(exist_ok=True)
    pairing=json.loads((BASE/'pairing_audit.json').read_text(encoding='utf-8'))
    available={r['name'] for r in predictions}
    samples=[]
    for plate in sorted({r['plate'] for r in pairing['pairs'] if r['name'] in available}):
        candidates=sorted([r for r in pairing['pairs'] if r['plate']==plate and r['name'] in available],key=lambda r:(r['user'],r['name']))
        samples.append(candidates[0])
        if len(samples)==2:break
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(12,10.5),facecolor='white')
    fig.text(.06,.955,'Image-to-measurement: held-out plate experiment',fontsize=19,fontweight='bold',color='#1d363a')
    fig.text(.06,.926,f"Our trained models | {result['rows']} photos / {result['plates']} plates | "+('Frozen pilot' if args.scope=='pilot' else 'Full public cohort'),fontsize=11,color='#52666b')
    gs=fig.add_gridspec(2,2,left=.08,right=.95,top=.87,bottom=.12,wspace=.27,hspace=.39)
    for n,sample in enumerate(samples):
        ax=fig.add_subplot(gs[0,n]);path=BASE/'replication_subset'/sample['member']
        with Image.open(path) as im:im.load();ax.imshow(im.copy())
        ax.set_anchor('N');ax.set_axis_off();ax.set_title(f"{chr(97+n)}  Real input / Plate {sample['plate']} / User {sample['user']}",loc='left',fontsize=11,fontweight='bold',pad=7)
        ax.text(0,-.035,path.name,transform=ax.transAxes,fontsize=8,color='#52666b',va='top')
    ax=fig.add_subplot(gs[1,0]);colors=['#267c84','#cf5c43','#705d98','#29946d','#ca9533','#555d65']
    distances=['A','B','C','AB','BC','AC']
    tree_rows=[r for r in predictions if r['method']=='own_extra_trees_32px']
    for d,c in zip(distances,colors):
        target=np.asarray([float(r['target_'+d]) for r in tree_rows]);pred=np.asarray([float(r['prediction_'+d]) for r in tree_rows])
        ax.scatter(target,pred,s=15,color=c,alpha=.65,label=d,edgecolors='none')
    values=[float(r[k]) for r in tree_rows for k in r if k.startswith(('target_','prediction_')) and np.isfinite(float(r[k]))]
    lo=min(0,min(values));hi=max(values)*1.04;ax.plot([lo,hi],[lo,hi],color='#52666b',lw=.8,ls='--')
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_xlabel('Manual reference (mm)');ax.set_ylabel('Held-out Extra Trees prediction (mm)')
    ax.set_title('c  All out-of-fold predictions',loc='left',fontweight='bold',fontsize=11)
    ax.legend(ncol=3,fontsize=8,frameon=False,loc='upper left')
    ax=fig.add_subplot(gs[1,1])
    for strategy,color in [('uncertainty','#267c84'),('image_quality','#cf5c43'),('random','#555d65')]:
        rows=[r for r in budgets if r['strategy']==strategy]
        xx=[100*r['reviewed_plates']/r['total_heldout_plates'] for r in rows]
        ax.plot(xx,[r['unresolved_mae_mean_mm'] for r in rows],marker='o',ms=4,lw=1.3,color=color,label=strategy.replace('_',' '))
        if strategy=='random':ax.fill_between(xx,[r['random_p05_mm'] for r in rows],[r['random_p95_mm'] for r in rows],color=color,alpha=.12)
    ax.set_xlabel('Actually reviewed plates (%)');ax.set_ylabel('Unresolved absolute error / original n (mm)')
    ax.set_title('d  Equal-cost reference-review simulation',loc='left',fontweight='bold',fontsize=11)
    ax.legend(frameon=False,fontsize=8);ax.set_ylim(bottom=0)
    for axis in fig.axes[2:]:
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',color='#e3e8e8',lw=.5);axis.set_axisbelow(True)
    fig.text(.06,.067,'Photos: Hallstrom et al. (2025), CC BY 4.0. Predictions and controls: our independently replayed fits.',fontsize=8.5,color='#52666b')
    fig.text(.06,.043,'Manual readings are references. No susceptibility labels, genomic linkage, new blind cohort, or clinical benefit inferred.',fontsize=8.5,color='#52666b')
    for ext in ('png','pdf','svg'):fig.savefig(figdir/f'own_vision_results.{ext}',dpi=350)
    plt.close(fig)
    source={'scope':args.scope,'result_sha256':sha(out/'results.json'),'validation_sha256':sha(out/'validation.json'),
        'prediction_sha256':sha(out/'predictions.csv'),'review_summary_sha256':sha(out/'review_budget_summary.csv'),
        'code_sha256':sha(Path(__file__)),'selection':'Smallest two available plate IDs, lowest user within each.',
        'photos':[{**r,'sha256':sha(BASE/'replication_subset'/r['member'])} for r in samples],
        'exports':{ext:sha(figdir/f'own_vision_results.{ext}') for ext in ('png','pdf','svg')}}
    (figdir/'source.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(report),'figure':str(figdir/'own_vision_results.png'),'scope':args.scope}))


if __name__=='__main__':main()
