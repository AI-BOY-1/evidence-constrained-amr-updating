"""Post-result descriptive audit of frozen predictions; no model fitting."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=ncs_paths.data_path('experiments/combiant_public_v1/own_vision_v1')
OUT=BASE/'error_audit_v1'
DIST=('A','B','C','AB','BC','AC')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def key(r): return tuple(r[k] for k in ('name','plate','user','fold'))

def main():
    v=read(BASE/'validation.json');r=read(BASE/'results.json')
    assert v['status']=='passed' and v['result_sha256']==sha(BASE/'results.json')
    assert r['predictions_sha256']==sha(BASE/'predictions.csv')
    rows=list(csv.DictReader((BASE/'predictions.csv').open(encoding='utf-8',newline='')))
    original_prediction_rows=len(rows)
    table=list(csv.DictReader((BASE.parent/'paired_distances.csv').open(encoding='utf-8',newline='')))
    measured={(x['name'],x['distance']):x for x in table}
    observed=[x for x in rows if x['method']=='own_extra_trees_32px']
    baseline_rows=[]
    for x in observed:
        plan=next(p for p in r['fold_plan'] if str(p['fold'])==x['fold'])
        for method in ('train_plate_mean','published_software_reference'):
            b={**x,'method':method,'risk_score':'0'}
            for d in DIST:
                raw=measured[(x['name'],d)]
                b['target_'+d]=raw['manual_mm'] or 'nan'
                b['prediction_'+d]=(float(raw['software_mm'] or 'nan') if method=='published_software_reference'
                    else float(np.nanmean([float(measured[(name,d)]['manual_mm'] or 'nan') for name in plan['train_names']])))
            baseline_rows.append(b)
    rows+=baseline_rows
    methods=sorted({x['method'] for x in rows})
    lookup={}
    for x in rows:
        k=(x['method'],key(x));assert k not in lookup;lookup[k]=x
    summaries=[]
    for method in methods:
        mr=[x for x in rows if x['method']==method]
        for user in ('all',*sorted({x['user'] for x in mr})):
            group=[x for x in mr if user=='all' or x['user']==user]
            for distance in (*DIST,'macro'):
                errors=[];plates=defaultdict(list)
                for x in group:
                    e=np.array([float(x['prediction_'+d])-float(x['target_'+d]) for d in DIST])
                    if distance=='macro':
                        if not np.all(np.isfinite(e)):continue
                        value=float(np.mean(np.abs(e)))
                    else:
                        value=float(abs(e[DIST.index(distance)]))
                        if not np.isfinite(value):continue
                    errors.append(value);plates[x['plate']].append(value)
                if errors:
                    summaries.append({'method':method,'user':user,'distance':distance,'n':len(errors),
                        'plates':len(plates),'mae_mm':float(np.mean(errors)),
                        'plate_equal_mae_mm':float(np.mean([np.mean(a) for a in plates.values()])),
                        'p95_absolute_mm':float(np.quantile(errors,.95)),'max_absolute_mm':max(errors),
                        'over_2mm':sum(e>2 for e in errors)})
    paired=[]
    for transfer,primary in (('operator_transfer_ridge','own_ridge_32px'),('operator_transfer_extra_trees','own_extra_trees_32px')):
        tests=[x for x in rows if x['method']==transfer]
        for reference in (primary,'train_plate_mean','published_software_reference'):
            for user in ('all',*sorted({x['user'] for x in tests})):
                pairs=[]
                for x in tests:
                    if user!='all' and x['user']!=user:continue
                    y=lookup[(reference,key(x))]
                    a=np.array([float(x['target_'+d]) for d in DIST]);b=np.array([float(y['target_'+d]) for d in DIST])
                    assert np.array_equal(a,b,equal_nan=True)
                    if not np.all(np.isfinite(a)):continue
                    te=float(np.mean(np.abs(np.array([float(x['prediction_'+d]) for d in DIST])-a)))
                    re=float(np.mean(np.abs(np.array([float(y['prediction_'+d]) for d in DIST])-a)))
                    pairs.append((key(x),te,re))
                if not pairs:continue
                pe=defaultdict(list)
                for k,t,e in pairs:pe[k[1]].append(t-e)
                paired.append({'transfer':transfer,'reference':reference,'user':user,'n':len(pairs),'plates':len(pe),
                    'transfer_mae_mm':float(np.mean([p[1] for p in pairs])),
                    'reference_mae_mm':float(np.mean([p[2] for p in pairs])),
                    'paired_delta_mm':float(np.mean([p[1]-p[2] for p in pairs])),
                    'plate_equal_delta_mm':float(np.mean([np.mean(p) for p in pe.values()])),
                    'improved_photos':sum(t<e for _,t,e in pairs),'worsened_photos':sum(t>e for _,t,e in pairs),
                    'unchanged_photos':sum(t==e for _,t,e in pairs),
                    'new_over_2mm':sum(t>2 and e<=2 for _,t,e in pairs),
                    'removed_over_2mm':sum(t<=2 and e>2 for _,t,e in pairs)})
    OUT.mkdir(exist_ok=True)
    inputs=[BASE/'results.json',BASE/'predictions.csv',BASE/'validation.json',BASE.parent/'paired_distances.csv',HERE/'COMBIANT_POSTHOC_ERROR_AUDIT_PROTOCOL.md',Path(__file__)]
    result={'status':'computed_pending_independent_validation','designation':'post-result descriptive audit',
        'binding':{str(p):sha(p) for p in inputs},'prediction_rows':original_prediction_rows,
        'baseline_rows_reconstructed':len(baseline_rows),'summaries':summaries,'paired':paired,
        'limitations':['Shared retrospective plates; no blind confirmation or p-values.',
            'Transfer changes operator and training-set size; not a causal operator effect.',
            'Published software is a reference comparator, not our rerun.']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 冻结照片预测：完整误差与同照片迁移审计','',
        '事后描述分析；不重训、不调参。迁移比较使用完全相同的测试照片和人工参考值，不将全队列均值当作同样本迁移基线。','',
        '## 全部方法、操作者、距离','',
        '|方法|操作者|距离|有效n|培养皿|照片加权MAE mm|培养皿等权MAE mm|95%误差 mm|最大 mm|>2mm数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summaries:lines.append(f"|{x['method']}|{x['user']}|{x['distance']}|{x['n']}|{x['plates']}|{x['mae_mm']:.4f}|{x['plate_equal_mae_mm']:.4f}|{x['p95_absolute_mm']:.4f}|{x['max_absolute_mm']:.4f}|{x['over_2mm']}|")
    lines+=['','## 相同测试照片上的完整迁移对照','',
        '|迁移模型|参考方案|操作者|n/培养皿|迁移MAE|参考MAE|配对差值|培养皿等权差值|改善/恶化/相同|新增/消除>2mm|',
        '|---|---|---|---|---:|---:|---:|---:|---|---|']
    for x in paired:lines.append(f"|{x['transfer']}|{x['reference']}|{x['user']}|{x['n']}/{x['plates']}|{x['transfer_mae_mm']:.4f}|{x['reference_mae_mm']:.4f}|{x['paired_delta_mm']:+.4f}|{x['plate_equal_delta_mm']:+.4f}|{x['improved_photos']}/{x['worsened_photos']}/{x['unchanged_photos']}|{x['new_over_2mm']}/{x['removed_over_2mm']}|")
    lines+=['','2 mm为固定技术误差阈值，不是药敏断点。参考缺失不填补。培养皿隔离不等同患者/菌株独立，跨操作者方案同时改变训练量，不据此识别因果效应。','',
        '独立校验状态见D盘error_audit_v1/validation.json；完整来源哈希见results.json。']
    (HERE/'COMBIANT_POSTHOC_ERROR_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'summaries':len(summaries),'paired':len(paired),'prediction_rows':len(rows)}))

if __name__=='__main__':main()
