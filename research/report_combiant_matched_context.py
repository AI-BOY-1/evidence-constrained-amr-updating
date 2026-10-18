"""Full tables for the independently validated matched-context experiment."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=ncs_paths.data_path('experiments/combiant_public_v1/matched_context_v1')
def main():
    r=json.loads((OUT/'results.json').read_text());v=json.loads((OUT/'validation.json').read_text())
    assert v['status']=='passed' and v['result_sha256']==hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()
    lines=['# 等培养皿、等照片数的采集上下文实验','',
        f"事后探索。40次真实训练与40次独立回放通过；{r['prediction_rows']}条预测、224组指标、126组配对对照。",
        '每折四种方案使用完全相同的训练培养皿及照片数，各视图保留其本人的人工测量参考。测试培养皿全部未见。','',
        '|折|共同训练培养皿/每方案照片数|混合方案操作者顺序|','|---|---:|---|']
    lines += [f"|{p['fold']}|{len(p['common_train_plates'])}|{p['mixed_user_order']}|" for p in r['plans']]
    lines += ['','## 全部性能表','',
        '|模型|训练方案|测试操作者|距离|n/培养皿|MAE mm|培养皿等权MAE|95%误差|最大误差|>2mm|',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for x in r['summaries']:lines.append(f"|{x['method']}|{x['regime']}|{x['test_user']}|{x['distance']}|{x['n']}/{x['plates']}|{x['mae_mm']:.4f}|{x['plate_equal_mae_mm']:.4f}|{x['p95_absolute_mm']:.4f}|{x['max_absolute_mm']:.4f}|{x['over_2mm']}|")
    lines += ['','## 相同测试照片相对同操作者训练的全部对照','',
        '|模型|训练方案|参考|测试操作者|距离|n/培养皿|配对MAE差|培养皿等权差|改善/恶化|新增/消除>2mm|',
        '|---|---|---|---|---|---|---:|---:|---|---|']
    for x in r['paired']:lines.append(f"|{x['method']}|{x['regime']}|{x['reference']}|{x['test_user']}|{x['distance']}|{x['n']}/{x['plates']}|{x['delta_mm']:+.4f}|{x['plate_equal_delta_mm']:+.4f}|{x['improved']}/{x['worsened']}|{x['new_over_2mm']}/{x['removed_over_2mm']}|")
    macros=[x for x in r['paired'] if x['distance']=='macro']
    lines+=['','## 描述性总览','']
    for method in ('ridge','extra_trees'):
        cross=[x for x in macros if x['method']==method and x['regime']!='mixed'];mixed=[x for x in macros if x['method']==method and x['regime']=='mixed']
        lines.append(f"- {method}：六种单操作者跨上下文比较中{sum(x['delta_mm']>0 for x in cross)}/6平均误差较高，差值范围{min(x['delta_mm'] for x in cross):+.4f}至{max(x['delta_mm'] for x in cross):+.4f} mm；混合输入相对同操作者方案有{sum(x['delta_mm']<0 for x in mixed)}/3测试操作者改善。")
    lines+=['','不同培养皿和视图共享有限旧队列；描述性比较无p值、非新盲态确认。等数量与等培养皿排除了这两项设计差异，但设备、预处理和人工参考测量仍共变，不能声称操作者因果效应或临床安全。2 mm为技术误差阈值，非MIC断点。完整预测和校验哈希位于D盘matched_context_v1。']
    (HERE/'COMBIANT_MATCHED_CONTEXT_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'passed','macro_comparisons':macros}))
if __name__=='__main__':main()
