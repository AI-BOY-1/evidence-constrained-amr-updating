"""Exploratory reanalysis of published measurements, retaining flags and gaps."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_combiant_pairing import OUT, SUBSET, sha

DISTANCES = ('A', 'B', 'C', 'AB', 'BC', 'AC')


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def summarize(rows, group, distance):
    valid = [r for r in rows if r['error_mm'] is not None]
    errors = np.array([r['error_mm'] for r in valid])
    absolute = np.abs(errors)
    return {'group': group, 'distance': distance, 'eligible_pairs': len(rows), 'finite_pairs': len(valid),
            'invalid_pairs': len(rows) - len(valid), 'plates': len({r['plate'] for r in valid}),
            'bias_mm': float(errors.mean()) if len(valid) else None,
            'mae_mm': float(absolute.mean()) if len(valid) else None,
            'median_absolute_mm': float(np.median(absolute)) if len(valid) else None,
            'p95_absolute_mm': float(np.quantile(absolute, .95)) if len(valid) else None,
            'max_absolute_mm': float(absolute.max()) if len(valid) else None}


def main():
    pairing_path = OUT / 'pairing_audit.json'
    pairing = json.loads(pairing_path.read_text())
    with (SUBSET / 'inference/output.csv').open(newline='') as stream:
        software = list(csv.DictReader(stream))
    with (SUBSET / 'CombiANT_raw_data/combiANT_user_measurements.csv').open(newline='') as stream:
        manual = list(csv.DictReader(stream))
    rows = []
    for pair in pairing['pairs']:
        s, h = software[pair['software_row']], manual[pair['manual_row']]
        flags = {}
        for key in ('Discard_inner', 'Discard_outer', 'Discard_no_island'):
            if s[key] not in ('True', 'False'):
                raise RuntimeError('Unexpected original discard flag')
            flags[key] = s[key] == 'True'
        for distance in DISTANCES:
            sv, hv = number(s[distance]), number(h[distance + '_user'])
            rows.append({'name': pair['name'], 'member': pair['member'], 'plate': pair['plate'],
                'user': pair['user'], 'distance': distance, 'software_mm': sv, 'manual_mm': hv,
                'error_mm': sv - hv if sv is not None and hv is not None else None,
                'flagged': any(flags.values()), **flags})
    summaries = []
    for group in ('all', 'flagged', 'unflagged', 'user1', 'user2', 'user3'):
        for distance in DISTANCES:
            subset = [r for r in rows if r['distance'] == distance and
                      (group == 'all' or group == 'flagged' and r['flagged'] or
                       group == 'unflagged' and not r['flagged'] or
                       group.startswith('user') and r['user'] == int(group[-1]))]
            summaries.append(summarize(subset, group, distance))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['plate'], r['distance'])].append(r)
    spans = []
    for (plate, distance), values in sorted(grouped.items()):
        if len({r['user'] for r in values}) < 2:
            continue
        record = {'plate': plate, 'distance': distance, 'users': len(values), 'flagged': any(r['flagged'] for r in values)}
        for kind in ('software', 'manual'):
            finite = [r[kind + '_mm'] for r in values if r[kind + '_mm'] is not None]
            record[kind + '_finite_users'] = len(finite)
            record[kind + '_span_mm'] = max(finite) - min(finite) if len(finite) >= 2 else None
        spans.append(record)
    span_summaries = []
    for group in ('at_least_two_users', 'exactly_three_users'):
        for distance in DISTANCES:
            values = [r for r in spans if r['distance'] == distance and (group == 'at_least_two_users' or r['users'] == 3)]
            for kind in ('software', 'manual'):
                finite = [r[kind + '_span_mm'] for r in values if r[kind + '_span_mm'] is not None]
                span_summaries.append({'group': group, 'distance': distance, 'method': kind,
                    'eligible_plates': len(values), 'finite_plates': len(finite),
                    'mean_span_mm': float(np.mean(finite)) if finite else None,
                    'median_span_mm': float(np.median(finite)) if finite else None,
                    'p95_span_mm': float(np.quantile(finite, .95)) if finite else None,
                    'max_span_mm': max(finite) if finite else None})
    result = {'status': 'measurement_reanalysis_complete_images_pending',
        'binding': {str(p): sha(p) for p in (pairing_path, SUBSET / 'inference/output.csv',
                SUBSET / 'CombiANT_raw_data/combiANT_user_measurements.csv', Path(__file__),
                Path(__file__).with_name('COMBIANT_REANALYSIS_PROTOCOL.md'))},
        'paired_images': pairing['paired_rows'], 'plates': pairing['paired_plates'],
        'distance_records': len(rows), 'flagged_images': len({r['name'] for r in rows if r['flagged']}),
        'summaries': summaries, 'span_summaries': span_summaries,
        'limitations': ['Reanalysis of author-supplied outputs, not rerun U-Net or original algorithm performance.',
                       'Distances are not clinical susceptibility labels or antibiotic interaction decisions.',
                       'Reference human measurements may vary between operators and are not independent gold standards.',
                       'Photos and plate IDs do not establish patient/isolate/genomic independence.']}
    save_csv('paired_distances.csv', rows)
    save_csv('measurement_summary.csv', summaries)
    save_csv('plate_distance_spans.csv', spans)
    save_csv('plate_span_summary.csv', span_summaries)
    (OUT / 'measurement_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 真实培养皿图像测量：公开结果再分析', '',
        '来源：Hallström等2025年CombiANT Reader，CC BY 4.0。本轮重算原作者输出与人工测量，不是本项目新模型性能。', '',
        f"软件299行、人工283行；严格唯一配对{pairing['paired_rows']}行、{pairing['paired_plates']}个培养皿、{len(rows)}条距离记录。",
        f"原始三个discard标志任一为True的图像共{result['flagged_images']}幅。全部保留。", '',
        '4个编号存在JPG/PNG两种软件记录，暂不任意选一种；另有未匹配数据。完整列表见pairing_audit.json。', '',
        '|距离|配对n|MAE mm|95%绝对差 mm|最大绝对差 mm|', '|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['group'] == 'all':
            lines.append(f"|{r['distance']}|{r['finite_pairs']}|{r['mae_mm']:.4f}|{r['p95_absolute_mm']:.4f}|{r['max_absolute_mm']:.4f}|")
    lines += ['', '全部标志分组、操作者分组和培养皿内跨度见四张完整CSV表；未定义值单列，不删除失败。', '',
              '不能从上述差值推出S/I/R错误、抗生素协同、新机制或临床安全。原论文已研究多操作者稳定性；本轮不宣称首次发现该问题。', '',
              '全299幅评估图像的下载与实际解码仍需单独核验；固定六幅照片图板的独立校验状态见advanced_figures/combiant_photos_v1/validation.json。']
    Path(__file__).with_name('COMBIANT_MEASUREMENT_REANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'paired_images', 'plates', 'distance_records', 'flagged_images')}))
    print(json.dumps([r for r in summaries if r['group'] == 'all']))


if __name__ == '__main__':
    main()
