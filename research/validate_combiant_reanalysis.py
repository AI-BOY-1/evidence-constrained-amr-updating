"""Independent raw-row joins, metric recomputation, and acquired-image checks."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
import math
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

BASE = ncs_paths.data_path('experiments/combiant_public_v1')
SUB = BASE / 'replication_subset'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def same(expected, actual):
    if expected is None:
        assert actual is None or actual == ''
    else:
        assert math.isclose(float(actual), expected, abs_tol=1e-11, rel_tol=1e-11), (expected, actual)


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main():
    pair = json.loads((BASE / 'pairing_audit.json').read_text())
    analysis = json.loads((BASE / 'measurement_analysis.json').read_text())
    for binding in (pair['binding'], analysis['binding']):
        for raw, expected in binding.items():
            assert digest(Path(raw)) == expected, raw
    software = load_rows(SUB / 'inference/output.csv')
    manual = load_rows(SUB / 'CombiANT_raw_data/combiANT_user_measurements.csv')
    smap = defaultdict(list); hmap = defaultdict(list)
    for r in software:
        smap[PurePosixPath(r['name']).stem].append(r)
    for r in manual:
        hmap[r['name']].append(r)
    unique_names = {n for n, rs in smap.items() if len(rs) == 1 and len(hmap[n]) == 1}
    assert unique_names == {p['name'] for p in pair['pairs']}
    assert len(software) == pair['software_rows'] and len(manual) == pair['manual_rows']
    assert len(software) == pair['paired_rows'] + len(pair['unmatched_software'])
    assert len(manual) == pair['paired_rows'] + len(pair['unmatched_manual'])
    assert {n for n, rs in smap.items() if len(rs) > 1} == set(pair['duplicate_software_names'])
    table = load_rows(BASE / 'paired_distances.csv')
    keys = [(r['name'], r['distance']) for r in table]
    assert len(keys) == len(set(keys)) == len(unique_names) * 6 == analysis['distance_records']
    for r in table:
        s, h = smap[r['name']][0], hmap[r['name']][0]
        assert int(r['plate']) == float(h['plate'])
        assert int(r['user']) == int(PurePosixPath(s['name']).parent.name[4:])
        for flag in ('Discard_inner', 'Discard_outer', 'Discard_no_island'):
            assert r[flag] == s[flag]
        assert (r['flagged'] == 'True') == any(s[f] == 'True' for f in ('Discard_inner', 'Discard_outer', 'Discard_no_island'))
        sv, hv = finite(s[r['distance']]), finite(h[r['distance'] + '_user'])
        same(sv, r['software_mm']); same(hv, r['manual_mm'])
        same(sv - hv if sv is not None and hv is not None else None, r['error_mm'])
    for summary in analysis['summaries']:
        group = summary['group']
        selected = [r for r in table if r['distance'] == summary['distance'] and
                    (group == 'all' or group == 'flagged' and r['flagged'] == 'True' or
                     group == 'unflagged' and r['flagged'] == 'False' or
                     group.startswith('user') and r['user'] == group[4:])]
        errors = [float(r['error_mm']) for r in selected if r['error_mm'] != '']
        absolute = [abs(e) for e in errors]
        assert len(selected) == summary['eligible_pairs'] and len(errors) == summary['finite_pairs']
        same(statistics.fmean(errors) if errors else None, summary['bias_mm'])
        same(statistics.fmean(absolute) if errors else None, summary['mae_mm'])
        same(statistics.median(absolute) if errors else None, summary['median_absolute_mm'])
        same(float(np.quantile(absolute, .95)) if errors else None, summary['p95_absolute_mm'])
        same(max(absolute) if errors else None, summary['max_absolute_mm'])
    spans = load_rows(BASE / 'plate_distance_spans.csv')
    for row in spans:
        members = [r for r in table if r['plate'] == row['plate'] and r['distance'] == row['distance']]
        assert len({r['user'] for r in members}) == int(row['users']) >= 2
        for kind in ('software', 'manual'):
            values = [float(r[kind + '_mm']) for r in members if r[kind + '_mm'] != '']
            same(max(values) - min(values) if len(values) >= 2 else None, row[kind + '_span_mm'])
    for summary in analysis['span_summaries']:
        rows = [r for r in spans if r['distance'] == summary['distance'] and
                (summary['group'] == 'at_least_two_users' or r['users'] == '3')]
        values = [float(r[summary['method'] + '_span_mm']) for r in rows if r[summary['method'] + '_span_mm'] != '']
        assert len(rows) == summary['eligible_plates'] and len(values) == summary['finite_plates']
        same(statistics.fmean(values) if values else None, summary['mean_span_mm'])
        same(statistics.median(values) if values else None, summary['median_span_mm'])
        same(float(np.quantile(values, .95)) if values else None, summary['p95_span_mm'])
        same(max(values) if values else None, summary['max_span_mm'])
    inventory = json.loads((BASE / 'archive_inventory.json').read_text())
    entries = {e['name']: e for e in inventory['entries']}
    acquisition = json.loads((BASE / 'image_acquisition_manifest.json').read_text())
    inspected = []; image_hashes = defaultdict(list)
    for name, record in acquisition['members'].items():
        path = SUB / name
        blob = path.read_bytes(); entry = entries[name]
        assert len(blob) == record['size'] == entry['size']
        assert hashlib.sha256(blob).hexdigest() == record['sha256']
        assert f'{zlib.crc32(blob) & 0xffffffff:08x}' == record['crc32'] == entry['crc32']
        with Image.open(path) as im:
            im.load(); size, mode = list(im.size), im.mode
        inspected.append({'member': name, 'sha256': record['sha256'], 'size': size, 'mode': mode})
        image_hashes[record['sha256']].append(name)
    result = {'status': 'passed', 'scope': 'paired_measurement_recomputation_and_current_acquired_images',
        'pairing_sha256': digest(BASE / 'pairing_audit.json'), 'analysis_sha256': digest(BASE / 'measurement_analysis.json'),
        'paired_images': len(unique_names), 'distance_records_checked': len(table), 'summary_rows_checked': len(analysis['summaries']),
        'plate_span_rows_checked': len(spans), 'span_summary_rows_checked': len(analysis['span_summaries']),
        'acquired_images_checked': len(inspected), 'expected_archive_images': 299,
        'all_evaluation_images_checked': len(inspected) == 299,
        'image_checks': inspected, 'exact_duplicate_image_groups': [v for v in image_hashes.values() if len(v) > 1],
        'table_hashes': {name: digest(BASE / name) for name in ('paired_distances.csv', 'measurement_summary.csv',
                     'plate_distance_spans.csv', 'plate_span_summary.csv')},
        'limitations': ['Numeric reanalysis validation does not rerun original U-Net.',
                       'Current image count is a snapshot; full image acquisition has its own pending status.']}
    (BASE / 'measurement_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('image_checks', 'table_hashes', 'limitations')}))


if __name__ == '__main__':
    main()
