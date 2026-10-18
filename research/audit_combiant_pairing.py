"""Preserve every public row and independently identify image/plate pairing."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

OUT = ncs_paths.data_path('experiments/combiant_public_v1')
SUBSET = OUT / 'replication_subset'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    soft_path = SUBSET / 'inference/output.csv'
    human_path = SUBSET / 'CombiANT_raw_data/combiANT_user_measurements.csv'
    with soft_path.open(newline='', encoding='utf-8-sig') as stream:
        soft = list(csv.DictReader(stream))
    with human_path.open(newline='', encoding='utf-8-sig') as stream:
        human = list(csv.DictReader(stream))
    inventory = json.loads((OUT / 'archive_inventory.json').read_text())
    images = {e['name']: e for e in inventory['entries'] if e['name'].startswith('inference/input/') and not e['directory']}
    soft_counts = Counter(PurePosixPath(r['name']).stem for r in soft)
    human_counts = Counter(r['name'] for r in human)
    human_map = {r['name']: r for r in human if human_counts[r['name']] == 1}
    paired, unmatched_soft = [], []
    for index, row in enumerate(soft):
        name = PurePosixPath(row['name']).stem
        member = 'inference/' + row['name']
        if soft_counts[name] != 1 or human_counts[name] != 1:
            unmatched_soft.append({'row_index': index, 'name': row['name'], 'reason': 'missing_or_nonunique_manual_name'})
            continue
        manual = human_map[name]
        user = int(PurePosixPath(row['name']).parent.name.removeprefix('user'))
        plate_value = float(manual['plate'])
        if not plate_value.is_integer():
            raise RuntimeError('Noninteger plate identifier')
        paired.append({'name': name, 'software_row': index, 'manual_row': human.index(manual),
                       'member': member, 'member_present_in_archive': member in images,
                       'user': user, 'plate': int(plate_value)})
    matched = {r['name'] for r in paired}
    plates = defaultdict(list)
    for row in paired:
        plates[row['plate']].append(row)
    duplicate_plate_users = [{'plate': p, 'users': [r['user'] for r in rows]}
                             for p, rows in plates.items() if len({r['user'] for r in rows}) != len(rows)]
    eligible = sorted(p for p, rows in plates.items()
                      if len(rows) == 3 and {r['user'] for r in rows} == {1, 2, 3}
                      and all(r['member_present_in_archive'] for r in rows))
    examples = eligible[:2]
    result = {'status': 'paired_metadata_audited_images_pending',
        'binding': {str(p): sha(p) for p in (soft_path, human_path, OUT / 'archive_inventory.json',
                   Path(__file__), Path(__file__).with_name('COMBIANT_REANALYSIS_PROTOCOL.md'))},
        'software_rows': len(soft), 'manual_rows': len(human), 'archive_evaluation_images': len(images),
        'paired_rows': len(paired), 'paired_plates': len(plates),
        'images_per_paired_plate': dict(Counter(len(v) for v in plates.values())),
        'paired_user_counts': dict(Counter(r['user'] for r in paired)),
        'unmatched_software': unmatched_soft,
        'unmatched_manual': [{'row_index': i, 'name': r['name']} for i, r in enumerate(human) if r['name'] not in matched],
        'duplicate_software_names': [n for n, c in soft_counts.items() if c > 1],
        'duplicate_manual_names': [n for n, c in human_counts.items() if c > 1],
        'duplicate_plate_user_assignments': duplicate_plate_users,
        'software_without_archive_image': [r['name'] for r in soft if 'inference/' + r['name'] not in images],
        'archive_image_without_software': sorted(set(images) - {'inference/' + r['name'] for r in soft}),
        'pairs': paired, 'example_plates': examples,
        'example_image_members': [r['member'] for p in examples for r in sorted(plates[p], key=lambda r: r['user'])],
        'limitations': ['Plate is a repeated physical assay identifier, not a unique patient or isolate.',
                       'Existing software outputs are original author results, not our own retraining.']}
    (OUT / 'pairing_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'binding', 'limitations', 'example_image_members')}))


if __name__ == '__main__':
    main()
