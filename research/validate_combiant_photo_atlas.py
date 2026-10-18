"""Audit source photos, measurement rows, immutable inputs and figure exports."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'advanced_figures/combiant_photos_v1'
BASE = ncs_paths.data_path('experiments/combiant_public_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads((FIG / 'source.json').read_text())
    pairing = json.loads((BASE / 'pairing_audit.json').read_text())
    for raw, value in source['binding'].items():
        assert sha(Path(raw)) == value
    assert [p['member'] for p in source['samples']] == pairing['example_image_members']
    with (BASE / 'paired_distances.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for sample in source['samples']:
        path = Path(sample['image_path'])
        assert sha(path) == sample['sha256']
        with Image.open(path) as im:
            im.load(); assert list(im.size) == sample['pixel_size']
        assert sample['measurements'] == [r for r in rows if r['name'] == sample['name']]
    svg = ET.parse(FIG / 'combiant_photo_atlas.svg')
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}image')) == 6
    exports = {ext: sha(FIG / ('combiant_photo_atlas.' + ext)) for ext in ('png', 'pdf', 'svg')}
    with Image.open(FIG / 'combiant_photo_atlas.png') as im:
        im.load(); assert im.width >= 5000 and im.height >= 4000
    with (FIG / 'example_measurements.csv').open(newline='') as stream:
        example_rows = list(csv.DictReader(stream))
    assert example_rows == [r for s in source['samples'] for r in s['measurements']]
    visual_path = FIG / 'visual_review.json'
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {'status': 'pending'}
    ready = visual.get('status') == 'passed' and visual.get('figure_sha256') == exports['png']
    validation = {'status': 'passed' if ready else 'source_passed_visual_pending',
        'source_sha256': sha(FIG / 'source.json'), 'physical_plates': 2, 'photographs': 6,
        'paired_measurement_rows': len(example_rows), 'exports': exports,
        'visual_review': visual,
        'limitations': ['Six photographs belong to two plates; no claim of six independent isolates.',
                       'Full 299-image acquisition is a separate audit.']}
    (FIG / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps({'status': validation['status'], 'photographs': 6, 'paired_measurement_rows': len(example_rows)}))


if __name__ == '__main__':
    main()
