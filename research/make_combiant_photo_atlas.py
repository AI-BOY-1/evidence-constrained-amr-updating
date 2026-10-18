"""Uncropped source photographs: fixed two plates, three acquisition users."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from audit_combiant_pairing import OUT, SUBSET, sha

FIG = Path(__file__).resolve().parent / 'advanced_figures/combiant_photos_v1'


def main():
    FIG.mkdir(exist_ok=True)
    pairing = json.loads((OUT / 'pairing_audit.json').read_text())
    acquisition = json.loads((OUT / 'image_acquisition_manifest.json').read_text())
    with (OUT / 'paired_distances.csv').open(newline='') as stream:
        distances = list(csv.DictReader(stream))
    samples = [p for plate in pairing['example_plates']
               for p in sorted(pairing['pairs'], key=lambda p: p['user']) if p['plate'] == plate]
    assert len(samples) == 6
    source = {'role': 'Public photo evidence for exploratory reanalysis of published measurements',
        'article_doi': '10.1371/journal.pdig.0000669', 'data_doi': '10.5281/zenodo.15005285',
        'attribution': 'Hallstrom et al., PLOS Digital Health (2025); CC BY 4.0',
        'selection': 'Smallest two plate IDs with exactly three uniquely paired users and archive inputs.',
        'transformations': 'Display scaling only; full source image extent; no added measurement overlays.',
        'binding': {str(p): sha(p) for p in (OUT / 'pairing_audit.json', OUT / 'paired_distances.csv', Path(__file__))},
        'samples': [], 'limitations': ['Two physical plates, six acquisition views; not six independent isolates.',
            'Published inputs may already include author cropping/rotation/preprocessing.',
            'Not linked to our genomic cohort; no new image-model performance claim.']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.facecolor': '#ffffff'})
    fig = plt.figure(figsize=(15, 11.5), facecolor='white')
    fig.text(.045, .953, 'One assay, three acquisitions', fontsize=25, fontweight='bold', color='#162932')
    fig.text(.045, .922, 'Real CombiANT photographs  |  Two plates, six views  |  Public-data reanalysis',
             fontsize=12, color='#54666e')
    gs = fig.add_gridspec(2, 3, left=.045, right=.965, top=.875, bottom=.095, wspace=.045, hspace=.22)
    for idx, pair in enumerate(samples):
        path = SUBSET / pair['member']
        assert sha(path) == acquisition['members'][pair['member']]['sha256']
        with Image.open(path) as original:
            original.load(); pixels = original.copy()
            size = list(original.size)
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.imshow(pixels, interpolation='antialiased')
        ax.set_anchor('N')
        ax.set_axis_off()
        ax.set_title(f"{chr(97+idx)}  Plate {pair['plate']} / User {pair['user']}",
                     loc='left', fontsize=12, fontweight='bold', color='#162932', pad=9)
        ax.text(0, -.035, path.name, transform=ax.transAxes, fontsize=8.5, color='#54666e', va='top')
        measurements = [r for r in distances if r['name'] == pair['name']]
        assert len(measurements) == 6
        source['samples'].append({**pair, 'image_path': str(path), 'sha256': sha(path),
                                  'pixel_size': size, 'measurements': measurements})
    fig.text(.045, .048, 'Images: Hallstrom et al. (2025), CC BY 4.0  |  Zenodo 15005285  |  Full image extent retained',
             fontsize=9, color='#54666e')
    fig.text(.045, .027, 'Repeated views of the same plate. Published inputs may be preprocessed. No genomic linkage or new model result.',
             fontsize=9, color='#54666e')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / ('combiant_photo_atlas.' + extension), dpi=350)
    plt.close(fig)
    (FIG / 'source.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    with (FIG / 'example_measurements.csv').open('w', newline='') as stream:
        rows = [r for s in source['samples'] for r in s['measurements']]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (FIG / 'README.md').write_text(
        '# CombiANT真实照片图板\n\n原作者Hallström等（2025），PLOS Digital Health，CC BY 4.0。'
        '数据版本Zenodo 15005285。固定选择最小两个具有三用户唯一配对的培养皿编号1和2。'
        '6幅完整照片只在版面中缩放，未加伪造分割或测量线。\n\n'
        '这是2个培养皿的6个视图；不是6株独立菌株、不是本项目基因组配对，也不是新模型输出。'
        '原作者输入中可能已有旋转/裁切等处理。36条配对距离见example_measurements.csv；全队列计数以独立验证为准。\n', encoding='utf-8')
    print(str(FIG / 'combiant_photo_atlas.png'))


if __name__ == '__main__':
    main()
