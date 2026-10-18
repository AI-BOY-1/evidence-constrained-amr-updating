"""Read-only artifact/data verification; never fits models or reruns experiments."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figures'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(name):
    return read(OUT / 'source_data' / (name + '.json'))['data']


def check_means(raw, exported, keys, fields, expected_runs=5):
    grouped = defaultdict(list)
    for r in raw:
        grouped[tuple(r[k] for k in keys)].append(r)
    assert len(exported) == len(grouped)
    seen = set()
    max_error = 0.0
    for r in exported:
        key = tuple(r[k] for k in keys)
        assert key not in seen
        seen.add(key)
        original = grouped[key]
        assert len(original) == r['runs'] == expected_runs
        for field in fields:
            values = [float(x[field]) for x in original if x.get(field) is not None]
            assert r[field + '_available_runs'] == len(values)
            if not values:
                assert r[field] is None
                continue
            error = abs(math.fsum(values) / len(values) - r[field])
            assert error < 1e-10, (key, field, error)
            max_error = max(max_error, error)
    return {'groups': len(grouped), 'max_absolute_mean_error': max_error}


def main():
    manifest = read(OUT / 'figure_manifest.json')
    assert (manifest['main_figures'], manifest['supplement_figures'], manifest['tables']) == (6, 10, 12)
    assert manifest['script_sha256'] == sha(ROOT / 'make_paper_figures.py')
    frozen = read(ROOT / 'completion_manifest.json')
    for name, expected in manifest['input_hashes'].items():
        assert sha(ROOT / name) == expected == frozen['hashes'][name]
    for name, expected in manifest['artifact_hashes'].items():
        assert sha(OUT / name) == expected, name
    assert len(list(OUT.glob('*.png'))) == 16
    assert len(list(OUT.glob('*.pdf'))) == 16
    assert len(list(OUT.glob('*.svg'))) == 16
    assert len(list((OUT / 'source_data').glob('*.json'))) == 16
    images = []
    for fig in manifest['figures']:
        name = fig['id']
        with Image.open(OUT / (name + '.png')) as im:
            assert im.format == 'PNG'
            dpi = im.info.get('dpi')
            assert dpi and all(abs(x - 350) < .1 for x in dpi)
            images.append({'figure': name, 'pixels': list(im.size), 'dpi': list(dpi)})
            im.verify()
        assert (OUT / (name + '.pdf')).read_bytes().startswith(b'%PDF-')
        assert ElementTree.parse(OUT / (name + '.svg')).getroot().tag.endswith('svg')
        data = read(OUT / 'source_data' / (name + '.json'))
        assert data['figure'] == name and data['sources'] == fig['sources']
    assert len(list((OUT / 'tables').glob('*.json'))) == 12
    for t in manifest['table_inventory']:
        assert len(read(OUT / 'tables' / (t['id'] + '.json'))['rows']) == t['rows']
        for ext in ['md', 'html']:
            assert (OUT / 'tables' / (t['id'] + '.' + ext)).is_file()

    internal = read(ROOT / 'decision_repair_results.json')['results']
    external = read(ROOT / 'external_transport_results.json')['results']
    auxiliary = read(ROOT / 'auxiliary_information_results.json')['results'] + read(ROOT / 'auxiliary_tree_results.json')['results']
    checks = {}
    hm = source('Fig02_all_strategy_brier')
    checks['internal_heatmap'] = check_means([r for r in internal if r['budget'] == 25], hm['internal'], ['protocol', 'drug', 'strategy'], ['brier'])
    checks['external_heatmap'] = check_means(external, hm['external'], ['internal_protocol', 'drug', 'strategy'], ['brier'])
    checks['auxiliary_information'] = check_means([r for r in auxiliary if r['budget'] == 25], source('Fig04_auxiliary_information_value')['means'], ['protocol', 'drug', 'mode'], ['brier', 'extra_training_assay_labels', 'extra_evaluation_assay_labels'])
    curves = source('S06_auxiliary_budget_curves')
    checks['auxiliary_curves'] = check_means(auxiliary, curves, ['protocol', 'drug', 'mode', 'budget'], ['brier'])
    plotted = defaultdict(set)
    for r in curves:
        eligible = [0, 10, 25, 50] if r['protocol'] == 'forward_2019' else [0, 10, 25]
        if r['budget'] in eligible:
            plotted[(r['protocol'], r['mode'], r['budget'])].add(r['drug'])
    assert len(plotted) == 28 and all(len(drugs) == 9 for drugs in plotted.values())
    checks['macro_curve_composition'] = {'plotted_groups': 28, 'drugs_per_group': 9}
    contrasts = read(ROOT / 'completion_analysis.json')['contrasts']
    expected = [r for r in contrasts if r['domain'] in ('internal', 'external') and (r['left'], r['right']) == ('decision_diagnostic', 'random_gene')]
    assert source('Fig03_paired_project_intervals') == expected
    expected = [r for r in contrasts if r['domain'] == 'auxiliary_information' and (r['left'], r['right']) in [('gene_auxiliary_AST', 'gene'), ('tree_gene_auxiliary_AST', 'tree_gene')]]
    assert source('Fig04_auxiliary_information_value')['contrasts'] == expected
    assert source('S04_typed_transport')['contrasts'] == [r for r in contrasts if r['domain'] == 'typed_transport']
    checks['paired_intervals'] = 'exactly match existing analysis; no new inference'
    cohort = read(OUT / 'tables' / 'Table01_external_cohort.json')['rows']
    assert len(cohort) == 9 and sum(r['n'] for r in cohort) == 16687
    assert all(r['S'] + r['R'] == r['n'] for r in cohort)
    checks['external_cohort'] = {'drug_tasks': 9, 'drug_sample_pairs': 16687, 'S_plus_R_equals_n': True}
    report = {'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(OUT / 'figure_manifest.json'), 'artifact_hashes_verified': len(manifest['artifact_hashes']), 'input_hashes_verified': len(manifest['input_hashes']), 'figures': 16, 'tables': 12, 'images': images, 'source_checks': checks, 'scope': 'Artifact integrity and exported summaries only. No model fitting, experiment reruns, or additional hypothesis tests. PDF header checked; PNGs visually inspected. This report is separate from the artifact manifest.'}
    (OUT / 'figure_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
