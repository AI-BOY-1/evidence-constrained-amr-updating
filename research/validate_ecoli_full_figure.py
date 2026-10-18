"""Check figure source values against frozen sequence and prediction artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ecoli_clusters import ROOT

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'advanced_figures'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_path = FIGURES / 'ecoli_genome_repair_landscape_source.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    matrix = ROOT / 'ecoli_assemblies_jaccard.csv'
    assert sha(matrix) == source['matrix_sha256']
    with matrix.open(encoding='utf-8', newline='') as f:
        names = [Path(p).stem for p in next(csv.reader(f))]
    assert names == source['sequence_ids'] and len(names) == len(set(names))
    coordinates = np.array(source['coordinates'])
    assert coordinates.shape == (len(names), 2) and np.isfinite(coordinates).all()
    with np.load(ROOT / 'ecoli_jaccard_profile_embedding.npz', allow_pickle=False) as cache:
        assert str(cache['matrix_sha256']) == source['matrix_sha256']
        assert np.array_equal(cache['sequence_ids'], names)
        assert np.array_equal(cache['coordinates'], coordinates)
    drug, seed = source['example_drug'], source['example_seed']
    threshold, budget = source['example_threshold'], source['example_budget']
    path = ROOT / 'repair_full_scaled5000' / drug.replace('/', '-') / f'{drug.replace("/", "-")}_t{threshold}_decision_diagnostic_{seed}.npz'
    assert sha(path) == source['prediction_sha256']
    with np.load(path, allow_pickle=False) as predictions:
        assert predictions['evaluation_samples'].tolist() == source['evaluation_samples']
        assert predictions['labels'].tolist() == source['evaluation_labels']
        improvement = (predictions['p_b0'] - predictions['labels']) ** 2 - (predictions[f'p_b{budget}'] - predictions['labels']) ** 2
        assert np.array_equal(improvement, source['heldout_squared_loss_improvement'])
        assert set(source['evaluation_samples']) <= set(names)
        assert set(source['queried_samples']) <= set(predictions['query_pool_samples'])
        assert not (set(source['queried_samples']) & set(source['evaluation_samples']))
    trace = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    snapshot = next(r for r in trace['results'] if r['budget'] == budget)
    assert snapshot['queried_samples'] == source['queried_samples']
    assert len(set(source['queried_samples'])) == budget
    analysis_path = HERE / 'ecoli_full_analysis.json'
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    contrasts = [r for r in analysis['paired_contrasts'] if r['threshold'] == threshold and r['seed'] == seed and r['budget'] == budget]
    assert contrasts == source['contrasts']
    figure = FIGURES / 'ecoli_genome_repair_landscape.png'
    with Image.open(figure) as raster:
        raster.load()
        size = list(raster.size)
        assert min(size) >= 2000
    paths = [FIGURES / f'ecoli_genome_repair_landscape.{ext}' for ext in ('png', 'pdf', 'svg')]
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
    report = {'status': 'complete', 'sequence_n': len(names), 'evaluation_n': len(source['evaluation_samples']),
              'queried_n': budget, 'contrast_rows': len(contrasts), 'pixel_dimensions': size,
              'figure_sha256': sha(figure), 'source_sha256': sha(source_path), 'analysis_sha256': sha(analysis_path),
              'export_sha256': {str(p): sha(p) for p in paths}, 'source_predictions_match': True,
              'query_evaluation_disjoint': True, 'embedding_cache_match': True,
              'visual_review': 'pending', 'limitations': 'Automated source correspondence does not verify visual layout or biological mechanism.'}
    (HERE / 'ecoli_full_figure_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'sequence_n', 'evaluation_n', 'queried_n')}), flush=True)


if __name__ == '__main__':
    main()
