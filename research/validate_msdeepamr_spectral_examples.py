"""Replay fixed illustrative samples and every plotted spectral bin."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'advanced_figures/msdeepamr_spectra_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = json.loads((OUT / 'source.json').read_text())
    assert source['asset_audit_sha256'] == sha(ROOT / 'msdeepamr_assets.json')
    assert source['code_sha256'] == sha(ROOT / 'make_msdeepamr_spectral_examples.py')
    assert source['figure_sha256'] == sha(OUT / 'real_mass_spectra.png')
    assert source['svg_sha256'] == sha(OUT / 'real_mass_spectra.svg')
    expected = {(s, c) for s in ('e_coli', 'k_pneumoniae', 's_aureus') for c in ('driams_a', 'driams_b')}
    assert {(e['species'], e['cohort']) for e in source['examples']} == expected
    assert len(source['examples']) == 6
    for item in source['examples']:
        p = Path(item['path'])
        assert item['sha256'] == sha(p)
        df = pd.read_csv(p)
        index = df.code.astype(str).sort_values(kind='stable').index[0]
        assert int(index) == item['original_row'] and str(df.loc[index, 'code']) == item['sample_code']
        assert item['mass_Da'] == list(range(2000, 20000, 3))
        x = df.loc[index, [str(m) for m in item['mass_Da']]].to_numpy(np.float32)
        x /= max(float(x.max()), 1.)
        assert np.array_equal(x.astype(float), np.array(item['normalized_intensity']))
        label = df.loc[index, item['target']]
        assert item['original_label'] == (None if pd.isna(label) else int(label))
    result = {'status': 'data_and_export_validation_passed', 'examples_checked': 6, 'spectral_bins_replayed': 36000,
              'source_sha256': sha(OUT / 'source.json'), 'figure_sha256': sha(OUT / 'real_mass_spectra.png'),
              'svg_sha256': sha(OUT / 'real_mass_spectra.svg'), 'visual_review': {'status': 'pending'},
              'limitations': ['Fixed illustrative examples do not demonstrate population shift or model performance.']}
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
