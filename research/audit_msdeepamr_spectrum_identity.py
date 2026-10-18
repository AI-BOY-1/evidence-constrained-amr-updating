"""Audit literal spectral identities; never infer patient identity from spectra."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'msdeepamr_assets.json'
FEATURES = [str(i) for i in range(2000, 20000, 3)]


def main():
    assets = json.loads(ASSETS.read_text())
    rows = []
    by_species = defaultdict(dict)
    inputs = {str(ASSETS): hashlib.sha256(ASSETS.read_bytes()).hexdigest(),
              str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for dataset in assets['datasets']:
        p = Path(dataset['path'])
        raw_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert raw_hash == dataset['sha256']
        inputs[str(p)] = raw_hash
        df = pd.read_csv(p)
        species = p.name.split('_driams_')[0]
        x = df[FEATURES].to_numpy(np.float32)
        finite = np.isfinite(x).all(axis=1)
        assert not (x[finite] < 0).any()
        maxima = np.max(x[finite], axis=1)
        normalized = x[finite] / np.where(maxima > 0, maxima, 1)[:, None]
        # Positive zero canonicalization; no rounded or approximate identity.
        normalized[normalized == 0] = 0
        fingerprints = defaultdict(list)
        for index, spectrum in zip(df.index[finite], normalized):
            key = hashlib.sha256(spectrum.astype('<f4').tobytes()).hexdigest()
            label = df.loc[index, dataset['target']]
            fingerprints[key].append({'code': str(df.loc[index, 'code']), 'original_row': int(index),
                                      'label': None if pd.isna(label) else int(label)})
        repeated = [v for v in fingerprints.values() if len(v) > 1]
        row = {'species': species, 'cohort': dataset['cohort'], 'target': dataset['target'],
               'rows': len(df), 'nonfinite_spectrum_rows': int((~finite).sum()),
               'zero_spectrum_rows': int((maxima == 0).sum()), 'unique_spectral_fingerprints': len(fingerprints),
               'repeated_fingerprint_groups': len(repeated), 'repeated_groups': repeated,
               'discordant_label_groups': sum(len({r['label'] for r in v if r['label'] is not None}) > 1
                                              for v in repeated)}
        rows.append(row)
        by_species[species][dataset['cohort']] = fingerprints
    overlaps = []
    for species, cohorts in sorted(by_species.items()):
        a, b = cohorts['driams_a'], cohorts['driams_b']
        common = sorted(set(a) & set(b))
        overlaps.append({'species': species, 'exact_normalized_overlap_groups': len(common),
                         'target_rows_in_overlap': sum(len(b[k]) for k in common),
                         'groups': [{'fingerprint': k, 'A': a[k], 'B': b[k]} for k in common]})
    result = {'status': 'passed', 'datasets': rows, 'source_target_overlap': overlaps,
              'input_sha256': inputs, 'normalization': 'float32 per-spectrum maximum; exact SHA256 without rounding',
              'limitations': ['Exact spectra are an input-content audit, not patient or laboratory identity.',
                              'No-overlap does not rule out same-patient repeat acquisition or approximate spectra.',
                              'Author original source-training membership and 6001st declared coordinate are unresolved.']}
    out = ROOT / 'msdeepamr_spectrum_identity_audit.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# MSDeepAMR 谱图内容身份核查', '',
             '全部六CSV输入哈希重核对通过。逐谱最大值归一化后以float32精确字节哈希比较，不使用标签选择近似阈值。', '',
             '|物种|队列|总行数|非有限谱|零谱|重复谱组|重复组标签冲突|', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['species']}|{r['cohort']}|{r['rows']}|{r['nonfinite_spectrum_rows']}|{r['zero_spectrum_rows']}|{r['repeated_fingerprint_groups']}|{r['discordant_label_groups']}|")
    lines += ['', '|物种|A/B完全相同归一化谱组|涉及B行数|', '|---|---:|---:|']
    for r in overlaps:
        lines.append(f"|{r['species']}|{r['exact_normalized_overlap_groups']}|{r['target_rows_in_overlap']}|")
    lines += ['', *('- ' + note for note in result['limitations'])]
    (ROOT / 'MSDEEPAMR_SPECTRUM_IDENTITY_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'datasets': len(rows), 'overlap': overlaps}))


if __name__ == '__main__':
    main()
