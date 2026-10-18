"""Actual locked MALDI intensities, selected by code before model-result review."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'advanced_figures/msdeepamr_spectra_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assets = json.loads((ROOT / 'msdeepamr_assets.json').read_text())
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': .65,
                        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.79, wspace=.18, hspace=.48)
    masses = np.arange(2000, 20000, 3)
    examples = []
    names = [('e_coli', 'Escherichia coli'), ('k_pneumoniae', 'Klebsiella pneumoniae'),
             ('s_aureus', 'Staphylococcus aureus')]
    colors = ['#206B83', '#A75643']
    for i, (species, name) in enumerate(names):
        for j, cohort in enumerate(('driams_a', 'driams_b')):
            dataset = next(d for d in assets['datasets'] if d['cohort'] == cohort and
                           Path(d['path']).name.startswith(species + '_driams_'))
            p = Path(dataset['path'])
            assert sha(p) == dataset['sha256']
            frame = pd.read_csv(p)
            index = frame.code.astype(str).sort_values(kind='stable').index[0]
            raw = frame.loc[index, [str(m) for m in masses]].to_numpy(np.float32)
            intensity = raw / max(float(raw.max()), 1.)
            assert np.isfinite(raw).all() and (raw >= 0).all()
            ax = axes[j, i]
            ax.plot(masses / 1000, intensity, color=colors[j], lw=.6)
            ax.fill_between(masses / 1000, 0, intensity, color=colors[j], alpha=.10, linewidth=0)
            ax.set(xlim=(2, 20), ylim=(0, 1.055), xticks=[2, 6, 10, 14, 18], yticks=[0, .5, 1])
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_color('#BCC5CB')
            ax.tick_params(color='#BCC5CB', length=3, labelcolor='#39434A')
            code = str(frame.loc[index, 'code'])
            ax.set_title(('Source A' if j == 0 else 'Target B') + f'  ·  original row {index}',
                         loc='left', fontsize=9, color=colors[j], pad=12)
            ax.text(0, 1.16, chr(ord('a') + j * 3 + i), transform=ax.transAxes,
                    weight='bold', fontsize=13)
            if j == 0:
                ax.text(.5, 1.32, name, transform=ax.transAxes, ha='center', fontstyle='italic', fontsize=11)
            if i == 0:
                ax.set_ylabel('Relative intensity')
            if j == 1:
                ax.set_xlabel('Mass / charge (10³ Da)')
            label = frame.loc[index, dataset['target']]
            examples.append({'species': species, 'cohort': cohort, 'path': str(p), 'sha256': sha(p),
                             'original_row': int(index), 'sample_code': code,
                             'target': dataset['target'], 'original_label': None if pd.isna(label) else int(label),
                             'mass_Da': masses.tolist(), 'normalized_intensity': intensity.astype(float).tolist()})
    fig.text(.075, .957, 'Real mass spectra across source and target cohorts', fontsize=18, weight='bold', color='#1C313C')
    fig.text(.075, .906, 'DRIAMS author-released measurements  |  6,000 spectral bins  |  3 Da resolution', fontsize=10, color='#59656E')
    fig.text(.075, .065, 'One fixed example per species and cohort: first sample code in lexical order. Labels and model scores do not select examples.',
             fontsize=8.3, color='#59656E')
    fig.text(.075, .037, 'Per-spectrum maximum normalization; all bins shown. These are separate retrospective specimens, not paired isolates or proof of site effects.',
             fontsize=8.3, color='#59656E')
    png, svg = OUT / 'real_mass_spectra.png', OUT / 'real_mass_spectra.svg'
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    source = {'selection_rule': 'lexically_first_sample_code_per_species_cohort; no label or score selection',
              'examples': examples, 'asset_audit_sha256': sha(ROOT / 'msdeepamr_assets.json'),
              'code_sha256': sha(Path(__file__).resolve()), 'figure_sha256': sha(png), 'svg_sha256': sha(svg),
              'scientific_claim': 'actual recorded spectral inputs; illustrative examples only'}
    (OUT / 'source.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'generated_pending_independent_validation_and_visual_review', 'examples': 6,
                      'png': str(png), 'figure_sha256': sha(png)}))


if __name__ == '__main__':
    main()
