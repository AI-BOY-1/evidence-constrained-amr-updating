"""Curate existing evidence without changing experiments or their protocols."""
import ncs_paths  # portable I/O paths; original computational rules preserved
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT = HERE / 'manuscript_package_v1_2026_10_12'
DRUGS = ['meropenem', 'imipenem', 'ertapenem', 'ceftazidime', 'ciprofloxacin',
         'trimethoprim/sulfamethoxazole', 'amikacin', 'gentamicin', 'cefepime']
provenance = []
figures = []
tables = []
checks = []


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def bind(p):
    p = Path(p).resolve()
    if not any(x['source'] == str(p) for x in provenance):
        provenance.append({'source': str(p), 'sha256': sha(p)})
    return p


def copy(p, dest):
    p = bind(p)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    assert sha(p) == sha(dest)
    checks.append({'check': 'byte_identical_copy', 'path': str(dest.relative_to(OUT))})


def blocks(p):
    p = bind(p)
    lines = p.read_text(encoding='utf-8').splitlines()
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'^\|[ :|\-]+\|$', lines[i + 1]):
            rows = []
            headers = [c.strip() for c in lines[i].strip()[1:-1].split('|')]
            i += 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip()[1:-1].split('|')])
                i += 1
            assert all(len(r) == len(headers) for r in rows), str(p)
            result.append((headers, rows))
        else:
            i += 1
    return result


def fmt(v):
    if isinstance(v, float):
        return f'{v:+.6f}'
    return str(v)


def table_html(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join('<th>' + html.escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(fmt(v)) + '</td>' for v in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def table_md(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in rows])


def add_table(identifier, title, sections, notes, rationale, sources):
    category = 'main_tables' if identifier.startswith('Table0') else 'supplementary_tables'
    stem = OUT / category / identifier
    md = '# ' + identifier + '. ' + title + '\n\n'
    body = '<h1>' + html.escape(identifier + '. ' + title) + '</h1>'
    for heading, headers, rows in sections:
        md += '## ' + heading + '\n\n' + table_md(headers, rows) + '\n\n'
        body += '<h2>' + html.escape(heading) + '</h2>' + table_html(headers, rows)
    md += 'Notes: ' + notes + '\n\n'
    body += '<p class="note"><strong>Notes.</strong> ' + html.escape(notes) + '</p>'
    for p in sources:
        bind(p)
    stem.with_suffix('.md').write_text(md, encoding='utf-8')
    stem.with_suffix('.html').write_text(page(title, body), encoding='utf-8')
    tables.append({'id': identifier, 'title': title, 'category': category, 'rationale_zh': rationale,
                   'sections': [{'title': h, 'rows': len(r), 'columns': len(c)} for h, c, r in sections],
                   'notes': notes, 'sources': [str(Path(p).resolve()) for p in sources],
                   'path': str(stem.with_suffix('.md').relative_to(OUT))})


CSS = '''body{font:16px/1.65 "Segoe UI",Arial,sans-serif;color:#202e36;margin:0;background:#f5f5f2}
main{max-width:1200px;margin:auto;padding:42px 36px;background:white}h1{font-size:30px;line-height:1.25}
h2{font-size:23px;margin-top:42px}h3{font-size:18px}a{color:#096b78}img{width:100%;height:auto}
figure{margin:35px 0;border-top:1px solid #d4dcdc;padding-top:24px}figcaption{font-size:14px;color:#39474f}
.note{font-size:14px;color:#47575f}.tablewrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;font-size:13px;width:100%}
thead{border-top:2px solid #243841;border-bottom:1px solid #243841}th,td{padding:9px 10px;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f4f6f6}tbody{border-bottom:2px solid #243841}
.nav{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0}.card{padding:18px;border:1px solid #dce2e2;margin:15px 0}
@media print{body{background:white}main{padding:0}figure{break-inside:avoid}.tablewrap{overflow:visible}}
'''


def page(title, body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>' + CSS + '</style><main>' + body + '</main></html>'


def add_figure(identifier, title, rationale, legend, source=None, category='main_figures'):
    stem = OUT / category / identifier
    if source:
        for ext in ('png', 'pdf', 'svg'):
            p = Path(source).with_suffix('.' + ext)
            if p.exists():
                copy(p, stem.with_suffix('.' + ext))
    assert stem.with_suffix('.png').exists()
    figures.append({'id': identifier, 'title': title, 'rationale_zh': rationale, 'legend': legend,
                    'category': category, 'path': str(stem.with_suffix('.png').relative_to(OUT)),
                    'formats': [x for x in ('png', 'pdf', 'svg') if stem.with_suffix('.' + x).exists()]})


def export_plot(fig, identifier, plot_data):
    stem = OUT / 'main_figures' / identifier
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=400, facecolor='white')
    plt.close(fig)
    (OUT / 'source_data' / (identifier + '.json')).write_text(json.dumps(plot_data, ensure_ascii=False), encoding='utf-8')


def short(drug):
    return 'TMP–SMX' if '/' in drug else drug.capitalize()


def axes_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def new_figures(raw, acquisition):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.titlesize': 10, 'axes.linewidth': .65, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    adjacent = [r for r in raw['transitions'] if r['adjacent']]
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.1))
    fig.subplots_adjust(left=.10, right=.88, top=.87, bottom=.10, hspace=.57, wspace=.48)
    fig.text(.10, .966, 'Model updating changes both loss and error direction', size=17, weight='bold')
    fig.text(.10, .929, 'ANI-screened cohort · both inputs and thresholds · all adjacent budget transitions', size=9, color='#536572')
    colors = {'homolog': '#167e83', 'sketch': '#79558b'}
    rate_counts = {}
    for feature, color in colors.items():
        rr = [r for r in adjacent if r['feature'] == feature]
        rate = [r for r in rr if r['R'] > 0]
        rate_counts[feature] = {'all': len(rr), 'R_positive': len(rate), 'R_zero_omitted_from_rate': len(rr) - len(rate)}
        ax[0, 0].scatter([r['brier_delta'] for r in rate], [100*(r['FS_after']-r['FS_before'])/r['R'] for r in rate], s=12, color=color, alpha=.32, lw=0, label=feature, rasterized=True)
        ax[0, 1].scatter([r['corrected'] for r in rr], [r['introduced'] for r in rr], s=12, color=color, alpha=.30, lw=0, rasterized=True)
    ax[0, 0].axhline(0, c='#a0a5aa', lw=.6); ax[0, 0].axvline(0, c='#a0a5aa', lw=.6)
    ax[0, 0].set(title='a  Lower mean loss can coexist with more false susceptibility', xlabel='Brier change (next − previous)', ylabel='False-susceptible rate change (percentage points)')
    ax[0, 0].legend(frameon=False, fontsize=8)
    upper = max(max(r['corrected'], r['introduced']) for r in adjacent)
    ax[0, 1].plot([0, upper], [0, upper], '--', c='#999da3', lw=.7)
    ax[0, 1].set(title='b  Error correction and introduction coexist', xlabel='Errors corrected', ylabel='Errors introduced')
    gate_counts = {}
    for rule, color, marker in [('brier_only', '#2e82ad', 'o'), ('brier_and_FS', '#c96e47', '^')]:
        rr = [r for r in raw['gates'] if r['rule'] == rule and r['status'] == 'complete' and r['selected_budget'] < r['consumed_budget']]
        rate = [r for r in rr if r['R'] > 0]
        gate_counts[rule] = {'rollbacks': len(rr), 'R_positive': len(rate), 'R_zero_omitted_from_rate': len(rr) - len(rate)}
        ax[1, 0].scatter([r['brier_delta'] for r in rate], [100*r['FS_delta']/r['R'] for r in rate], color=color, marker=marker, s=14, lw=0, alpha=.35, label=rule.replace('_', ' '), rasterized=True)
    ax[1, 0].axhline(0, c='#a0a5aa', lw=.6); ax[1, 0].axvline(0, c='#a0a5aa', lw=.6)
    ax[1, 0].set(title='c  Audit-based rollback has mixed target outcomes', xlabel='Brier change (rollback − same-cost update)', ylabel='False-susceptible rate change (percentage points)')
    ax[1, 0].legend(frameon=False, fontsize=8)
    mat = np.full((9, 2), np.nan)
    cells = []
    for i, drug in enumerate(DRUGS):
        for j, rule in enumerate(('brier_only', 'brier_and_FS')):
            rr = [r for r in raw['gate_summary'] if r['drug'] == drug and r['threshold'] == .99 and r['feature'] == 'sketch' and r['strategy'] == 'decision_diagnostic' and r['rule'] == rule]
            if rr:
                mat[i, j] = rr[0]['mean_brier_delta']; cells.append(rr[0])
    lim = max(abs(mat[np.isfinite(mat)]))
    im = ax[1, 1].imshow(mat, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(9):
        for j in range(2):
            v = mat[i, j]
            ax[1, 1].text(j, i, f'{v:+.4f}' if np.isfinite(v) else 'n.a.', ha='center', va='center', fontsize=8, color='white' if abs(v) > .65*lim else '#26333b')
    ax[1, 1].set(yticks=range(9), yticklabels=[short(d) for d in DRUGS], xticks=[0, 1], xticklabels=['Brier only', 'Brier and FS'], title='d  All drugs at ANI 0.99, sketch input, controller')
    ax[1, 1].tick_params(length=0)
    fig.colorbar(im, ax=ax[1, 1], fraction=.035, pad=.04).set_label('Mean Brier: rollback − update', size=8)
    for a in ax.flat: axes_style(a)
    fig.text(.10, .035, 'Repeated configurations share a retrospective cohort. FS rates exclude R = 0; all zero-R cases remain in tables.', fontsize=8)
    export_plot(fig, 'Fig03_update_error_dynamics', {'adjacent': adjacent, 'gates': raw['gates'], 'display_cells': cells, 'rate_denominator_counts': rate_counts, 'gate_rate_counts': gate_counts})
    checks.append({'check': 'Fig03_full_adjacent_coverage', 'transitions': len(adjacent), 'rate_counts': rate_counts, 'gate_counts': gate_counts})

    pairpath = bind(DATA / 'audit_acquisition_action_v1/sampling_pairs.csv')
    with pairpath.open(encoding='utf-8-sig', newline='') as f: pairs = list(csv.DictReader(f))
    direction = [r for r in pairs if r['rule'] == 'audit_direction']
    for acquisition_name in ('predicted_R', 'component_coverage'):
        rr = [r for r in direction if r['acquisition'] == acquisition_name]
        summary = next(r for r in acquisition['global_sampling_summary'] if r['rule'] == 'audit_direction' and r['acquisition'] == acquisition_name)
        assert len(rr) == summary['paired_configurations'] == 5808
        assert math.isclose(math.fsum(float(r['delta']) for r in rr)/len(rr), summary['mean_brier_delta'], abs_tol=1e-12)
        assert sum(float(r['delta']) > 0 for r in rr) == summary['higher_loss']
        assert sum(int(r['FS_delta']) > 0 for r in rr) == summary['FS_increased']
    checks.append({'check': 'acquisition_matched_pairs_numeric_replay', 'pairs': len(direction)})
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.1))
    fig.subplots_adjust(left=.12, right=.97, top=.87, bottom=.10, hspace=.58, wspace=.40)
    fig.text(.10, .966, 'Changing audit acquisition yields gains and residual failures', size=16, weight='bold')
    fig.text(.10, .929, 'Frozen candidate predictions · identical query + audit label cost within each comparison', size=9, color='#536572')
    means = acquisition['drug_sampling_summary']
    for acquisition_name, color, offset, label in [('predicted_R', '#167e83', -.10, 'Predicted resistance'), ('component_coverage', '#c96e47', .10, 'Component coverage')]:
        rr = [next(r for r in means if r['drug'] == d and r['rule'] == 'audit_direction' and r['acquisition'] == acquisition_name) for d in DRUGS]
        ax[0, 0].scatter([r['mean_brier_delta'] for r in rr], np.arange(9)+offset, color=color, s=28, label=label)
    ax[0, 0].axvline(0, c='#999da3', lw=.7)
    ax[0, 0].set(yticks=range(9), yticklabels=[short(d) for d in DRUGS], xlabel='Mean Brier difference versus random acquisition', title='a  Complete drug-wise direction-rule comparisons')
    ax[0, 0].invert_yaxis(); ax[0, 0].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5, 1.20), ncol=2)
    labels = {'audit_direction': 'Audit direction', 'audit_loss': 'Audit loss', 'component_direction': 'Component direction', 'retain': 'Retain', 'update': 'Update'}
    for i, r in enumerate(acquisition['global_rule_summary']):
        x, y = r['mean_brier_delta_from_frozen'], r['eval_FS_increased']
        ax[0, 1].scatter(x, y, color=['#167e83', '#2e82ad', '#79558b', '#818a92', '#c96e47'][i], s=38)
        offsets = {'audit_direction': (9, -17), 'component_direction': (9, 9), 'audit_loss': (8, 6), 'update': (8, 4), 'retain': (-36, 9)}
        ax[0, 1].annotate(labels[r['rule']], (x, y), xytext=offsets[r['rule']], textcoords='offset points', fontsize=8)
    ax[0, 1].set(xlabel='Mean Brier change versus frozen model', ylabel='Configurations with net FS increase', title='b  Probability loss and false susceptibility trade off')
    ax[0, 1].set_xlim(-.0055, .001)
    for a, acquisition_name, color, title in [(ax[1, 0], 'predicted_R', '#167e83', 'c  Predicted-resistance acquisition versus random'), (ax[1, 1], 'component_coverage', '#c96e47', 'd  Component-coverage acquisition versus random')]:
        rr = [r for r in direction if r['acquisition'] == acquisition_name]
        a.scatter([float(r['delta']) for r in rr], [int(r['FS_delta']) for r in rr], s=12, color=color, alpha=.22, lw=0, rasterized=True)
        a.axhline(0, c='#999da3', lw=.6); a.axvline(0, c='#999da3', lw=.6)
        a.set(title=title + '\nAll 5,808 matched configurations; audit direction rule', xlabel='Brier difference versus random acquisition', ylabel='False-susceptible count difference')
    for a in ax.flat: axes_style(a)
    fig.text(.10, .035, 'Configuration counts describe shared data, not independent events. Rejected updates still pay the full query and audit cost.', fontsize=8)
    export_plot(fig, 'Fig04_audit_acquisition', {'drug_summary': means, 'rule_summary': acquisition['global_rule_summary'], 'direction_pairs': direction})


def main():
    for sub in ('main_figures', 'supplementary_figures', 'main_tables', 'supplementary_tables', 'source_data', 'provenance'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    valnames = [HERE/'ecoli_frozen_query_validation.json', HERE/'ecoli_frozen_query_figure_validation.json', HERE/'ecoli_ani_annotation_validation.json', HERE/'audit_acquisition_analysis_validation.json', HERE/'reliability_controls_validation.json', HERE/'reliability_controls_figure_validation.json', HERE/'msdeepamr_reexecution_validation.json', HERE/'citable_literature_metrics_validation.json', DATA/'repair_dynamics_ani_v1/validation.json', DATA/'audit_acquisition_action_v1/validation.json']
    for p in valnames:
        v = read(bind(p)); assert v['status'].startswith('passed'), str(p)
        copy(p, OUT/'provenance'/p.name)
    rawpath = bind(DATA/'repair_dynamics_ani_v1/results.json')
    raw = read(rawpath)
    assert read(DATA/'repair_dynamics_ani_v1/validation.json')['result_sha256'] == sha(rawpath)
    acqpath = bind(HERE/'audit_acquisition_action_analysis.json'); acquisition = read(acqpath)
    assert read(HERE/'audit_acquisition_analysis_validation.json')['analysis_sha256'] == sha(acqpath)
    for p, h in acquisition['outputs_sha256'].items(): assert sha(bind(p)) == h
    new_figures(raw, acquisition)
    add_figure('Fig01_sequence_evidence', 'Actual loci, nucleotide alignments and linked AST records', '用真实序列及原始药敏记录建立对象感；属于固定实例，不能当机制证实。',
        'Three prespecified homolog-family/drug examples use the first two eligible sequence identifiers in lexical order. The upper row shows actual contig coordinates of qualifying BLAST HSPs relative to their midpoint. The middle row shows alignments of extracted loci in the pinned reference orientation. The lower row shows a fixed central 64-reference-base window; insertions and allele ambiguity remain in the source alignments. Recorded AST labels and available laboratory standards are retained. The NDM-homolog/susceptible record is a linkage and label-context audit flag, not evidence of a new resistance mechanism or loss of NDM function. These six examples are illustrative, not a prevalence estimate.', HERE/'advanced_figures/sequence_evidence_v1/sequence_evidence_portraits.png')
    add_figure('Fig02_query_action_decomposition', 'Matched decomposition of query-path and action contributions', '说明选样收益与动作收益可互相抵消；旧sketch隔离清楚标识，完整九药物表配套。',
        'R is the original random-query sketch logistic model; G retrains the same model on the frozen diagnostic query path; D is the diagnostic controller. Panels a and b show five-partition descriptive mean Brier differences, G−R and D−G, over all nine drugs, two original sketch-component thresholds and three nonzero query budgets. Panel c includes all 270 nonzero-budget matched contrasts; D−R=(G−R)+(D−G) is an algebraic decomposition, not causal identification. Panel d reports resistant-error correction and introduction relative to budget zero at sketch threshold 0.99 and query budget 25. Counts are partition means. The old sketch components are not ANI or SNP isolation. This exploratory ablation retains negative results and does not replace the later ANI-screened analysis.', HERE/'advanced_figures/ecoli_query_action_decomposition.png')
    add_figure('Fig03_update_error_dynamics', 'Error direction and target outcomes of model updating', '全ANI计算版本的完整预算变化；去掉条形外观，保留零R不适用与门控失败。',
        'Panels a and b include every adjacent transition from the 680 validated ANI-screened prediction trajectories (2,040 transitions), both homolog/sketch inputs, both ANI thresholds, all drugs and strategies. Panel a uses only R>0 evaluation sets for false-susceptible rates; zero-R cases remain in tables and source data. Panel b includes all transitions and contrasts old errors corrected with new errors introduced. Panel c shows all feasible rollback decisions with R>0 evaluation sets, relative to updating the identical strategy to its consumed budget. All consumed query and audit labels are charged. Panel d uses the fixed ANI 0.99/sketch/controller display condition, includes all nine drugs, and averages feasible partitions without imputing unavailable values. Repeated partitions share the retrospective cohort. Approximate ANI grouping is not SNP phylogeny or patient/laboratory independence. Lower overall Brier does not guarantee fewer false-susceptible calls.')
    add_figure('Fig04_audit_acquisition', 'Matched audit-acquisition comparisons and residual failures', '新增最新正式产物可视化，覆盖全部九药物及简单保留/更新对照，不只画获益。',
        'Frozen candidate predictions and query paths are reused; this experiment does not retrain candidates. Panel a shows complete drug-wise mean Brier differences versus random audit acquisition for predicted-resistance priority and component coverage under audit_direction. Each arm has 5,808 matched feasible configurations; drug-wise counts differ and are given in Table05. Panel b shows all five action rules across the same 17,424 feasible acquisition conditions per rule, comparing mean Brier with the number of configurations showing a net increase in false susceptibility relative to the budget-zero candidate. Counts include retained fallbacks when direction assessment is infeasible. Panels c and d show every matched direction-rule acquisition comparison, including zero changes and adverse results. Costs equal consumed query labels plus selected audit labels even if updates are rejected. Predictions and allowed unlabeled metadata guide acquisition; audit labels determine action; evaluation labels only score it. These shared-data configuration descriptions are not event rates, significance tests or clinical safety guarantees. All 7,056 unavailable acquisition conditions are retained in TableS08.')
    add_figure('FigS01_own_photo_experiment', 'Real photographs and independently trained image-to-distance models', '独立公开照片实验，作为测量任务扩展；明确自己的模型未胜过原作者专用软件。',
        'Panels a and b show the fixed Plate 1/User 1 and Plate 2/User 1 public CombiANT photographs. Panel c shows held-out ExtraTrees predictions for all six distance targets in five plate-isolated folds, using the complete public cohort of 279 photographs and 98 plates. Panel d is an equal-plate-cost reference-review simulation: reviewed plate predictions are replaced with manual reference distances, without a real prospective review intervention. Random-review bands describe 100 random orderings, not confidence intervals. One A reference is missing; valid denominators differ between macro and distance-level metrics. The published software reference outperforms our pixel-based baselines, as shown in TableS04. Photographs: Hallström et al. (2025), CC BY 4.0, https://doi.org/10.1371/journal.pdig.0000669, Zenodo 15005285. These plates are not paired with genomic or MALDI samples and distance prediction is not S/R prediction.', HERE/'advanced_figures/combiant_own_v1/own_vision_results.png', 'supplementary_figures')
    add_figure('FigS02_actual_mass_spectra', 'Actual source and target MALDI spectral inputs', '真实6000bin输入展示；不用六条示例宣称站点差异已证实。',
        'Six actual author-released DRIAMS spectra, one per species and source A/target B cohort, are selected by the first lexical sample code without consulting labels or predictions. All 6,000 bins are shown at 3 Da resolution after per-spectrum maximum normalization; no smoothing or synthetic peaks are used. Source records retain all 36,000 spectral values and original CSV row identities. These are illustrative unpaired specimens, not evidence of site effects or model performance. Dataset/method source: MSDeepAMR, https://doi.org/10.3389/fmicb.2024.1361795. Local B reexecution results and its limits are reported separately in TableS05.', HERE/'advanced_figures/msdeepamr_spectra_v1/real_mass_spectra.png', 'supplementary_figures')
    add_figure('FigS03_reliability_controls', 'Negative controls and transport stress tests', '全部负对照和不可行格保留，防止只看原任务表现。',
        'Complete fixed-display comparisons at ANI 0.99 and query budget 25 for both inputs and all nine drugs. Seven locked conditions include global label permutation, within-component permutation, within-context feature permutation, two temporal cutoffs and two cross-standard directions. Cell values are controller-minus-random Brier differences and k denotes feasible partitions; grey cells are unavailable and are not zero. The full grid includes 1,440 trajectories, 5,328 metric rows and 1,080 skipped strategy configurations. One permutation per seed is not a significance null distribution. Standard stratification does not identify a causal standards effect. Full comparisons, resistant denominators and all skips accompany TableS02.', DATA/'reliability_controls_v1/reliability_controls_landscape.png', 'supplementary_figures')
    add_figure('FigS04_photo_acquisition_views', 'Repeated genuine views of two fixed plates', '六张真实照片增强采集场景解释；不是六个独立样本。',
        'The two smallest plate identifiers with unique three-user matches (plates 1 and 2) provide six genuine public images. Full published image extents are retained and resized only for layout; no inferred segmentation or fabricated measurement overlays are added. Original author preprocessing may be present. These are six views of two plates, not six independent isolates or new model outputs. Attribution: Hallström et al. (2025), CC BY 4.0, https://doi.org/10.1371/journal.pdig.0000669; Zenodo dataset 15005285. There is no sample-level genomic linkage.', HERE/'advanced_figures/combiant_photos_v1/combiant_photo_atlas.png', 'supplementary_figures')
    figures.sort(key=lambda x: x['id'])

    # All main numeric tables come from bound validated analyses, not manually entered claims.
    full = blocks(HERE/'ECOLI_FULL_RESULTS.md')
    add_table('Table01_cohort', 'Genomic cohort and drug-specific recorded AST support', [('Recorded S/R-labelled genomes', ['Drug', 'Valid labelled sequences', 'Resistant sequences', 'Excluded conflicting sequence groups'], [[r[0], r[1], r[2], r[4]] for r in full[1][1]])],
        'There are 3,516 valid FASTA files; drug-specific labelled subsets overlap and must not be summed as independent genomes or patients. Counts are from the fixed cohort audit. The original sketch component counts are intentionally not relabelled as ANI counts. The updated evaluation uses separately validated approximate ANI grouping. Recorded AST is not harmonized MIC/CLSI evidence. Recovery of 3,259 previously missing sequences is included in the 3,516 total, not additional samples.', '先交代对象、九药物支持度和小耐药分母。', [HERE/'ECOLI_FULL_RESULTS.md', HERE/'ecoli_full_validation.json'])
    ab = blocks(HERE/'ECOLI_FROZEN_QUERY_ABLATION.md')
    add_table('Table02_query_action', 'Matched query-path and action decomposition for every drug', [('Budget 25, original sketch threshold 0.99', ['Drug', 'Query G−R', 'Action D−G', 'Total D−R', 'Lower-total-loss partitions / 5'], ab[0][1])],
        'Negative Brier differences favor the first model. G/R share the same sketch logistic estimator; D is the diagnostic controller. All three share evaluation identities and labels. Displayed values are rounded independently, so displayed components may not sum exactly; the full-precision identity was validated to 2.78e−17. The five partitions reuse one cohort. These original sketch groups are not ANI/SNP isolation. Resistant denominators and directional errors are in TableS07.', '主贡献拆解；TMP–SMX动作抵消选样、头孢他啶动作挽救均保留。', [HERE/'ECOLI_FROZEN_QUERY_ABLATION.md', HERE/'ecoli_frozen_query_validation.json'])
    findings = read(bind(DATA/'repair_dynamics_ani_v1/findings.json'))
    assert findings['result_sha256'] == sha(rawpath)
    gate_rows = []
    for rule, g in findings['gate_counts'].items():
        rr = [r for r in raw['gates'] if r['rule'] == rule and r['status'] == 'complete']
        rejected = [r for r in rr if r['selected_budget'] < r['consumed_budget']]
        assert len(rr) == g['feasible'] and len(rejected) == g['rollback']
        assert sum(r['brier_delta'] > 0 for r in rejected) == g['higher_eval_brier']
        gate_rows.append([rule, g['feasible'], g['infeasible'], g['rollback'], g['lower_eval_brier'], g['higher_eval_brier'], g['equal_eval_brier'], g['increased_eval_FS'], g['decreased_eval_FS']])
    h = findings['headline']
    add_table('Table03_ANI_dynamics', 'ANI-screened update dynamics and matched stopping outcomes', [('Adjacent budget transitions', ['Outcome', 'Configuration count', 'Denominator'], [['Higher Brier', h['brier_worsened'], h['adjacent_transitions']], ['Lower Brier with net FS increase', h['improved_brier_increased_FS'], h['adjacent_transitions']], ['Errors corrected and introduced', h['both_introduced_and_corrected'], h['adjacent_transitions']], ['Any newly introduced FS', h['introduced_FS_transitions'], h['adjacent_transitions']]]), ('Stopping decisions', ['Gate', 'Feasible', 'Unavailable', 'Rollback', 'Lower loss', 'Higher loss', 'Equal loss', 'More FS', 'Fewer FS'], gate_rows)],
        'The 680 trajectories yield 2,040 adjacent transitions. Outcomes overlap. Each gate has 680 configured replays; target outcomes are counted among rollbacks only (528 and 395). Rollback is compared with the same strategy updated to the consumed budget; rejected query labels and audit labels remain charged. More/fewer FS refers to the target evaluation set, not the audit. All R=0 configurations are retained in counts; their FS rate is undefined. ANI is approximate and does not establish SNP or patient independence. Configuration counts are not independent event rates.', '主文保留负结果和门控失败，说明平均指标与关键错误并不等价。', [DATA/'repair_dynamics_ani_v1/results.json', DATA/'repair_dynamics_ani_v1/findings.json'])
    rule_fields = ['rule', 'configurations', 'eligible', 'accepted', 'eval_loss_higher', 'eval_FS_increased', 'any_new_FS', 'lower_loss_more_FS', 'mean_brier_delta_from_frozen']
    sampling_fields = ['acquisition', 'rule', 'paired_configurations', 'lower_loss', 'higher_loss', 'equal_loss', 'FS_increased', 'FS_decreased', 'mean_brier_delta']
    add_table('Table04_audit_rules', 'Complete action-rule and same-cost audit-acquisition comparisons', [('Action versus budget-zero frozen candidate', ['Rule', 'Configs', 'Eligible', 'Updates', 'Higher loss', 'Net FS increase', 'Any new FS', 'Lower loss + more FS', 'Mean ΔBrier'], [[r[k] for k in rule_fields] for r in acquisition['global_rule_summary']]), ('Acquisition versus random, same rule and cost', ['Acquisition', 'Rule', 'Pairs', 'Lower loss', 'Higher loss', 'Equal', 'More FS', 'Fewer FS', 'Mean ΔBrier'], [[r[k] for k in sampling_fields] for r in acquisition['global_sampling_summary']])],
        'All five rules use the same 17,424 feasible acquisition conditions per rule. Direction rules retain the model when audit R=0 and mark the judgment unavailable; these fallbacks remain in the configuration denominator and are not evidence of protection. Acquisitions use only allowed unlabeled information. Each nonrandom acquisition has 5,808 matched comparisons per rule. All rules pay consumed query + selected audit labels, including retain and rejected updates. Retain/update equality across acquisitions is an invariance check, not replication. Means pool correlated and unequally supported configurations; inspect Table05 and full strata. No new candidate training or audit/training budget reallocation is performed.', '把简单对照、全部规则和组件覆盖的负结果放在主文可见位置。', [acqpath])
    drug_rows = [r for r in acquisition['drug_sampling_summary'] if r['rule'] == 'audit_direction']
    drug_rows.sort(key=lambda r: (DRUGS.index(r['drug']), r['acquisition']))
    fields = ['drug', 'acquisition', 'paired_configurations', 'lower_loss', 'higher_loss', 'equal_loss', 'FS_increased', 'FS_decreased', 'mean_brier_delta']
    add_table('Table05_all_drug_acquisition', 'Drug-specific audit-direction acquisition outcomes, including adverse results', [('All nine drugs, both acquisition controls', ['Drug', 'Acquisition vs random', 'Pairs', 'Lower loss', 'Higher loss', 'Equal', 'More FS', 'Fewer FS', 'Mean ΔBrier'], [[r[k] for k in fields] for r in drug_rows])],
        'All thresholds, inputs, strategies, seeds and feasible query/audit sizes contribute. Predicted-resistance priority has a negative drug-wise mean in 7/9 drugs, with positive means for meropenem and TMP–SMX. Component coverage has positive means for four drugs. Small differences, especially near numerical zero, are descriptive and are not practical or significant improvements. Evaluation R ranges include 0–279 across the full grid; detailed audit/evaluation denominators remain in source_data/audit_detailed.csv. FS counts are paired target differences, not patient outcomes.', '九药物完整边界，不能只报7/9获益。', [acqpath])

    def supplemental(identifier, title, source, selected, notes, rationale, headings=None):
        bs = blocks(source)
        sections = [(headings[i] if headings else f'Part {i+1}', bs[j][0], bs[j][1]) for i, j in enumerate(selected)]
        add_table(identifier, title, sections, notes, rationale, [source])

    supplemental('TableS01_ANI_complete_comparisons', 'Complete ANI-screened input, budget and strategy comparisons', HERE/'ECOLI_ANI_ANNOTATION_RESULTS.md', [0], 'All available drug × input × ANI threshold × budget × control comparisons are retained. Lower Brier is better. Approximate ANI grouping is not patient or SNP isolation. Original skips and denominator strata are preserved in source reports.', '正式ANI完整大表，替代只展示旧sketch结果。', ['All matched contrasts'])
    supplemental('TableS02_controls', 'Complete negative-control and transport comparisons', HERE/'RELIABILITY_CONTROLS_RESULTS.md', [0], 'All feasible fixed-display cells are retained. Full configuration coverage is 1,440 trajectories and 1,080 skipped strategy configurations. A single permutation per seed is not a significance null distribution. Test-standard stratification is not a causal experiment. Temporal and cross-standard failures/unavailable cases remain in the copied source report.', '负对照完整数值表。', ['Fixed ANI 0.99, budget 25'])
    supplemental('TableS03_auxiliary_AST', 'Auxiliary AST information with all additional assay costs', HERE/'paper_figures/tables/Table03_auxiliary_information.md', [0], 'The fixed budget-25 forward-2017 and forward-2019 comparisons use matched complete cases. Both auxiliary drugs are charged for every enriched training/evaluation sample. ExtraTrees is an exploratory stronger baseline added after inspecting logistic results. Additional assay information is not free and is not equivalent to obtaining 25 target labels.', '新增信息收益及测量成本全部列出。', ['All protocols, drugs and estimators'])
    supplemental('TableS04_own_photo_models', 'Own image-to-distance models and reference-review simulation', HERE/'COMBIANT_OWN_VISION_RESULTS.md', [0, 1], '279 photos, 98 plates, five held-out plate folds. Complete six-target macro metrics use 278 photos; one A reference is absent. The 95% error column is a 95th percentile, not a 95% confidence interval. Our ExtraTrees macro MAE 1.5081 mm is worse than the published software reference 0.7018 mm. Operator-transfer support differs and is separately reported. Reference replacement is a simulation; random-order intervals are descriptive spreads, not CIs.', '公开照片自己的实验，正视不及原专用系统的结果。', ['All baselines and operator transfer', 'All review budgets and controls'])
    ms = blocks(HERE/'MSDEEPAMR_REEXECUTION_RESULTS.md')
    er = blocks(HERE/'MSDEEPAMR_ERROR_TRANSITIONS.md')
    add_table('TableS05_MALDI_reexecution', 'Public CNN reexecution and full directional-error comparisons', [('All three tasks and four strategies', ms[0][0], ms[0][1]), ('All six strategy pairs in each task', er[0][0], er[0][1])],
        '120 local configurations use ten fixed splits per task. Resistant test denominators are 9/4/4. These splits share small target cohorts and are not independent external studies. Means ± SD describe splits, not confidence intervals. This is compatible-runtime DRIAMS-B reexecution of author assets, not exact full-paper replication (author split identities and full C/D acquisition are absent). The author CNN is not our original architecture. Pair directions are fixed; AUROC/loss improvement with FS increase is retained. Full pairs and sample records are linked in the source report.', '质谱更新的另一任务案例；不声称自己的控制器跨模态胜出。', [HERE/'MSDEEPAMR_REEXECUTION_RESULTS.md', HERE/'MSDEEPAMR_ERROR_TRANSITIONS.md'])
    lit = blocks(HERE/'CITABLE_LITERATURE_METRICS.md')
    literature_titles = ['MSDeepAMR Table 4, DOI 10.3389/fmicb.2024.1361795', 'MSDeepAMR Table 5, DOI 10.3389/fmicb.2024.1361795', 'MSDeepAMR Table 6, DOI 10.3389/fmicb.2024.1361795', 'Kervancı–Adebiyi Table 2, DOI 10.1093/bioadv/vbag193', 'Kervancı–Adebiyi Table 3, DOI 10.1093/bioadv/vbag193', 'Kervancı–Adebiyi Table 4, DOI 10.1093/bioadv/vbag193', 'CombiANT reader Table 1, DOI 10.1371/journal.pdig.0000669', 'CarbaDetector Table 3, DOI 10.1038/s41467-025-66183-z', 'MALCA Table 1, DOI 10.1038/s41467-026-72713-0']
    assert len(lit) == len(literature_titles) == 9
    add_table('TableS06_author_reported_literature', 'Author-reported literature tables, separately sourced', [(literature_titles[i], h, rs) for i, (h, rs) in enumerate(lit)],
        'Five papers and nine original tables. Values are author-reported and cell-replayed against preserved XML. They must not be mixed into a local SOTA ranking or used for direct percentage gains because tasks, cohorts, labels, protocols and endpoints differ. Source inconsistencies and footnote ambiguities remain uncorrected and are documented in provenance/CITABLE_LITERATURE_METRICS.md. Source table headings and DOI links in that report are required when citing any values. A ± value is not relabelled as a CI.', '相关工作引用表与我们的结果分开。', [HERE/'CITABLE_LITERATURE_METRICS.md', HERE/'citable_literature_metrics_validation.json'])
    add_table('TableS07_original_sketch_context', 'Original sketch evaluation, all decomposition settings and class-specific errors', [('Original sketch complete four-strategy comparison', full[0][0], full[0][1]), ('All decomposition budgets and original thresholds', ab[1][0], ab[1][1]), ('Resistant/susceptible losses and all D/G/R false-susceptible counts', ab[2][0], ab[2][1])],
        'These results use the original sketch component grouping, not the updated ANI grouping. Five partition descriptions share one cohort. Resistant-class support and small R ranges are retained; mean FS counts are not independent patient events. All thresholds, original source definitions and full precision values remain in source reports.', '历史口径不混入ANI强独立性，保留类别损失和小分母。', [HERE/'ECOLI_FULL_RESULTS.md', HERE/'ECOLI_FROZEN_QUERY_ABLATION.md'])
    skips = acquisition['skips']
    add_table('TableS08_unavailable_and_support', 'Unavailable audit-acquisition conditions and support ranges', [('All 7,056 unavailable conditions', ['Query budget', 'Audit size', 'Reason', 'Conditions'], [[r['query_budget'], r['audit_size'], r['reason'], r['conditions']] for r in skips]), ('Pooled rule support (shared configuration grid)', ['Rule', 'Configs', 'Audit R=0', 'Audit R min', 'Audit R max', 'Evaluation R min', 'Evaluation R max'], [[r['rule'], r['configurations'], r['audit_R_zero'], r['audit_R_min'], r['audit_R_max'], r['eval_R_min'], r['eval_R_max']] for r in acquisition['global_rule_summary']])],
        'Audit-pool size failures are retained, not replaced with zero effects. The inherited ten input/partition skips correspond to forty unavailable strategy trajectories and are separate from these 7,056 acquisition conditions. Each rule repeats the same acquisition grid, so R=0 rows must not be summed as independent audits. All 24,120 detailed summary rows include drug/input/threshold/strategy/budget/size/acquisition/rule strata and resistant support in the copied audit_detailed.csv.', '所有跳过与小R支撑列项，不隐藏不可行。', [acqpath])

    # Preserve full-data companions, source reports and existing independent audits.
    companions = {
        DATA/'audit_acquisition_action_v1/detailed.csv': 'audit_detailed.csv',
        DATA/'audit_acquisition_action_v1/summary.csv': 'audit_summary.csv',
        DATA/'audit_acquisition_action_v1/drug_sampling_summary.csv': 'audit_drug_sampling_summary.csv',
        DATA/'audit_acquisition_action_v1/sampling_pair_summary.csv': 'audit_sampling_pair_summary.csv',
        DATA/'repair_dynamics_ani_v1/transitions.csv': 'ANI_transitions.csv',
        DATA/'repair_dynamics_ani_v1/gate_summary.csv': 'ANI_gate_summary.csv',
    }
    for p, n in companions.items(): copy(p, OUT/'source_data'/n)
    reports = ['ECOLI_FULL_RESULTS.md', 'ECOLI_FROZEN_QUERY_ABLATION.md', 'ECOLI_ANI_ANNOTATION_RESULTS.md', 'AUDIT_ACQUISITION_ACTION_RESULTS.md', 'RELIABILITY_CONTROLS_RESULTS.md', 'COMBIANT_OWN_VISION_RESULTS.md', 'COMBIANT_MATCHED_CONTEXT_RESULTS.md', 'MSDEEPAMR_REEXECUTION_RESULTS.md', 'MSDEEPAMR_ERROR_TRANSITIONS.md', 'CITABLE_LITERATURE_METRICS.md', 'SCIENTIFIC_EVIDENCE_MATRIX.md', 'SCIENTIFIC_STORY_V2_2026_10_11.md', 'FINAL_COMPLETION_STATUS.md', 'FIGURE_TABLE_INVENTORY_2026_10_12.md', 'MANUSCRIPT_VISUAL_POLICY.md']
    for n in reports: copy(HERE/n, OUT/'provenance'/n)
    for folder in ('sequence_evidence_v1', 'combiant_own_v1', 'combiant_photos_v1', 'msdeepamr_spectra_v1'):
        for name in ('source.json', 'validation.json', 'visual_review.json', 'README.md'):
            p = HERE/'advanced_figures'/folder/name
            if p.exists(): copy(p, OUT/'provenance'/folder/name)
    bind(Path(__file__))

    legends = '# Manuscript figure legends\n\n' + '\n\n'.join('## ' + f['id'] + '. ' + f['title'] + '\n\n' + f['legend'] for f in figures)
    (OUT/'FIGURE_LEGENDS_EN.md').write_text(legends, encoding='utf-8')
    body = '<h1>论文图表候选包</h1><p>4张主文图、4张补充图、5组主文表、8组补充表。按目前可支持的科学主线筛选；重复导出格式不重复计数。补充文献表组含9张原文子表。</p><p>主问题：有限检测预算下，什么证据足以支持耐药预测模型更新？现有研究支持探索性边界分析，不代表新盲态临床确认。</p><div class="nav"><a href="FIGURE_LEGENDS_EN.md">英文图注</a><a href="ALL_TABLES.html">全部可编辑表</a><a href="README.md">取舍与使用说明</a><a href="manifest.json">来源与哈希</a></div>'
    readme = '# 论文图表候选包\n\n本包精选 **4张主文图 + 4张补充图 + 5组主文表 + 8组补充表**。每张图的PNG/PDF/SVG是同一图的不同格式，不重复计数。TableS06为一个相关工作表组，内部包含9张原文子表。当前38张历史图板不等于38张适合主文的图。\n\n工作题目：Evidence-constrained updating of antimicrobial resistance prediction under limited testing。\n\n推荐主线：真实序列/药敏记录 → 匹配选样与更新动作分解 → ANI分组下的纠错/引错 → 同成本审计采集与行动判断。照片和质谱作为不同任务的补充证据。\n\n[图表浏览页](index.html)；[完整表页](ALL_TABLES.html)；[英文图注](FIGURE_LEGENDS_EN.md)。MD与HTML表均为可编辑文本，复制到稿件后按目标期刊版式排版。表的CSV附件为原始产物的逐字节副本，不是截图表。\n\n## 图的取舍\n\n'
    for category, title in [('main_figures', '主文图'), ('supplementary_figures', '补充图')]:
        body += '<h2>' + title + '</h2>'
        for f in [f for f in figures if f['category'] == category]:
            links = ' · '.join('<a href="' + str(Path(f['path']).with_suffix('.'+ext)).replace('\\', '/') + '">' + ext.upper() + '</a>' for ext in f['formats'])
            body += '<figure><h3>' + f['id'] + '. ' + html.escape(f['title']) + '</h3><p>' + html.escape(f['rationale_zh']) + '</p><p>' + links + '</p><img loading="lazy" src="' + f['path'].replace('\\', '/') + '"><figcaption>' + html.escape(f['legend']) + '</figcaption></figure>'
            readme += f"- **{f['id']}** [{f['title']}]({f['path'].replace(chr(92), '/')})：{f['rationale_zh']}\n"
    readme += '\n## 表的取舍\n\n'
    tablebody = '<h1>论文表格</h1><p>5组主文表、8组补充表；每组保留标题、分母、比较方向、来源和限制。文献表与本地实验结果分列。</p>'
    allmd = '# 论文表格合集\n\n'
    for t in tables:
        p = OUT/t['path']; text = p.read_text(encoding='utf-8')
        allmd += text + '\n\n'
        content = (p.with_suffix('.html')).read_text(encoding='utf-8').split('<main>',1)[1].split('</main>',1)[0]
        tablebody += content
        body += '<div class="card"><a href="' + t['path'].replace('\\', '/') + '">' + t['id'] + ' ' + html.escape(t['title']) + '</a><p>' + html.escape(t['rationale_zh']) + '</p></div>'
        readme += f"- **{t['id']}** [{t['title']}]({t['path'].replace(chr(92), '/')})：{t['rationale_zh']}\n"
    readme += '''
## 使用边界

- Fig02/Table02/TableS07沿用旧sketch组件；不能改写为ANI/SNP隔离。Fig03/Fig04及其正式对照使用后来验证的近似ANI分组；ANI仍不证明患者/实验室独立。
- 序列sketch不是AMR注释；BLAST同源候选不是完整AMR机制或官方机制鉴定。NDM同源与敏感标签的例子是记录审计线索。
- 主文保留负结果：获益不单调、Brier改善伴随FS增加、门控失败、组件覆盖无普遍收益，以及meropenem/TMP–SMX采集均值变差。共享队列的配置次数不作独立事件率或显著性。
- 照片、质谱和基因组是三个不同任务，没有逐株多模态配对。公开照片的原专用软件优于我们自己的像素树模型。
- 不收入S05_auxiliary_selection_cost、S10_repair_actions等柱状图；历史sketch美化图、pilot、重复版本不靠外观进入主文。未选项目见provenance/FIGURE_TABLE_INVENTORY_2026_10_12.md，原始文件保留。
- MIC精确版本与CLSI统一、完整AMR专项已发表基线复现、新盲态确认仍有缺口。此前已发表基线执行失败与SIRscan全数据获取受阻状态未被本次整理抹除。
- 文献表为作者报告值，来源XML逐格校验；表义/源文矛盾不静默修改。TableS06使用时连同原DOI和具体表号引用，不当本方法SOTA对照。

## 交付与核验

原始实验与阈值不变。本包两张新图只重新绘制现有验证数据；其余图逐字节复制。新的PNG均需实际目视检查，记录当前SHA256到visual_review.json。manifest.json绑定源文件与全部交付产物。package_validation.json核查字节副本、已有验证哈希、完整配对复算、文件链接与图像目视记录；不替代科学外部确认。
'''
    (OUT/'README.md').write_text(readme, encoding='utf-8')
    (OUT/'index.html').write_text(page('论文图表候选包', body), encoding='utf-8')
    (OUT/'ALL_TABLES.html').write_text(page('论文表格', tablebody), encoding='utf-8')
    (OUT/'ALL_TABLES.md').write_text(allmd, encoding='utf-8')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selection_scope': 'exploratory manuscript candidates, not scientific closure',
                'counts': {'main_figures': 4, 'supplementary_figures': 4, 'main_table_groups': 5, 'supplementary_table_groups': 8, 'literature_original_subtables': len(lit)},
                'figures': figures, 'tables': tables, 'sources': provenance, 'checks': checks,
                'exports': {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json', 'package_validation.json', 'visual_review.json')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'package': str(OUT), 'counts': manifest['counts'], 'numeric_and_copy_checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
