"""Describe local audit decisions on already-validated frozen transitions."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT = DATA / 'audit_transfer_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bin_name(row):
    if not row['audit_n']:
        return 'empty_audit'
    n = row['audit_R']
    return '0' if n == 0 else '1-4' if n < 5 else '5-19' if n < 20 else '>=20'


def aggregate(rows, fields):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, group in sorted(buckets.items()):
        accepted = [r for r in group if r['accepted']]
        result.append({**dict(zip(fields, key)), 'transitions': len(group),
            'eligible': sum(r['eligible'] for r in group), 'accepted': len(accepted),
            'evaluation_R_zero': sum(r['evaluation_R'] == 0 for r in accepted),
            'eval_loss_lower': sum(r['brier_delta'] < 0 for r in accepted),
            'eval_loss_higher': sum(r['brier_delta'] > 0 for r in accepted),
            'eval_loss_equal': sum(r['brier_delta'] == 0 for r in accepted),
            'any_new_FS': sum(r['FS_introduced'] > 0 for r in accepted),
            'net_FS_increase': sum(r['FS_delta'] > 0 for r in accepted),
            'lower_loss_more_FS': sum(r['brier_delta'] < 0 and r['FS_delta'] > 0 for r in accepted),
            'audit_n_range': [min(r['audit_n'] for r in group), max(r['audit_n'] for r in group)],
            'audit_R_range': [min(r['audit_R'] for r in group), max(r['audit_R'] for r in group)],
            'evaluation_R_range': [min(r['evaluation_R'] for r in group), max(r['evaluation_R'] for r in group)]})
    return result


def main():
    OUT.mkdir(exist_ok=True)
    inputs = {}
    records = []
    for dataset in ('full', 'ani'):
        root = DATA / ('repair_dynamics_' + dataset + '_v1')
        result_path, validation_path = root / 'results.json', root / 'validation.json'
        validation = read(validation_path)
        assert validation['status'] == 'passed' and validation['result_sha256'] == sha(result_path)
        inputs.update({str(p): sha(p) for p in (result_path, validation_path)})
        for index, row in enumerate(read(result_path)['transitions']):
            if not row['adjacent']:
                continue
            for rule in ('brier_only', 'brier_and_FS'):
                eligible = row['audit_n'] > 0 and (rule == 'brier_only' or row['audit_R'] > 0)
                accepted = eligible and row['audit_brier_delta'] < 0
                if rule == 'brier_and_FS':
                    accepted = accepted and row['audit_FS_delta'] <= 0
                records.append({**{k: row[k] for k in ('drug', 'threshold', 'feature', 'seed', 'strategy',
                    'from_budget', 'to_budget', 'audit_n', 'audit_R', 'brier_delta', 'FS_introduced')},
                    'dataset': dataset, 'transition_index': index, 'rule': rule,
                    'audit_R_bin': bin_name(row), 'eligible': eligible, 'accepted': bool(accepted),
                    'evaluation_R': row['R'], 'FS_delta': row['FS_after'] - row['FS_before']})
    protocol = HERE / 'AUDIT_TRANSFER_ANALYSIS_PROTOCOL.md'
    binding = {'inputs': inputs, 'code_sha256': sha(Path(__file__)), 'protocol_sha256': sha(protocol)}
    pooled = aggregate(records, ('dataset', 'rule', 'audit_R_bin'))
    detailed = aggregate(records, ('dataset', 'drug', 'threshold', 'feature', 'strategy', 'rule', 'audit_R_bin'))
    result = {'status': 'complete_pending_validation', 'binding': binding, 'records': records,
              'pooled': pooled, 'detailed': detailed,
              'limitations': ['Post-result descriptive configuration counts; no causal sample-size effect.',
                              'Local acceptance is not sequential stopping; no new training or blind confirmation.']}
    path = OUT / 'results.json'
    if path.exists():
        assert read(path)['binding'] == binding, 'Refuse mixed analysis versions'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, rows in (('pooled', pooled), ('detailed', detailed)):
        with (OUT / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Audit decisions and evaluation transfer', '',
        'Post-result descriptive analysis of validated frozen probabilities. Counts are correlated configuration observations, not independent patients. Local acceptance is not sequential stopping.', '',
        '|Dataset|Local rule|Audit R bin|Eligible / transitions|Accepted|Eval loss higher|Any new FS|Net FS increase|Lower loss, more FS|Audit n range|Eval R range|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in pooled:
        lines.append(f"|{r['dataset']}|{r['rule']}|{r['audit_R_bin']}|{r['eligible']}/{r['transitions']}|{r['accepted']}|{r['eval_loss_higher']}|{r['any_new_FS']}|{r['net_FS_increase']}|{r['lower_loss_more_FS']}|{r['audit_n_range']}|{r['evaluation_R_range']}|")
    lines += ['', 'The denominator for the adverse evaluation columns is accepted updates, not all transitions. Evaluation-R=0 counts are retained in pooled.csv and results.json. All drug/input/threshold/strategy strata are in detailed.csv. Empty audit sets and rules without audit-resistant support are retained as ineligible.', '',
        'Bin differences can reflect cohort composition, audit size, prevalence, lineage or shift. No significance or causal interpretation is assigned. A factorial audit-acquisition experiment followed by holder-blind confirmation would be needed to distinguish explanations.', '',
        f'Complete data: {path}', f'Independent validation: {OUT / "validation.json"}']
    (HERE / 'AUDIT_TRANSFER_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'records': len(records), 'pooled_rows': len(pooled), 'detailed_rows': len(detailed),
                      'status': result['status']}))


if __name__ == '__main__':
    main()
