"""Independent source-record and aggregate checks for audit transfer tables."""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = ncs_paths.data_path('experiments/ecoli_bvbrc')
OUT = DATA / 'audit_transfer_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = OUT / 'results.json'
    result = read(path)
    for name, value in result['binding']['inputs'].items():
        assert digest(Path(name)) == value
    assert digest(HERE / 'analyze_audit_transfer.py') == result['binding']['code_sha256']
    assert digest(HERE / 'AUDIT_TRANSFER_ANALYSIS_PROTOCOL.md') == result['binding']['protocol_sha256']
    sources = {}
    expected = set()
    for dataset in ('full', 'ani'):
        root = DATA / ('repair_dynamics_' + dataset + '_v1')
        validation = read(root / 'validation.json')
        assert validation['status'] == 'passed'
        assert validation['result_sha256'] == digest(root / 'results.json')
        for index, row in enumerate(read(root / 'results.json')['transitions']):
            if row['adjacent']:
                sources[(dataset, index)] = row
                for rule in ('brier_only', 'brier_and_FS'):
                    expected.add((dataset, index, rule))
    seen = set()
    for row in result['records']:
        key = (row['dataset'], row['transition_index'], row['rule'])
        assert key not in seen and key in expected
        seen.add(key)
        source = sources[key[:2]]
        eligible = bool(source['audit_n']) and (key[2] == 'brier_only' or source['audit_R'] > 0)
        accepted = eligible and source['audit_brier_delta'] < 0
        if key[2] == 'brier_and_FS':
            accepted = accepted and source['audit_FS_delta'] <= 0
        r = source['audit_R']
        category = 'empty_audit' if not source['audit_n'] else '0' if not r else '1-4' if r <= 4 else '5-19' if r <= 19 else '>=20'
        assert row['eligible'] == eligible and row['accepted'] == accepted and row['audit_R_bin'] == category
        assert row['evaluation_R'] == source['R'] and row['FS_delta'] == source['FS_after'] - source['FS_before']
        for field in ('drug', 'threshold', 'feature', 'seed', 'strategy', 'from_budget', 'to_budget',
                      'audit_n', 'audit_R', 'brier_delta', 'FS_introduced'):
            assert row[field] == source[field]
    assert seen == expected
    for name, keys in [('pooled', ('dataset', 'rule', 'audit_R_bin')),
                       ('detailed', ('dataset', 'drug', 'threshold', 'feature', 'strategy', 'rule', 'audit_R_bin'))]:
        groups = defaultdict(list)
        for row in result['records']:
            groups[tuple(row[k] for k in keys)].append(row)
        assert len(result[name]) == len(groups)
        checked = set()
        for summary in result[name]:
            key = tuple(summary[k] for k in keys)
            assert key in groups and key not in checked
            checked.add(key)
            rows = groups[key]
            accepted = [r for r in rows if r['accepted']]
            counts = {'transitions': len(rows), 'eligible': sum(r['eligible'] for r in rows),
                      'accepted': len(accepted), 'evaluation_R_zero': sum(r['evaluation_R'] == 0 for r in accepted),
                      'eval_loss_lower': sum(r['brier_delta'] < 0 for r in accepted),
                      'eval_loss_higher': sum(r['brier_delta'] > 0 for r in accepted),
                      'eval_loss_equal': sum(r['brier_delta'] == 0 for r in accepted),
                      'any_new_FS': sum(r['FS_introduced'] > 0 for r in accepted),
                      'net_FS_increase': sum(r['FS_delta'] > 0 for r in accepted),
                      'lower_loss_more_FS': sum(r['brier_delta'] < 0 and r['FS_delta'] > 0 for r in accepted)}
            for field, value in counts.items():
                assert summary[field] == value
            for field in ('audit_n', 'audit_R', 'evaluation_R'):
                assert summary[field + '_range'] == [min(r[field] for r in rows), max(r[field] for r in rows)]
    validation = {'status': 'passed', 'result_sha256': digest(path),
        'validator_sha256': digest(Path(__file__)), 'records_checked': len(seen),
        'pooled_rows': len(result['pooled']), 'detailed_rows': len(result['detailed']),
        'checks': {'source_hashes': True, 'configuration_coverage': True,
                   'local_acceptance_and_denominators': True, 'aggregate_recomputation': True}}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
