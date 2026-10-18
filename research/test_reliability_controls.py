"""Small fixtures for exclusion, permutation and sealed-label invariants."""
import unittest
from unittest.mock import patch
from collections import Counter
import numpy as np
from counterexample_guided_repair import Record
from run_reliability_controls import partition, transform, CONDITIONS, ROLES
from validate_reliability_controls import expected_roles, expected_donors
from run_decision_repair import trajectory, mask
from decision_repair_core import EXPERTS


class ControlFixtures(unittest.TestCase):
    def records(self):
        rows=[]
        for c in range(100):
            for i in range(4):
                rows.append(Record(f'n{c:03}_{i}',f'c{c:03}',
                    (2011,2013,2016,2019,0)[c%5],('EUCAST','CLSI','')[c%3],
                    (f'gene{i}',),('fixture',),i%2))
        rows.extend([
            Record('span0','span',2011,'EUCAST',('a',),('fixture',),0),
            Record('span1','span',2018,'CLSI',('b',),('fixture',),1)])
        return sorted(rows,key=lambda r:r.sample)

    def test_roles_and_exclusions(self):
        rows=self.records()
        metadata={r.sample:{'component':r.project,'year':r.year,'standard':r.country} for r in rows}
        for condition in CONDITIONS:
            for seed in (7,17):
                gs,ex=partition(rows,condition,seed)
                assigned,reasons=expected_roles(metadata,condition,seed)
                self.assertEqual({r.sample:role for role,g in zip(ROLES,gs) for r in g},assigned)
                self.assertEqual(ex,reasons)
                components=[{r.project for r in g} for g in gs]
                self.assertTrue(all(not components[i]&components[j] for i in range(4) for j in range(i)))
                if condition.startswith('time_') or '_to_' in condition:
                    self.assertFalse({'span0','span1'}&set(assigned))
                    self.assertGreater(ex['component_spans_windows_or_standards'],0)
                if condition.startswith('time_'):
                    self.assertGreater(ex['unknown_or_ambiguous_date'],0)

    def test_donor_replay_and_protected_roles(self):
        rows=self.records()
        lookup={r.sample:r for r in rows}
        metadata={r.sample:{'component':r.project,'year':r.year,'standard':r.country} for r in rows}
        for condition in CONDITIONS:
            gs,_=partition(rows,condition,7)
            transformed,mapping,seed=transform(gs,condition,'fixture',.99,7,'homolog')
            rng=np.random.default_rng(seed)
            for role,g,tg in zip(ROLES,gs,transformed):
                ns=[r.sample for r in g]
                donors=expected_donors(ns,metadata,condition,role,rng)
                actual={m['sample']:m['donor'] for m in mapping if m['role']==role}
                self.assertEqual(donors,actual)
                self.assertEqual(set(actual.values()),set(ns))
                if role in ('audit','evaluation'):self.assertEqual(g,tg)
                if condition=='features_within_context':
                    self.assertEqual([r.label for r in g],[r.label for r in tg])
                    self.assertEqual(Counter(r.features for r in g),Counter(r.features for r in tg))
                    for r in g:
                        donor=lookup[actual[r.sample]]
                        self.assertEqual(r.country,donor.country)
                if condition=='label_within_component':
                    for c in {r.project for r in g}:
                        self.assertEqual(Counter(r.label for r in g if r.project==c),
                                         Counter(r.label for r in tg if r.project==c))

    def test_sealed_trajectory_and_query_provenance(self):
        rows=self.records();source=rows[:20];pool=rows[20:40];evaluation=rows[40:50];audit=rows[50:55]
        class Bank:
            def __init__(self,records,seed):self.seen=[]
            def predict(self,records):
                return np.tile(np.linspace(.2,.8,len(EXPERTS))[:,None],(1,len(records)))
            def update(self,records):self.seen=list(records)
        oracle={r.sample:r for r in pool};calls=[]
        def reveal(n):
            self.assertNotIn(n,calls);self.assertIn(n,oracle);calls.append(n);return oracle[n]
        for strategy in ('decision_diagnostic','random_gene'):
            calls.clear()
            with patch('run_decision_repair.ExpertBank',Bank):
                snapshots,events=trajectory(source,mask(pool),mask(evaluation),mask(audit),reveal,strategy,7,[0,10])
            self.assertEqual([s['budget'] for s in snapshots],[0,10])
            self.assertEqual(len(calls),10)
            self.assertEqual(snapshots[-1]['queried_samples'],calls)
            for e in events:self.assertEqual(e['revealed_labels'],[oracle[n].label for n in e['samples']])
        with self.assertRaises(AssertionError):
            trajectory(source,mask(pool),evaluation,mask(audit),reveal,'random_gene',7,[0])

    def test_real_model_smoke(self):
        rows=self.records();source=rows[:24];pool=rows[24:44];evaluation=rows[44:52]
        oracle={r.sample:r for r in pool}
        for strategy in ('decision_diagnostic','random_gene'):
            snapshots,events=trajectory(source,mask(pool),mask(evaluation),[],oracle.__getitem__,strategy,7,[0,10])
            self.assertEqual(len(events),2)
            for s in snapshots:
                self.assertEqual(s['evaluation_p'].shape,(8,))
                self.assertEqual(len(s['audit_p']),0)
                self.assertTrue(np.all(np.isfinite(s['evaluation_p'])))
                self.assertTrue(np.all((s['evaluation_p']>=0)&(s['evaluation_p']<=1)))
                self.assertEqual(len(s['queried_samples']),s['budget'])


if __name__=='__main__':unittest.main()
