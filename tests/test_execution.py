import sys,unittest,ast,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ncs_amr import Record,run_trajectory
from ncs_amr.records import validate_roles
from ncs_amr.controller import action_matrix,update_weights,query_scores
from ncs_amr.models import ExpertBank

class ExecutionTests(unittest.TestCase):
    def groups(self):
        source=[Record(f's{i}',f'S{i//2}',2010+i%3,'EUCAST',(f'k{i%4}',),i%2) for i in range(20)]
        pool=[Record(f'q{i}',f'Q{i//2}',2020,'CLSI',(f'k{i%4}',),-1) for i in range(15)]
        audit=[Record('a','A',2021,'CLSI',('k0',),-1)]
        evaluation=[Record('e','E',2022,'CLSI',('k1',),-1)]
        return dict(source=source,query_pool=pool,audit=audit,evaluation=evaluation)
    def test_sealed_labels_four_strategies_and_no_repeat(self):
        for strategy in ('decision_diagnostic','random_gene','uncertainty_gene','random_tree'):
            groups=self.groups();calls=[];oracle={r.sample:Record(r.sample,r.component,r.year,r.standards,r.features,i%2) for i,r in enumerate(groups['query_pool'])}
            def reveal(s):calls.append(s);return oracle[s]
            snapshots,events=run_trajectory(groups,reveal,7,(0,5,10),strategy=strategy)
            self.assertEqual([x['budget'] for x in snapshots],[0,5,10])
            self.assertEqual(len(calls),len(set(calls)));self.assertEqual(len(calls),10)
            self.assertEqual(len(events),2)
            for x in snapshots:
                self.assertTrue(np.isfinite(x['evaluation_p']).all());self.assertEqual(len(x['audit_p']),1)
            if strategy in ('random_gene','uncertainty_gene'):self.assertEqual(snapshots[-1]['action'],'updated_gene')
    def test_overlap_labels_budgets_and_reveal_identity_rejected(self):
        groups=self.groups();validate_roles(groups)
        groups['evaluation']=[groups['query_pool'][0]]
        with self.assertRaises(ValueError):validate_roles(groups)
        groups=self.groups();r=groups['evaluation'][0];groups['evaluation']=[Record(r.sample,r.component,r.year,r.standards,r.features,1)]
        with self.assertRaises(ValueError):validate_roles(groups)
        with self.assertRaises(ValueError):run_trajectory(self.groups(),lambda _:None,7,(0,25))
        with self.assertRaises(ValueError):run_trajectory(self.groups(),lambda _:None,7,(0,7))
        with self.assertRaises(ValueError):run_trajectory(self.groups(),lambda _:Record('wrong','wrong',0,'',(),1),7,(0,5))
    def test_extracted_functions_equal_historical_AST(self):
        root=Path(__file__).resolve().parents[1]
        mapping=[('decision_repair_core.py','controller.py',['action_matrix','update_weights','query_scores','diverse_indices']),
                 ('run_decision_repair.py','controller.py',['trajectory','tree_predict']),
                 ('decision_repair_core.py','models.py',['ExpertBank']),
                 ('counterexample_guided_repair.py','models.py',['ProbabilityModel','compact_sparse','feature_dict','entropy_score'])]
        for old,new,names in mapping:
            a=ast.parse((root/'historical_frozen/source_snapshots'/old).read_text(encoding='utf-8'))
            b=ast.parse((root/'src/ncs_amr'/new).read_text(encoding='utf-8'))
            for name in names:
                x=next(n for n in a.body if getattr(n,'name',None)==name)
                y=next(n for n in b.body if getattr(n,'name',None)==name)
                self.assertEqual(ast.dump(x,include_attributes=False),ast.dump(y,include_attributes=False))
    def test_query_and_action_formula_properties(self):
        rng=np.random.default_rng(731)
        for _ in range(32):
            p=rng.random((5,15));w=rng.dirichlet(np.ones(5));mix=w@p
            variance=sum(w[j]*np.mean((p[j]-mix)**2) for j in range(5))
            distance=np.mean((p-mix[None,:])**2,axis=1)
            np.testing.assert_allclose(w@action_matrix(p),variance+distance,atol=1e-12)
            self.assertTrue(np.all(query_scores(p,w,'decision')>=0))
            actual=update_weights(w,p[:,:3],np.array([1,0,1]));expected=w*p[:,0]*(1-p[:,1])*p[:,2];expected/=expected.sum()
            np.testing.assert_allclose(actual,expected,atol=1e-12)
    def test_array_validation(self):
        from ncs_amr.metrics import binary_metrics
        for y,p in [([0,1],[.1,float('nan')]),([0,2],[.1,.4]),([0],[2.]),([],[])]:
            with self.assertRaises(ValueError):binary_metrics(y,p)
    def test_command_line_and_no_overwrite(self):
        import tempfile,subprocess,json
        root=Path(__file__).resolve().parents[1];groups=self.groups()
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'input.jsonl';out=Path(td)/'result.json'
            rows=[]
            for role,records in groups.items():
                for i,r in enumerate(records):
                    rows.append(json.dumps(dict(sample=r.sample,component=r.component,year=r.year,
                                                standards=r.standards,features=list(r.features),
                                                label=(i%2 if role=='query_pool' else r.label),role=role)))
            inp.write_text('\n'.join(rows),encoding='utf-8')
            cmd=[sys.executable,str(root/'scripts/run_controller.py'),'--input',str(inp),'--out',str(out),'--query-budget','0','5','10']
            proc=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            value=json.loads(out.read_text(encoding='utf-8'));self.assertEqual(len(value['events']),2)
            self.assertEqual(len(value['input_sha256']),64)
            self.assertIn('src/ncs_amr/controller.py',value['code_sha256'])
            before=out.read_bytes();proc=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0);self.assertEqual(out.read_bytes(),before)

if __name__=='__main__':unittest.main()
