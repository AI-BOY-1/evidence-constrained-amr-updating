import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ncs_amr import Record, ExpertBank, binary_metrics, transition_metrics
from ncs_amr.controller import update_weights, query_scores
from ncs_amr.acquisition import acquisition_order, audit_evidence, decide_action

class PortableTests(unittest.TestCase):
    def setUp(self):
        self.rows=[Record(f's{i}',f'c{i}',2020,'EUCAST',(f'g{i%5}',),i%2) for i in range(20)]
    def test_mask_and_weights(self):
        x=update_weights(np.ones(5)/5,np.full((5,3),.7),np.array([1,0,1]));self.assertAlmostEqual(float(x.sum()),1.0)
    def test_query_nonnegative(self):
        p=np.array([[.1,.2],[.9,.8],[.4,.5],[.3,.7],[.6,.2]]);s=query_scores(p,np.ones(5)/5,'decision');self.assertTrue(np.all(s>=-1e-12))
    def test_metrics_keep_zero_denominator(self):
        m=binary_metrics([0,0],[.2,.8]);self.assertIsNone(m['FS_rate']);self.assertEqual(m['FR'],1)
    def test_transition_identity(self):
        t=transition_metrics([1,0],[.4,.8],[.7,.2]);self.assertEqual(t['corrected'],2);self.assertEqual(t['introduced'],0)
    def test_label_free_order_and_rules(self):
        samples=['a','b','c','d'];components=['A','A','B','C'];p=np.array([.2,.8,.3,.9]);o=acquisition_order(samples,components,p,7,'component_coverage');self.assertEqual(sorted(o),list(range(4)))
        e=audit_evidence(np.array([1,0]),np.array([.4,.8]),np.array([.7,.2]),np.array(['x','y']));self.assertEqual(decide_action(e,'audit_direction'),(True,'eligible'))
if __name__=='__main__':unittest.main()
