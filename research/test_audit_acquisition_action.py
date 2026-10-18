"""Nontrivial label-blindness, direction, equality and target-transfer fixtures."""
import numpy as np
from run_audit_acquisition_action import acquisition_order,audit_evidence,decide,score

samples=np.array(['a','b','c','d','e']);clusters=np.array(['A','A','A','B','C']);p=np.array([.8,.2,.4,.9,.1])
for method in ('random','predicted_R','component_coverage'):
    a=acquisition_order(samples,clusters,p,7,method)
    assert sorted(a)==list(range(5))
    assert a==acquisition_order(samples,clusters,p,7,method)
    # Labels do not appear in the acquisition interface; permutation changes no ordering.
    assert a[:2]==acquisition_order(samples,clusters,p,7,method)[:2]
assert len(set(clusters[acquisition_order(samples,clusters,p,7,'component_coverage')[:3]]))==3
y=np.array([1,0]);old=np.array([.4,.8]);new=np.array([.7,.2]);c=np.array(['x','y'])
e=audit_evidence(y,old,new,c)
assert decide(e,'audit_direction')==(True,'eligible')
assert score(y,np.array([.2,.9]))['brier']>score(y,old)['brier'] # accepted audit is not evaluation protection
equal=audit_evidence(y,old,old,c)
assert not decide(equal,'audit_loss')[0] and not decide(equal,'component_direction')[0]
noR=audit_evidence(np.array([0,0]),old,new,c)
assert decide(noR,'audit_direction')==(False,'infeasible_no_audit_R_fallback_retain')
print('passed: deterministic label-blind nested sampling, component round-robin, audit/target mismatch, ties, no-R fallback')
