"""Synthetic formula-to-function checks only; no training, real data or pipeline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import numpy as np
from decision_repair_core import action_matrix, query_scores, update_weights
from run_audit_acquisition_action import audit_evidence, decide, score, acquisition_order
from run_repair_dynamics import transition

HERE = Path(__file__).resolve().parent
OUT = HERE/'manuscript_package_v1_2026_10_12'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    rng=np.random.default_rng(20261012)
    checks=[]; max_error=0.
    def close(a,b):
        nonlocal max_error
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        max_error=max(max_error,error)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    for case in range(64):
        p=rng.uniform(0,1,(5,13));p[0,0]=0.;p[1,1]=1.
        w=rng.dirichlet(np.ones(5));y=rng.integers(0,2,3)
        # Independent loops characterize raw disagreement, action and E09.
        A=np.array([[sum((p[j,i]-p[k,i])**2 for i in range(13))/13
                     for k in range(5)] for j in range(5)])
        close(action_matrix(p),A)
        mixture=w@p
        variance=sum(w[j]*np.mean((p[j]-mixture)**2) for j in range(5))
        distances=np.array([np.mean((mixture-p[k])**2) for k in range(5)])
        close(w@A,variance+distances)
        q=np.clip(p,1e-6,1-1e-6);Ac=action_matrix(q);cur=float(np.min(w@Ac))
        scores=[]
        for i in range(13):
            m=float(w@q[:,i]);post1=w*q[:,i]/m;post0=w*(1-q[:,i])/(1-m)
            close(m*post1+(1-m)*post0,w)
            cost1=min(sum(post1[j]*Ac[j,k] for j in range(5)) for k in range(5))
            cost0=min(sum(post0[j]*Ac[j,k] for j in range(5)) for k in range(5))
            raw=cur-m*cost1-(1-m)*cost0
            assert raw>=-1e-12 and raw<=cur+1e-12
            scores.append(max(0.,raw))
        close(query_scores(p,w,'decision'),scores)
        # Independent product likelihood update and protected-zero behavior.
        expected=w*np.prod(np.where(y[None,:]==1,q[:,:3],1-q[:,:3]),axis=1)
        close(update_weights(w,p[:,:3],y),expected/expected.sum())
        result=update_weights(np.array([0.,.1,.2,.3,.4]),p[:,:3],y)
        assert np.isfinite(result).all() and np.all(result>=0)
        close(result.sum(),1.)
        # Information identity using the independent Bernoulli entropy form.
        m=w@q
        entropy=lambda a:-(a*np.log(a)+(1-a)*np.log1p(-a))
        close(query_scores(p,w,'information'),entropy(m)-w@entropy(q))
        labels=rng.integers(0,2,13);old=p[2];new=p[3]
        t=transition(labels,old,new)
        assert t['errors_after']-t['errors_before']==t['introduced']-t['corrected']
        assert t['FS_after']-t['FS_before']==t['FS_introduced']-t['FS_corrected']
        close(t['brier_delta'],np.mean((new-old)*(new+old-2*labels)))
    checks += [
        {'check':'raw_disagreement_and_mixture_action_identity_E08_E09','cases':64},
        {'check':'clipped_one_step_score_E10_E11_and_nonnegative_surrogate','cases':64},
        {'check':'batch_likelihood_update_E07_and_simplex','cases':64},
        {'check':'optional_information_identity_E12','cases':64},
        {'check':'error_flow_and_Brier_identities_E16_E18','cases':64}]
    old=np.array([.51,.4]);new=np.array([.49,.1]);y=np.array([1,0])
    a,b=score(y,old),score(y,new)
    close(a['brier'],.20005);close(b['brier'],.13505)
    assert b['FS']-a['FS']==1 and b['brier']<a['brier']
    audit=audit_evidence(np.array([1]),np.array([.4]),np.array([.6]),np.array(['a']))
    assert decide(audit,'audit_direction')==(True,'eligible')
    assert decide(audit,'component_direction')==(True,'eligible')
    target_old,target_new=score(np.array([1]),np.array([.6])),score(np.array([1]),np.array([.4]))
    assert target_new['brier']>target_old['brier'] and target_new['FS']>target_old['FS']
    noR=audit_evidence(np.array([0]),np.array([.6]),np.array([.4]),np.array(['a']))
    assert decide(noR,'audit_direction')==(False,'infeasible_no_audit_R_fallback_retain')
    tied=audit_evidence(y,old,old,np.array(['a','b']))
    assert decide(tied,'audit_loss')==(False,'eligible')
    assert decide(tied,'audit_direction')==(False,'eligible')
    checks.append({'check':'synthetic_counterexamples_C1_P8_and_strict_tie_noR_rules','cases':4})
    samples=np.array(['s'+str(i) for i in range(13)])
    clusters=np.array(['c'+str(i//3) for i in range(13)])
    assert list(inspect.signature(acquisition_order).parameters)==['samples','clusters','base_p','seed','method']
    for method in ('random','predicted_R','component_coverage'):
        order=acquisition_order(samples,clusters,old.repeat(7)[:13],7,method)
        assert sorted(order)==list(range(13))
        assert set(order[:5])<=set(order[:10])
    checks.append({'check':'label_free_acquisition_signature_permutation_and_nesting_E19','methods':3})
    source_names=['decision_repair_core.py','run_audit_acquisition_action.py','run_repair_dynamics.py']
    hashes={n:sha(HERE/n) for n in source_names}
    for n,h in hashes.items():assert sha(OUT/'methods/source_code'/n)==h
    record={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'finite synthetic numerical agreement with formulas; not scientific results or substitutes for analytic proofs',
            'source_hashes':hashes,'verification_script_sha256':sha(__file__),
            'checks':checks,'max_absolute_formula_difference':max_error,
            'limitations':['No new model fitting or empirical experiment.',
                           'Does not prove target risk monotonicity, calibration, clinical safety or batch optimality.',
                           'P1/P2/E05 rely on stated analytic definitions and solver settings, not these numerical cases.']}
    (OUT/'METHODS_MATH_IMPLEMENTATION_CHECK.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','checks':len(checks),'max_absolute_difference':max_error}))

if __name__=='__main__':main()
