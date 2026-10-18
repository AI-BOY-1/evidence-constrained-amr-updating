"""Label-blind audit ordering and fixed acceptance rules."""
import hashlib
from collections import defaultdict
import numpy as np
from .metrics import binary_metrics
def acquisition_order(samples, components, base_p, seed, method):
    ranks={i:hashlib.sha256(f"audit-acquisition-v1:{seed}:{s}".encode()).hexdigest() for i,s in enumerate(samples)}
    ranked=sorted(range(len(samples)), key=lambda i:(ranks[i],str(samples[i])))
    if method=='random': return ranked
    if method=='predicted_R': return sorted(ranked,key=lambda i:-float(base_p[i]))
    if method!='component_coverage': raise ValueError(method)
    groups=defaultdict(list)
    for i in ranked: groups[str(components[i])].append(i)
    keys=sorted(groups,key=lambda c:hashlib.sha256(f"audit-component-v1:{seed}:{c}".encode()).hexdigest())
    return [groups[c][level] for level in range(max(map(len,groups.values()),default=0)) for c in keys if level<len(groups[c])]
def audit_evidence(y, old, new, components):
    y,old,new=np.asarray(y),np.asarray(old),np.asarray(new)
    a,b=binary_metrics(y,old),binary_metrics(y,new); groups=[np.asarray(components)==c for c in sorted(set(components))]
    rgroups=[g&(np.asarray(y)==1) for g in groups if np.any(g&(np.asarray(y)==1))]
    return {"old":a,"new":b,"brier_delta":b["brier"]-a["brier"],"FS_delta":b["FS"]-a["FS"],
            "component_brier_delta":float(np.mean([np.mean((new[g]-y[g])**2-(old[g]-y[g])**2) for g in groups])),
            "component_FS_rate_delta":float(np.mean([np.mean(new[g]<.5)-np.mean(old[g]<.5) for g in rgroups])) if rgroups else None}
def decide_action(e, rule):
    if rule not in ('retain','update','audit_loss','audit_direction','component_direction'): raise ValueError(rule)
    if rule=='retain': return False,'fixed'
    if rule=='update': return True,'fixed'
    if rule=='audit_loss': return e['brier_delta']<0,'eligible'
    if e['old']['R']==0: return False,'infeasible_no_audit_R_fallback_retain'
    if rule=='audit_direction': return e['brier_delta']<0 and e['FS_delta']<=0,'eligible'
    if rule=='component_direction': return e['component_brier_delta']<0 and e['component_FS_rate_delta']<=0,'eligible'
    raise ValueError(rule)
