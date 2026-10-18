"""Explicit binary metrics with unavailable denominators preserved."""
import numpy as np
def binary_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y) != len(p) or not len(y): raise ValueError("y and p must have equal nonzero length")
    if not np.all(np.isin(y,(0,1))) or not np.all(np.isfinite(p)) or not np.all((p>=0)&(p<=1)):
        raise ValueError('labels must be binary and probabilities finite in [0,1]')
    r, s, pred = y==1, y==0, p>=.5
    return {"n":int(len(y)),"R":int(r.sum()),"S":int(s.sum()),"brier":float(np.mean((p-y)**2)),
            "FS":int((r&~pred).sum()),"FR":int((s&pred).sum()),
            "FS_rate":float((r&~pred).sum()/r.sum()) if r.any() else None,
            "FR_rate":float((s&pred).sum()/s.sum()) if s.any() else None}
def transition_metrics(y, old, new):
    y, old, new = np.asarray(y), np.asarray(old), np.asarray(new)
    binary_metrics(y,old);binary_metrics(y,new)
    olderr=(old>=.5)!=y; newerr=(new>=.5)!=y
    return {"corrected":int((olderr&~newerr).sum()),"introduced":int((~olderr&newerr).sum()),
            "FS_corrected":int(((y==1)&olderr&~newerr).sum()),"FS_introduced":int(((y==1)&~olderr&newerr).sum()),
            "brier_delta":float(np.mean((new-y)**2-(old-y)**2))}
