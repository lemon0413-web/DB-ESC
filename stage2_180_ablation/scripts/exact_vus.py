"""Exact full-resolution TSB-AD VUS, with the official 250 thresholds.

This evaluates the same finite sums as RangeAUC_volume_opt. It groups points
by their FIRST satisfied official threshold; it does not subsample time points,
thresholds, or buffer widths. Direct official implementation is bundled for audit.
"""
import importlib.util
from pathlib import Path
import numpy as np

def load_official():
    path=Path(__file__).parent/'official_evaluation/basic_metrics.py'
    spec=importlib.util.spec_from_file_location('official_basic_metrics',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

OFFICIAL=load_official()

def normalize_scores(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model scores; no silent imputation permitted')
    span=float(np.ptp(scores))
    return (scores-float(scores.min()))/span if span>1e-12 else np.zeros_like(scores)

def exact_vus(labels,scores,window,thre=250):
    y=(np.asarray(labels)>0).astype(np.int32)
    s=normalize_scores(scores)
    if not 0<y.sum()<len(y):
        raise ValueError('VUS requires both label classes')
    grader=OFFICIAL.basic_metricor()
    seq=grader.range_convers_new(y)
    thresholds=-np.sort(-s)[np.linspace(0,len(s)-1,thre).astype(int)]
    bins=np.searchsorted(-thresholds,-s,side='left')
    npred=np.cumsum(np.bincount(bins,minlength=thre+1))[:thre].astype(float)
    tp_original=np.cumsum(np.bincount(bins[y==1],minlength=thre+1))[:thre].astype(float)
    P=float(y.sum())
    aucs=[]; aps=[]
    for w in range(int(window)+1):
        ext=grader.sequencing(y,seq,w)
        L=grader.new_sequence(ext,seq,w)
        tails=(y==0)&(ext>0)
        tail_tp=np.cumsum(np.bincount(bins[tails],weights=ext[tails],minlength=thre+1))[:thre]
        tp=tp_original+tail_tp
        # Official labels are multiplied by predictions in extended ranges,
        # then original anomaly samples are restored to 1.
        nlabels=P+tail_tp
        pnew=(P+nlabels)/2.0
        first=np.array([int(np.min(bins[a:b+1])) for a,b in L])
        existence=np.cumsum(np.bincount(first,minlength=thre+1))[:thre]/len(L)
        tpr=np.minimum(tp/pnew,1.0)*existence
        fpr=(npred-tp)/(len(y)-pnew)
        precision=tp/npred
        t=np.r_[0.,tpr,1.]; f=np.r_[0.,fpr,1.]
        aucs.append(float(np.dot(np.diff(f),(t[1:]+t[:-1])/2)))
        aps.append(float(np.dot(np.diff(t[:-1]),precision)))
    return float(sum(aps)/len(aps)),float(sum(aucs)/len(aucs))

def direct_official_vus(labels,scores,window,thre=250):
    result=OFFICIAL.generate_curve((np.asarray(labels)>0).astype(np.int32),normalize_scores(scores),int(window),'opt',thre)
    return float(result[-1]),float(result[-2])
