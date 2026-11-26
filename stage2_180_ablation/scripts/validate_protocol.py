import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='1'
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from exact_vus import exact_vus, direct_official_vus
from experiment_model import SeriesEngine, UNIQUE, independent_score
from run_experiments import OUT, DATA, load_series

def main():
    rows=[]; rng=np.random.default_rng(20261126)
    for n in [19,73,211]:
        cases=[]
        y=(rng.random(n)<.2).astype(int); y[0]=1; y[-1]=0
        cases.append(('random_ties',y,rng.integers(0,5,n).astype(float)))
        cases.append(('constant',y,np.ones(n)))
        y=np.zeros(n,dtype=int);y[:2]=1;y[-3:]=1;y[n//2:n//2+2]=1
        cases.append(('boundary_overlap',y,rng.normal(size=n)))
        cases.append(('almost_constant',y,1+1e-15*rng.normal(size=n)))
        for case,labels,scores in cases:
            for sw in [0,1,2,9,17]:
                t=time.perf_counter(); vp,vr=exact_vus(labels,scores,sw)
                dp,dr=direct_official_vus(labels,scores,sw)
                err=max(abs(vp-dp),abs(vr-dr))
                rows.append({'case':case,'n':n,'sw':sw,'VUS-PR_exact':vp,'VUS-PR_official':dp,
                             'VUS-ROC_exact':vr,'VUS-ROC_official':dr,'max_error':err,
                             'elapsed':time.perf_counter()-t,'status':'pass' if err<=2e-12 else 'FAIL'})
                assert err<=2e-12,(case,n,sw,err)
    files=pd.read_csv(OUT/'TSB-AD-M-Eva.csv').file_name.tolist()
    # All distinct ablation score generators are audited on a real raw series.
    name=files[1];X,y,tr,sw,base=load_series(name);engine=SeriesEngine(X,tr)
    for config in UNIQUE:
        s,meta,_=engine.score(config)
        vp,vr=exact_vus(y,s,sw);dp,dr=direct_official_vus(y,s,sw)
        independent=independent_score(X,tr,config)
        score_err=float(np.max(np.abs(independent-s)))
        err=max(abs(vp-dp),abs(vr-dr))
        rows.append({'case':name+' / '+config,'n':len(y),'sw':sw,'VUS-PR_exact':vp,'VUS-PR_official':dp,
                     'VUS-ROC_exact':vr,'VUS-ROC_official':dr,'max_error':err,'independent_model_score_error':score_err,
                     'status':'pass' if err<=2e-12 and score_err<=1e-12 else 'FAIL'})
        assert err<=2e-12 and score_err<=1e-12,(config,err,score_err)
    pd.DataFrame(rows).to_csv(OUT/'official_vus_equivalence_audit.csv',index=False)
    print('PASS',len(rows),'cases; max error',max(r['max_error'] for r in rows),flush=True)

if __name__=='__main__':main()
