"""Ablation wrappers. dbesc.py is frozen and never edited.

Branch-B Stage-1 POT is retained throughout S2-A (including Raw/ECDF).
Only the explicitly named S2-B stage controls disable it. Default wrappers
are checked against the original fit_score_dbesc on every one of 180 files.
"""
import inspect, json, time
import numpy as np
import dbesc as db

CORE=['Branch A only','Branch B only','A+B Raw Fusion','A+B Std-Scaled Fusion',
      'A+B ECDF without Smoothing','A+B ECDF with Smoothing',
      'A+B POT without Smoothing','Full DB-ESC']
SCALES=[(48,),(96,),(192,),(48,96),(48,192),(96,192),(48,96,192)]
GROUPS={
 's2a_core_ablation_180.csv':CORE,
 's2b_multiscale_ablation_180.csv':['W = {'+','.join(map(str,w))+'}' for w in SCALES],
 's2b_pot_stage_ablation_180.csv':['No Stage-1 POT','No Stage-2 POT','No Stage-1 and Stage-2 POT','Full Two-Stage POT'],
 's2b_period_ablation_180.csv':['Fixed Window = 32','Fixed Window = 64','Fixed Window = 128','FFT Period-Adaptive Window'],
 's2b_branch_a_internal_180.csv':['k-NN only','PCA Orthogonal Residual only','k-NN + PCA Orthogonal Residual']}
ALIASES={'W = {48,96,192}':'Full DB-ESC','Full Two-Stage POT':'Full DB-ESC',
         'FFT Period-Adaptive Window':'Full DB-ESC','k-NN + PCA Orthogonal Residual':'Full DB-ESC'}
UNIQUE=CORE+[x for group in list(GROUPS.values())[1:] for x in group if x not in ALIASES]

def smooth(s): return 0.5*s+0.5*db.moving_average(s,16,causal=False)
def ecdf(s,end):
    ref=np.sort(s[:end]); n=len(ref)
    return -np.log(np.clip((n-np.searchsorted(ref,s,side='right')+0.5)/(n+1),1e-12,1.))

def a_branch(Z,tr,fixed=None,component='both'):
    fn=db.branch_a_temporal_manifold
    if fixed is not None or component!='both':
        source=inspect.getsource(fn).replace('def branch_a_temporal_manifold(','def modified_branch_a(')
        if fixed is not None:
            source=source.replace('w = estimate_period_acf(Z[:tr], min_p=24, max_p=160)',f'w = {int(fixed)}')
        if component!='both':
            replacement='sc = s1' if component=='knn' else 'sc = 0.5 * s2'
            source=source.replace('sc = s1 + 0.5 * s2',replacement)
        scope=dict(db.__dict__)
        exec(compile(source,'<explicit_branch_a_ablation>','exec'),scope)
        fn=scope['modified_branch_a']
    score,la=fn(Z,tr,strict_stride_truncation=True)
    fallback=tr-la<32
    if fallback: score,la=fn(Z,tr,causal=True,strict_stride_truncation=True)
    requested=int(fixed if fixed is not None else db.estimate_period_acf(Z[:tr]))
    actual=min(requested,max(8,tr//5))
    return score,la,{'branch_a_requested_window':requested,'branch_a_actual_window':actual,
                    'branch_a_window_capped':actual!=requested,'branch_a_causal_fallback':fallback,
                    'branch_a_component':component}

def b_branch(Z,tr,windows=(48,96,192),stage1=True,cache=None):
    cache={} if cache is None else cache
    T,C=Z.shape; step=max(1,T//3000)
    def calculate(causal):
        scores=[]; las=[]; actual=[]; detail=[]
        if C<=1:
            return np.zeros(T),0,[],[],False,'univariate_zero_branch'
        for w in windows:
            req_la=0 if causal else w-w//2+step-1
            if tr<2*w or tr-req_la<32: continue
            key=(w,causal)
            if key not in cache: cache[key]=db._single_scale_spd_cov(Z,tr,w,causal=causal,strict_stride_truncation=True)
            sw,la=cache[key]; end=tr-la
            scores.append(db.evt_pot_calibrate(sw,end) if stage1 else sw)
            las.append(la); actual.append(min(w,max(12,tr//4)))
            detail.append({'requested':w,'actual':actual[-1],'cal_end':end,'causal':causal,'stage1_applied':stage1})
        if not scores:
            actual_w=min(48,max(12,tr//4))
            use_causal=causal or tr-(actual_w//2+step-1)<32
            key=(48,use_causal)
            if key not in cache: cache[key]=db._single_scale_spd_cov(Z,tr,48,causal=use_causal,strict_stride_truncation=True)
            score,la=cache[key]
            detail=[{'requested':48,'actual':actual_w,'cal_end':tr-la,'causal':use_causal,'stage1_applied':False}]
            return score,la,[actual_w],detail,True,'no_eligible_requested_scale'
        return np.mean(scores,axis=0),max(las),actual,detail,False,''
    score,la,actual,detail,fallback,reason=calculate(False)
    outer=tr-la<32
    if outer: score,la,actual,detail,fallback,reason=calculate(True)
    meta={'branch_b_requested_scales':json.dumps(windows),'branch_b_effective_scales':json.dumps(actual),
          'branch_b_scale_details':json.dumps(detail),'branch_b_scale_fallback':fallback,
          'branch_b_fallback_reason':reason,'branch_b_outer_causal_fallback':outer,
          'branch_b_stage1_enabled':stage1,'branch_b_stage1_actually_applied':any(x['stage1_applied'] for x in detail)}
    return score,la,meta

class SeriesEngine:
    def __init__(self,X,tr,cache=True):
        t=time.perf_counter(); self.Z=db.robust_standardize(X,tr)
        self.standardize_seconds=time.perf_counter()-t
        self.X=X; self.tr=tr; self.use_cache=cache; self.ac={}; self.bc={}; self.scale_cache={}
    def a(self,fixed=None,component='both'):
        key=(fixed,component)
        if key not in self.ac:
            t=time.perf_counter(); value=a_branch(self.Z,self.tr,fixed,component)
            self.ac[key]=(value,time.perf_counter()-t)
        return self.ac[key]
    def b(self,windows=(48,96,192),stage1=True):
        key=(windows,stage1)
        if key not in self.bc:
            t=time.perf_counter(); value=b_branch(self.Z,self.tr,windows,stage1,self.scale_cache if self.use_cache else None)
            self.bc[key]=(value,time.perf_counter()-t)
        return self.bc[key]
    def score(self,name):
        name=ALIASES.get(name,name)
        fixed=int(name.split('=')[1]) if name.startswith('Fixed Window') else None
        component='knn' if name=='k-NN only' else 'pca' if name=='PCA Orthogonal Residual only' else 'both'
        windows=tuple(map(int,name.split('{')[1].split('}')[0].split(','))) if name.startswith('W =') else (48,96,192)
        stage1=name not in ['No Stage-1 POT','No Stage-1 and Stage-2 POT']
        stage2=name not in ['No Stage-2 POT','No Stage-1 and Stage-2 POT']
        (sa,la,ma),ta=self.a(fixed,component)
        (sb,lb,mb),tb=self.b(windows,stage1)
        ca=max(1,self.tr-la); cb=max(1,self.tr-lb)
        t=time.perf_counter()
        za=db.evt_pot_calibrate(sa,ca)
        zb=db.evt_pot_calibrate(sb,cb) if stage2 else sb
        std_fallback_a=std_fallback_b=False
        if name=='Branch A only': result=smooth(za)
        elif name=='Branch B only': result=smooth(zb)
        elif name=='A+B Raw Fusion': result=smooth(0.5*(sa+sb))
        elif name=='A+B Std-Scaled Fusion':
            astd=float(np.std(sa[:ca])); bstd=float(np.std(sb[:cb]))
            std_fallback_a=astd<=1e-12; std_fallback_b=bstd<=1e-12
            result=smooth(0.5*(sa/(astd if not std_fallback_a else 1.)+sb/(bstd if not std_fallback_b else 1.)))
        elif name.startswith('A+B ECDF'):
            result=0.5*(ecdf(sa,ca)+ecdf(sb,cb))
            if name.endswith('with Smoothing'): result=smooth(result)
        else:
            result=0.5*(za+zb)
            if name!='A+B POT without Smoothing': result=smooth(result)
        post=time.perf_counter()-t
        meta={**ma,**mb,'cal_end_a':ca,'cal_end_b':cb,'cal_end_all':min(ca,cb),
              'branch_a_used_for_final_score':name!='Branch B only','branch_b_used_for_final_score':name!='Branch A only',
              'branch_b_stage2_enabled':stage2,'branch_a_stage2':'ECDF' if name.startswith('A+B ECDF') else 'none' if name in ['A+B Raw Fusion','A+B Std-Scaled Fusion'] else 'POT',
              'std_scaling_zero_a':std_fallback_a,'std_scaling_zero_b':std_fallback_b,
              'component_accounted_batch_seconds':self.standardize_seconds+ta+tb+post}
        return result,meta,{'sa':sa,'sb':sb,'za':za,'zb':zb,'fuse':0.5*(za+zb)}

def independent_score(X,tr,name):
    if ALIASES.get(name,name)=='Full DB-ESC':
        return db.fit_score_dbesc(X,tr)['score_final']
    if name in ['Branch A only','Branch B only']:
        Z=db.robust_standardize(X,tr)
        if name=='Branch A only': raw,la,_=a_branch(Z,tr)
        else: raw,la,_=b_branch(Z,tr)
        return smooth(db.evt_pot_calibrate(raw,max(1,tr-la)))
    return SeriesEngine(X,tr,cache=False).score(name)[0]
