import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[_key]='1'
os.environ['PYTHONIOENCODING']='utf-8'
import argparse, hashlib, importlib.util, json, platform, re, sys, time, traceback, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits, threadpool_info
import dbesc as db
from exact_vus import exact_vus, direct_official_vus, OFFICIAL
from experiment_model import SeriesEngine, GROUPS, UNIQUE, ALIASES, independent_score

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
WORK=Path(os.environ.get('STAGE2_WORK_DIR',str(ROOT/'work')))
CHECK=WORK/'checkpoints'
DATA=Path(os.environ.get('STAGE2_DATA_DIR',str(WORK/'data')))
THREAD_LIMIT=threadpool_limits(limits=1)

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.generic): return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v): return None
        return v
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def load_series(name):
    frame=pd.read_csv(DATA/name)
    X=frame.iloc[:,:-1].to_numpy(dtype=np.float64)
    y=frame.iloc[:,-1].to_numpy(dtype=np.int32)
    if not np.isfinite(X).all(): raise ValueError('Nonfinite raw input')
    if not np.isin(y,[0,1]).all(): raise ValueError('Nonbinary raw label')
    tr=max(64,min(int(re.search(r'_tr_(\d+)_',name).group(1)),len(y)-1))
    first=int(re.search(r'_1st_(\d+)',name).group(1))
    meta={'file':name,'dataset':name.split('_')[1],'T':len(y),'C':X.shape[1],
          'train_prefix_length':tr,'nominal_train_prefix_length':int(re.search(r'_tr_(\d+)_',name).group(1)),
          'first_anomaly_filename':first,'clean160':first>tr,'actual_prefix_anomaly_points':int(y[:tr].sum()),
          'raw_nan_count':0,'status':'success'}
    meta['parallel_workers']=int(os.environ.get('STAGE2_WORKERS','4'))
    spec=importlib.util.spec_from_file_location('official_sliding',OUT/'scripts/official_evaluation/slidingWindows.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with warnings.catch_warnings(record=True) as warn:
        sw=int(mod.find_length_rank(X[:,0].reshape(-1,1),rank=1))
    meta['evaluation_sliding_window']=sw
    meta['sliding_window_warnings']=' | '.join(str(w.message) for w in warn)
    return X,y,tr,sw,meta

def metrics(y,s,sw,direct=False):
    if not np.isfinite(s).all(): raise ValueError('Nonfinite model score')
    t=time.perf_counter()
    vp,vr=exact_vus(y,s,sw,thre=250)
    ap=float(average_precision_score(y,s)); ar=float(roc_auc_score(y,s))
    ys=y[np.argsort(-s,kind='mergesort')]
    legacy=float(np.sum((np.cumsum(ys)/np.arange(1,len(y)+1))[ys==1])/np.sum(y))
    # The official TSB-AD metric_PR also uses sklearn Average Precision.
    official_pr=float(OFFICIAL.basic_metricor().metric_PR(y,s))
    result={'VUS-PR':vp,'VUS-ROC':vr,'AUC-PR':ap,'AUC-ROC':ar,'AUC-PR_legacy_order_ties':legacy,
            'AUC-PR_official_average_precision':official_pr,'AP_minus_legacy':ap-legacy,
            'unique_score_count':int(len(np.unique(s))),'metric_engine':'exact_full_resolution_official_formula',
            'metric_seconds':time.perf_counter()-t}
    if direct:
        t=time.perf_counter(); dp,dr=direct_official_vus(y,s,sw,250)
        result.update({'direct_official_VUS-PR':dp,'direct_official_VUS-ROC':dr,
                       'official_parity_error':max(abs(vp-dp),abs(vr-dr)),
                       'direct_official_seconds':time.perf_counter()-t})
        if result['official_parity_error']>2e-12: raise AssertionError(result)
        result['VUS-PR']=dp; result['VUS-ROC']=dr
        result['metric_engine']='direct_official_generate_curve_opt_thre250'
    if not all(np.isfinite(result[k]) for k in ['VUS-PR','VUS-ROC','AUC-PR','AUC-ROC']): raise ValueError('Nonfinite metrics')
    return result

def safe_ratio(a,b): return float(a/b) if b!=0 and np.isfinite(a) and np.isfinite(b) else None

def variance_audit(out,y,base):
    end=int(out['cal_end_all']); a=out['z_a'][:end]; b=out['z_b'][:end]
    va=float(np.var(a)); vb=float(np.var(b)); vf=float(np.var(out['score_fuse'][:end]))
    cov=float(np.mean((a-a.mean())*(b-b.mean())))
    rho=float(np.corrcoef(a,b)[0,1]) if va>0 and vb>0 else None
    rmodel=float(out['r_ab']); rshrink=0.85*float(np.clip(rho,-.99,.99)) if rho is not None else None
    theo=0.25*(va+vb+2*cov)
    approx=0.25*(va+vb)*(1+rho) if rho is not None else None
    shrunk=0.25*(va+vb)*(1+rshrink) if rho is not None else None
    ratio=safe_ratio(vf,0.5*(va+vb))
    row={**base,'experiment_config':'Full DB-ESC variance audit','cal_end_a':int(out['cal_end_a']),
         'cal_end_b':int(out['cal_end_b']),'cal_end_all':end,'variance_ddof':0,
         'var_a':va,'var_b':vb,'std_a':float(np.sqrt(va)),'std_b':float(np.sqrt(vb)),
         'var_ratio_a_over_b':safe_ratio(va,vb),'std_ratio_a_over_b':safe_ratio(np.sqrt(va),np.sqrt(vb)),
         'covariance_ab':cov,'pearson_unshrunk':rho,'pearson_shrunk_clipped_model':rshrink,
         'model_returned_r_ab':rmodel,'model_correlation_fallback':rho is None,
         'actual_fusion_variance':vf,'general_covariance_formula':theo,
         'general_formula_absolute_error':abs(vf-theo),'equal_variance_approximation':approx,
         'shrunk_equal_variance_approximation':shrunk,'fusion_variance_over_mean_branch_variance':ratio,
         'variance_reduction_percent':100*(1-ratio) if ratio is not None else None,
         'maximum_score_ratio_b_over_a':safe_ratio(float(out['z_b'].max()),float(out['z_a'].max())),
         'maximum_labeled_anomaly_ratio_b_over_a':safe_ratio(float(out['z_b'][y==1].max()),float(out['z_a'][y==1].max())),
         'equal_approximation_signed_error':approx-vf if approx is not None else None,
         'equal_approximation_absolute_error':abs(approx-vf) if approx is not None else None,
         'equal_approximation_relative_error':safe_ratio(approx-vf,vf) if approx is not None else None,
         'zero_variance_a':va==0,'zero_variance_b':vb==0,'near_zero_variance_a':va<=1e-12,'near_zero_variance_b':vb<=1e-12,
         'undefined_correlation':rho is None,'undefined_variance_ratio':vb==0,
         'undefined_reduction_ratio':va+vb==0,
         'equal_variance_within_2x':0.5<=va/vb<=2 if vb>0 else None}
    return row

def task_baseline(name):
    target=CHECK/'baseline'/f'{name}.json'
    if target.exists(): return json.loads(target.read_text(encoding='utf-8'))
    try:
        X,y,tr,sw,base=load_series(name)
        t=time.perf_counter(); out=db.fit_score_dbesc(X,tr)
        secs=time.perf_counter()-t
        engine=SeriesEngine(X,tr); score,meta,components=engine.score('Full DB-ESC')
        parity=max(float(np.max(np.abs(score-out['score_final']))),float(np.max(np.abs(components['za']-out['z_a']))),float(np.max(np.abs(components['zb']-out['z_b']))))
        if parity>1e-12 or meta['cal_end_all']!=out['cal_end_all']: raise AssertionError('Wrapper failed baseline equivalence')
        measured=metrics(y,out['score_final'],sw,direct=True)
        row={**base,**meta,**measured,'experiment_config':'Full DB-ESC','runtime_parallel_fit_seconds':secs,
             'wrapper_max_score_error':parity}
        variance=variance_audit(out,y,{**base,**meta})
        variance['full_model_VUS-PR']=measured['VUS-PR']
        # Matched unsmoothed branch controls for variance/performance analysis.
        vp_f,_=exact_vus(y,out['score_fuse'],sw)
        vp_a,_=exact_vus(y,out['z_a'],sw); vp_b,_=exact_vus(y,out['z_b'],sw)
        variance.update({'unsmoothed_fusion_VUS-PR':vp_f,'unsmoothed_branch_a_VUS-PR':vp_a,
                         'unsmoothed_branch_b_VUS-PR':vp_b,
                         'fusion_gain_over_mean_branch_VUS-PR':vp_f-0.5*(vp_a+vp_b),
                         'fusion_gain_over_best_branch_VUS-PR':vp_f-max(vp_a,vp_b)})
        result={'baseline':row,'variance':variance}
    except Exception:
        result={'error':traceback.format_exc(),'file':name,'phase':'baseline'}
    dump(target,result)
    return result

def task_ablations(name):
    target=CHECK/'ablations'/f'{name}.json'
    if target.exists(): return json.loads(target.read_text(encoding='utf-8'))
    try:
        X,y,tr,sw,base=load_series(name); engine=SeriesEngine(X,tr)
        baseline=json.loads((CHECK/'baseline'/f'{name}.json').read_text(encoding='utf-8'))['baseline']
        rows=[]
        for config in UNIQUE:
            if config=='Full DB-ESC':
                score,_,_=engine.score(config)
                baseline['score_sha256']=hashlib.sha256(np.asarray(score,dtype=np.float64).tobytes()).hexdigest()
                rows.append(baseline); continue
            t=time.perf_counter(); score,meta,components=engine.score(config)
            secs=time.perf_counter()-t
            row={**base,**meta,**metrics(y,score,sw),'experiment_config':config,
                 'runtime_parallel_cached_call_seconds':secs,
                 'score_sha256':hashlib.sha256(np.asarray(score,dtype=np.float64).tobytes()).hexdigest()}
            rows.append(row)
        result={'rows':rows}
    except Exception:
        result={'error':traceback.format_exc(),'file':name,'phase':'ablations'}
    dump(target,result)
    return result

def log(msg):
    msg=time.strftime('%Y-%m-%d %H:%M:%S')+' '+msg
    print(msg,flush=True)
    with (OUT/'logs/execution.log').open('a',encoding='utf-8') as fh: fh.write(msg+'\n')

def parallel_phase(files,fn,workers):
    t=time.perf_counter(); errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,name):name for name in files}
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            if 'error' in result:
                errors.append(result); log('ERROR '+result['file']+' '+result['error'])
            log(f'{fn.__name__} {i}/{len(files)} {futures[future]} elapsed={time.perf_counter()-t:.1f}s')
    if errors:
        dump(OUT/'logs/errors.json',errors)
        raise RuntimeError(f'{len(errors)} errors; all retained in logs/errors.json')
    return time.perf_counter()-t

def write_baseline():
    values=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((CHECK/'baseline').glob('*.json'))]
    for v in values:
        row=v['baseline']
        if 'AUC-PR_official_trapezoidal' in row:
            row['AUC-PR_official_average_precision']=row.pop('AUC-PR_official_trapezoidal')
    rows=pd.DataFrame([v['baseline'] for v in values])
    variance=pd.DataFrame([v['variance'] for v in values])
    rows.to_csv(OUT/'baseline_per_series_180.csv',index=False)
    variance.to_csv(OUT/'s2c_real_branch_variance_180.csv',index=False)
    reference=pd.read_csv(OUT/'reference/eva180_final_results.csv')
    merged=rows.merge(reference[['file','DB_ESC__VUSPR_off','DB_ESC__AUCPR','sw_official']],on='file',validate='one_to_one')
    merged['reference_VUS-PR_difference']=merged['VUS-PR']-merged['DB_ESC__VUSPR_off']
    merged['reference_legacy_AP_difference']=merged['AUC-PR_legacy_order_ties']-merged['DB_ESC__AUCPR']
    merged.to_csv(OUT/'baseline_reference_per_series_differences.csv',index=False)
    ser=float(rows['VUS-PR'].mean()); ds=float(rows.groupby('dataset')['VUS-PR'].mean().mean())
    records=[]
    for metric,target,actual in [('Series-Average VUS-PR',.4773,ser),('Dataset-Average VUS-PR',.3758,ds),
                                  ('Legacy Series-Average AUC-PR',.4653,float(rows['AUC-PR_legacy_order_ties'].mean()))]:
        records.append({'metric':metric,'manuscript_reference':target,'measured':actual,'difference':actual-target,
                        'matches_4_decimal_reference':round(actual,4)==target,'series_n':len(rows),'dataset_n':rows.dataset.nunique(),
                        'max_per_series_reference_VUS_error':merged['reference_VUS-PR_difference'].abs().max(),
                        'max_official_engine_parity_error':rows.official_parity_error.max(),
                        'max_wrapper_error':rows.wrapper_max_score_error.max()})
    pd.DataFrame(records).to_csv(OUT/'baseline_reproduction_check.csv',index=False)
    log(f'BASELINE Series VUS-PR={ser:.12f}; Dataset VUS-PR={ds:.12f}; standard AP={rows["AUC-PR"].mean():.12f}; legacy AP={rows["AUC-PR_legacy_order_ties"].mean():.12f}')
    if len(rows)!=180 or rows.dataset.nunique()!=17 or not all(r['matches_4_decimal_reference'] for r in records[:2]):
        raise RuntimeError('Baseline reference failed. Frozen model unchanged; investigate before ablations.')

def write_ablations():
    values=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((CHECK/'ablations').glob('*.json'))]
    rows=[row for v in values for row in v['rows']]
    for row in rows:
        if 'AUC-PR_official_trapezoidal' in row:
            row['AUC-PR_official_average_precision']=row.pop('AUC-PR_official_trapezoidal')
    allrows=pd.DataFrame(rows)
    for filename,configs in GROUPS.items():
        frames=[]
        for config in configs:
            frame=allrows[allrows.experiment_config==ALIASES.get(config,config)].copy()
            frame['experiment_config']=config
            frame['canonical_config']=ALIASES.get(config,config)
            frames.append(frame)
        pd.concat(frames,ignore_index=True).to_csv(OUT/filename,index=False)

def isolated_timing(files,configs):
    target=OUT/'runtime_single_thread_independent.csv'
    rows=pd.read_csv(target).to_dict('records') if target.exists() else []
    done={(r['file'],r['experiment_config']) for r in rows}
    log('Starting isolated timing: one process, one CPU thread, no concurrent experiment workers, fitting and complete scoring only; 1 timed call/config/file after per-file warmup.')
    for i,name in enumerate(files,1):
        X,y,tr,sw,meta=load_series(name)
        # Warm default model once per file, outside measured interval.
        if any((name,c) not in done for c in configs): db.fit_score_dbesc(X,tr)
        for config in configs:
            if (name,config) in done: continue
            t=time.perf_counter(); score=independent_score(X,tr,config); elapsed=time.perf_counter()-t
            if not np.isfinite(score).all(): raise ValueError('Nonfinite timing-run score')
            rows.append({'file':name,'dataset':meta['dataset'],'experiment_config':config,
                         'runtime_single_thread_seconds':elapsed,'timed_repetitions':1,
                         'score_sha256':hashlib.sha256(np.asarray(score,dtype=np.float64).tobytes()).hexdigest(),
                         'protocol':'isolated_one_process_one_thread_full_fit_and_score',
                         'status':'success'})
        pd.DataFrame(rows).to_csv(target,index=False)
        log(f'isolated_timing {i}/{len(files)}; records={len(rows)}')

def config(workers):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    settings={'frozen_model_sha256':sha(OUT/'scripts/dbesc.py'),'original_model_path':'D:/Downloads/Google/dbesc.py',
       'raw_archive':'D:/ms_cmesc/TSB-AD-M.zip','raw_archive_sha256':sha('D:/ms_cmesc/TSB-AD-M.zip') if Path('D:/ms_cmesc/TSB-AD-M.zip').exists() else None,
       'file_list_sha256':sha(OUT/'TSB-AD-M-Eva.csv'),'official_TSB_AD_commit':'303dffbfc11612995b17b06633f8c9d0ea0e38e5',
       'official_source':'https://github.com/TheDatumOrg/TSB-AD','official_basic_metrics_sha256':sha(OUT/'scripts/official_evaluation/basic_metrics.py'),
       'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
       'pandas':pd.__version__,'workers':workers,'threads_per_worker':1,'threadpools':threadpool_info(),
       'defaults':{'causal':False,'k_nn':3,'d_emb_max':12,'c_eff_max':24,'windows':[48,96,192],'lam':.15,'q_tail':.98,'w_ds':16,'reg_alpha':.15,'strict_stride_truncation':True},
       'seed_bootstrap':20261126,'bootstrap_resamples':10000,'zero_variance_audit_tolerance':1e-12,
       'AUC_PR':'sklearn.metrics.average_precision_score with grouped ties',
       'legacy_AP':'stable descending mergesort, point-by-point precision (order-dependent on ties)',
       'VUS':{'thre':250,'full_time_resolution':True,'all_integer_buffer_widths':True,'baseline':'unmodified official generate_curve version opt for all 180',
              'ablations':'algebraically exact threshold-bin sums; parity checked against unmodified official implementation',
              'score_normalization':'original runner min-max, span <= 1e-12 -> constant zeros',
              'sliding_window':'official find_length_rank(raw first channel, rank=1), recomputed and crosschecked against original'},
       'prefix':'original runner max(64,min(filename _tr_,T-1)); no label-assisted truncation',
       'preprocessing':'frozen model prefix median/IQR, std fallback and clipping [-15,15]; no external scaling',
       'clean160':'filename first_anomaly > training_prefix_length; actual prefix labels also audited',
       'S2A_stage1':'retained in all S2A branch-B controls; Raw means no Stage-2 branch calibration',
       'S2A_smoothing':'retained except explicitly without Smoothing controls',
       'S2B_stage2':'branch A POT always retained; Stage-2 ablation refers only to branch B',
       'std_scaled_zero_rule':'if prefix std <= 1e-12, divisor=1 and flag; no arbitrary epsilon added to measured variance',
       'variance_ddof':0,'equal_variance_approximation':'(var_a+var_b)*(1+raw_rho)/4',
       'shrinkage':'0.85*clip(raw_rho,-0.99,0.99); undefined audit correlation left null, model diagnostic fallback separately reported',
       'families':GROUPS,'unique_inference_configs':UNIQUE,'optional_S2E_S2F':'deferred until core experiments complete; not required for core package',
       'provided_manuscript':'DBESCNeurocomputingManuscript_provided.pdf (copied unchanged, not revised)'}
    previous=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8')) if (OUT/'experiment_config.json').exists() else {}
    invocations=previous.get('execution_invocations',[])
    invocations.append({'argv':sys.argv,'workers':workers,'data_dir':str(DATA),'checkpoint_dir':str(CHECK),
                        'script_sha256':{p.name:sha(p) for p in (OUT/'scripts').glob('*.py')}})
    settings['execution_invocations']=invocations
    dump(OUT/'experiment_config.json',settings)

def main():
    global DATA,WORK,CHECK
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['baseline','ablations','timing','all'],default='all')
    parser.add_argument('--workers',type=int,default=4); parser.add_argument('--timing-core-only',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--work-dir',type=Path,default=WORK)
    args=parser.parse_args()
    DATA=args.data_dir.resolve();WORK=args.work_dir.resolve();CHECK=WORK/'checkpoints'
    os.environ['STAGE2_DATA_DIR']=str(DATA);os.environ['STAGE2_WORK_DIR']=str(WORK)
    os.environ['STAGE2_WORKERS']=str(args.workers)
    config(args.workers)
    files=pd.read_csv(OUT/'TSB-AD-M-Eva.csv').file_name.tolist()
    if len(files)!=180 or len(set(files))!=180: raise ValueError('Expected exact unique official Eva-180 manifest')
    if args.phase in ['baseline','all']:
        parallel_phase(files,task_baseline,args.workers); write_baseline()
    if args.phase in ['ablations','all']:
        parallel_phase(files,task_ablations,args.workers); write_ablations()
    if args.phase in ['timing','all']:
        isolated_timing(files,UNIQUE[:8] if args.timing_core_only else UNIQUE)
    log('PHASE COMPLETE '+args.phase)

if __name__=='__main__': main()
