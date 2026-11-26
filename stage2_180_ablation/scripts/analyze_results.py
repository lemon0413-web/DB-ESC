"""Dataset-cluster inference and transparent result audit (no selected exclusions)."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json, zipfile, hashlib, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, shapiro, skew, spearmanr, binomtest
from experiment_model import GROUPS, ALIASES, UNIQUE

OUT=Path(__file__).resolve().parents[1]
METRICS=['VUS-PR','VUS-ROC','AUC-PR','AUC-ROC']
SEED=20261126

def holm(p):
    p=np.asarray(p);order=np.argsort(p); adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    result=np.empty(len(p));result[order]=np.minimum(adjusted,1.)
    return result

def bootstrap(group,metric):
    g=group.groupby('dataset')[metric].agg(['sum','count','mean']).sort_index()
    n=len(g);idx=np.random.default_rng(SEED).integers(0,n,(10000,n))
    ser=g['sum'].to_numpy()[idx].sum(axis=1)/g['count'].to_numpy()[idx].sum(axis=1)
    ds=g['mean'].to_numpy()[idx].mean(axis=1)
    return (*np.quantile(ser,[.025,.975]),*np.quantile(ds,[.025,.975]))

def paired_stats(variant,full,metric):
    paired=variant[['file','dataset',metric]].merge(full[['file',metric]],on='file',suffixes=('_variant','_full'),validate='one_to_one')
    paired['difference']=paired[metric+'_variant']-paired[metric+'_full']
    delta=paired.groupby('dataset').difference.mean().sort_index().to_numpy(copy=True)
    delta[np.abs(delta)<=1e-12]=0.0
    nonzero=delta[np.abs(delta)>1e-14]
    p=float(wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if len(nonzero) else 1.
    # Exact paired sign-flip test using dataset means (all 2^17 assignments).
    signs=2*((np.arange(2**len(delta))[:,None]>>np.arange(len(delta)))&1)-1
    perm=np.abs(signs@delta/len(delta))
    pperm=float(np.mean(perm>=abs(delta.mean())-1e-15))
    signp=float(binomtest(int((nonzero>0).sum()),len(nonzero),.5).pvalue) if len(nonzero) else 1.
    idx=np.random.default_rng(SEED).integers(0,len(delta),(10000,len(delta)))
    boot=delta[idx].mean(axis=1);ci=np.quantile(boot,[.025,.975])
    q1,q3=np.quantile(delta,[.25,.75]);iqr=q3-q1
    absr=pd.Series(np.abs(nonzero)).rank(method='average').to_numpy()
    rbc=float((absr[nonzero>0].sum()-absr[nonzero<0].sum())/absr.sum()) if len(nonzero) else 0.
    return {'metric':metric,'n_datasets':len(delta),'n_series':len(paired),'mean_dataset_delta_variant_minus_full':float(delta.mean()),
      'mean_series_delta_variant_minus_full':float(paired.difference.mean()),
      'delta_dataset_cluster_ci95_low':ci[0],'delta_dataset_cluster_ci95_high':ci[1],
      'paired_test':'two-sided Wilcoxon signed-rank on dataset means','p_raw':p,
      'exact_sign_flip_p':pperm,'exact_sign_test_p':signp,'rank_biserial_correlation':rbc,
      'shapiro_delta_p':float(shapiro(delta).pvalue) if np.ptp(delta)>1e-14 else 1.,
      'dataset_delta_skew':float(skew(delta,bias=False)) if np.ptp(delta)>1e-14 else 0.,
      'dataset_delta_iqr_outliers':int(((delta<q1-1.5*iqr)|(delta>q3+1.5*iqr)).sum()),
      'variant_series_win_rate_vs_full':float((paired.difference>1e-12).mean()),
      'full_series_win_rate_vs_variant':float((paired.difference< -1e-12).mean()),
      'series_tie_rate':float((np.abs(paired.difference)<=1e-12).mean())}

def main():
    manifest=pd.read_csv(OUT/'TSB-AD-M-Eva.csv').file_name.tolist()
    baseline=pd.read_csv(OUT/'baseline_per_series_180.csv')
    baseline=baseline.rename(columns={'AUC-PR_official_trapezoidal':'AUC-PR_official_average_precision'})
    if 'parallel_workers' in baseline:baseline['parallel_workers']=baseline.parallel_workers.fillna(4).astype(int)
    else:baseline['parallel_workers']=4
    baseline['branch_a_used_for_final_score']=True
    baseline['branch_b_used_for_final_score']=True
    baseline.to_csv(OUT/'baseline_per_series_180.csv',index=False)
    source=pd.read_csv(OUT/'reference/eva180_final_results.csv')
    window_check=baseline[['file','evaluation_sliding_window']].merge(source[['file','sw_official']],on='file',validate='one_to_one')
    window_check['match']=window_check.evaluation_sliding_window==window_check.sw_official
    window_check.to_csv(OUT/'evaluation_window_reference_check.csv',index=False)
    if not window_check['match'].all():raise AssertionError('Evaluation window differs from original protocol')
    timing=pd.read_csv(OUT/'runtime_single_thread_independent.csv')
    all_unique=[];group_data={};qa=[];dataset_rows=[];summaries=[];tests=[]
    for filename,configs in GROUPS.items():
        data=pd.read_csv(OUT/filename)
        if 'AUC-PR_official_trapezoidal' in data:
            data=data.rename(columns={'AUC-PR_official_trapezoidal':'AUC-PR_official_average_precision'})
        data['parallel_workers']=data.parallel_workers.fillna(4).astype(int)
        data['branch_a_used_for_final_score']=data.experiment_config!='Branch B only'
        data['branch_b_used_for_final_score']=data.experiment_config!='Branch A only'
        def scale_caps(s):
            return any(item['requested']!=item['actual'] for item in json.loads(s))
        data['branch_b_any_scale_capped']=data.branch_b_scale_details.map(scale_caps)
        data['branch_b_requested_scales_fully_realized']=[
            sorted(json.loads(req))==sorted(json.loads(actual)) and not bool(fallback)
            for req,actual,fallback in zip(data.branch_b_requested_scales,data.branch_b_effective_scales,data.branch_b_scale_fallback)]
        data=data.merge(timing[['file','experiment_config','runtime_single_thread_seconds','score_sha256']],left_on=['file','canonical_config'],right_on=['file','experiment_config'],suffixes=('','_timing'),how='left',validate='many_to_one')
        data=data.drop(columns=['experiment_config_timing'])
        data['isolated_score_hash_matches']=data.score_sha256==data.score_sha256_timing
        for c in configs:
            subset=data[data.experiment_config==c]
            check={'file':filename,'experiment_config':c,'expected_rows':180,'actual_rows':len(subset),
                   'missing_files':len(set(manifest)-set(subset.file)),'unexpected_files':len(set(subset.file)-set(manifest)),
                   'duplicates':int(subset.file.duplicated().sum()),'non_success':int((subset.status!='success').sum()),
                   'nonfinite_metrics':int((~np.isfinite(subset[METRICS].to_numpy())).sum()),
                   'isolated_score_hash_mismatches':int((~subset.isolated_score_hash_matches).sum()),
                   'missing_isolated_runtime':int(subset.runtime_single_thread_seconds.isna().sum())}
            check['passed']=all(check[k]==0 for k in ['missing_files','unexpected_files','duplicates','non_success','nonfinite_metrics','missing_isolated_runtime','isolated_score_hash_mismatches']) and len(subset)==180
            qa.append(check)
            for metric in METRICS:
                serlo,serhi,dslo,dshi=bootstrap(subset,metric)
                stats=paired_stats(subset,baseline,metric)
                summary={'group':filename.replace('_180.csv',''),'experiment_config':c,'canonical_config':ALIASES.get(c,c),
                  'metric':metric,'n_series':len(subset),'n_datasets':subset.dataset.nunique(),
                  'series_average':subset[metric].mean(),'series_cluster_bootstrap_ci95_low':serlo,'series_cluster_bootstrap_ci95_high':serhi,
                  'dataset_average':subset.groupby('dataset')[metric].mean().mean(),'dataset_bootstrap_ci95_low':dslo,'dataset_bootstrap_ci95_high':dshi,
                  'runtime_isolated_mean_seconds':subset.runtime_single_thread_seconds.mean(),'runtime_isolated_median_seconds':subset.runtime_single_thread_seconds.median(),
                  'branch_a_window_capped_n':int(subset.branch_a_window_capped.sum()),
                  'branch_b_fallback_n':int(subset.branch_b_scale_fallback.sum()),
                  'branch_b_any_scale_capped_n':int(subset.branch_b_any_scale_capped.sum()),
                  'branch_b_requested_scales_fully_realized_n':int(subset.branch_b_requested_scales_fully_realized.sum()),
                  **{k:v for k,v in stats.items() if k not in ['metric','n_series','n_datasets']}}
                summaries.append(summary)
                if ALIASES.get(c,c)!='Full DB-ESC':
                    key=(ALIASES.get(c,c),metric)
                    if key not in [(t['experiment_config'],t['metric']) for t in tests]:
                        tests.append({'experiment_config':ALIASES.get(c,c),**stats})
                for ds,g in subset.groupby('dataset'):
                    dataset_rows.append({'group':filename.replace('_180.csv',''),'experiment_config':c,'dataset':ds,
                                         'n_series':len(g),'metric':metric,'mean':g[metric].mean()})
        data.to_csv(OUT/filename,index=False)
        group_data[filename]=data
    metadata_columns=['train_prefix_length','branch_a_actual_window','branch_b_effective_scales',
                      'cal_end_a','cal_end_b','cal_end_all',*METRICS]
    scored=pd.concat(group_data.values(),ignore_index=True).drop_duplicates(['file','canonical_config'])
    timing=timing.drop(columns=metadata_columns,errors='ignore').merge(
        scored[['file','canonical_config',*metadata_columns]].rename(columns={'canonical_config':'experiment_config'}),
        on=['file','experiment_config'],how='left',validate='one_to_one')
    timing.to_csv(OUT/'runtime_single_thread_independent.csv',index=False)
    tests=pd.DataFrame(tests)
    tests['p_holm']=holm(tests.p_raw)
    tests['sign_flip_p_holm']=holm(tests.exact_sign_flip_p)
    tests['holm_family']='all 21 unique ablations x 4 metrics = 84 dataset-level paired comparisons'
    tests.to_csv(OUT/'statistical_tests.csv',index=False)
    summary=pd.DataFrame(summaries).drop(columns=['p_raw'])
    summary=summary.merge(tests[['experiment_config','metric','p_raw','p_holm','sign_flip_p_holm']].rename(columns={'experiment_config':'canonical_config'}),on=['canonical_config','metric'],how='left')
    summary.loc[summary.canonical_config=='Full DB-ESC',['p_raw','p_holm','sign_flip_p_holm']]=1.
    summary[summary.group=='s2a_core_ablation'].to_csv(OUT/'s2a_core_ablation_summary.csv',index=False)
    summary[summary.group!='s2a_core_ablation'].to_csv(OUT/'s2b_summary.csv',index=False)
    pd.DataFrame(dataset_rows).to_csv(OUT/'dataset_level_results.csv',index=False)
    qa=pd.DataFrame(qa);qa.to_csv(OUT/'completeness_audit.csv',index=False)
    if not qa.passed.all(): raise AssertionError('Completeness audit failed; inspect CSV')
    variance=pd.read_csv(OUT/'s2c_real_branch_variance_180.csv')
    variance=variance.drop(columns=[m for m in METRICS if m in variance.columns])
    variance=variance.merge(baseline[['file',*METRICS]],on='file',validate='one_to_one')
    variance['parallel_workers']=variance.parallel_workers.fillna(4).astype(int)
    variance['branch_a_used_for_final_score']=True
    variance['branch_b_used_for_final_score']=True
    for quantity in ['var_ratio_a_over_b','std_ratio_a_over_b','pearson_unshrunk','pearson_shrunk_clipped_model',
                     'equal_variance_approximation','shrunk_equal_variance_approximation',
                     'fusion_variance_over_mean_branch_variance','variance_reduction_percent',
                     'maximum_score_ratio_b_over_a','maximum_labeled_anomaly_ratio_b_over_a',
                     'equal_approximation_relative_error']:
        variance[quantity+'_undefined']=variance[quantity].isna()
    variance.to_csv(OUT/'s2c_real_branch_variance_180.csv',index=False)
    if not np.allclose(variance.actual_fusion_variance,variance.general_covariance_formula,rtol=1e-12,atol=1e-12):
        raise AssertionError('General covariance identity failed numerical audit')
    if len(variance)!=180 or variance.clean160.sum()!=160:raise AssertionError('Incorrect variance/Clean-160 cohort')
    fields=['var_a','var_b','std_a','std_b','var_ratio_a_over_b','std_ratio_a_over_b','covariance_ab','pearson_unshrunk',
            'pearson_shrunk_clipped_model','actual_fusion_variance','general_covariance_formula','general_formula_absolute_error',
            'equal_variance_approximation','shrunk_equal_variance_approximation','fusion_variance_over_mean_branch_variance',
            'variance_reduction_percent','maximum_score_ratio_b_over_a','maximum_labeled_anomaly_ratio_b_over_a',
            'equal_approximation_signed_error','equal_approximation_absolute_error','equal_approximation_relative_error',
            'fusion_gain_over_mean_branch_VUS-PR','fusion_gain_over_best_branch_VUS-PR']
    vsummary=[];props=[];relationships=[]
    for cohort,g in [('All-180',variance),('Clean-160',variance[variance.clean160])]:
        for field in fields:
            x=g[field].dropna()
            vsummary.append({'cohort':cohort,'quantity':field,'n_total':len(g),'n_defined':len(x),'n_undefined':len(g)-len(x),
                             'mean':x.mean(),'median':x.median(),'std_sample':x.std(ddof=1),'q25':x.quantile(.25),'q75':x.quantile(.75),
                             'min':x.min(),'max':x.max()})
        for key,values in [('negative_correlation',g.pearson_unshrunk.dropna()<0),
                           ('actual_variance_reduction',g.fusion_variance_over_mean_branch_variance.dropna()<1),
                           ('fusion_improves_mean_branch_VUSPR',g['fusion_gain_over_mean_branch_VUS-PR'].dropna()>0),
                           ('fusion_improves_best_branch_VUSPR',g['fusion_gain_over_best_branch_VUS-PR'].dropna()>0),
                           ('equal_variance_within_2x',g.equal_variance_within_2x.dropna().astype(bool)),
                           *[(field,g[field]) for field in ['zero_variance_a','zero_variance_b','near_zero_variance_a','near_zero_variance_b','undefined_correlation','undefined_variance_ratio','undefined_reduction_ratio']]]:
            props.append({'cohort':cohort,'quantity':key,'n_total':len(g),'n_defined':len(values),'n_true':int(values.sum()),
                          'proportion_among_defined':float(values.mean()) if len(values) else None})
        for outcome in ['fusion_gain_over_mean_branch_VUS-PR','fusion_gain_over_best_branch_VUS-PR']:
            for level in ['series_descriptive','dataset_means_primary_exploratory']:
                pair=g[['dataset','variance_reduction_percent',outcome]].dropna()
                if level.startswith('dataset'): pair=pair.groupby('dataset').mean()
                a=pair.variance_reduction_percent.to_numpy();b=pair[outcome].to_numpy()
                r,p=spearmanr(a,b)
                relationships.append({'cohort':cohort,'level':level,'outcome':outcome,'n':len(pair),
                                      'spearman_rho':r,'p_raw':p,'interpretation':'exploratory association; not causal evidence'})
    pd.DataFrame(vsummary).to_csv(OUT/'s2c_variance_summary.csv',index=False)
    pd.DataFrame(props).to_csv(OUT/'s2c_variance_proportions.csv',index=False)
    relationships=pd.DataFrame(relationships);relationships['p_holm_exploratory']=holm(relationships.p_raw)
    relationships.to_csv(OUT/'s2c_variance_performance_relationship.csv',index=False)
    # Descriptive sensitivity only: preserve the complete-180 primary results.
    scale_rows=[]
    for filename in ['s2b_multiscale_ablation_180.csv','s2b_period_ablation_180.csv']:
        data=group_data[filename]
        for c,group in data.groupby('experiment_config',sort=False):
            realized=group.branch_b_requested_scales_fully_realized if 'multiscale' in filename else ~group.branch_a_window_capped
            for state,mask in [('requested_setting_fully_realized',realized),('capped_dropped_or_fallback',~realized)]:
                sub=group[mask]
                for metric in METRICS:
                    scale_rows.append({'group':filename,'experiment_config':c,'stratum':state,'n':len(sub),'metric':metric,
                                       'series_mean':sub[metric].mean() if len(sub) else None,
                                       'status':'defined' if len(sub) else 'empty_stratum_not_an_experiment_failure',
                                       'interpretation':'descriptive execution audit; different cohorts, not a controlled performance comparison'})
    pd.DataFrame(scale_rows).to_csv(OUT/'requested_vs_realized_window_sensitivity.csv',index=False)
    ap=baseline[['file','dataset','experiment_config','train_prefix_length','branch_a_actual_window','branch_b_effective_scales',
                 *METRICS,'AUC-PR_legacy_order_ties','AP_minus_legacy','unique_score_count','status']].copy()
    ap.to_csv(OUT/'average_precision_implementation_audit.csv',index=False)
    audit={'completed':True,'expected_unique_score_configurations':len(UNIQUE),'expected_grouped_metric_rows':sum(len(c) for c in GROUPS.values())*180,
           'actual_grouped_metric_rows':int(qa.actual_rows.sum()),'all_26_configuration_checks_passed':bool(qa.passed.all()),
           'variance_rows':len(variance),'clean160_rows':int(variance.clean160.sum()),'undefined_variance_quantities_intentionally_preserved':True,
           'metric_nan_count':int(qa.nonfinite_metrics.sum()),'isolated_timing_rows':len(timing),'holm_comparisons':len(tests),
           'max_covariance_identity_error':float(variance.general_formula_absolute_error.max()),
           'max_official_baseline_parity_error':float(baseline.official_parity_error.max()),
           'frozen_model_sha256':hashlib.sha256((OUT/'scripts/dbesc.py').read_bytes()).hexdigest()}
    (OUT/'validation_report.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
