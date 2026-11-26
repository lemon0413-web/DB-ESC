from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parents[1]

def main():
    a=pd.read_csv(OUT/'s2a_core_ablation_summary.csv')
    b=pd.read_csv(OUT/'s2b_summary.csv')
    v=pd.read_csv(OUT/'s2c_variance_summary.csv')
    p=pd.read_csv(OUT/'s2c_variance_proportions.csv')
    r=pd.read_csv(OUT/'s2c_variance_performance_relationship.csv')
    valid=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'))
    baseline=pd.read_csv(OUT/'baseline_per_series_180.csv')
    tests=pd.read_csv(OUT/'statistical_tests.csv')
    std=a[(a.experiment_config=='A+B Std-Scaled Fusion')&(a.metric=='VUS-PR')].iloc[0]
    lines=['Stage-2 final core experiments: result notes','',
           'Scope: S2-A, S2-B and S2-C, all 180 official TSB-AD-M-Eva raw CSV sequences; 17 datasets.',
           'SKAB was not rerun. Optional ExCal and S2-F multiseed stress tests were not executed.',
           'The manuscript PDF is the latest supplied local PDF, copied unchanged; these new results have not been inserted into it.','',
           'BASELINE REPRODUCTION',
           f'Series-average VUS-PR: {baseline["VUS-PR"].mean():.12f}',
           f'Dataset-average VUS-PR: {baseline.groupby("dataset")["VUS-PR"].mean().mean():.12f}',
           f'Standard Average Precision: {baseline["AUC-PR"].mean():.12f}',
           f'Legacy pointwise tie-dependent AP: {baseline["AUC-PR_legacy_order_ties"].mean():.12f}',
           f'Series with changed AP: {(baseline.AP_minus_legacy.abs()>1e-12).sum()}/180',
           f'Max official VUS equivalence discrepancy: {valid["max_official_baseline_parity_error"]:.4g}',
           '', 'IMPORTANT NEGATIVE / QUALIFIED FINDINGS',
           f'Std-Scaled Fusion has higher point estimates than the frozen Full model: Series VUS-PR={std.series_average:.6f}, Dataset VUS-PR={std.dataset_average:.6f}. Its paired raw p={std.p_raw:.6g}, Holm p={std.p_holm:.6g}; a higher mean is not a significant advantage under the prespecified correction.',
           'Removing Branch-B Stage-1 alone or Stage-2 alone also raises both VUS-PR means. The full two-stage design is not empirically best in every control.',
           f'VUS-PR comparisons significant after the global 84-comparison Holm correction: {int(((tests.metric=="VUS-PR")&(tests.p_holm<.05)).sum())}/21.',
           'These results do not support a claim that all proposed modules significantly improve VUS-PR. The main model and its defaults remain frozen.',
           '', 'CORE ABLATION (standard AP, complete 180)',
           'Method | Series VUS-PR | Dataset VUS-PR | AP | isolated mean seconds']
    for name in a.experiment_config.drop_duplicates():
        sub=a[a.experiment_config==name].set_index('metric')
        vp=sub.loc['VUS-PR'];ap=sub.loc['AUC-PR','series_average']
        lines.append(f'{name} | {vp.series_average:.6f} | {vp.dataset_average:.6f} | {ap:.6f} | {vp.runtime_isolated_mean_seconds:.6f}')
    lines+=['','CALIBRATION AND FALLBACK CAUTIONS',
            'All S2-A Branch-B controls retain Stage-1 POT. Raw/Std/ECDF are not fully calibration-free models.',
            'S2-B Stage-2 removal affects Branch B only; Branch-A POT stays active.',
            'Requested scales/windows can be capped, dropped or replaced by the frozen fallback; primary means keep all 180.',
            'Do not interpret a fallback result as the originally requested window. Consult effective scales, scale details and requested_vs_realized_window_sensitivity.csv.',
            '', 'INTERNAL ABLATION SUMMARY (all 180; no selected exclusions)']
    for _,row in b[b.metric=='VUS-PR'].iterrows():
        lines.append(f'{row["group"]} / {row.experiment_config}: Ser={row.series_average:.6f}, DS={row.dataset_average:.6f}, '
                     f'paired p={row.p_raw:.6g}, Holm p={row.p_holm:.6g}, '
                     f'Branch-B requested scales fully realized={row.branch_b_requested_scales_fully_realized_n:.0f}/180, '
                     f'Branch-A window capped={row.branch_a_window_capped_n:.0f}/180')
    lines+=['','VARIANCE AUDIT']
    for cohort in ['All-180','Clean-160']:
        lines.append(cohort)
        for field in ['var_ratio_a_over_b','pearson_unshrunk','variance_reduction_percent','equal_approximation_relative_error']:
            row=v[(v.cohort==cohort)&(v.quantity==field)].iloc[0]
            lines.append(f'  {field}: mean={row["mean"]:.6g}, median={row["median"]:.6g}, '
                         f'IQR=[{row.q25:.6g}, {row.q75:.6g}], defined={row.n_defined:.0f}/{row.n_total:.0f}')
        for field in ['negative_correlation','actual_variance_reduction','equal_variance_within_2x','undefined_correlation','undefined_variance_ratio']:
            row=p[(p.cohort==cohort)&(p.quantity==field)].iloc[0]
            lines.append(f'  {field}: {row.n_true:.0f}/{row.n_defined:.0f} ({row.proportion_among_defined:.2%} of defined cases)')
        for _,row in r[(r.cohort==cohort)&r.level.str.startswith('dataset')].iterrows():
            lines.append(f'  Exploratory dataset-level association with {row.outcome}: rho={row.spearman_rho:.4f}, '
                         f'p={row.p_raw:.6g}, exploratory Holm p={row.p_holm_exploratory:.6g}, n={row.n}')
    lines+=['','INTERPRETATION LIMITS',
            'The general covariance formula is an algebraic identity; its agreement is a computational check, not evidence of detection effectiveness.',
            'Negative correlation and reduced variance do not by themselves prove detection gains. Use matched unsmoothed branch/fusion performance comparisons.',
            'Equal-variance approximation errors and all negative or nonsignificant results are retained.',
            'The variance-ratio band [0.5,2] is a descriptive diagnostic, not a theoretical validation threshold.',
            'Primary tests operate on 17 dataset means. Holm spans 84 unique ablation/metric comparisons. Bootstrap CIs resample dataset clusters 10,000 times.',
            'Independent runtime uses one process and one CPU thread after parallel workers exit, one timed fresh fit/score per file/config after per-file warmup.',
            'Internal component-ablation wrappers preserve common computations even when a resulting score term is disabled; their runtimes describe this implementation.',
            'No metric evaluation, CSV parsing, disk writing or shared inter-configuration score cache is inside the independent timed interval.',
            '', 'COMPLETENESS',json.dumps(valid,indent=2),'',
            'REPRODUCTION',
            'Install scripts/requirements.txt into a scientific Python environment. Supply the original 180 raw CSV files separately.',
            'PowerShell example:',
            '.\\run_all.ps1 -DataDir D:\\path\\to\\TSB-AD-M -PythonExe D:\\path\\to\\python.exe -Workers 4',
            'The 180-file official manifest, frozen code, exact VUS code, official metric source, scripts and input hashes are included.',
            'Check PROTOCOL.md for full configuration definitions, inference conventions, and runtime limitations.',
            'CSV nulls in variance/diagnostic quantities are intentional and accompanied by flags/counts. Core detection metrics must all be finite.',
            'Raw data, model score-array caches and temporary package installations are excluded from the final ZIP.']
    lines+=['','RUN NOTES',
            'One postprocessing attempt encountered a pandas 3 read-only array assignment; copying the 17 dataset differences fixed it. The original traceback is retained in logs/analysis_attempt1_readonly_array.log. No model scores or detection metrics were changed.',
            'The initial 4-worker baseline batch was restarted with 8 workers after a memory check; completed per-series checkpoints were reused. Partial baseline and full resumed execution logs are retained.',
            'The input-preparation source in reference/prepare_inputs_original_run.py is an archival copy of the actual original workspace script; use run_all.ps1 with an external raw-CSV directory for portable reproduction.']
    (OUT/'RESULT_NOTES.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved RESULT_NOTES.txt',flush=True)

if __name__=='__main__':main()
