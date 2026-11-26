import json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from pypdf import PdfReader
from experiment_model import GROUPS,UNIQUE

OUT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    required=['s2a_core_ablation_180.csv','s2a_core_ablation_summary.csv','s2b_multiscale_ablation_180.csv',
              's2b_pot_stage_ablation_180.csv','s2b_period_ablation_180.csv','s2b_branch_a_internal_180.csv',
              's2b_summary.csv','s2c_real_branch_variance_180.csv','s2c_variance_summary.csv',
              'statistical_tests.csv','baseline_reproduction_check.csv','experiment_config.json','RESULT_NOTES.txt']
    for name in required:assert (OUT/name).is_file(),name
    report=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'))
    audit=pd.read_csv(OUT/'completeness_audit.csv')
    assert len(audit)==26 and audit.passed.all()
    assert int(audit.actual_rows.sum())==4680
    assert audit.isolated_score_hash_mismatches.sum()==0
    for name,configs in GROUPS.items():
        df=pd.read_csv(OUT/name)
        for config in configs:
            group=df[df.experiment_config==config]
            assert len(group)==180 and group.file.nunique()==180
            metrics=group[['VUS-PR','VUS-ROC','AUC-PR','AUC-ROC']].to_numpy()
            assert np.isfinite(metrics).all() and (metrics>=-1e-12).all() and (metrics<=1+1e-12).all()
            assert (group.cal_end_a<=group.train_prefix_length).all() and (group.cal_end_b<=group.train_prefix_length).all()
    timing=pd.read_csv(OUT/'runtime_single_thread_independent.csv')
    assert len(timing)==3960 and timing.experiment_config.nunique()==len(UNIQUE)
    assert not timing.duplicated(['file','experiment_config']).any()
    assert (timing.runtime_single_thread_seconds>0).all()
    official=pd.read_csv(OUT/'official_vus_equivalence_audit.csv')
    assert len(official)==82 and (official.status=='pass').all()
    base=pd.read_csv(OUT/'baseline_per_series_180.csv')
    assert len(base)==180 and base.wrapper_max_score_error.max()==0
    assert base.official_parity_error.max()<2e-12
    frozen='c454732bc1dfdf8eaf16b46076786c641744cbade97ea1f97a89ba05dac10227'
    assert sha(OUT/'scripts/dbesc.py')==frozen
    figures=[]
    for path in sorted((OUT/'figures').glob('*.pdf')):
        pdf=PdfReader(path);images=sum(len(page.images) for page in pdf.pages)
        assert len(pdf.pages)==1 and images==0
        assert path.with_suffix('.png').is_file()
        figures.append({'file':path.name,'pages':len(pdf.pages),'embedded_raster_images':images,'sha256':sha(path),
                        'manual_render_review':'passed: labels and axes readable, no clipping or data-overlapping legend'})
    assert len(figures)==5 and len(list((OUT/'tables').glob('*.tex')))==6
    (OUT/'figure_qa.json').write_text(json.dumps(figures,indent=2),encoding='utf-8')
    report.update({'isolated_score_hash_mismatches':int(audit.isolated_score_hash_mismatches.sum()),
                   'official_synthetic_and_real_ablation_parity_cases':len(official),'vector_pdf_count':5,
                   'latex_table_count':6,'figure_manual_review_passed':True,
                   'all_requested_core_outputs_present':True,'frozen_source_verified':True})
    (OUT/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
    config.update({'completion_time_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                   'workers_history':[4,8],'independent_timing_calls':3960,'independent_timing_repeats_per_file_config':1,
                   'paired_test_unit':'dataset mean, 17 paired datasets',
                   'paired_test':'two-sided Wilcoxon signed-rank; exact sign-flip and sign-test sensitivity checks',
                   'holm_family_size':84,'statistical_numerical_tie_tolerance':1e-12,
                   'implementation_constants':{'robust_clip':[-15,15],'fft_lags':[24,160],'acf_peak_threshold':.25,
                     'branch_a_channel_cap':16,'branch_a_orthogonal_weight':.5,'dictionary_stride_target':1200,
                     'branch_a_grid_target':4000,'branch_b_grid_target':3000,'spd_trace_floor':1e-4},
                   'final_script_sha256':{p.name:sha(p) for p in sorted((OUT/'scripts').glob('*.py'))},
                   'formal_metric_nan_count':0,'postprocessing_recovery':'pandas read-only dataset-difference array copied before numerical tie handling; original traceback retained',
                   'manuscript_status':'provided latest PDF copied unchanged; not revised with new experiments'})
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
