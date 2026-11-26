# Frozen Stage-2 experiment protocol

The model is the user-provided `D:/Downloads/Google/dbesc.py`, SHA-256
`c454732bc1dfdf8eaf16b46076786c641744cbade97ea1f97a89ba05dac10227`.
Its source is copied unchanged. No parameter is fitted to evaluation performance.
The raw data are extracted from the supplied ZIP, restricted to the 180 unique
names in the official TSB-AD-M-Eva manifest. Input hashes and ZIP CRCs are saved.
The 34 SKAB series are not recomputed.

## Evaluation

Training prefix length and preprocessing follow the original supplied benchmark
runner: `max(64, min(filename_tr, T-1))`, median/IQR estimated only from that prefix,
the original standard-deviation fallback and clipping. All timestamps, including
the training prefix, are evaluated. No label-based correction of a contaminated
prefix is made. Clean-160 follows `filename_first_anomaly > prefix_length`;
actual anomalous points in each prefix are separately counted.

The evaluation buffer is recomputed using the official
`find_length_rank(X[:, 0], rank=1)` on the raw first channel, exactly as in the
original runner. This evaluation convention differs from the model's prefix-only
FFT window estimator; it does not tune the model. The window is also checked
against the supplied historical per-series reference.

All 180 Full DB-ESC baseline scores are evaluated by the unmodified official
`generate_curve(..., version='opt', thre=250)`. For ablations, `exact_vus.py`
reorders the same finite sums using the first satisfied official threshold for
each point. It preserves all samples, all 250 official threshold entries
(including duplicates), every integer buffer width from 0 through the official
window, the tie rule `score >= threshold`, and the original endpoint conventions.
It is an exact computational reformulation, without temporal downsampling or
approximation. The algebra uses:

```
TP = detected_original_anomaly_mass + detected_extended_tail_mass
N_labels = original_anomaly_mass + detected_extended_tail_mass
P_new = (original_anomaly_mass + N_labels) / 2
```

The official interval helpers construct the extensions and existence groups.
Equivalence is checked on all 180 complete baseline sequences, 60 synthetic
boundary/tie cases and all 22 unique real-series ablation scores. The maximum
per-case discrepancy is retained in the audit CSVs. The bundled official source
is from local TSB-AD commit `303dffbfc11612995b17b06633f8c9d0ea0e38e5`.
Upstream: https://github.com/TheDatumOrg/TSB-AD/tree/303dffbfc11612995b17b06633f8c9d0ea0e38e5/TSB_AD/evaluation

Primary pointwise AUC-PR is sklearn's standard Average Precision, with equal
scores grouped correctly. The historical point-by-point precision sum with
stable sorting is also saved, as is the official TSB-AD Average Precision.
The legacy implementation can depend on timestamp order within a tied score.

## Configurations

S2-A has exactly eight controls. Branch-only controls retain the corresponding
final POT calibration and dual-scale smoothing. Raw Fusion removes Stage-2
branch calibration; it retains Branch-B's Stage-1 per-scale POT. Std-Scaled Fusion
divides each raw branch by its own safe-prefix standard deviation (no centering),
also retaining Stage-1 POT. A zero/near-zero standard deviation (<=1e-12) uses
divisor 1 with an explicit flag. ECDF replaces only branch-level calibration,
retaining the original Stage-1 POT. Thus Raw/Std/ECDF controls must never be
described as completely calibration-free. Smoothing is kept except in the two
explicitly named without-smoothing controls. The Full model is untouched.

S2-B covariance controls keep all other settings. The original eligibility rule
`tr >= 2*requested_window` and strict calibration-boundary rule are applied.
The single-scale function may cap the actual window to `max(12,tr//4)`, even for
an eligible requested scale. If no requested scale is eligible, the frozen model
falls back to its (possibly capped and causal) 48-point branch, without Stage-1
POT in that fallback. Every requested/actual window, dropped scale, Stage-1
application, causal fallback and per-scale calibration cutoff is recorded.
Default multiscale windows can consequently coincide on short prefixes.

The Stage-1/Stage-2 controls change only Branch B. Branch A always retains its
original POT. Removing both Branch-B stages is not a completely uncalibrated
model. Fixed Branch-A windows retain the original `min(w,max(8,tr//5))` cap and
the original strict-boundary causal fallback. PCA-only retains the original 0.5
residual coefficient; k-NN-only keeps its original coefficient 1. The wrapper's
default scores, both calibrated branches and common cutoff are checked against
the original function on every baseline file. All 26 reported group/configuration
combinations use 22 distinct scoring configurations; four default rows are aliases.

## Variance audit and interpretation

All variance calculations use the exact returned common range
`[0, cal_end_all)`, unmodified `z_a`, `z_b`, and unsmoothed `score_fuse`, with
population moments (ddof=0). No epsilon is added to a measured variance.
Zero variances, near-zero variances (<=1e-12), undefined correlations and ratios
are flagged explicitly. Undefined audit quantities remain empty in CSV/JSON.
The model's own diagnostic zero-correlation fallback is recorded separately.
The empirical Pearson correlation is saved before clipping or shrinkage;
the model's actual shrunk value is `0.85*clip(r,-0.99,0.99)` where defined.

The general formula is `(var_a+var_b+2*cov_ab)/4`. Its agreement with measured
fusion variance is an algebraic identity and only a computational check.
The equal-variance approximation uses the average measured branch variance:
`(var_a+var_b)*(1+r)/4`. The shrunk approximation substitutes the model's shrunk
correlation. Maximum-score ratios are reported both over the entire sequence
and over labeled anomalous timestamps. The 0.5-to-2 variance-ratio band is a
descriptive diagnostic, not a validity theorem or a post-hoc exclusion rule.

Performance association compares the unsmoothed fusion with equally unsmoothed
and equally calibrated branch scores, evaluating their full-resolution VUS-PR.
The mean-branch and best-branch improvements are both preserved. Spearman
correlations at series and dataset-mean levels are exploratory associations.
Variance reduction alone, or negative correlation alone, does not establish
improved anomaly detection or a causal denoising mechanism.

## Inference and runtime

Main paired tests use 17 paired dataset means, with two-sided Wilcoxon signed
ranks. Shapiro p values, skew, IQR outlier counts, rank-biserial effect sizes,
exact paired sign-flip p values and exact sign-test sensitivity p values are
reported. Wilcoxon relies on symmetry of differences; no assumption-test result
is used to switch to whichever test looks more favorable. Holm correction spans
all 21 nondefault unique ablations and all four metrics (84 comparisons).
Differences within 1e-12 are treated as numerical ties for wins and paired tests,
well above the measured sub-1e-15 official-engine rounding discrepancies. Raw
metric values remain unmodified in every per-series CSV.
Exploratory variance/performance tests have a separate Holm family.

95% intervals use 10,000 resamples of the 17 dataset clusters, seed 20261126.
Dataset averages weight resampled datasets equally; series averages resample
the complete datasets, retaining their original sequence counts. Paired
difference intervals resample paired dataset means. These intervals quantify
between-dataset uncertainty conditional on this benchmark, not random model seeds.

The parallel batch uses multiple processes, each explicitly limited to one
BLAS/OpenMP CPU thread. Shared deterministic branch calculations are reused in
the batch; cached-call and component-accounted batch times are not standalone
algorithm runtimes. Independent timing is a separate phase after all workers
have exited, with one process and one thread, a per-file warmup, fresh fitting
and full scoring for each configuration, no metric/CSV loading inside the timed
interval, and one timed call per file/configuration. Mean and median across all
180 files are saved. This single-call protocol does not estimate repeated-run
timing variability. Other unrelated host activity is not controlled.

The provided latest manuscript PDF is copied unchanged for provenance. It has
not been revised with these new results. Optional ExCal and multiseed stress
tests are outside the required core run and must not be claimed as completed.
