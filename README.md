# DB-ESC: Dual-Branch Empirical Survival Calibration for Efficient Multivariate Time Series Anomaly Detection

Official reference implementation, unit tests, and complete reproducible evaluation logs for:

> **Dual-Branch Empirical Survival Calibration for Efficient Multivariate Time Series Anomaly Detection**  
> Jiayang Yang, Yi Li  
> Chang'an University, Xi'an 710064, Shaanxi, China  
> Submitted to *Neurocomputing* (Elsevier)

---

## Overview

**DB-ESC** is a closed-form, gradient-free, CPU-only multivariate time series anomaly detection framework that integrates:
1. **Branch A (Temporal Morphology)**: FFT period-adaptive subsequence PCA + $k$-NN manifold scoring with an explicit temporal Exclusion Zone ($|s' - s| \ge w_A$) and orthogonal projection residual.
2. **Branch B (Cross-Channel Geometry)**: Multi-scale fixed-shrinkage SPD covariance pyramid ($\mathcal{W}=\{48, 96, 192\}$, $\lambda=0.15$) scored via LogDet Bregman divergence with Stage-1 per-scale POT equalization.
3. **Boundary-Strict Anchored Exp-POT Calibration & Equal-Weight Fusion**: Reference-prefix-strict piecewise negative log-survival calibration ($q_{\mathrm{tail}}=0.98$, monotone anchor $b_v$) followed by equal-weight fusion and dual-scale smoothing ($w_{\mathrm{ds}}=16$).

---

## Key Verified Results (Frozen `dbesc.py`)

| Benchmark / Suite | Protocol | Series VUS-PR | Dataset VUS-PR | Std. AP / AUC-PR | Single-Thread CPU Time |
|---|---|---:|---:|---:|---:|
| **TSB-AD-M-Eva** (180 series, 17 datasets) | Official full-resolution VUS-PR | **0.4773** (3rd / 32) | **0.3758** (3rd / 32) | 0.4654 (AP) / 0.4653 (legacy) | 1.05 s/series (0.766 s on i7-13650HX) |
| **DB-ESC-Causal** (180 series) | Fixed-grid 0-lookahead post-prefix | 0.3810 | 0.3189 | — | ~1.05 s/series |
| **SKAB** (34 water-pump series, frozen) | Label-informed prefix ($t_{\mathrm{tr}}\ge 64$) | **0.8213** | — | 0.7881 (legacy) | 0.543 s/series |
| **SKAB** (34 series, frozen) | Label-free fixed prefix ($t_{\mathrm{tr}}=200$) | **0.8020** | — | 0.7723 (legacy) | — |
| **S2-C Real Variance Audit** (180 series) | Common safe prefix $\mathcal{I}_{\mathrm{cal}}$ | Median var ratio **1.0031** | 70.56% $r_{AB}<0$ | **58.76%** mean var reduction | — |

---

## Repository Structure

```text
.
├── dbesc.py                         # Frozen reference implementation (zero external DL dependencies)
├── run_tsb_ad_m.py                  # Full TSB-AD-M-Eva & Tuning evaluation runner
├── run_neurocomputing_suite.py      # Exp 1 (substitutions), Exp 2 (stress tests), Exp 5 (SKAB)
├── run_stage2_core_vus.py           # Stage-2 S2-A/B/C/D ablation & SKAB fixed-prefix runner
├── verify_reproducibility.py        # Reproducibility & leaderboard verification script
├── test_edge_cases.py               # Unit tests (constant, tied, C=1, short-prefix, zero-leakage)
├── check_pot_ties.py                # Monotone POT anchor tie-handling verification
├── requirements.txt                 # Minimal Python dependencies (numpy, scipy, pandas, scikit-learn)
├── results/                         # Primary leaderboard, Exp 1/2/3/5, and SKAB evaluation CSVs
└── stage2_180_ablation/             # Complete 180-series S2-A/B/C CSVs, Holm tests, logs & SHA-256
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run unit & zero-leakage edge-case tests
python test_edge_cases.py

# 2. Verify saved benchmark summary metrics
python verify_reproducibility.py

# 3. Minimal Python API usage
python -c "
import numpy as np
from dbesc import fit_score_dbesc
X = np.random.randn(2000, 8)
res = fit_score_dbesc(X, tr=600, causal=False)
print('Score shape:', res['score_final'].shape, 'r_AB:', round(res['r_ab'], 4))
"
```

---

## License

Released under the MIT License (see `LICENSE`). Official TSB-AD evaluation utilities under `stage2_180_ablation/scripts/official_evaluation/` retain their original upstream license.
