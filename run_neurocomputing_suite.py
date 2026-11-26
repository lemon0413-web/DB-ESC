#!/usr/bin/env python3
"""
Neurocomputing Upgrade — One-Click Pure-CPU Experiment Suite (Exact dbesc.py Wrapper)
=====================================================================================
Place this script in the same directory as `dbesc.py` (03_Code_and_Verification/).
Directly imports `dbesc.py` so the default DB-ESC row is bit-for-bit identical to
`eva180_final_results.csv` (0.4773 Ser VUS-PR / 0.4653 Ser AUC-PR).

Experiments:
  Exp 1: Single-variable Operator / Calibration / Fusion substitutions on TSB-AD-M-Eva (180 series)
           1A. SPD Metric & Shrinkage: Fixed-Shrink LogDet (Ours) vs Frobenius
               vs Log-Euclidean vs Adaptive Ledoit-Wolf LogDet
           1B. Calibration: Z-score vs Robust MAD-Z vs Discrete ECDF vs GPD-MLE
               vs Anchored Exp-POT (Ours)
           1C. Fusion: Max vs Geometric Mean vs Rank Avg vs Inv-Var vs Equal Min-Var (Ours)
  Exp 2: Robustness Stress Tests on TSB-AD-M-Eva / Clean-160
           2A. Cold-Start Prefix Truncation ({25%, 50%, 75%, 100%}, verifying Eq. 9 fallback)
               DB-ESC vs PCA vs IForest
           2B. Additive Gaussian Noise (SNR in {30, 20, 15, 10} dB on Clean-160)
               DB-ESC vs PCA vs MCD vs IForest
           2C. Controlled Prefix Contamination ({1%, 2%, 5%, 10%} bursts on Clean-160)
               DB-ESC vs PCA vs MCD vs IForest
  Exp 5: Frozen Zero-Tuning External Benchmark — SKAB (34 water-pump series)
           DB-ESC (Frozen 2-Branch) vs ThreeView_Proto (+Sparse-HBOS) vs PCA, IForest,
           MCD, OCSVM, KMeansAD (VUS-PR, VUS-ROC, AUC-PR, AUC-ROC, runtime)

Usage (same CLI style as run_tsb_ad_m.py):
  python run_neurocomputing_suite.py \
    --data_dir "D:/ms_cmesc/TSB-AD-M" \
    --file_list "D:/ms_cmesc/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv" \
    --out_dir "./nc_results" \
    --exp 1 2 5 --download_skab --fast_auc_exp12 --workers 4
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import glob
import re
import time
import urllib.request
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import genpareto, rankdata
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from dbesc import (
    branch_a_temporal_manifold,
    branch_b_spd_pyramid,
    causal_step_interp,
    evt_pot_calibrate,
    fit_score_dbesc,
    moving_average,
    robust_standardize,
    sparse_hbos_branch,
)

warnings.filterwarnings("ignore")

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    HAS_TSB_AD = True
except Exception:
    HAS_TSB_AD = False


# =============================================================================
# 1. Exact Metric & File Utilities (Matching run_tsb_ad_m.py verbatim)
# =============================================================================

def parse_train_len(filename: str, T: int) -> int:
    m = re.search(r"_tr_(\d+)_", os.path.basename(filename))
    return max(64, min(int(m.group(1)), T - 1)) if m else max(64, int(0.2 * T))


def parse_first_anom(filename: str):
    m = re.search(r"_1st_(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else None


def parse_dataset_name(filename: str) -> str:
    parts = os.path.basename(filename).split("_")
    return parts[1] if len(parts) > 1 else "Unknown"


def is_prefix_contaminated(filename: str, tr: int) -> bool:
    first = parse_first_anom(filename)
    return (first is not None) and (first <= tr)


def compute_auc_pr_roc(y_true: np.ndarray, scores: np.ndarray):
    y = (y_true > 0).astype(np.int32)
    n_pos = int(np.sum(y))
    n = len(y)
    if n_pos == 0 or n_pos == n:
        return np.nan, np.nan
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, n + 1, dtype=np.float64)
    auc_pr = float(np.sum(prec[y_sorted == 1]) / n_pos)
    ranks = rankdata(s)
    auc_roc = float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos)))
    return auc_pr, auc_roc


def compute_official_vus(y_true: np.ndarray, scores: np.ndarray, sw: int):
    if not HAS_TSB_AD:
        return np.nan, np.nan
    y = (y_true > 0).astype(np.int32)
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ptp = float(np.ptp(s))
    s = (s - float(np.min(s))) / ptp if ptp > 1e-12 else np.zeros_like(s)
    res = get_metrics(s, y, slidingWindow=int(sw))
    return float(res.get("VUS-PR", res.get("VUS_PR", np.nan))), float(res.get("VUS-ROC", res.get("VUS_ROC", np.nan)))


def fit_score_3view_proto(X: np.ndarray, tr: int) -> np.ndarray:
    """Exact ThreeView_Proto from run_tsb_ad_m.py for SKAB 2-branch vs 3-branch check."""
    Z = robust_standardize(X, tr)
    s_1 = sparse_hbos_branch(Z, tr)
    s_a, la_a = branch_a_temporal_manifold(Z, tr, strict_stride_truncation=True)
    if tr - la_a < 32:
        s_a, la_a = branch_a_temporal_manifold(Z, tr, causal=True, strict_stride_truncation=True)
    s_b, la_b = branch_b_spd_pyramid(Z, tr, strict_stride_truncation=True)
    if tr - la_b < 32:
        s_b, la_b = branch_b_spd_pyramid(Z, tr, causal=True, strict_stride_truncation=True)
    z_1 = evt_pot_calibrate(s_1, tr, q_tail=0.98)
    z_a = evt_pot_calibrate(s_a, max(1, tr - la_a), q_tail=0.98)
    z_b = evt_pot_calibrate(s_b, max(1, tr - la_b), q_tail=0.98)
    cal_all = max(2, tr - max(la_a, la_b))
    M = np.column_stack([z_1, z_a, z_b])
    M_cal = M[:cal_all]
    R = np.eye(3, dtype=np.float64)
    for i in range(3):
        for j in range(i + 1, 3):
            if np.std(M_cal[:, i]) > 1e-12 and np.std(M_cal[:, j]) > 1e-12:
                c = float(np.corrcoef(M_cal[:, i], M_cal[:, j])[0, 1])
                if np.isfinite(c):
                    R[i, j] = R[j, i] = np.clip(c, -0.99, 0.99)
    R_reg = 0.85 * R + 0.15 * np.eye(3)
    w = np.clip(np.linalg.solve(R_reg, np.ones(3)), 0.05, None)
    w = w / np.sum(w)
    f = M @ w
    return 0.5 * f + 0.5 * moving_average(f, 16, causal=False)


# =============================================================================
# 2. Exp 1 Single-Variable Substitution Operators (on Exact dbesc.py Backbone)
# =============================================================================

def calibrate_alt(raw: np.ndarray, cal_end: int, method: str = "pot_exp", q_tail: float = 0.98) -> np.ndarray:
    if method == "pot_exp":
        return evt_pot_calibrate(raw, cal_end, q_tail=q_tail)
    ref = raw[:cal_end]
    n_ref = len(ref)
    if method == "zscore":
        return (raw - float(np.mean(ref))) / (float(np.std(ref)) + 1e-6)
    if method == "mad_z":
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med))) * 1.4826 + 1e-6
        return (raw - med) / mad
    ref_sorted = np.sort(ref)
    idx = np.searchsorted(ref_sorted, raw, side="right")
    z_body = -np.log(np.clip((n_ref - idx + 0.5) / (n_ref + 1.0), 1e-12, 1.0))
    if method == "ecdf":
        return z_body
    if method == "gpd_mle":
        u = float(np.quantile(ref, q_tail))
        p_u = max(1.0 - q_tail, 1.0 / (n_ref + 1.0))
        idx_u = int(np.searchsorted(ref_sorted, u, side="right"))
        z_u = -np.log(max((n_ref - idx_u + 0.5) / (n_ref + 1.0), 1e-12))
        b_v = max(-np.log(p_u), z_u)
        exc = ref[ref > u] - u
        if len(exc) >= 5:
            try:
                xi, _, beta = genpareto.fit(exc, floc=0.0)
                xi = float(np.clip(xi, -0.35, 0.50))
                beta = max(float(beta), 1e-6)
                sf = genpareto.sf(np.maximum(0.0, raw - u), xi, loc=0.0, scale=beta)
                return np.where(raw > u, b_v - np.log(np.clip(sf, 1e-12, 1.0)), z_body)
            except Exception:
                pass
        return evt_pot_calibrate(raw, cal_end, q_tail=q_tail)
    raise ValueError(method)


def _single_scale_spd_alt(Z, tr, w_cov, spd_mode="fixed_logdet", c_eff_max=24, lam=0.15, eps_tr=1e-4):
    T, C = Z.shape
    if C <= 1:
        return np.zeros(T, dtype=np.float64), 0
    if C > c_eff_max:
        var_c = np.var(Z[:tr], axis=0)
        Zc = Z[:, np.argsort(var_c, kind="mergesort")[-c_eff_max:]]
        C_eff = c_eff_max
    else:
        Zc = Z
        C_eff = C

    w = min(w_cov, max(12, tr // 4))
    step = max(1, T // 3000)
    eye_c = np.eye(C_eff, dtype=np.float64)

    def make_cov(seg):
        if spd_mode == "lw_logdet" and len(seg) >= 5:
            try:
                S_lw = LedoitWolf().fit(seg).covariance_
                return S_lw + (lam * eps_tr) * eye_c
            except Exception:
                pass
        m = seg - np.mean(seg, axis=0, keepdims=True)
        S = (m.T @ m) / max(len(seg) - 1, 1)
        tau = max(float(np.trace(S)) / C_eff, eps_tr)
        return (1.0 - lam) * S + lam * tau * eye_c

    S0 = make_cov(Zc[:tr])
    S0_inv = np.linalg.inv(S0)
    _, ld0 = np.linalg.slogdet(S0)
    if spd_mode == "log_euclidean":
        w0, v0 = np.linalg.eigh(S0)
        L0 = (v0 * np.log(np.maximum(w0, 1e-12))) @ v0.T

    centers = np.arange(0, T, step)
    raw_s = np.zeros(len(centers), dtype=np.float64)
    half = w // 2
    for idx_i, t in enumerate(centers):
        a = max(0, t - half)
        b = min(T, a + w)
        a = max(0, b - w)
        Sw = make_cov(Zc[a:b])
        if spd_mode == "frobenius":
            raw_s[idx_i] = float(np.linalg.norm(Sw - S0, ord="fro"))
        elif spd_mode == "log_euclidean":
            ww, vw = np.linalg.eigh(Sw)
            Lw = (vw * np.log(np.maximum(ww, 1e-12))) @ vw.T
            raw_s[idx_i] = float(np.linalg.norm(Lw - L0, ord="fro"))
        else:
            _, ldw = np.linalg.slogdet(Sw)
            div = 0.5 * (float(np.sum(S0_inv * Sw)) - (ldw - ld0) - C_eff)
            raw_s[idx_i] = max(0.0, div)

    la = (w - half) + (step - 1)
    return np.interp(np.arange(T, dtype=np.float64), centers.astype(np.float64), raw_s), la


def branch_b_alt(Z, tr, spd_mode="fixed_logdet", cal_mode="pot_exp"):
    if spd_mode == "fixed_logdet" and cal_mode == "pot_exp":
        return branch_b_spd_pyramid(Z, tr, strict_stride_truncation=True)
    T, C = Z.shape
    if C <= 1:
        return np.zeros(T, dtype=np.float64), 0
    sc_list, max_la = [], 0
    step_b = max(1, T // 3000)
    for w in (48, 96, 192):
        if tr >= w * 2:
            req_la = (w - w // 2) + (step_b - 1)
            if tr - req_la < 32:
                continue
            s_w, la_w = _single_scale_spd_alt(Z, tr, w_cov=w, spd_mode=spd_mode)
            sc_list.append(calibrate_alt(s_w, tr - la_w, method=cal_mode, q_tail=0.98))
            max_la = max(max_la, la_w)
    if not sc_list:
        s_w, la_w = _single_scale_spd_alt(Z, tr, w_cov=48, spd_mode=spd_mode)
        return s_w, la_w
    return np.mean(sc_list, axis=0), max_la


def fuse_alt(z_a, z_b, cal_all, rule="equal"):
    if rule == "equal":
        return 0.5 * (z_a + z_b)
    if rule == "max":
        return np.maximum(z_a, z_b)
    if rule == "geom":
        pa = np.maximum(z_a - np.min(z_a) + 1e-3, 1e-3)
        pb = np.maximum(z_b - np.min(z_b) + 1e-3, 1e-3)
        return np.sqrt(pa * pb)
    if rule == "rank":
        return 0.5 * (rankdata(z_a) / len(z_a) + rankdata(z_b) / len(z_b))
    if rule == "inv_var":
        va = float(np.var(z_a[:cal_all])) + 1e-6
        vb = float(np.var(z_b[:cal_all])) + 1e-6
        wa, wb = (1.0 / va), (1.0 / vb)
        return (wa * z_a + wb * z_b) / (wa + wb)
    raise ValueError(rule)


# =============================================================================
# 3. Classical Baselines (for Exp 2 & Exp 5)
# =============================================================================

def run_baseline(name: str, X: np.ndarray, tr: int, seed: int = 42) -> np.ndarray:
    Z = robust_standardize(X, tr)
    C = Z.shape[1]
    if name == "PCA":
        k = max(1, min(C - 1, max(1, C // 2)))
        pca = PCA(n_components=k).fit(Z[:tr])
        return np.sum((Z - pca.inverse_transform(pca.transform(Z))) ** 2, axis=1)
    if name == "IForest":
        clf = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(Z[:tr])
        return -clf.score_samples(Z)
    if name == "MCD":
        try:
            Z_use = PCA(n_components=max(2, min(10, tr // 4))).fit_transform(Z) if C > max(2, tr // 4) else Z
            return MinCovDet(random_state=seed).fit(Z_use[:tr]).mahalanobis(Z_use)
        except Exception:
            return np.sum(Z ** 2, axis=1)
    if name == "OCSVM":
        sub = Z[:tr]
        if len(sub) > 1500:
            sub = sub[np.linspace(0, len(sub) - 1, 1500, dtype=int)]
        return -OneClassSVM(kernel="rbf", nu=0.05).fit(sub).decision_function(Z)
    if name == "KMeansAD":
        k = min(8, max(2, tr // 20))
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Z[:tr])
        return np.min(cdist(Z, km.cluster_centers_, metric="euclidean"), axis=1)
    raise ValueError(name)


# =============================================================================
# 4. Parallel Worker Functions for Exp 1 & Exp 2
# =============================================================================

EXP1_CONFIGS = [
    ("1A_SPD", "Fixed-Shrink LogDet (Ours)", "fixed_logdet", "pot_exp", "equal"),
    ("1A_SPD", "Frobenius Distance", "frobenius", "pot_exp", "equal"),
    ("1A_SPD", "Log-Euclidean Distance", "log_euclidean", "pot_exp", "equal"),
    ("1A_SPD", "Adaptive Ledoit-Wolf LogDet", "lw_logdet", "pot_exp", "equal"),
    ("1B_Cal", "Z-Score", "fixed_logdet", "zscore", "equal"),
    ("1B_Cal", "Robust MAD-Z", "fixed_logdet", "mad_z", "equal"),
    ("1B_Cal", "Discrete ECDF", "fixed_logdet", "ecdf", "equal"),
    ("1B_Cal", "GPD-MLE", "fixed_logdet", "gpd_mle", "equal"),
    ("1C_Fuse", "Max Fusion", "fixed_logdet", "pot_exp", "max"),
    ("1C_Fuse", "Geometric Mean", "fixed_logdet", "pot_exp", "geom"),
    ("1C_Fuse", "Rank Average", "fixed_logdet", "pot_exp", "rank"),
    ("1C_Fuse", "Inverse-Variance Weighted", "fixed_logdet", "pot_exp", "inv_var"),
]


def _worker_exp1(task):
    fn, data_dir, use_vus = task
    df = pd.read_csv(os.path.join(data_dir, fn))
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int32)
    T = X.shape[0]
    tr = parse_train_len(fn, T)
    ds = parse_dataset_name(fn)
    sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1)) if (use_vus and HAS_TSB_AD) else None

    Z = robust_standardize(X, tr)
    s_a, la_a = branch_a_temporal_manifold(Z, tr, strict_stride_truncation=True)
    if tr - la_a < 32:
        s_a, la_a = branch_a_temporal_manifold(Z, tr, causal=True, strict_stride_truncation=True)
    cal_a = max(1, tr - la_a)

    b_cache = {}
    out_rows = []
    for grp, label, spd_m, cal_m, fuse_m in EXP1_CONFIGS:
        if spd_m == "fixed_logdet" and cal_m == "pot_exp" and fuse_m == "equal":
            sc = fit_score_dbesc(X, tr, causal=False, strict_stride_truncation=True)["score_final"]
        else:
            key = (spd_m, cal_m)
            if key not in b_cache:
                s_b, la_b = branch_b_alt(Z, tr, spd_mode=spd_m, cal_mode=cal_m)
                b_cache[key] = (s_b, max(1, tr - la_b))
            s_b, cal_b = b_cache[key]
            z_a = calibrate_alt(s_a, cal_a, method=cal_m, q_tail=0.98)
            z_b = calibrate_alt(s_b, cal_b, method=cal_m, q_tail=0.98)
            f = fuse_alt(z_a, z_b, min(cal_a, cal_b), rule=fuse_m)
            sc = 0.5 * f + 0.5 * moving_average(f, 16, causal=False)

        ap, ar = compute_auc_pr_roc(y, sc)
        vp, vr = compute_official_vus(y, sc, sw) if sw is not None else (np.nan, np.nan)
        out_rows.append({"file": fn, "dataset": ds, "group": grp, "variant": label,
                         "AUC-PR": ap, "AUC-ROC": ar, "VUS-PR": vp, "VUS-ROC": vr})
    return out_rows


def _worker_exp2(task):
    fn, data_dir, use_vus, seed = task
    rng = np.random.default_rng(seed)
    df = pd.read_csv(os.path.join(data_dir, fn))
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int32)
    T = X.shape[0]
    tr = parse_train_len(fn, T)
    ds = parse_dataset_name(fn)
    contam = is_prefix_contaminated(fn, tr)
    sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1)) if (use_vus and HAS_TSB_AD) else None

    r_cs, r_noise, r_contam = [], [], []

    # 2A: Cold-start prefix truncation on long prefixes (tr >= 400)
    if tr >= 400:
        for frac in [0.25, 0.50, 0.75, 1.00]:
            eff_tr = max(40, int(tr * frac))
            X_sub = np.vstack([X[tr - eff_tr : tr], X[tr:]])
            y_sub = np.concatenate([y[tr - eff_tr : tr], y[tr:]])
            sw_sub = int(find_length_rank(X_sub[:, 0].reshape(-1, 1), rank=1)) if (use_vus and HAS_TSB_AD) else None
            for mname in ["DB-ESC", "PCA", "IForest"]:
                sc = fit_score_dbesc(X_sub, eff_tr)["score_final"] if mname == "DB-ESC" else run_baseline(mname, X_sub, eff_tr)
                ap, ar = compute_auc_pr_roc(y_sub, sc)
                vp, vr = compute_official_vus(y_sub, sc, sw_sub) if sw_sub is not None else (np.nan, np.nan)
                r_cs.append({"file": fn, "dataset": ds, "frac": frac, "method": mname,
                             "AUC-PR": ap, "AUC-ROC": ar, "VUS-PR": vp, "VUS-ROC": vr})

    # 2B & 2C: Strictly on Clean-160 (uncontaminated prefix)
    if not contam:
        sig_pow = np.mean(X ** 2, axis=0, keepdims=True) + 1e-9
        for snr in [30, 20, 15, 10]:
            Xn = X + rng.standard_normal(X.shape) * np.sqrt(sig_pow / (10.0 ** (snr / 10.0)))
            for mname in ["DB-ESC", "PCA", "MCD", "IForest"]:
                sc = fit_score_dbesc(Xn, tr)["score_final"] if mname == "DB-ESC" else run_baseline(mname, Xn, tr)
                ap, ar = compute_auc_pr_roc(y, sc)
                vp, vr = compute_official_vus(y, sc, sw) if sw is not None else (np.nan, np.nan)
                r_noise.append({"file": fn, "dataset": ds, "snr_db": snr, "method": mname,
                                "AUC-PR": ap, "AUC-ROC": ar, "VUS-PR": vp, "VUS-ROC": vr})

        std_tr = np.std(X[:tr], axis=0) + 1e-6
        for rate in [0.01, 0.02, 0.05, 0.10]:
            Xc = X.copy()
            n_anom = max(5, int(tr * rate))
            st = int(rng.integers(max(1, tr // 10), max(2, tr - n_anom - 1)))
            Xc[st : st + n_anom] += rng.choice([-1.0, 1.0], size=(1, X.shape[1])) * 4.5 * std_tr
            for mname in ["DB-ESC", "PCA", "MCD", "IForest"]:
                sc = fit_score_dbesc(Xc, tr)["score_final"] if mname == "DB-ESC" else run_baseline(mname, Xc, tr)
                ap, ar = compute_auc_pr_roc(y, sc)
                vp, vr = compute_official_vus(y, sc, sw) if sw is not None else (np.nan, np.nan)
                r_contam.append({"file": fn, "dataset": ds, "contam_rate": rate, "method": mname,
                                 "AUC-PR": ap, "AUC-ROC": ar, "VUS-PR": vp, "VUS-ROC": vr})

    return r_cs, r_noise, r_contam


# =============================================================================
# 5. Experiment 5: Frozen Zero-Tuning External Benchmark (SKAB, 34 Series)
# =============================================================================

def download_skab(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "skab.zip")
    url = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"
    print(f"  Downloading SKAB from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return os.path.join(dest_dir, "SKAB-master", "data")


def run_exp5_skab(skab_dir: str, out_dir: str, auto_download: bool = False):
    if (not skab_dir or not os.path.isdir(skab_dir)) and auto_download:
        skab_dir = download_skab(os.path.join(out_dir, "skab_data"))
    if not skab_dir or not os.path.isdir(skab_dir):
        print("[Exp 5] SKAB directory not found. Pass --skab_dir or --download_skab.")
        return

    csv_files = sorted(glob.glob(os.path.join(skab_dir, "**", "*.csv"), recursive=True))
    csv_files = [f for f in csv_files if "anomaly-free" not in os.path.basename(f).lower()]
    print(f"\n[Exp 5] Frozen DB-ESC External Blind Benchmark on SKAB ({len(csv_files)} series)...")
    rows = []
    for fp in csv_files:
        df = pd.read_csv(fp, sep=";")
        if "anomaly" not in df.columns:
            continue
        y = (df["anomaly"].values > 0).astype(np.int32)
        drop_cols = [c for c in ["datetime", "anomaly", "changepoint"] if c in df.columns]
        X = df.drop(columns=drop_cols).values.astype(np.float64)
        first_pos = int(np.argmax(y > 0)) if y.sum() > 0 else len(y) // 3
        tr = max(64, min(400, first_pos))
        sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1)) if HAS_TSB_AD else None
        for mname in ["DB-ESC (Frozen 2B)", "ThreeView_Proto (+HBOS)", "PCA", "IForest", "MCD", "OCSVM", "KMeansAD"]:
            t0 = time.perf_counter()
            if mname == "DB-ESC (Frozen 2B)":
                sc = fit_score_dbesc(X, tr, causal=False, strict_stride_truncation=True)["score_final"]
            elif mname == "ThreeView_Proto (+HBOS)":
                sc = fit_score_3view_proto(X, tr)
            else:
                sc = run_baseline(mname, X, tr)
            dt = time.perf_counter() - t0
            ap, ar = compute_auc_pr_roc(y, sc)
            vp, vr = compute_official_vus(y, sc, sw) if sw is not None else (np.nan, np.nan)
            rows.append({"file": os.path.relpath(fp, skab_dir), "method": mname, "time_s": dt,
                         "VUS-PR": vp, "VUS-ROC": vr, "AUC-PR": ap, "AUC-ROC": ar})

    df_skab = pd.DataFrame(rows)
    df_skab.to_csv(os.path.join(out_dir, "exp5_skab_per_series.csv"), index=False)
    agg = df_skab.groupby("method")[["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC", "time_s"]].mean().reset_index()
    agg.to_csv(os.path.join(out_dir, "exp5_skab_summary.csv"), index=False)
    print(agg.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Exact dbesc.py Neurocomputing CPU Experiment Suite")
    parser.add_argument("--data_dir", type=str, default="", help="Path to TSB-AD-M directory")
    parser.add_argument("--file_list", type=str, default="", help="Path to TSB-AD-M-Eva.csv (180 series)")
    parser.add_argument("--skab_dir", type=str, default="", help="Path to local SKAB data directory")
    parser.add_argument("--out_dir", type=str, default="./nc_results", help="Output directory")
    parser.add_argument("--exp", nargs="+", type=int, default=[1, 2, 5], help="Experiments: 1 2 5")
    parser.add_argument("--workers", type=int, default=4, help="Parallel CPU workers")
    parser.add_argument("--download_skab", action="store_true", help="Auto-download SKAB from GitHub")
    parser.add_argument("--fast_auc_exp12", action="store_true", help="AUC-PR/ROC only for Exp 1 & 2 (fast)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    use_vus_12 = (not args.fast_auc_exp12) and HAS_TSB_AD
    print(f"TSB_AD installed: {HAS_TSB_AD} | Exp 1&2 VUS: {use_vus_12} | Workers: {args.workers}")

    files = []
    if args.file_list and os.path.isfile(args.file_list):
        fl = pd.read_csv(args.file_list)
        files = fl["file_name"].tolist() if "file_name" in fl.columns else fl.iloc[:, 0].tolist()
    elif args.data_dir and os.path.isdir(args.data_dir):
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, "*.csv")))

    if 1 in args.exp and files:
        t0 = time.perf_counter()
        print(f"\n[Exp 1] Running 12 single-variable substitutions on {len(files)} series...")
        tasks = [(fn, args.data_dir, use_vus_12) for fn in files]
        all_rows = []
        if args.workers <= 1:
            for i, t in enumerate(tasks, 1):
                all_rows.extend(_worker_exp1(t))
                if i % 20 == 0 or i == len(tasks):
                    print(f"  Exp 1: {i}/{len(tasks)}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for i, fut in enumerate(as_completed([ex.submit(_worker_exp1, t) for t in tasks]), 1):
                    all_rows.extend(fut.result())
                    if i % 20 == 0 or i == len(tasks):
                        print(f"  Exp 1: {i}/{len(tasks)}")
        df1 = pd.DataFrame(all_rows)
        df1.to_csv(os.path.join(args.out_dir, "exp1_substitutions_per_series.csv"), index=False)
        ser = df1.groupby(["group", "variant"])[["AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC"]].mean().reset_index()
        ds_m = df1.groupby(["group", "variant", "dataset"])[["AUC-PR", "VUS-PR"]].mean().groupby(["group", "variant"]).mean().reset_index()
        agg1 = ser.merge(ds_m, on=["group", "variant"], suffixes=("_Ser", "_DS"))
        agg1.to_csv(os.path.join(args.out_dir, "exp1_substitutions_summary.csv"), index=False)
        print(f"[Exp 1] Done in {time.perf_counter() - t0:.1f}s")
        print(agg1.to_string(index=False))

    if 2 in args.exp and files:
        t0 = time.perf_counter()
        print(f"\n[Exp 2] Running Cold-Start / Noise SNR / Contamination stress tests on {len(files)} series...")
        tasks = [(fn, args.data_dir, use_vus_12, 2026 + idx) for idx, fn in enumerate(files)]
        cs_all, n_all, c_all = [], [], []
        if args.workers <= 1:
            for i, t in enumerate(tasks, 1):
                a, b, c = _worker_exp2(t)
                cs_all.extend(a); n_all.extend(b); c_all.extend(c)
                if i % 20 == 0 or i == len(tasks):
                    print(f"  Exp 2: {i}/{len(tasks)}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for i, fut in enumerate(as_completed([ex.submit(_worker_exp2, t) for t in tasks]), 1):
                    a, b, c = fut.result()
                    cs_all.extend(a); n_all.extend(b); c_all.extend(c)
                    if i % 20 == 0 or i == len(tasks):
                        print(f"  Exp 2: {i}/{len(tasks)}")
        pd.DataFrame(cs_all).to_csv(os.path.join(args.out_dir, "exp2a_cold_start.csv"), index=False)
        pd.DataFrame(n_all).to_csv(os.path.join(args.out_dir, "exp2b_noise_snr.csv"), index=False)
        pd.DataFrame(c_all).to_csv(os.path.join(args.out_dir, "exp2c_prefix_contamination.csv"), index=False)
        print(f"[Exp 2] Done in {time.perf_counter() - t0:.1f}s")

    if 5 in args.exp:
        run_exp5_skab(args.skab_dir, args.out_dir, auto_download=args.download_skab)


if __name__ == "__main__":
    main()
