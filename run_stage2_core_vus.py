#!/usr/bin/env python3
"""
Stage-2 Core Scientific Evidence Suite for DB-ESC (Neurocomputing Final Revision)
=================================================================================
Place alongside `dbesc.py` in `03_Code_and_Verification/`.
Runs the 4 high-value local CPU experiments identified in Round-2 Review:
  S2-A: Core VUS-PR Ablation on Final Frozen `dbesc.py` (180 TSB-AD-M-Eva series)
        (Branch A only, Branch B only, A+B uncalibrated, A+B ECDF, A+B POT no-DS, Full DB-ESC)
  S2-B: Multi-Scale Covariance ({48}, {96}, {192}, {48,96}, {48,96,192}), Stage-1/Stage-2 POT,
        and Branch A Fixed-Window (32, 64, 128) vs. FFT Period-Adaptive Ablation
  S2-C: Real 180-Series Branch Variance Ratio & Empirical Noise-Cancellation Audit
        (Var(zA)/Var(zB), realized Var(s_fuse) vs theoretical (1+r_AB)/2, max branch ratio)
  S2-D: SKAB Label-Free Fixed-Prefix Controls (t_tr = 64, 128; plus count of t_1st < 64)

Usage:
  python run_stage2_core_vus.py \
    --data_dir "D:/ms_cmesc/TSB-AD-M" \
    --file_list "D:/ms_cmesc/TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv" \
    --skab_dir "./nc_results/skab_data/SKAB-master/data" \
    --out_dir "./stage2_results" --workers 4
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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from dbesc import (
    branch_a_temporal_manifold,
    branch_b_spd_pyramid,
    _single_scale_spd_cov,
    evt_pot_calibrate,
    fit_score_dbesc,
    moving_average,
    robust_standardize,
)

warnings.filterwarnings("ignore")

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    HAS_TSB_AD = True
except Exception:
    HAS_TSB_AD = False


def parse_train_len(fn: str, T: int) -> int:
    m = re.search(r"_tr_(\d+)_", os.path.basename(fn))
    return max(64, min(int(m.group(1)), T - 1)) if m else max(64, int(0.2 * T))


def parse_dataset(fn: str) -> str:
    parts = os.path.basename(fn).split("_")
    return parts[1] if len(parts) > 1 else "Unknown"


def auc_pr_roc(y_true, scores):
    y = (y_true > 0).astype(np.int32)
    n_pos = int(np.sum(y))
    n = len(y)
    if n_pos == 0 or n_pos == n:
        return np.nan, np.nan
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ap = float(np.sum((np.cumsum(ys) / np.arange(1, n + 1, dtype=np.float64))[ys == 1]) / n_pos)
    ranks = rankdata(s)
    ar = float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos)))
    return ap, ar


def vus_pr_roc(y_true, scores, sw):
    if not HAS_TSB_AD or sw is None:
        return np.nan, np.nan
    y = (y_true > 0).astype(np.int32)
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ptp = float(np.ptp(s))
    s = (s - float(np.min(s))) / ptp if ptp > 1e-12 else np.zeros_like(s)
    res = get_metrics(s, y, slidingWindow=int(sw))
    return float(res.get("VUS-PR", res.get("VUS_PR", np.nan))), float(res.get("VUS-ROC", res.get("VUS_ROC", np.nan)))


def ecdf_cal(raw, cal_end):
    ref = np.sort(raw[:cal_end])
    n_ref = len(ref)
    idx = np.searchsorted(ref, raw, side="right")
    return -np.log(np.clip((n_ref - idx + 0.5) / (n_ref + 1.0), 1e-12, 1.0))


def branch_a_fixed_window(Z, tr, w_fixed):
    """Same as branch_a_temporal_manifold but with fixed window w_fixed."""
    import dbesc as _db
    orig = _db.estimate_period_acf
    _db.estimate_period_acf = lambda *args, **kwargs: w_fixed
    try:
        return branch_a_temporal_manifold(Z, tr, strict_stride_truncation=True)
    finally:
        _db.estimate_period_acf = orig


def _worker_tsb(task):
    fn, data_dir, compute_vus = task
    df = pd.read_csv(os.path.join(data_dir, fn))
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int32)
    T = X.shape[0]
    tr = parse_train_len(fn, T)
    ds = parse_dataset(fn)
    sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1)) if (compute_vus and HAS_TSB_AD) else None

    Z = robust_standardize(X, tr)
    out = fit_score_dbesc(X, tr, causal=False, strict_stride_truncation=True)
    s_a, la_a = branch_a_temporal_manifold(Z, tr, strict_stride_truncation=True)
    if tr - la_a < 32:
        s_a, la_a = branch_a_temporal_manifold(Z, tr, causal=True, strict_stride_truncation=True)
    cal_a = max(1, tr - la_a)
    s_b, la_b = branch_b_spd_pyramid(Z, tr, strict_stride_truncation=True)
    if tr - la_b < 32:
        s_b, la_b = branch_b_spd_pyramid(Z, tr, causal=True, strict_stride_truncation=True)
    cal_b = max(1, tr - la_b)

    # S2-C: Real prefix variance & correlation audit
    cal_all = min(cal_a, cal_b)
    za, zb = out["z_a"], out["z_b"]
    var_a = float(np.var(za[:cal_all])) + 1e-12
    var_b = float(np.var(zb[:cal_all])) + 1e-12
    var_fuse = float(np.var(out["score_fuse"][:cal_all]))
    r_emp = float(np.corrcoef(za[:cal_all], zb[:cal_all])[0, 1]) if (var_a > 1e-10 and var_b > 1e-10) else 0.0
    var_row = {
        "file": fn, "dataset": ds, "var_a": var_a, "var_b": var_b,
        "var_ratio_a_over_b": var_a / var_b,
        "r_ab_emp": r_emp,
        "var_fuse_emp": var_fuse,
        "var_fuse_theo_equal": 0.5 * (1.0 + r_emp) * 0.5 * (var_a + var_b),
        "max_ratio_b_over_a": float(np.max(zb) / (np.max(za) + 1e-9)),
    }

    # S2-A: Core VUS-PR Ablations
    uncal_ab = 0.5 * (s_a / (np.std(s_a[:cal_a]) + 1e-6) + s_b / (np.std(s_b[:cal_b]) + 1e-6))
    ecdf_ab = 0.5 * (ecdf_cal(s_a, cal_a) + ecdf_cal(s_b, cal_b))
    ecdf_ab_ds = 0.5 * ecdf_ab + 0.5 * moving_average(ecdf_ab, 16, causal=False)

    variants_s2a = {
        "BranchA_Only": 0.5 * za + 0.5 * moving_average(za, 16, causal=False),
        "BranchB_Only": 0.5 * zb + 0.5 * moving_average(zb, 16, causal=False),
        "AB_Uncalibrated": 0.5 * uncal_ab + 0.5 * moving_average(uncal_ab, 16, causal=False),
        "AB_ECDF_DS": ecdf_ab_ds,
        "AB_POT_NoDS": out["score_fuse"],
        "Full_DB_ESC": out["score_final"],
    }

    # S2-B: Multi-Scale Covariance + Stage-1/2 + Fixed Window Branch A
    variants_s2b = {}
    for w_set, tag in [((48,), "Cov_48"), ((96,), "Cov_96"), ((192,), "Cov_192"), ((48, 96), "Cov_48_96")]:
        sb_w, lab_w = branch_b_spd_pyramid(Z, tr, windows=w_set, strict_stride_truncation=True)
        zb_w = evt_pot_calibrate(sb_w, max(1, tr - lab_w), q_tail=0.98)
        f_w = 0.5 * (za + zb_w)
        variants_s2b[tag] = 0.5 * f_w + 0.5 * moving_average(f_w, 16, causal=False)

    # No Stage-2 POT (average Stage-1 scales directly into fusion)
    f_nos2 = 0.5 * (za + s_b)
    variants_s2b["No_Stage2_POT"] = 0.5 * f_nos2 + 0.5 * moving_average(f_nos2, 16, causal=False)

    # Fixed Window Branch A (32, 64, 128)
    for wf in (32, 64, 128):
        sa_f, la_f = branch_a_fixed_window(Z, tr, wf)
        za_f = evt_pot_calibrate(sa_f, max(1, tr - la_f), q_tail=0.98)
        f_f = 0.5 * (za_f + zb)
        variants_s2b[f"BranchA_FixedW{wf}"] = 0.5 * f_f + 0.5 * moving_average(f_f, 16, causal=False)

    s2a_rows, s2b_rows = [], []
    for tag, sc in variants_s2a.items():
        ap, ar = auc_pr_roc(y, sc)
        vp, vr = vus_pr_roc(y, sc, sw)
        s2a_rows.append({"file": fn, "dataset": ds, "variant": tag, "VUS-PR": vp, "VUS-ROC": vr, "AUC-PR": ap, "AUC-ROC": ar})
    for tag, sc in variants_s2b.items():
        ap, ar = auc_pr_roc(y, sc)
        vp, vr = vus_pr_roc(y, sc, sw)
        s2b_rows.append({"file": fn, "dataset": ds, "variant": tag, "VUS-PR": vp, "VUS-ROC": vr, "AUC-PR": ap, "AUC-ROC": ar})
    return s2a_rows, s2b_rows, var_row


def run_skab_labelfree(skab_dir, out_dir):
    csv_files = sorted(glob.glob(os.path.join(skab_dir, "**", "*.csv"), recursive=True))
    csv_files = [f for f in csv_files if "anomaly-free" not in os.path.basename(f).lower()]
    if not csv_files:
        print("[S2-D] No SKAB CSVs found.")
        return
    rows = []
    n_lt64 = 0
    for fp in csv_files:
        df = pd.read_csv(fp, sep=";")
        if "anomaly" not in df.columns:
            continue
        y = (df["anomaly"].values > 0).astype(np.int32)
        first_pos = int(np.argmax(y > 0)) if y.sum() > 0 else len(y)
        if first_pos < 64:
            n_lt64 += 1
        drop_cols = [c for c in ["datetime", "anomaly", "changepoint"] if c in df.columns]
        X = df.drop(columns=drop_cols).values.astype(np.float64)
        sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1)) if HAS_TSB_AD else None
        for tr_fixed in (64, 128, 200):
            sc = fit_score_dbesc(X, tr_fixed, causal=False, strict_stride_truncation=True)["score_final"]
            ap, ar = auc_pr_roc(y, sc)
            vp, vr = vus_pr_roc(y, sc, sw)
            rows.append({"file": os.path.relpath(fp, skab_dir), "tr_fixed": tr_fixed, "t_1st": first_pos,
                         "VUS-PR": vp, "VUS-ROC": vr, "AUC-PR": ap, "AUC-ROC": ar})
    df_s = pd.DataFrame(rows)
    df_s.to_csv(os.path.join(out_dir, "s2d_skab_labelfree.csv"), index=False)
    print(f"\n[S2-D SKAB Label-Free Control] Series with t_1st < 64: {n_lt64}/{len(csv_files)}")
    print(df_s.groupby("tr_fixed")[["VUS-PR", "VUS-ROC", "AUC-PR"]].mean().round(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--file_list", type=str, default="")
    ap.add_argument("--skab_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="./stage2_results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_vus", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.data_dir and args.file_list and os.path.isfile(args.file_list):
        fl = pd.read_csv(args.file_list)
        files = fl["file_name"].tolist() if "file_name" in fl.columns else fl.iloc[:, 0].tolist()
        tasks = [(fn, args.data_dir, (not args.skip_vus)) for fn in files]
        s2a_all, s2b_all, var_all = [], [], []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(_worker_tsb, t) for t in tasks]), 1):
                a, b, v = fut.result()
                s2a_all.extend(a); s2b_all.extend(b); var_all.append(v)
                if i % 20 == 0 or i == len(tasks):
                    print(f"Progress: {i}/{len(tasks)} ({time.perf_counter()-t0:.1f}s)")
        df_a = pd.DataFrame(s2a_all)
        df_b = pd.DataFrame(s2b_all)
        df_v = pd.DataFrame(var_all)
        df_a.to_csv(os.path.join(args.out_dir, "s2a_core_vus_ablation.csv"), index=False)
        df_b.to_csv(os.path.join(args.out_dir, "s2b_multiscale_period_ablation.csv"), index=False)
        df_v.to_csv(os.path.join(args.out_dir, "s2c_real_branch_variance.csv"), index=False)
        print("\n=== S2-A CORE VUS-PR ABLATION ===")
        print(df_a.groupby("variant")[["VUS-PR", "AUC-PR"]].mean().round(4))
        print("\n=== S2-B MULTISCALE & PERIOD ABLATION ===")
        print(df_b.groupby("variant")[["VUS-PR", "AUC-PR"]].mean().round(4))
        print("\n=== S2-C REAL 180-SERIES VARIANCE AUDIT ===")
        print(df_v[["var_ratio_a_over_b", "r_ab_emp", "var_fuse_emp", "var_fuse_theo_equal", "max_ratio_b_over_a"]].median().round(4))

    if args.skab_dir and os.path.isdir(args.skab_dir):
        run_skab_labelfree(args.skab_dir, args.out_dir)


if __name__ == "__main__":
    main()
