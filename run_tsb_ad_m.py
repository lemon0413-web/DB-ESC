"""
Official TSB-AD-M Benchmark Runner for DB-ESC (Supports Tuning-20 & Eva-180).

Example usage:
  # 1) Quick 20-series TSB-AD-M-Tuning verification (~20-40s with 4 workers)
  python run_tsb_ad_m.py \
    --data_dir ./TSB-AD-M \
    --file_list ./TSB-AD/Datasets/File_List/TSB-AD-M-Tuning.csv \
    --out_csv ./dbesc_tuning20_results.csv \
    --workers 4

  # 2) Full 180-series TSB-AD-M-Eva evaluation (optional --strict_stride_truncation)
  python run_tsb_ad_m.py \
    --data_dir ./TSB-AD-M \
    --file_list ./TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv \
    --out_csv ./dbesc_eva180_results.csv \
    --workers 4
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
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
    evt_pot_calibrate,
    fit_score_dbesc,
    moving_average,
    robust_standardize,
    sparse_hbos_branch,
)

warnings.filterwarnings("ignore")


def parse_train_len(filename: str, T: int) -> int:
    m = re.search(r"_tr_(\d+)_", os.path.basename(filename))
    return max(64, min(int(m.group(1)), T - 1)) if m else max(64, int(0.2 * T))


def parse_dataset_name(filename: str) -> str:
    parts = os.path.basename(filename).split("_")
    return parts[1] if len(parts) > 1 else "Unknown"


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
    from TSB_AD.evaluation.metrics import get_metrics
    y = (y_true > 0).astype(np.int32)
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ptp = float(np.ptp(s))
    s = (s - float(np.min(s))) / ptp if ptp > 1e-12 else np.zeros_like(s)
    res = get_metrics(s, y, slidingWindow=int(sw))
    return float(res.get("VUS-PR", res.get("VUS_PR", np.nan))), float(res.get("VUS-ROC", res.get("VUS_ROC", np.nan)))


def fit_score_3view_proto(X: np.ndarray, tr: int, strict_stride_truncation: bool = True) -> np.ndarray:
    Z = robust_standardize(X, tr)
    s_1 = sparse_hbos_branch(Z, tr)
    s_a, la_a = branch_a_temporal_manifold(Z, tr, strict_stride_truncation=strict_stride_truncation)
    s_b, la_b = branch_b_spd_pyramid(Z, tr, strict_stride_truncation=strict_stride_truncation)
    z_1 = evt_pot_calibrate(s_1, tr, q_tail=0.98)
    z_a = evt_pot_calibrate(s_a, max(32, tr - la_a), q_tail=0.98)
    z_b = evt_pot_calibrate(s_b, max(32, tr - la_b), q_tail=0.98)
    cal_all = max(32, tr - max(la_a, la_b))
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


def eval_one(task):
    fn, data_dir, skip_vus, strict_stride = task
    df = pd.read_csv(os.path.join(data_dir, fn))
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int32)
    T, C = X.shape
    tr = parse_train_len(fn, T)
    ds = parse_dataset_name(fn)

    t0 = time.perf_counter()
    out_2v = fit_score_dbesc(X, tr, causal=False, strict_stride_truncation=strict_stride)
    sec_2v = time.perf_counter() - t0
    out_cau = fit_score_dbesc(X, tr, causal=True)
    s_3v = fit_score_3view_proto(X, tr, strict_stride_truncation=strict_stride)

    row = {
        "dataset": ds, "file": fn, "T": T, "C": C, "tr": tr,
        "sec_dbesc": sec_2v, "r_ab": out_2v["r_ab"], "nu_eff": out_2v["nu_eff"],
    }

    sw = None
    if not skip_vus:
        from TSB_AD.utils.slidingWindows import find_length_rank
        sw = int(find_length_rank(X[:, 0].reshape(-1, 1), rank=1))
        row["sw_official"] = sw

    for tag, sc in [("DB_ESC", out_2v["score_final"]), ("ThreeView_Proto", s_3v), ("DB_ESC_Causal", out_cau["score_final"])]:
        ap, ar = compute_auc_pr_roc(y, sc)
        row[f"{tag}__AUCPR"] = ap
        row[f"{tag}__AUCROC"] = ar
        if not skip_vus:
            vp, vr = compute_official_vus(y, sc, sw)
            row[f"{tag}__VUSPR"] = vp
            row[f"{tag}__VUSROC"] = vr
    return row


def main():
    parser = argparse.ArgumentParser(description="DB-ESC TSB-AD-M Benchmark Runner")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--file_list", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="dbesc_results.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_vus", action="store_true", help="Compute AUC-PR/ROC only (fast)")
    parser.add_argument("--no_strict_stride", action="store_true", help="Disable default strict stride truncation")
    args = parser.parse_args()

    fl = pd.read_csv(args.file_list)
    files = fl["file_name"].tolist() if "file_name" in fl.columns else fl.iloc[:, 0].tolist()
    order = {fn: i for i, fn in enumerate(files)}
    tasks = [(fn, args.data_dir, args.skip_vus, (not args.no_strict_stride)) for fn in files]

    rows = []
    t0 = time.perf_counter()
    if args.workers <= 1:
        for t in tasks:
            rows.append(eval_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(eval_one, t) for t in tasks]):
                rows.append(fut.result())

    res = pd.DataFrame(rows)
    res["_ord"] = res["file"].map(order)
    res = res.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    res.to_csv(args.out_csv, index=False)

    print(f"Completed {len(res)} series in {time.perf_counter()-t0:.1f}s -> {args.out_csv}")
    for tag in ["DB_ESC", "ThreeView_Proto", "DB_ESC_Causal"]:
        ap = res[f"{tag}__AUCPR"].mean()
        msg = f"{tag:16s} | Ser AUC-PR={ap:.4f}"
        if not args.skip_vus:
            vp = res[f"{tag}__VUSPR"].mean()
            vp_ds = res.groupby("dataset")[f"{tag}__VUSPR"].mean().mean()
            msg += f" | Ser VUS-PR={vp:.4f} | DS VUS-PR={vp_ds:.4f}"
        print(msg)


if __name__ == "__main__":
    main()
