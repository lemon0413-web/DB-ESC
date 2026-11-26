"""
Exact v5 vs. v6/v7 POT Anchor Audit Script (Resolves Reviewer Item 1).

Checks across TSB-AD-M-Eva (180 series) and/or TSB-AD-M-Tuning (20 series):
  1. How many series trigger z_body(u_v) > -log(p_u) in Branch A, Branch B Stage-1, or Branch B Stage-2;
  2. Maximum pointwise anomaly score difference max |s_final_anchored - s_final_unanchored|;
  3. Exact AUC-PR difference across all series;
  4. For ANY series with max_score_diff > 1e-12, automatically computes official full-resolution VUS-PR
     for both versions and prints the exact impact on Series-Average and Dataset-Average VUS-PR.

Usage (PowerShell, ~25s for all 200 series):
  python check_pot_ties.py --data_dir .\TSB-AD-M `
    --file_list .\TSB-AD\Datasets\File_List\TSB-AD-M-Eva.csv `
    --prev_vus_csv .\p1_eva180_strict.csv `
    --out_csv .\pot_anchor_audit_eva180.csv

  python check_pot_ties.py --data_dir .\TSB-AD-M `
    --file_list .\TSB-AD\Datasets\File_List\TSB-AD-M-Tuning.csv `
    --prev_vus_csv .\p1_tuning20.csv `
    --out_csv .\pot_anchor_audit_tuning20.csv
"""
import argparse, os, re, time
import numpy as np
import pandas as pd
from dbesc import (
    robust_standardize,
    branch_a_temporal_manifold,
    _single_scale_spd_cov,
    moving_average,
)


def pot_pair(raw, cal_end, q_tail=0.98):
    ref = raw[:cal_end]
    n_ref = len(ref)
    u = float(np.quantile(ref, q_tail))
    excess = ref[ref > u] - u
    beta = max(float(np.mean(excess)) if len(excess) >= 3 else float(np.std(ref)) * 0.5, 1e-6)
    ref_sorted = np.sort(ref)
    idx = np.searchsorted(ref_sorted, raw, side="right")
    p_emp = np.clip((n_ref - idx + 0.5) / (n_ref + 1.0), 1e-12, 1.0)
    nlogp_emp = -np.log(p_emp)
    p_u = max(1.0 - q_tail, 1.0 / (n_ref + 1.0))
    mlpu = -np.log(p_u)
    idx_u = int(np.searchsorted(ref_sorted, u, side="right"))
    z_u_body = -np.log(max((n_ref - idx_u + 0.5) / (n_ref + 1.0), 1e-12))
    b_v = max(mlpu, z_u_body)
    triggered = (z_u_body > mlpu + 1e-12)
    z_old = np.where(raw > u, mlpu + np.maximum(0.0, raw - u) / beta, nlogp_emp)
    z_new = np.where(raw > u, b_v + np.maximum(0.0, raw - u) / beta, nlogp_emp)
    return z_old, z_new, triggered, z_u_body, mlpu


def auc_pr(y_true, scores):
    y = (y_true > 0).astype(np.int32)
    n_pos, n = int(np.sum(y)), len(y)
    if n_pos == 0 or n_pos == n:
        return np.nan
    s = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_sorted = y[np.argsort(-s, kind="mergesort")]
    prec = np.cumsum(y_sorted) / np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum(prec[y_sorted == 1]) / n_pos)


def score_both_versions(X, tr):
    Z = robust_standardize(X, tr)
    sa, la_a = branch_a_temporal_manifold(Z, tr, strict_stride_truncation=True)
    if tr - la_a < 32:
        sa, la_a = branch_a_temporal_manifold(Z, tr, causal=True, strict_stride_truncation=True)
    cal_a = max(1, tr - la_a)
    za_old, za_new, trig_a, _, _ = pot_pair(sa, cal_a)

    T, C = Z.shape
    step_b = max(1, T // 3000)
    sc_old, sc_new, max_la, trig_b1 = [], [], 0, 0
    if C > 1:
        for w in (48, 96, 192):
            if tr >= 2 * w:
                req_la = (w - w // 2) + (step_b - 1)
                if tr - req_la < 32:
                    continue
                sw, lw = _single_scale_spd_cov(Z, tr, w_cov=w, strict_stride_truncation=True)
                zo, zn, tr_w, _, _ = pot_pair(sw, tr - lw)
                sc_old.append(zo)
                sc_new.append(zn)
                trig_b1 += int(tr_w)
                max_la = max(max_la, lw)
        if not sc_old:
            use_c = (tr - (min(48, max(12, tr // 4)) // 2 + (step_b - 1)) < 32)
            sw, lw = _single_scale_spd_cov(Z, tr, w_cov=48, causal=use_c, strict_stride_truncation=True)
            sb_old, sb_new, max_la = sw, sw, lw
        else:
            sb_old, sb_new = np.mean(sc_old, axis=0), np.mean(sc_new, axis=0)
    else:
        sb_old = sb_new = np.zeros(T, dtype=np.float64)

    cal_b = max(1, tr - max_la)
    zb_old, _, trig_b2_old, _, _ = pot_pair(sb_old, cal_b)
    _, zb_new, trig_b2_new, _, _ = pot_pair(sb_new, cal_b)

    f_old = 0.5 * (za_old + zb_old)
    f_new = 0.5 * (za_new + zb_new)
    s_old = 0.5 * f_old + 0.5 * moving_average(f_old, 16, causal=False)
    s_new = 0.5 * f_new + 0.5 * moving_average(f_new, 16, causal=False)
    any_trig = bool(trig_a or trig_b1 > 0 or trig_b2_old or trig_b2_new)
    return s_old, s_new, any_trig, int(trig_a), trig_b1, int(trig_b2_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--file_list", required=True)
    ap.add_argument("--prev_vus_csv", default="")
    ap.add_argument("--out_csv", default="pot_anchor_audit.csv")
    args = ap.parse_args()

    fl = pd.read_csv(args.file_list)
    files = fl["file_name"].tolist() if "file_name" in fl.columns else fl.iloc[:, 0].tolist()
    prev_df = pd.read_csv(args.prev_vus_csv) if args.prev_vus_csv and os.path.exists(args.prev_vus_csv) else None
    sw_map = dict(zip(prev_df["file"], prev_df["sw_official"])) if prev_df is not None and "sw_official" in prev_df.columns else {}
    vus_map = dict(zip(prev_df["file"], prev_df["DB_ESC_Strict__VUSPR_off"])) if prev_df is not None and "DB_ESC_Strict__VUSPR_off" in prev_df.columns else {}

    rows = []
    t0 = time.perf_counter()
    for fn in files:
        df = pd.read_csv(os.path.join(args.data_dir, fn))
        X = df.iloc[:, :-1].values.astype(np.float64)
        y = df.iloc[:, -1].values.astype(np.int32)
        T = len(X)
        m = re.search(r"_tr_(\d+)_", os.path.basename(fn))
        tr = max(64, min(int(m.group(1)), T - 1)) if m else max(64, int(0.2 * T))
        ds = os.path.basename(fn).split("_")[1]

        s_old, s_new, any_trig, ta, tb1, tb2 = score_both_versions(X, tr)
        max_diff = float(np.max(np.abs(s_new - s_old)))
        ap_old, ap_new = auc_pr(y, s_old), auc_pr(y, s_new)

        vus_old = vus_map.get(fn, np.nan)
        vus_new = vus_old
        if max_diff > 1e-12:
            from TSB_AD.evaluation.metrics import get_metrics
            sw = int(sw_map[fn]) if fn in sw_map else 100
            y_bin = (y > 0).astype(np.int32)
            def _vus(sc):
                ptp = float(np.ptp(sc))
                sn = (sc - float(np.min(sc))) / ptp if ptp > 1e-12 else np.zeros_like(sc)
                res = get_metrics(sn, y_bin, slidingWindow=sw)
                return float(res.get("VUS-PR", res.get("VUS_PR", np.nan)))
            vus_old = _vus(s_old)
            vus_new = _vus(s_new)

        rows.append({
            "dataset": ds, "file": fn,
            "any_trigger": any_trig, "trig_A": ta, "trig_B1": tb1, "trig_B2": tb2,
            "max_score_diff": max_diff,
            "AUCPR_v5": ap_old, "AUCPR_v7": ap_new, "AUCPR_diff": ap_new - ap_old,
            "VUSPR_v5": vus_old, "VUSPR_v7": vus_new, "VUSPR_diff": vus_new - vus_old,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    n_trig = int(out["any_trigger"].sum())
    n_diff = int((out["max_score_diff"] > 1e-12).sum())
    print(f"Audited {len(out)} series in {time.perf_counter()-t0:.1f}s -> {args.out_csv}")
    print(f"  Triggered z_body(u_v) > -log(p_u): {n_trig}/{len(out)}")
    print(f"  Non-zero score diff (>1e-12):      {n_diff}/{len(out)} (max={out['max_score_diff'].max():.3e})")
    print(f"  Mean AUC-PR: v5={out['AUCPR_v5'].mean():.6f} -> v7={out['AUCPR_v7'].mean():.6f} (diff={out['AUCPR_diff'].mean():+.6f})")
    if out["VUSPR_v7"].notna().all():
        ds_v5 = out.groupby("dataset")["VUSPR_v5"].mean().mean()
        ds_v7 = out.groupby("dataset")["VUSPR_v7"].mean().mean()
        print(f"  Mean VUS-PR: v5={out['VUSPR_v5'].mean():.6f} (DS {ds_v5:.6f}) -> v7={out['VUSPR_v7'].mean():.6f} (DS {ds_v7:.6f})")
    if n_diff > 0:
        print("\nTriggered/changed series:")
        print(out[out["max_score_diff"] > 1e-12][["dataset", "file", "trig_A", "trig_B1", "trig_B2", "max_score_diff", "AUCPR_diff", "VUSPR_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
