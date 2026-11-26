"""
DB-ESC: Dual-Branch Empirical Survival Calibration (Reference Implementation).

Closed-form, gradient-free, pure-CPU multivariate time-series anomaly detector.
Matches Equations (1)-(11) and Algorithm 1 of the manuscript:
  - Prefix-strict median/IQR standardization (Eq. 1)
  - Branch A: Period-adaptive PCA + k-NN manifold with temporal Exclusion Zone (Eqs. 2-4)
  - Branch B: Multi-scale fixed-shrinkage SPD covariance pyramid (Eqs. 5-7) with
              Stage-1 per-scale POT equalization and scale averaging
  - Stage-2 branch-level boundary-strict exponential POT calibration (Eq. 8)
  - Equal-weight minimum-variance fusion (Eq. 9), optional approximate Satterthwaite
    null-variance degrees of freedom nu_eff (Eq. 10), and dual-scale smoothing (Eq. 11).
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def robust_standardize(X: np.ndarray, tr: int) -> np.ndarray:
    """Prefix-strict robust standardization (Eq. 1)."""
    X_tr = X[:tr]
    med = np.median(X_tr, axis=0, keepdims=True)
    q75 = np.percentile(X_tr, 75, axis=0, keepdims=True)
    q25 = np.percentile(X_tr, 25, axis=0, keepdims=True)
    scale = q75 - q25
    std = np.std(X_tr, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    Z = (X - med) / scale
    return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -15.0, 15.0)


def estimate_period_acf(Z_tr: np.ndarray, min_p: int = 24, max_p: int = 160) -> int:
    """FFT autocorrelation dominant cycle estimator strictly on normal prefix."""
    T_tr, _ = Z_tr.shape
    max_lag = min(max_p, T_tr // 4)
    if max_lag <= min_p + 4:
        return 32
    ch = int(np.argmax(np.var(Z_tr, axis=0)))
    x = Z_tr[:, ch] - np.mean(Z_tr[:, ch])
    if np.std(x) < 1e-8:
        return 32
    n_fft = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(f * np.conj(f), n=n_fft)[:max_lag]
    acf = acf / (acf[0] + 1e-12)
    seg = acf[min_p:max_lag]
    if len(seg) < 3:
        return 32
    p_rel = int(np.argmax(seg))
    return int(min_p + p_rel) if seg[p_rel] > 0.25 else 32


def causal_step_interp(t_grid: np.ndarray, anchors: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Strictly causal zero-order hold: timestamp t uses ONLY latest anchor <= t."""
    idx = np.searchsorted(anchors, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def evt_pot_calibrate(raw: np.ndarray, cal_end: int, q_tail: float = 0.98) -> np.ndarray:
    """Boundary-strict exponential POT negative log-survival calibration (Eq. 8)."""
    ref = raw[:cal_end]
    n_ref = len(ref)
    u = float(np.quantile(ref, q_tail))
    excess = ref[ref > u] - u
    beta = float(np.mean(excess)) if len(excess) >= 3 else float(np.std(ref)) * 0.5
    beta = max(beta, 1e-6)

    ref_sorted = np.sort(ref)
    idx = np.searchsorted(ref_sorted, raw, side="right")
    p_emp = np.clip((n_ref - idx + 0.5) / (n_ref + 1.0), 1e-12, 1.0)
    nlogp_emp = -np.log(p_emp)
    p_u = max(1.0 - q_tail, 1.0 / (n_ref + 1.0))
    idx_u = int(np.searchsorted(ref_sorted, u, side="right"))
    z_u_body = -np.log(max((n_ref - idx_u + 0.5) / (n_ref + 1.0), 1e-12))
    b_v = max(-np.log(p_u), z_u_body)
    nlogp_tail = b_v + np.maximum(0.0, raw - u) / beta
    return np.where(raw > u, nlogp_tail, nlogp_emp)


def branch_a_temporal_manifold(
    Z: np.ndarray,
    tr: int,
    d_emb_max: int = 12,
    k_nn: int = 3,
    causal: bool = False,
    strict_stride_truncation: bool = True,
) -> Tuple[np.ndarray, int]:
    """Branch A: Period-adaptive PCA + k-NN manifold with temporal Exclusion Zone (Eqs. 2-4)."""
    T, C = Z.shape
    w = estimate_period_acf(Z[:tr], min_p=24, max_p=160)
    w = min(w, max(8, tr // 5))
    if C > 16:
        var_c = np.var(Z[:tr], axis=0)
        Zc = Z[:, np.argsort(var_c, kind="mergesort")[-16:]]
    else:
        Zc = Z

    ds_in = max(1, w // 24)
    step_tr = max(1, (tr - w) // 1200)
    starts_tr = np.arange(0, max(1, tr - w + 1), step_tr)
    W_tr = np.array([Zc[s : s + w : ds_in].reshape(-1) for s in starts_tr], dtype=np.float64)
    mu = np.mean(W_tr, axis=0, keepdims=True)
    W_tr_c = W_tr - mu

    d_emb = min(d_emb_max, W_tr_c.shape[0] - 1, W_tr_c.shape[1] - 1)
    if d_emb < 1:
        return np.zeros(T, dtype=np.float64), 0

    if W_tr_c.shape[1] <= W_tr_c.shape[0]:
        cov = (W_tr_c.T @ W_tr_c) / max(len(W_tr_c) - 1, 1)
        _, evec = np.linalg.eigh(cov)
        V = evec[:, -d_emb:][:, ::-1]
    else:
        _, _, Vt = np.linalg.svd(W_tr_c, full_matrices=False)
        V = Vt[:d_emb].T

    emb_tr = W_tr_c @ V
    tree = cKDTree(emb_tr)

    step_all = max(1, min(w // 6, max(1, T // 4000)))
    starts_all = np.arange(0, max(1, T - w + 1), step_all)
    if starts_all[-1] != max(0, T - w):
        starts_all = np.append(starts_all, max(0, T - w))
    W_all = np.array([Zc[s : s + w : ds_in].reshape(-1) for s in starts_all], dtype=np.float64) - mu
    emb_all = W_all @ V

    k_eff = min(k_nn, max(1, len(emb_tr) - 1))
    tr_mask = (starts_all + w) <= tr
    n_excl = int(math.ceil(w / max(step_tr, 1))) * 2 + 2
    k_query_tr = min(len(emb_tr), k_eff + n_excl + 4)

    d_knn = np.zeros(len(starts_all), dtype=np.float64)
    idx_tr_q = np.where(tr_mask)[0]
    if len(idx_tr_q) > 0:
        d_raw, i_raw = tree.query(emb_all[idx_tr_q], k=k_query_tr)
        if k_query_tr == 1:
            d_raw, i_raw = d_raw[:, None], i_raw[:, None]
        for row_pos, q_i in enumerate(idx_tr_q):
            s_q = starts_all[q_i]
            ref_starts = starts_tr[i_raw[row_pos]]
            valid = np.abs(ref_starts - s_q) >= w
            d_valid = d_raw[row_pos][valid]
            if len(d_valid) >= k_eff:
                d_knn[q_i] = float(np.mean(d_valid[:k_eff]))
            elif len(d_valid) > 0:
                d_knn[q_i] = float(np.mean(d_valid))
            else:
                d_knn[q_i] = float(np.mean(d_raw[row_pos, -k_eff:]))

    idx_te_q = np.where(~tr_mask)[0]
    if len(idx_te_q) > 0:
        d_te, _ = tree.query(emb_all[idx_te_q], k=k_eff)
        d_knn[idx_te_q] = np.mean(d_te, axis=1) if k_eff > 1 else d_te

    recon = emb_all @ V.T
    d_orth = np.sqrt(np.mean((W_all - recon) ** 2, axis=1) + 1e-12)
    s1 = d_knn / (np.median(d_knn[tr_mask]) + 1e-6) if np.any(tr_mask) else d_knn
    s2 = d_orth / (np.median(d_orth[tr_mask]) + 1e-6) if np.any(tr_mask) else d_orth
    sc = s1 + 0.5 * s2

    t_grid = np.arange(T, dtype=np.float64)
    if causal:
        anchors = starts_all + w - 1.0
        return causal_step_interp(t_grid, anchors, sc), 0
    anchors = starts_all + w / 2.0
    la = (w - w // 2) + ((step_all - 1) if strict_stride_truncation else 0)
    return np.interp(t_grid, anchors, sc), la


def _single_scale_spd_cov(
    Z: np.ndarray,
    tr: int,
    w_cov: int,
    c_eff_max: int = 24,
    lam: float = 0.15,
    eps_tr: float = 1e-4,
    causal: bool = False,
    strict_stride_truncation: bool = True,
) -> Tuple[np.ndarray, int]:
    """Single-scale fixed-shrinkage SPD covariance LogDet divergence (Eqs. 5-7)."""
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

    Z_tr = Zc[:tr]
    Z_tr_m = Z_tr - np.mean(Z_tr, axis=0, keepdims=True)
    S_glob = (Z_tr_m.T @ Z_tr_m) / max(tr - 1, 1)
    tau_0 = max(float(np.trace(S_glob)) / C_eff, eps_tr)
    S0 = (1.0 - lam) * S_glob + lam * tau_0 * eye_c
    S0_inv = np.linalg.inv(S0)
    _, logdet_S0 = np.linalg.slogdet(S0)

    centers = np.arange(0, T, step)
    raw_s = np.zeros(len(centers), dtype=np.float64)
    half = w // 2
    for idx_i, t in enumerate(centers):
        if causal:
            b = min(T, t + 1)
            a = max(0, b - w)
            if b - a < 4:
                b = min(tr, a + w)
        else:
            a = max(0, t - half)
            b = min(T, a + w)
            a = max(0, b - w)
        seg = Zc[a:b]
        seg_m = seg - np.mean(seg, axis=0, keepdims=True)
        Sw = (seg_m.T @ seg_m) / max(len(seg) - 1, 1)
        tau_w = max(float(np.trace(Sw)) / C_eff, eps_tr)
        Sw_reg = (1.0 - lam) * Sw + lam * tau_w * eye_c
        _, logdet_Sw = np.linalg.slogdet(Sw_reg)
        div = 0.5 * (float(np.sum(S0_inv * Sw_reg)) - (logdet_Sw - logdet_S0) - C_eff)
        raw_s[idx_i] = max(0.0, div)

    t_grid = np.arange(T, dtype=np.float64)
    if causal:
        return causal_step_interp(t_grid, centers.astype(np.float64), raw_s), 0
    la = (w - half) + ((step - 1) if strict_stride_truncation else 0)
    return np.interp(t_grid, centers.astype(np.float64), raw_s), la


def branch_b_spd_pyramid(
    Z: np.ndarray,
    tr: int,
    windows: Tuple[int, ...] = (48, 96, 192),
    c_eff_max: int = 24,
    lam: float = 0.15,
    q_tail: float = 0.98,
    causal: bool = False,
    strict_stride_truncation: bool = True,
) -> Tuple[np.ndarray, int]:
    """Branch B: Multi-scale SPD covariance pyramid with Stage-1 per-scale POT equalization."""
    T, C = Z.shape
    if C <= 1:
        return np.zeros(T, dtype=np.float64), 0
    sc_list = []
    max_la = 0
    step_b = max(1, T // 3000)
    for w in windows:
        if tr >= w * 2:
            half_b = w - w // 2
            req_la = 0 if causal else (half_b + ((step_b - 1) if strict_stride_truncation else 0))
            if tr - req_la < 32:
                continue  # Drop wider scale rather than pushing calibration past safe prefix boundary
            s_w, la_w = _single_scale_spd_cov(
                Z, tr, w_cov=w, c_eff_max=c_eff_max, lam=lam,
                causal=causal, strict_stride_truncation=strict_stride_truncation,
            )
            cal_end = tr - la_w
            sc_list.append(evt_pot_calibrate(s_w, cal_end, q_tail=q_tail))
            max_la = max(max_la, la_w)
    if not sc_list:
        use_causal = causal or (tr - (min(48, max(12, tr // 4)) // 2 + ((step_b - 1) if strict_stride_truncation else 0)) < 32)
        s_w, la_w = _single_scale_spd_cov(
            Z, tr, w_cov=48, c_eff_max=c_eff_max, lam=lam,
            causal=use_causal, strict_stride_truncation=strict_stride_truncation,
        )
        return s_w, la_w
    return np.mean(sc_list, axis=0), max_la


def moving_average(x: np.ndarray, w: int, causal: bool = False) -> np.ndarray:
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    if causal:
        return np.convolve(np.pad(x, (w - 1, 0), mode="edge"), kernel, mode="valid")
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), kernel, mode="valid")


def fit_score_dbesc(
    X: np.ndarray,
    tr: int,
    causal: bool = False,
    k_nn: int = 3,
    d_emb_max: int = 12,
    c_eff_max: int = 24,
    windows: Tuple[int, ...] = (48, 96, 192),
    lam: float = 0.15,
    q_tail: float = 0.98,
    w_ds: int = 16,
    reg_alpha: float = 0.15,
    strict_stride_truncation: bool = True,
) -> Dict[str, object]:
    """
    Run DB-ESC on multivariate series X (shape T x C) with normal prefix length tr.

    Returns dict with keys:
      - score_final : final dual-scale ranking score s_t^{final} (Eq. 11)
      - score_fuse  : unsmoothed equal-weight fused score s_t^{fuse} (Eq. 9)
      - z_a, z_b    : calibrated Branch A and Branch B scores
      - r_ab        : prefix Pearson correlation between z_a and z_b
      - nu_eff      : approximate Satterthwaite DoF for s_t^{fuse} (Eq. 10)
      - cal_end_a, cal_end_b, cal_end_all : exact calibration cutoffs
    """
    Z = robust_standardize(X, tr)
    s_a, la_a = branch_a_temporal_manifold(
        Z, tr, d_emb_max=d_emb_max, k_nn=k_nn, causal=causal,
        strict_stride_truncation=strict_stride_truncation,
    )
    if (not causal) and (tr - la_a < 32):
        # Deterministic ultra-short prefix fallback: causal windows guarantee zero test leakage
        s_a, la_a = branch_a_temporal_manifold(
            Z, tr, d_emb_max=d_emb_max, k_nn=k_nn, causal=True,
            strict_stride_truncation=strict_stride_truncation,
        )
    s_b, la_b = branch_b_spd_pyramid(
        Z, tr, windows=windows, c_eff_max=c_eff_max, lam=lam, q_tail=q_tail,
        causal=causal, strict_stride_truncation=strict_stride_truncation,
    )
    if (not causal) and (tr - la_b < 32):
        s_b, la_b = branch_b_spd_pyramid(
            Z, tr, windows=windows, c_eff_max=c_eff_max, lam=lam, q_tail=q_tail,
            causal=True, strict_stride_truncation=strict_stride_truncation,
        )

    cal_a = max(1, tr - la_a)
    cal_b = max(1, tr - la_b)
    cal_all = min(cal_a, cal_b)

    # Branch-level POT (Stage 2 for Branch B)
    z_a = evt_pot_calibrate(s_a, cal_a, q_tail=q_tail)
    z_b = evt_pot_calibrate(s_b, cal_b, q_tail=q_tail)

    # Equal-weight fusion (Eq. 9) & optional approximate Satterthwaite nu_eff (Eq. 10)
    s_fuse = 0.5 * (z_a + z_b)
    va, vb = z_a[:cal_all], z_b[:cal_all]
    if np.std(va) > 1e-12 and np.std(vb) > 1e-12:
        r_ab = float(np.clip(np.corrcoef(va, vb)[0, 1], -0.99, 0.99))
    else:
        r_ab = 0.0
    r_tilde = (1.0 - reg_alpha) * r_ab
    nu_eff = 4.0 / max(1.0 + r_tilde, 1e-9)

    # Dual-scale temporal consistency (Eq. 11)
    s_final = 0.5 * s_fuse + 0.5 * moving_average(s_fuse, w_ds, causal=causal)

    return {
        "score_final": s_final,
        "score_fuse": s_fuse,
        "z_a": z_a,
        "z_b": z_b,
        "r_ab": r_ab,
        "nu_eff": nu_eff,
        "cal_end_a": cal_a,
        "cal_end_b": cal_b,
        "cal_end_all": cal_all,
    }


def sparse_hbos_branch(Z: np.ndarray, tr: int, n_bins: int = 16) -> np.ndarray:
    """Exploratory third marginal branch (pruned in final DB-ESC; included for LOVO audit)."""
    T, C = Z.shape
    Z_tr = Z[:tr]
    S_mat = np.zeros((T, C), dtype=np.float64)
    active = 0
    eps = 1e-6
    for c in range(C):
        col_tr = Z_tr[:, c]
        if np.std(col_tr) < 1e-8 and np.std(Z[:, c]) < 1e-8:
            continue
        active += 1
        col_all = Z[:, c]
        lo, hi = float(np.min(col_tr)), float(np.max(col_tr))
        span = max(hi - lo, 1e-6)
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(col_tr, bins=edges)
        probs = (counts + 1.0) / (tr + n_bins)
        bin_idx = np.clip(np.searchsorted(edges, col_all, side="right") - 1, 0, n_bins - 1)
        s_c = -np.log(probs[bin_idx] + eps)
        under = np.maximum(0.0, lo - col_all) / span
        over = np.maximum(0.0, col_all - hi) / span
        s_c += 2.0 * (under + over)
        med_c = float(np.median(s_c[:tr]))
        iqr_c = float(np.percentile(s_c[:tr], 75) - np.percentile(s_c[:tr], 25))
        sc_c = iqr_c if iqr_c > 1e-6 else max(float(np.std(s_c[:tr])), 0.25)
        S_mat[:, c] = np.maximum(0.0, (s_c - med_c) / sc_c)
    if C == 1:
        return S_mat[:, 0]
    k_top = max(1, min(C, int(math.ceil(math.sqrt(max(active, 1))))))
    part = np.partition(S_mat, C - k_top, axis=1)[:, C - k_top:]
    return 0.5 * np.mean(S_mat, axis=1) + 0.5 * np.mean(part, axis=1)
