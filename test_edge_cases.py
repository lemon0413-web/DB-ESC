"""Unit tests for DB-ESC edge cases, prefix isolation (E03), and fixed-grid causality (E04)."""
import numpy as np
from dbesc import fit_score_dbesc, robust_standardize, estimate_period_acf


def test_edge_cases():
    rng = np.random.default_rng(123)
    for name, X, tr in [
        ("single_channel", rng.standard_normal((500, 1)), 200),
        ("constant_signal", np.ones((500, 5)), 200),
        ("tied_prefix", np.vstack([np.zeros((250, 4)), np.full((250, 4), 10.0)]), 200),
        ("short_prefix", rng.standard_normal((300, 6)), 20),
        ("high_dim_300", rng.standard_normal((400, 300)), 150),
    ]:
        out = fit_score_dbesc(X, tr)
        assert np.all(np.isfinite(out["score_final"])), f"Non-finite in {name}"

    X_nan = rng.standard_normal((400, 5))
    X_nan[10, 0] = np.nan
    X_nan[250, 2] = np.inf
    assert np.all(np.isfinite(fit_score_dbesc(X_nan, 150)["score_final"]))


def test_prefix_isolation_and_causal():
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((1200, 8))
    X2 = X1.copy()
    tr = 500
    X2[tr:] = rng.standard_normal((1200 - tr, 8)) * 1e4 + 5e3
    o1 = fit_score_dbesc(X1, tr, causal=False, strict_stride_truncation=True)
    o2 = fit_score_dbesc(X2, tr, causal=False, strict_stride_truncation=True)
    assert np.max(np.abs(o1["z_a"][:o1["cal_end_a"]] - o2["z_a"][:o2["cal_end_a"]])) == 0.0
    assert np.max(np.abs(o1["z_b"][:o1["cal_end_b"]] - o2["z_b"][:o2["cal_end_b"]])) == 0.0
    assert abs(o1["r_ab"] - o2["r_ab"]) == 0.0

    X3 = X1.copy()
    X3[800:] = 9999.0
    c1 = fit_score_dbesc(X1, tr, causal=True)
    c3 = fit_score_dbesc(X3, tr, causal=True)
    assert np.max(np.abs(c1["score_final"][:800] - c3["score_final"][:800])) == 0.0


if __name__ == "__main__":
    test_edge_cases()
    test_prefix_isolation_and_causal()
    print("ALL_DBESC_UNIT_TESTS_PASSED")
