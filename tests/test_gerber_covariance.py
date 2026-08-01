import numpy as np
import pandas as pd

from portfolio.gerber_covariance import (
    gerber_correlation,
    gerber_covariance,
    build_gerber_cov_dict,
    _nearest_psd_correlation,
    drop_until_complete,
    DEFAULT_WINDOW,
)


def _clean_returns(n=400, n_assets=3, seed=0):
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(seed)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0, 0.01, (n, n_assets)), index=dates, columns=cols)


def test_gerber_correlation_diagonal_is_exactly_one():
    returns = _clean_returns(n=300, n_assets=4)
    g = gerber_correlation(returns, c=0.5)
    assert np.allclose(np.diag(g.values), 1.0)


def test_gerber_correlation_bounded_in_unit_interval():
    returns = _clean_returns(n=300, n_assets=4)
    for c in (0.5, 0.7, 0.9):
        g = gerber_correlation(returns, c=c)
        vals = g.values[~np.isnan(g.values)]
        assert (vals >= -1.0).all()
        assert (vals <= 1.0).all()


def test_gerber_correlation_is_symmetric():
    returns = _clean_returns(n=300, n_assets=4)
    g = gerber_correlation(returns, c=0.5).values
    assert np.allclose(g, g.T)


def test_gerber_correlation_c_zero_reduces_to_kendalls_tau():
    # At c=0, H_k=0 for every asset, so (for continuous, essentially-never-
    # exactly-zero returns) every observation is Up or Down and none are
    # Neutral - Eq. 11's numerator/denominator then reduces exactly to
    # Kendall's Tau-a (concordant - discordant, over all pairs of dates).
    returns = _clean_returns(n=200, n_assets=2, seed=1)
    g = gerber_correlation(returns, c=0.0)
    a, b = returns["A0"], returns["A1"]
    sign_a = np.sign(a.values)
    sign_b = np.sign(b.values)
    concordant = (sign_a == sign_b).sum()
    discordant = (sign_a != sign_b).sum()
    expected_tau = (concordant - discordant) / len(a)
    assert np.isclose(g.loc["A0", "A1"], expected_tau, atol=1e-9)


def test_gerber_correlation_per_pair_t_uses_only_jointly_valid_dates():
    returns = _clean_returns(n=300, n_assets=3, seed=2)
    gapped = returns.copy()
    # A0 only has history for its first half - A1/A2 are unaffected by this,
    # but any pair involving A0 must use a smaller T than a pair that
    # doesn't (per-pair T_ij, not one global T).
    gapped.iloc[150:, 0] = np.nan

    g = gerber_correlation(gapped, c=0.5)
    # A1-A2 never touch the gapped asset, so their correlation should match
    # a computation over the full 300 rows exactly.
    g_full_pair = gerber_correlation(returns[["A1", "A2"]], c=0.5)
    assert np.isclose(g.loc["A1", "A2"], g_full_pair.loc["A1", "A2"])
    # A0 still produces a real (non-NaN) number against A1/A2, computed off
    # only its own 150 valid rows.
    assert not np.isnan(g.loc["A0", "A1"])


def test_gerber_correlation_all_neutral_pair_is_nan():
    # c so large that nothing ever pierces the threshold - every date is
    # Neutral for every asset, so T_ij == n_NN and the denominator is zero.
    returns = _clean_returns(n=100, n_assets=2, seed=3)
    g = gerber_correlation(returns, c=100.0)
    assert np.isnan(g.loc["A0", "A1"])


def test_nearest_psd_correlation_is_noop_when_already_psd():
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    result = _nearest_psd_correlation(corr)
    assert np.allclose(result, corr)


def test_nearest_psd_correlation_fixes_a_non_psd_matrix():
    # A deliberately invalid "correlation" matrix (not achievable from real
    # data, constructed directly to force a negative eigenvalue).
    non_psd = np.array([
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ])
    assert np.linalg.eigh(non_psd)[0].min() < 0
    fixed = _nearest_psd_correlation(non_psd)
    eigvals = np.linalg.eigh(fixed)[0]
    assert eigvals.min() >= -1e-8
    assert np.allclose(np.diag(fixed), 1.0)


def test_drop_until_complete_removes_the_worst_offender():
    g = pd.DataFrame(
        [[1.0, 0.5, np.nan], [0.5, 1.0, 0.2], [np.nan, 0.2, 1.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    result = drop_until_complete(g)
    # Either A or C must go (they're the only NaN pair) - B survives either way.
    assert "B" in result.columns
    assert not result.isna().to_numpy().any()


def test_gerber_covariance_matches_sigma_diag_times_correlation():
    returns = _clean_returns(n=300, n_assets=3, seed=4)
    cov = gerber_covariance(returns, c=0.5)
    sigma = returns.std(ddof=1)
    for a in cov.columns:
        assert np.isclose(cov.loc[a, a], sigma[a] ** 2, rtol=1e-6)


def test_build_gerber_cov_dict_produces_matrices_with_full_universe_shape():
    returns = _clean_returns(n=400, n_assets=3, seed=5)
    cov_dict = build_gerber_cov_dict(returns, window=252, freq="ME")
    assert len(cov_dict) > 0
    for date, cov in cov_dict.items():
        assert cov.shape == (3, 3)
        assert list(cov.index) == list(returns.columns)
        assert list(cov.columns) == list(returns.columns)


def test_build_gerber_cov_dict_skips_dates_without_full_warmup_window():
    returns = _clean_returns(n=100, n_assets=3)  # fewer than DEFAULT_WINDOW=252
    cov_dict = build_gerber_cov_dict(returns, window=252, freq="ME")
    assert len(cov_dict) == 0


def test_build_gerber_cov_dict_min_frac_gate_leaves_sparse_asset_nan_not_whole_date_dropped():
    returns = _clean_returns(n=400, n_assets=3, seed=6)
    gapped = returns.copy()
    gapped.iloc[100:300, 0] = np.nan  # ~200 of a typical 252-row window gone for A0

    cov_dict = build_gerber_cov_dict(gapped, window=252, freq="ME")
    assert len(cov_dict) > 0  # unlike Ledoit-Wolf, the date itself is NOT dropped
    last_date = list(cov_dict.keys())[-1]
    last_cov = cov_dict[last_date]
    # A1/A2 (unaffected by the gap) must still have a real, non-NaN entry.
    assert not np.isnan(last_cov.loc["A1", "A2"])


def test_default_window_matches_ledoit_wolf_default():
    assert DEFAULT_WINDOW == 252
