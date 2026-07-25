import numpy as np
import pandas as pd

from portfolio.covariance import real_period_end_dates, build_cov_dict, DEFAULT_WINDOW


def test_real_period_end_dates_uses_real_trading_days_not_calendar_labels():
    # Business-day index already skips weekends - 2021-01-31 is a Sunday, so the
    # real month-end trading day is 2021-01-29 (Friday), not the calendar label.
    index = pd.bdate_range("2020-12-01", "2021-02-28")
    result = real_period_end_dates(index, freq="ME")
    assert pd.Timestamp("2021-01-31") not in result
    assert pd.Timestamp("2021-01-29") in result
    # Every returned date must be a real member of the original index.
    assert set(result).issubset(set(index))


def test_real_period_end_dates_one_per_period():
    index = pd.bdate_range("2020-01-01", "2020-06-30")
    result = real_period_end_dates(index, freq="ME")
    assert len(result) == 6  # Jan through Jun


def _clean_returns(n=400, n_assets=3, seed=0):
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(seed)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0, 0.01, (n, n_assets)), index=dates, columns=cols)


def test_build_cov_dict_skips_dates_without_full_warmup_window():
    returns = _clean_returns(n=100)  # fewer than DEFAULT_WINDOW=252 trading days total
    cov_dict = build_cov_dict(returns, window=252, freq="ME")
    assert len(cov_dict) == 0  # never reaches a full 252-day window


def test_build_cov_dict_produces_matrices_with_correct_shape_and_labels():
    returns = _clean_returns(n=400, n_assets=3)
    cov_dict = build_cov_dict(returns, window=252, freq="ME")
    assert len(cov_dict) > 0
    for date, cov in cov_dict.items():
        assert cov.shape == (3, 3)
        assert list(cov.index) == list(returns.columns)
        assert list(cov.columns) == list(returns.columns)
        # A real covariance matrix must be symmetric.
        assert np.allclose(cov.values, cov.values.T)


def test_build_cov_dict_keys_are_real_trading_days():
    returns = _clean_returns(n=400)
    cov_dict = build_cov_dict(returns, window=252, freq="ME")
    assert set(cov_dict.keys()).issubset(set(returns.index))


def test_build_cov_dict_min_frac_gate_skips_sparse_windows():
    returns = _clean_returns(n=400, n_assets=2)
    # Make one asset entirely NaN for a big chunk of the window right before a
    # rebalance date - fewer than window*min_frac clean rows should skip that date.
    returns_gapped = returns.copy()
    returns_gapped.iloc[100:300, 0] = np.nan  # ~200 of 252 rows in a typical window gone

    cov_dict_clean = build_cov_dict(returns, window=252, freq="ME")
    cov_dict_gapped = build_cov_dict(returns_gapped, window=252, freq="ME")
    assert len(cov_dict_gapped) < len(cov_dict_clean)


def test_build_cov_dict_default_window_constant():
    assert DEFAULT_WINDOW == 252
