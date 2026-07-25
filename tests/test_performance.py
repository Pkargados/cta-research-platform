import numpy as np
import pandas as pd

from backtest.performance import simple_sharpe, performance_stats, MIN_OBS


def test_simple_sharpe_matches_hand_computed():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 10)  # 50 obs, > MIN_OBS
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert np.isclose(simple_sharpe(returns), expected)


def test_simple_sharpe_below_min_obs_is_nan():
    returns = pd.Series([0.01, -0.01, 0.02])  # well under MIN_OBS
    assert np.isnan(simple_sharpe(returns, min_obs=MIN_OBS))


def test_simple_sharpe_zero_std_is_nan():
    returns = pd.Series([0.0] * 30)  # all-zero -> exactly zero std, not just constant
    assert np.isnan(simple_sharpe(returns))


def test_performance_stats_below_min_obs_returns_all_nan_series():
    returns = pd.Series([0.01, -0.01])
    stats = performance_stats(returns)
    assert stats.isna().all()
    assert list(stats.index) == ["Ann Return", "Ann Vol", "Sharpe", "Sortino", "Calmar", "Max DD", "Win Rate"]


def test_performance_stats_positive_drift_gives_positive_sharpe():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    stats = performance_stats(returns)
    assert stats["Sharpe"] > 0
    assert stats["Ann Return"] > 0


def test_performance_stats_max_dd_is_negative_or_zero():
    dates = pd.bdate_range("2020-01-01", periods=100)
    returns = pd.Series([0.01] * 50 + [-0.02] * 50, index=dates)
    stats = performance_stats(returns)
    assert stats["Max DD"] <= 0


def test_performance_stats_win_rate_matches_hand_computed():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03] * 6)  # 30 obs: 18 positive, 12 negative
    stats = performance_stats(returns)
    expected_win_rate = (returns > 0).mean()
    assert np.isclose(stats["Win Rate"], expected_win_rate)


def test_performance_stats_sortino_uses_downside_deviation_only():
    dates = pd.bdate_range("2020-01-01", periods=60)
    # All positive returns except a few VARIED negative ones (not identical
    # values, to avoid a degenerate exactly-zero downside std) - downside dev
    # should be computed only from the negative subset, not the full sample's std.
    downside_values = [-0.01, -0.02, -0.015, -0.03, -0.005]
    returns = pd.Series([0.01] * 55 + downside_values, index=dates)
    stats = performance_stats(returns)
    downside = returns[returns < 0]
    expected_downside_dev = downside.std() * np.sqrt(252)
    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    assert np.isclose(stats["Sortino"], ann_return / expected_downside_dev)


def test_performance_stats_no_negative_returns_gives_nan_sortino():
    returns = pd.Series([0.01] * 30)
    stats = performance_stats(returns)
    assert np.isnan(stats["Sortino"])


def test_performance_stats_dropna_before_min_obs_check():
    # A series with enough raw length but too many NaNs to clear MIN_OBS after
    # dropna() must still be treated as below-threshold.
    values = [np.nan] * 25 + [0.01, -0.01, 0.02]
    returns = pd.Series(values)
    stats = performance_stats(returns)
    assert stats.isna().all()
