import numpy as np
import pandas as pd

from portfolio.risk_metrics import (
    historical_var, expected_shortfall, expanding_var, expanding_expected_shortfall,
    expanding_var_and_es,
)


def test_historical_var_matches_hand_computed_quantile():
    # 100 evenly spaced returns from -0.99 to 0.99 (step 0.02) - the 5th percentile
    # (95% confidence) is well-defined and easy to hand-check.
    returns = pd.Series(np.linspace(-0.99, 0.99, 100))
    var = historical_var(returns, confidence=0.95)
    expected = returns.quantile(0.05)
    assert np.isclose(var, expected)
    assert var < 0  # a 95% VaR on a symmetric, zero-centered distribution is negative


def test_expected_shortfall_is_more_negative_than_var():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.02, 500))
    var = historical_var(returns, confidence=0.95)
    es = expected_shortfall(returns, confidence=0.95)
    assert es <= var  # ES averages the tail beyond VaR, so it's at least as extreme


def test_expected_shortfall_equals_mean_of_worst_five_percent():
    returns = pd.Series(np.arange(1, 101) / 100.0)  # 0.01, 0.02, ..., 1.00
    es = expected_shortfall(returns, confidence=0.95)
    var = historical_var(returns, confidence=0.95)
    tail = returns[returns <= var]
    assert np.isclose(es, tail.mean())


def test_empty_series_returns_nan():
    empty = pd.Series(dtype=float)
    assert np.isnan(historical_var(empty))
    assert np.isnan(expected_shortfall(empty))


def test_expanding_var_respects_min_periods():
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 30), index=dates)
    series = expanding_var(returns, confidence=0.95, min_periods=24)

    assert series.iloc[:23].isna().all()
    assert series.iloc[23:].notna().all()


def test_expanding_var_is_point_in_time_correct():
    # A huge negative shock late in the series must NOT affect an earlier date's
    # expanding VaR - the whole point of using .iloc[:i+1] rather than the full
    # series for every date.
    dates = pd.bdate_range("2020-01-01", periods=40)
    returns = pd.Series(0.001, index=dates)
    var_before_shock = expanding_var(returns, confidence=0.95, min_periods=24).iloc[25]

    returns_with_shock = returns.copy()
    returns_with_shock.iloc[35] = -0.90  # shock occurs AFTER index 25
    var_after_shock_added = expanding_var(returns_with_shock, confidence=0.95, min_periods=24).iloc[25]

    assert np.isclose(var_before_shock, var_after_shock_added)


def test_expanding_expected_shortfall_respects_min_periods_and_matches_static():
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.Series(np.random.default_rng(2).normal(0, 0.01, 30), index=dates)
    series = expanding_expected_shortfall(returns, confidence=0.95, min_periods=24)

    assert series.iloc[:23].isna().all()
    # The LAST point's expanding ES should equal the static ES over the full sample.
    assert np.isclose(series.iloc[-1], expected_shortfall(returns, confidence=0.95))


def test_expanding_var_and_es_matches_the_two_separate_functions():
    # expanding_var_and_es exists purely to avoid recomputing the same quantile
    # twice - its output must be numerically identical to calling the two
    # standalone functions separately, not just directionally similar.
    dates = pd.bdate_range("2020-01-01", periods=40)
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.015, 40), index=dates)

    combined = expanding_var_and_es(returns, confidence=0.95, min_periods=24)
    var_series = expanding_var(returns, confidence=0.95, min_periods=24)
    es_series = expanding_expected_shortfall(returns, confidence=0.95, min_periods=24)

    pd.testing.assert_series_equal(combined["var"], var_series, check_names=False)
    pd.testing.assert_series_equal(combined["es"], es_series, check_names=False)
