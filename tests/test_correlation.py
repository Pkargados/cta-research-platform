import numpy as np
import pandas as pd

from portfolio.correlation import rolling_correlation, ewma_correlation, correlation_summary


def test_rolling_correlation_perfect_positive():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 50), index=dates)
    b = a * 2.0 + 1.0  # perfectly linearly related -> correlation 1.0
    result = rolling_correlation(a, b, window=20)
    assert np.allclose(result.dropna(), 1.0, atol=1e-8)


def test_rolling_correlation_perfect_negative():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 50), index=dates)
    b = -a
    result = rolling_correlation(a, b, window=20)
    assert np.allclose(result.dropna(), -1.0, atol=1e-8)


def test_rolling_correlation_no_partial_window_by_default():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    a = pd.Series(np.arange(30.0), index=dates)
    b = pd.Series(np.arange(30.0) * -1, index=dates)
    result = rolling_correlation(a, b, window=20)
    # min_periods defaults to window - first 19 values must be NaN.
    assert result.iloc[:19].isna().all()
    assert result.iloc[19:].notna().all()


def test_rolling_correlation_aligns_on_inner_join():
    dates_a = pd.date_range("2020-01-01", periods=30, freq="D")
    dates_b = pd.date_range("2020-01-10", periods=30, freq="D")
    a = pd.Series(1.0, index=dates_a)
    b = pd.Series(2.0, index=dates_b)
    result = rolling_correlation(a, b, window=5)
    expected_index = dates_a.intersection(dates_b)
    assert set(result.index) == set(expected_index)


def test_ewma_correlation_perfect_positive():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 50), index=dates)
    b = a * 3.0
    result = ewma_correlation(a, b, halflife=10)
    assert np.allclose(result.dropna(), 1.0, atol=1e-6)


def test_ewma_correlation_uses_book_style_halflife_formula():
    # halflife -> alpha via 1 - exp(-ln(2)/halflife), matching Book.run()'s own
    # EWMA convention - spot-check against a hand-computed alpha for halflife=1
    # (alpha should be 0.5 exactly).
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    a = pd.Series([1.0] * 10, index=dates)
    b = pd.Series([1.0] * 10, index=dates)
    result = ewma_correlation(a, b, halflife=1.0)
    expected_alpha = 1.0 - np.exp(-np.log(2.0) / 1.0)
    assert np.isclose(expected_alpha, 0.5)
    # constant, perfectly correlated series -> correlation should be 1.0 wherever defined
    assert np.allclose(result.dropna(), 1.0, atol=1e-6)


def test_correlation_summary_matches_hand_computed_stats():
    series = pd.Series([0.5, -0.5, 0.2, -0.8, np.nan, 0.1])
    summary = correlation_summary(series)
    clean = series.dropna()
    assert summary["n"] == len(clean)
    assert np.isclose(summary["mean"], clean.mean())
    assert np.isclose(summary["std"], clean.std())
    assert np.isclose(summary["min"], clean.min())
    assert np.isclose(summary["max"], clean.max())
    assert np.isclose(summary["pct_negative"], (clean < 0).mean() * 100)


def test_correlation_summary_empty_series():
    summary = correlation_summary(pd.Series(dtype=float))
    assert summary["n"] == 0
    assert np.isnan(summary["mean"])
