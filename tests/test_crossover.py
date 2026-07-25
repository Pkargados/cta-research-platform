import numpy as np
import pandas as pd

from signals.crossover import crossover_raw, crossover_signal, crossover_pair_signal, all_pair_signals, PAIRS


def _sparse_calendar_close(n=400):
    """~17% of trading days scattered-missing for one asset — the real gap density
    documented in the module docstring (Corn: 849 of 5015 panel dates, ~17%)."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    dense = pd.Series(np.linspace(100, 150, n), index=dates)
    sparse = dense.copy()
    rng = np.random.RandomState(0)
    gap_mask = rng.rand(n) < 0.17
    sparse[gap_mask] = np.nan
    return pd.DataFrame({"Dense": dense, "Sparse": sparse})


def test_crossover_raw_is_fast_minus_slow_sma():
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 200, 250)}, index=dates)

    raw = crossover_raw(close, fast_window=50, slow_window=100)

    fast = close["A"].rolling(50, min_periods=int(50 * 0.7)).mean()
    slow = close["A"].rolling(100, min_periods=int(100 * 0.7)).mean()
    expected = fast - slow
    pd.testing.assert_series_equal(raw["A"], expected, check_names=False)


def test_uptrend_produces_positive_crossover_after_warmup():
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 300, 250)}, index=dates)

    raw = crossover_raw(close, fast_window=50, slow_window=100)

    # In a steady uptrend, the fast (shorter) SMA sits above the slow SMA.
    assert raw["A"].dropna().gt(0).all()


def test_sparse_calendar_asset_is_not_100_percent_nan():
    close = _sparse_calendar_close()
    raw = crossover_raw(close, fast_window=50, slow_window=100)

    # Regression test for the documented bug: a strict min_periods==window rolling
    # mean pushed sparse-calendar assets to 100% NaN across their entire history.
    assert raw["Sparse"].notna().any()


def test_crossover_signal_scales_by_target_vol_over_vol():
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 300, 250)}, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=["A"])

    signal = crossover_signal(close, vol, fast_window=50, slow_window=100, target_vol=0.40)
    valid = signal["A"].dropna()

    assert np.allclose(valid.abs(), 0.40 / 0.2)


def test_crossover_pair_signal_uses_documented_windows():
    assert PAIRS == {"50_100": (50, 100), "50_200": (50, 200), "100_200": (100, 200)}

    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 300, 250)}, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=["A"])

    golden_cross = crossover_pair_signal(close, vol, "50_200")
    assert golden_cross.shape == close.shape


def test_all_pair_signals_returns_three_parallel_books():
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.DataFrame({"A": np.linspace(100, 300, 250)}, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=["A"])

    signals = all_pair_signals(close, vol)

    assert set(signals.keys()) == {"50_100", "50_200", "100_200"}
