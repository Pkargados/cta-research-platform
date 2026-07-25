import numpy as np
import pandas as pd

from signals.momentum import (
    raw_momentum,
    tsmom_signal,
    momentum_grid_signals,
    build_momentum_features,
    TRADING_DAYS_PER_MONTH,
)


def _close_series(n=400, start=100.0, drift=0.001):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    values = start * (1 + drift) ** np.arange(n)
    return pd.DataFrame({"A": values}, index=dates)


def test_raw_momentum_matches_manual_return_no_skip():
    close = _close_series()
    lookback_days = 3 * TRADING_DAYS_PER_MONTH
    mom = raw_momentum(close, lookback_months=3, skip_months=0)

    expected = close["A"] / close["A"].shift(lookback_days) - 1.0
    pd.testing.assert_series_equal(mom["A"], expected, check_names=False)


def test_raw_momentum_with_skip_shifts_base_first():
    close = _close_series()
    skip_days = TRADING_DAYS_PER_MONTH
    lookback_days = 3 * TRADING_DAYS_PER_MONTH
    mom = raw_momentum(close, lookback_months=3, skip_months=1)

    shifted = close["A"].shift(skip_days)
    expected = shifted / shifted.shift(lookback_days) - 1.0
    pd.testing.assert_series_equal(mom["A"], expected, check_names=False)


def test_tsmom_signal_sign_matches_momentum_sign():
    close = _close_series(drift=0.002)  # positive trend throughout
    vol = pd.DataFrame(0.2, index=close.index, columns=close.columns)

    signal = tsmom_signal(close, vol, lookback_months=3, target_vol=0.40)

    valid = signal["A"].dropna()
    assert (valid > 0).all()  # positive trend -> long position throughout


def test_tsmom_signal_magnitude_uses_target_vol_over_vol():
    close = _close_series(drift=0.002)
    vol = pd.DataFrame(0.2, index=close.index, columns=close.columns)

    signal = tsmom_signal(close, vol, lookback_months=3, target_vol=0.40)
    valid = signal["A"].dropna()

    assert np.allclose(valid.abs(), 0.40 / 0.2)


def test_momentum_grid_signals_returns_one_entry_per_lookback():
    close = _close_series()
    vol = pd.DataFrame(0.2, index=close.index, columns=close.columns)

    grid = momentum_grid_signals(close, vol, lookback_months_grid=(1, 3, 12))

    assert set(grid.keys()) == {1, 3, 12}
    for k, sig in grid.items():
        assert sig.shape == close.shape


def test_build_momentum_features_returns_three_transforms():
    close = _close_series()
    vol = pd.DataFrame(0.2, index=close.index, columns=close.columns)

    features = build_momentum_features(close, vol)

    assert set(features.keys()) == {"binary", "continuous", "rank"}
    # binary should only ever take values in {-1, 0, 1} (ignoring NaN)
    binary_values = features["binary"]["A"].dropna().unique()
    assert set(binary_values).issubset({-1.0, 0.0, 1.0})
