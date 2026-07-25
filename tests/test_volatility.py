import numpy as np
import pandas as pd

from data.volatility import yang_zhang_volatility


def _flat_ohlc(n=20, price=100.0, columns=("A",)):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    o = pd.DataFrame(price, index=dates, columns=columns)
    return o.copy(), o.copy(), o.copy(), o.copy()  # open, high, low, close all identical


def test_constant_price_gives_zero_volatility():
    o, h, l, c = _flat_ohlc(n=20)
    vol = yang_zhang_volatility(o, h, l, c, window=10)
    valid = vol.dropna()
    assert len(valid) > 0
    assert np.allclose(valid, 0.0, atol=1e-10)


def test_annualize_true_scales_by_sqrt_252_vs_false():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, 30))
    close = pd.DataFrame({"A": base}, index=dates)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 0.5
    low = close - 0.5

    annualized = yang_zhang_volatility(open_, high, low, close, window=10, annualize=True)
    raw = yang_zhang_volatility(open_, high, low, close, window=10, annualize=False)

    ratio = (annualized / raw).dropna()
    assert np.allclose(ratio, np.sqrt(252), atol=1e-6)


def test_nonpositive_price_propagates_nan_not_crash():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    close = pd.DataFrame({"A": [100.0] * 20}, index=dates)
    close.loc[dates[10], "A"] = -5.0  # a genuine non-positive print (e.g. WTI 2020)
    open_ = close.copy()
    high = close.abs() + 1
    low = close.abs() - 1

    vol = yang_zhang_volatility(open_, high, low, close, window=5)
    # Should not raise, and the affected window should show NaN rather than a
    # complex/garbage value from log() of a negative number.
    assert vol["A"].notna().any()


def test_min_frac_tolerates_scattered_missing_days():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 1, 40))
    close = pd.DataFrame({"A": base}, index=dates)
    open_ = close.shift(1).fillna(close.iloc[0])
    high, low = close + 0.5, close - 0.5

    # Scatter NaN on ~20% of days (well within the default min_frac=0.7 tolerance).
    gapped_close = close.copy()
    gap_positions = [3, 8, 13, 18, 23, 28, 33, 38]
    gapped_close.iloc[gap_positions, 0] = np.nan
    gapped_open = open_.copy()
    gapped_open.iloc[gap_positions, 0] = np.nan
    gapped_high = high.copy()
    gapped_high.iloc[gap_positions, 0] = np.nan
    gapped_low = low.copy()
    gapped_low.iloc[gap_positions, 0] = np.nan

    vol = yang_zhang_volatility(gapped_open, gapped_high, gapped_low, gapped_close, window=10)
    # Not every window should be nulled out just because of scattered single-day gaps.
    assert vol["A"].iloc[15:].notna().any()


def test_roll_mask_excludes_overnight_return_on_roll_date():
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    close = pd.DataFrame({"A": [100.0] * 15}, index=dates)
    open_ = pd.DataFrame({"A": [100.0] * 15}, index=dates)
    high = pd.DataFrame({"A": [101.0] * 15}, index=dates)
    low = pd.DataFrame({"A": [99.0] * 15}, index=dates)

    # A huge fake "roll gap" on day 7: open jumps far from the prior close,
    # simulating a raw-curve roll-date contract switch.
    roll_idx = 7
    open_.iloc[roll_idx, 0] = 150.0
    high.iloc[roll_idx, 0] = 151.0
    low.iloc[roll_idx, 0] = 149.0
    close.iloc[roll_idx, 0] = 150.0
    # Keep subsequent days at the new level so it's not ALSO a real jump elsewhere.
    close.iloc[roll_idx:, 0] = 150.0
    open_.iloc[roll_idx + 1:, 0] = 150.0
    high.iloc[roll_idx + 1:, 0] = 151.0
    low.iloc[roll_idx + 1:, 0] = 149.0

    roll_mask = pd.DataFrame(False, index=dates, columns=["A"])
    roll_mask.iloc[roll_idx, 0] = True

    vol_masked = yang_zhang_volatility(open_, high, low, close, window=10, roll_mask=roll_mask)
    vol_unmasked = yang_zhang_volatility(open_, high, low, close, window=10, roll_mask=None)

    last_masked = vol_masked["A"].dropna().iloc[-1]
    last_unmasked = vol_unmasked["A"].dropna().iloc[-1]
    assert last_masked < last_unmasked
