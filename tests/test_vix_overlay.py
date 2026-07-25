import numpy as np
import pandas as pd

from signals.vix_overlay import vix_size_multiplier, apply_size_multiplier
from backtest.engine import normalized_positions


def test_vix_size_multiplier_is_shifted_no_lookahead():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    vix = pd.Series(20.0, index=dates)
    vix.iloc[-1] = 1000.0  # a huge, same-day spike

    multiplier = vix_size_multiplier(vix, lookback=60, min_periods=10)

    # Today's own huge spike must not appear in today's multiplier (shift(1)).
    assert multiplier.iloc[-1] < 100


def test_vix_size_multiplier_scales_with_relative_level():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    vix = pd.Series(20.0, index=dates)
    vix.iloc[70:] = 40.0  # a sustained doubling

    multiplier = vix_size_multiplier(vix, lookback=60, min_periods=10)

    # Well after the level doubles, the multiplier should exceed 1 (VIX high
    # relative to its own trailing average).
    assert multiplier.iloc[-1] > 1.0


def test_apply_size_multiplier_scales_all_assets_uniformly():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    positions = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=dates)
    multiplier = pd.Series([1.0, 2.0, 0.5], index=dates)

    scaled = apply_size_multiplier(positions, multiplier)

    assert np.isclose(scaled.loc[dates[1], "A"], 1.0)
    assert np.isclose(scaled.loc[dates[1], "B"], -1.0)
    assert np.isclose(scaled.loc[dates[2], "A"], 0.25)


def test_uniform_multiplier_pre_normalization_is_a_no_op_regression():
    """Regression test for the documented bug: applying a uniform daily multiplier
    BEFORE gross-exposure normalization cancels out identically, making the
    overlay silently do nothing. Applying it AFTER normalization must NOT cancel."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    raw_signal = pd.DataFrame({"A": 1.0, "B": -2.0}, index=dates)
    multiplier = pd.Series(np.linspace(0.5, 2.0, 40), index=dates)

    # Bug path: multiply raw signal first, then normalize.
    bugged = normalized_positions(raw_signal.mul(multiplier, axis=0), frequency="daily")
    baseline = normalized_positions(raw_signal, frequency="daily")
    pd.testing.assert_frame_equal(bugged, baseline)

    # Fixed path: normalize first, then apply the multiplier -> genuinely different.
    normalized = normalized_positions(raw_signal, frequency="daily")
    fixed = apply_size_multiplier(normalized, multiplier)
    assert not fixed.equals(baseline)
