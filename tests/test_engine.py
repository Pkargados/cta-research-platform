import numpy as np
import pandas as pd

from backtest.engine import (
    weekly_positions, holding_period_positions, monthly_positions,
    normalized_positions, backtest_signal, backtest_signal_per_asset,
)


def _signal_and_returns(n=40, freq="D"):
    dates = pd.date_range("2020-01-01", periods=n, freq=freq)
    signal = pd.DataFrame({"A": np.linspace(1, 2, n), "B": np.linspace(-1, -2, n)}, index=dates)
    returns = pd.DataFrame({"A": 0.01, "B": -0.01}, index=dates)
    return signal, returns


def test_monthly_positions_is_holding_period_positions_with_holding_1():
    signal, _ = _signal_and_returns(200)
    a = monthly_positions(signal)
    b = holding_period_positions(signal, holding_months=1)
    pd.testing.assert_frame_equal(a, b)


def test_holding_period_positions_blends_trailing_months():
    # Two months of distinct month-end signal values - holding_months=2 should
    # average them, not just carry forward the latest.
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    signal = pd.DataFrame({"A": 1.0}, index=dates)
    signal.loc[dates >= "2020-02-01", "A"] = 3.0

    blended = holding_period_positions(signal, holding_months=2)
    # By March, the rolling-2 mean of Jan (1.0) and Feb (3.0) month-end values is 2.0.
    march_value = blended.loc[blended.index >= "2020-03-01", "A"].iloc[0]
    assert np.isclose(march_value, 2.0)


def test_weekly_positions_only_changes_on_week_boundaries():
    signal, _ = _signal_and_returns(30)
    weekly = weekly_positions(signal)
    # Within a single week, the forward-filled value must be constant.
    week = weekly.loc["2020-01-01":"2020-01-03"]
    assert week["A"].nunique() == 1


def test_normalized_positions_sums_to_unit_gross_exposure():
    signal, _ = _signal_and_returns(60)
    positions = normalized_positions(signal, frequency="daily")
    gross = positions.abs().sum(axis=1)
    nonzero = gross[gross > 0]
    assert np.allclose(nonzero, 1.0)


def test_normalized_positions_is_shifted_no_lookahead():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    signal = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    positions = normalized_positions(signal, frequency="daily")
    # Day 0 has nothing to shift into -> flat (fillna(0)), not today's own signal.
    assert positions["A"].iloc[0] == 0.0
    # Day 1's position reflects day 0's signal (1.0), normalized to unit gross (1.0).
    assert np.isclose(positions["A"].iloc[1], 1.0)


def test_backtest_signal_matches_hand_computed_return():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    signal = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0, -1.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.0, 0.02, 0.02, 0.02], "B": [0.0, -0.01, -0.01, -0.01]}, index=dates)

    result = backtest_signal(signal, returns, frequency="daily")
    # positions shift(1) -> flat (0) on day0. day1 position: |A|=|B|=1 normalized
    # to 0.5/-0.5 each. day1 return = 0.5*0.02 + (-0.5)*(-0.01) = 0.01 + 0.005 = 0.015
    assert np.isclose(result.iloc[0], 0.0)
    assert np.isclose(result.iloc[1], 0.015)


def test_backtest_signal_with_cost_bps_reduces_return():
    signal, returns = _signal_and_returns(40)
    cost_bps = pd.Series({"A": 5.0, "B": 5.0})
    gross = backtest_signal(signal, returns, frequency="daily")
    net = backtest_signal(signal, returns, frequency="daily", cost_bps=cost_bps)
    common = gross.index.intersection(net.index)
    assert (net.loc[common] <= gross.loc[common] + 1e-12).all()
    assert (net.loc[common] < gross.loc[common]).any()


def test_backtest_signal_per_asset_not_normalized_across_book():
    signal, returns = _signal_and_returns(10)
    per_asset = backtest_signal_per_asset(signal, returns, frequency="daily")
    # Unlike backtest_signal's pooled book, per-asset positions are the raw
    # (shifted, but NOT gross-exposure-normalized) signal times its own return.
    raw_shifted = signal.shift(1)
    expected = raw_shifted * returns
    pd.testing.assert_frame_equal(per_asset, expected)


def test_backtest_signal_per_asset_with_cost_bps():
    signal, returns = _signal_and_returns(20)
    cost_bps = pd.Series({"A": 10.0, "B": 10.0})
    gross = backtest_signal_per_asset(signal, returns, frequency="daily")
    net = backtest_signal_per_asset(signal, returns, frequency="daily", cost_bps=cost_bps)
    # Cost only bites on days where the position actually changed (turnover > 0).
    diff = (gross - net).dropna()
    assert (diff >= -1e-12).all().all()


def test_invalid_frequency_raises():
    signal, _ = _signal_and_returns(5)
    try:
        normalized_positions(signal, frequency="yearly")
        assert False, "expected ValueError"
    except ValueError:
        pass
