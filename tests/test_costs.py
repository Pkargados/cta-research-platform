import numpy as np
import pandas as pd

from backtest.costs import liquidity_tiered_cost_bps, transaction_cost_drag, turnover
from backtest.engine import backtest_signal, backtest_signal_per_asset


def test_turnover_is_zero_when_position_unchanged():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    positions = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)

    result = turnover(positions)

    assert result["A"].iloc[1:].eq(0).all()


def test_turnover_captures_position_change_magnitude():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    positions = pd.DataFrame({"A": [0.0, 1.0, -0.5]}, index=dates)

    result = turnover(positions)

    assert result["A"].iloc[1] == 1.0
    assert result["A"].iloc[2] == 1.5


def test_transaction_cost_drag_scales_with_cost_bps():
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    positions = pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, 1.0]}, index=dates)
    cost_bps = pd.Series({"A": 10.0, "B": 20.0})

    drag = transaction_cost_drag(positions, cost_bps)

    # Day 1 turnover = 1.0 for each asset; drag = 1.0*10bp + 1.0*20bp = 30bp = 0.003
    assert np.isclose(drag.iloc[1], 0.003)
    assert drag.iloc[0] == 0.0


def test_backtest_signal_cost_bps_none_matches_prior_gross_behavior():
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    signal = pd.DataFrame({"A": 1.0, "B": -1.0}, index=dates)
    returns = pd.DataFrame({"A": 0.01, "B": 0.01}, index=dates)

    with_default = backtest_signal(signal, returns, frequency="monthly")
    explicit_none = backtest_signal(signal, returns, frequency="monthly", cost_bps=None)

    pd.testing.assert_series_equal(with_default, explicit_none)


def test_backtest_signal_net_of_cost_is_lower_than_gross():
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    signal = pd.DataFrame({"A": 1.0, "B": -1.0}, index=dates)
    returns = pd.DataFrame({"A": 0.01, "B": 0.01}, index=dates)
    cost_bps = pd.Series({"A": 5.0, "B": 5.0})

    gross = backtest_signal(signal, returns, frequency="monthly")
    net = backtest_signal(signal, returns, frequency="monthly", cost_bps=cost_bps)

    common = gross.index.intersection(net.index)
    assert (net.loc[common] <= gross.loc[common]).all()
    assert (net.loc[common] < gross.loc[common]).any()


def test_backtest_signal_per_asset_nets_cost_independently_per_column():
    # Month-end values alternate sign so turnover (and therefore cost drag) recurs
    # every period, not just once during warmup - a constant signal only trades once.
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    signal = pd.DataFrame({"A": values, "B": values}, index=dates)
    daily = pd.date_range("2020-01-01", periods=200, freq="D")
    signal = signal.reindex(daily.union(dates)).ffill().reindex(daily)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=daily)
    cost_bps = pd.Series({"A": 5.0, "B": 50.0})

    net = backtest_signal_per_asset(signal, returns, frequency="monthly", cost_bps=cost_bps)

    valid = net.dropna()
    assert not valid.empty
    # Same gross signal/returns (zero, here, to isolate cost) - B's higher assumed
    # cost should drag its net return below A's on any day with real turnover.
    assert (valid["B"] < valid["A"]).any()


def test_liquidity_tiered_cost_bps_ranks_low_adv_as_most_expensive():
    dates = pd.date_range("2024-07-14", periods=30, freq="D")
    volume = pd.DataFrame({
        "Liquid": 100_000.0,
        "Thin": 500.0,
    }, index=dates)

    cost_bps = liquidity_tiered_cost_bps(volume, window_start="2024-07-14", tier_bps=(1.0, 10.0))

    assert cost_bps["Thin"] > cost_bps["Liquid"]
