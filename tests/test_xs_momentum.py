import inspect

import numpy as np
import pandas as pd

from signals.xs_momentum import mom2_12, xs_momentum_signal


def _monthly_close(prices, start="2020-01-31"):
    dates = pd.date_range(start, periods=len(prices), freq="ME")
    return pd.DataFrame({"A": prices}, index=dates)


def test_mom2_12_skip_month_and_lookback_matches_manual_calc():
    # 14 month-end prices, one row per month so resample("ME").last() is
    # identity - isolates the skip-month + lookback arithmetic from any
    # resampling ambiguity.
    prices = [100 + i for i in range(14)]  # 100..113
    close = _monthly_close(prices)

    result = mom2_12(close, lookback_months=12, skip_months=1)

    # raw[t] = price[t - skip] / price[t - skip - lookback] - 1
    # at t = last index (13): price[12]=112, price[0]=100 -> 112/100 - 1 = 0.12
    expected_last = prices[12] / prices[0] - 1.0
    assert np.isclose(result["A"].iloc[-1], expected_last)
    assert np.isclose(expected_last, 0.12)


def test_mom2_12_is_nan_before_enough_history():
    prices = [100 + i for i in range(15)]
    close = _monthly_close(prices)

    result = mom2_12(close, lookback_months=12, skip_months=1)

    # base = monthly.shift(skip=1) needs 1 month of history; base.shift(lookback=12)
    # needs 12 more on top of that, so the earliest month with a defined
    # base[t] AND base[t-12] is index 13 (0-indexed) - indices 0..12 (13
    # months) are NaN.
    monthly_result = result.resample("ME").last()
    assert monthly_result["A"].iloc[:13].isna().all()
    assert not monthly_result["A"].iloc[13:].isna().any()


def test_mom2_12_skips_most_recent_month_not_included_in_return():
    # A price spike only in the very last month should NOT move the momentum
    # value at that same date, since skip_months=1 excludes it from the window.
    prices = [100.0] * 13 + [1000.0]  # flat for 13 months, spike in month 14
    close = _monthly_close(prices)

    result = mom2_12(close, lookback_months=12, skip_months=1)

    # base = monthly.shift(1), so at the last date base looks at month 12
    # (100.0), not the month-14 spike (1000.0) - momentum should be ~0.
    assert np.isclose(result["A"].iloc[-1], 0.0)


def test_mom2_12_reindexed_and_ffilled_onto_daily_index():
    prices = [100 + i for i in range(14)]
    monthly_close = _monthly_close(prices)
    daily_index = pd.date_range(monthly_close.index[0], monthly_close.index[-1], freq="D")
    daily_close = monthly_close.reindex(daily_index).ffill()

    result = mom2_12(daily_close, lookback_months=12, skip_months=1)

    assert result.index.equals(daily_index)
    # Value should be constant (ffilled) across the days within a given month.
    last_month_mask = result.index >= monthly_close.index[-1].replace(day=1)
    assert result["A"][last_month_mask].nunique(dropna=True) == 1


def _two_sector_close():
    dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    # Sector1: A trends up strongly, B trends up mildly -> A should rank above B.
    # Sector2: C trends down mildly, D trends down strongly -> C should rank above D.
    a = [100 * (1.05 ** i) for i in range(15)]
    b = [100 * (1.01 ** i) for i in range(15)]
    c = [100 * (0.99 ** i) for i in range(15)]
    d = [100 * (0.95 ** i) for i in range(15)]
    close = pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=dates)
    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}
    return close, sectors


def test_xs_momentum_signal_ranks_within_sector_not_across():
    close, sectors = _two_sector_close()

    signal = xs_momentum_signal(close, sectors)
    last = signal.iloc[-1]

    assert last["A"] > last["B"]
    assert last["C"] > last["D"]
    # Sector-scoped rank-demean with 2 members per sector always produces
    # +/-0.5 regardless of magnitude - cross-sector momentum spread (A vs C)
    # must not leak into the within-sector ranks.
    assert np.isclose(abs(last["A"]), 0.5)
    assert np.isclose(abs(last["B"]), 0.5)
    assert np.isclose(abs(last["C"]), 0.5)
    assert np.isclose(abs(last["D"]), 0.5)


def test_xs_momentum_signal_does_not_compare_across_sectors_by_magnitude():
    close, sectors = _two_sector_close()

    signal = xs_momentum_signal(close, sectors)
    last = signal.iloc[-1]

    # A's momentum is far larger in magnitude than C's, but both are the
    # top-ranked member of their own sector - a magnitude-aware (non-rank)
    # construction would separate them; a sector-scoped rank must not.
    assert np.isclose(last["A"], last["C"])


def test_xs_momentum_signal_has_no_vol_scaling_parameter():
    # The paper's headline XSMOM construction has no vol-scaling step at all
    # (unlike momentum/breakout/crossover's vol_targeted_sign_signal)  -
    # confirm the public signature never grew a vol/target_vol argument.
    params = inspect.signature(xs_momentum_signal).parameters
    assert "vol" not in params
    assert "target_vol" not in params
    assert set(params) == {"close", "sectors"}


def test_xs_momentum_signal_output_bounded_by_group_size_regardless_of_return_magnitude():
    # If vol-scaling were silently applied, blowing up one asset's return
    # magnitude would blow up its signal value too. Since this is a pure
    # rank-demean, an extreme magnitude change that preserves rank order
    # must leave the signal completely unchanged.
    close, sectors = _two_sector_close()
    signal_base = xs_momentum_signal(close, sectors)

    close_extreme = close.copy()
    close_extreme["A"] = close_extreme["A"] ** 2  # preserves ordering, inflates magnitude
    signal_extreme = xs_momentum_signal(close_extreme, sectors)

    pd.testing.assert_frame_equal(signal_base, signal_extreme)
