import numpy as np
import pandas as pd

from data.continuous_curve import (
    build_contract_chain, assign_front_contract, build_raw_series, back_adjust,
)


def _asset_df(rows):
    """rows: list of dicts with date, contract_symbol, expiry_year, expiry_code,
    open, high, low, close, volume."""
    return pd.DataFrame(rows)


def test_build_contract_chain_sorted_chronologically_by_real_expiry():
    df = _asset_df([
        {"contract_symbol": "C_G20", "expiry_year": 2020, "expiry_code": "G", "date": "2020-01-01",
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"contract_symbol": "C_F20", "expiry_year": 2020, "expiry_code": "F", "date": "2020-01-01",
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"contract_symbol": "C_F21", "expiry_year": 2021, "expiry_code": "F", "date": "2020-01-01",
         "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ])
    chain = build_contract_chain(df)
    # F (month 1) 2020 < G (month 2) 2020 < F (month 1) 2021
    assert chain == ["C_F20", "C_G20", "C_F21"]


def _rolling_dataset():
    dates = pd.bdate_range("2020-01-06", periods=10)  # Mon-Fri x2
    c1_volume = [100, 100, 100, 80, 60, 40, 20, 10, 5, 5]
    c2_volume = [np.nan, np.nan, 50, 60, 90, 120, 150, 180, 190, 195]
    c1_close = [100.0] * 10
    c2_close = [110.0, 111.0, 112.0, 113.0, 114.0, 110.0, 115.0, 120.0, 125.0, 130.0]

    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "contract_symbol": "C1", "expiry_year": 2020, "expiry_code": "F",
                     "open": c1_close[i], "high": c1_close[i], "low": c1_close[i], "close": c1_close[i],
                     "volume": c1_volume[i]})
        if not np.isnan(c2_volume[i]):
            rows.append({"date": d, "contract_symbol": "C2", "expiry_year": 2020, "expiry_code": "G",
                         "open": c2_close[i], "high": c2_close[i], "low": c2_close[i], "close": c2_close[i],
                         "volume": c2_volume[i]})
    return _asset_df(rows), dates


def test_assign_front_contract_rolls_after_confirmation_days_of_volume_crossover():
    df, dates = _rolling_dataset()
    chain = build_contract_chain(df)
    front = assign_front_contract(df, chain, confirmation_days=2, backstop_days=1)

    # C2's volume first exceeds C1's on day index 4, and again on day index 5 -
    # two CONSECUTIVE days confirms the roll, which takes effect on day index 5
    # (the day confirmation completes), not day 4 or day 6.
    expected = ["C1"] * 5 + ["C2"] * 5
    assert front.tolist() == expected


def test_assign_front_contract_never_selects_a_contract_with_no_row_that_day():
    df, dates = _rolling_dataset()
    chain = build_contract_chain(df)
    front = assign_front_contract(df, chain, confirmation_days=2, backstop_days=1)

    # On the first two days, C2 has no row at all - front must never point to it.
    assert front.iloc[0] == "C1"
    assert front.iloc[1] == "C1"


def test_assign_front_contract_skips_dead_intermediate_contracts():
    # C_mid has NO rows at all (a thin serial-month contract that never trades) -
    # the forward scan for "next contract" must skip it and reach C_far instead
    # of getting permanently stuck.
    dates = pd.bdate_range("2020-01-06", periods=8)
    near_volume = [100, 100, 90, 80, 20, 10, 5, 5]
    far_volume = [np.nan, np.nan, 40, 90, 150, 180, 190, 195]

    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "contract_symbol": "C_near", "expiry_year": 2020, "expiry_code": "F",
                     "open": 100, "high": 100, "low": 100, "close": 100, "volume": near_volume[i]})
        if not np.isnan(far_volume[i]):
            rows.append({"date": d, "contract_symbol": "C_far", "expiry_year": 2020, "expiry_code": "N",
                         "open": 120, "high": 120, "low": 120, "close": 120, "volume": far_volume[i]})
    df = _asset_df(rows)

    # C_mid is a real chain entry (expiry between near and far) but contributes
    # NO rows to df at all.
    chain = ["C_near", "C_mid", "C_far"]

    front = assign_front_contract(df, chain, confirmation_days=2, backstop_days=1)
    assert front.iloc[-1] == "C_far"  # rolled all the way through to C_far, not stuck


def test_back_adjust_ratio_matches_manual_calc():
    df, dates = _rolling_dataset()
    chain = build_contract_chain(df)
    front = assign_front_contract(df, chain, confirmation_days=2, backstop_days=1)
    raw = build_raw_series(df, front)
    adjusted = back_adjust(df, raw)

    # Single roll at dates[5] (front C1->C2): ratio = C2_close(day5)/C1_close(day5) = 110/100 = 1.10
    assert adjusted["is_roll_date"].loc[dates[5]]
    assert not adjusted["is_roll_date"].loc[dates[4]]

    # Most recent segment (days 5-9, front=C2) is UNADJUSTED - equals raw close exactly.
    for i in range(5, 10):
        assert np.isclose(adjusted["adj_close"].loc[dates[i]], adjusted["raw_close"].loc[dates[i]])

    # Earlier segment (days 0-4, front=C1) carries the single roll ratio (1.10).
    for i in range(0, 5):
        assert np.isclose(adjusted["adj_close"].loc[dates[i]], 100.0 * 1.10)


def test_back_adjust_preserves_ohlc_ordering():
    dates = pd.bdate_range("2020-01-06", periods=6)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "contract_symbol": "C1", "expiry_year": 2020, "expiry_code": "F",
                     "open": 100 + i, "high": 105 + i, "low": 95 + i, "close": 102 + i, "volume": 100})
    df = _asset_df(rows)
    chain = build_contract_chain(df)
    front = assign_front_contract(df, chain, confirmation_days=2, backstop_days=1)
    raw = build_raw_series(df, front)
    adjusted = back_adjust(df, raw)

    assert (adjusted["adj_low"] <= adjusted["adj_open"]).all()
    assert (adjusted["adj_open"] <= adjusted["adj_high"]).all()
    assert (adjusted["adj_low"] <= adjusted["adj_close"]).all()
    assert (adjusted["adj_close"] <= adjusted["adj_high"]).all()
