import inspect

import numpy as np
import pandas as pd

from signals.value import (
    negative_5yr_return_value,
    bond_yield_change_value,
    fx_ppp_value_feature,
    build_value_panel,
    value_signal,
    BOND_YIELD_MATURITY_MAP,
    FX_CPI_COUNTRY_MAP,
)


def _monthly_frame(columns_to_values: dict, start="2020-01-31", periods=None):
    n = periods or len(next(iter(columns_to_values.values())))
    dates = pd.date_range(start, periods=n, freq="ME")
    return pd.DataFrame(columns_to_values, index=dates)


# ---------------------------------------------------------------------------
# negative_5yr_return_value
# ---------------------------------------------------------------------------

def test_negative_5yr_return_value_matches_manual_calc():
    n = 70
    prices = [100 + i for i in range(n)]  # linear ramp
    close = _monthly_frame({"Gold": prices}, periods=n)

    result = negative_5yr_return_value(close, avg_end_months=54, avg_window_months=12)

    # avg window = monthly.shift(54).rolling(12).mean() -> at the last index (69),
    # this averages monthly[69-65 .. 69-54] = monthly[4..15] (12 points).
    window_prices = prices[4:16]
    expected_avg = sum(window_prices) / len(window_prices)
    expected = np.log(expected_avg / prices[-1])

    assert np.isclose(result["Gold"].iloc[-1], expected)


def test_negative_5yr_return_value_is_nan_before_enough_history():
    n = 70
    prices = [100 + i for i in range(n)]
    close = _monthly_frame({"Gold": prices}, periods=n)

    result = negative_5yr_return_value(close, avg_end_months=54, avg_window_months=12)
    monthly_result = result.resample("ME").last()

    # First valid index needs shift(54).rolling(12) to have a full window ->
    # index 54 + 12 - 1 = 65 is the earliest valid observation.
    assert monthly_result["Gold"].iloc[:65].isna().all()
    assert not monthly_result["Gold"].iloc[65:].isna().any()


def test_negative_5yr_return_value_is_scale_invariant():
    # log(avg_price / price_now) - multiplying the whole price path by a
    # constant factor must leave the value unchanged (the constant cancels in
    # the ratio). A real dependence on price *level* (rather than shape) would
    # break this.
    n = 70
    prices = [100 + i for i in range(n)]
    close = _monthly_frame({"Gold": prices}, periods=n)
    close_scaled = close * 37.5

    result = negative_5yr_return_value(close)
    result_scaled = negative_5yr_return_value(close_scaled)

    pd.testing.assert_frame_equal(result, result_scaled)


def test_negative_5yr_return_value_higher_when_price_fell():
    n = 70
    # Asset A: price roughly flat -> value near 0.
    # Asset B: price rose a lot recently -> avg_price << price_now -> value very negative (expensive).
    flat = [100.0] * n
    risen = [100.0] * 60 + [100.0 * 1.10 ** i for i in range(1, n - 60 + 1)]
    close = _monthly_frame({"A": flat, "B": risen}, periods=n)

    result = negative_5yr_return_value(close, avg_end_months=54, avg_window_months=12)

    assert result["A"].iloc[-1] > result["B"].iloc[-1]


# ---------------------------------------------------------------------------
# bond_yield_change_value
# ---------------------------------------------------------------------------

def test_bond_yield_change_value_matches_manual_calc_and_maturity_mapping():
    n = 65
    two_yr = [1.0 + 0.01 * i for i in range(n)]
    thirty_yr = [3.0 + 0.02 * i for i in range(n)]
    yield_curve = _monthly_frame({"2Y": two_yr, "30Y": thirty_yr}, periods=n)

    result = bond_yield_change_value(
        yield_curve,
        maturity_map={"US_2Y": "2Y", "US_30Y": "30Y", "UltraBond": "30Y"},
        lookback_months=60,
    )

    expected_2y = two_yr[-1] - two_yr[-1 - 60]
    assert np.isclose(result["US_2Y"].iloc[-1], expected_2y)

    # UltraBond and US_30Y both map to the same "30Y" column -> identical series.
    pd.testing.assert_series_equal(result["UltraBond"], result["US_30Y"], check_names=False)


def test_bond_yield_change_value_nan_before_lookback():
    n = 65
    two_yr = [1.0 + 0.01 * i for i in range(n)]
    yield_curve = _monthly_frame({"2Y": two_yr}, periods=n)

    result = bond_yield_change_value(yield_curve, maturity_map={"US_2Y": "2Y"}, lookback_months=60)

    assert result["US_2Y"].iloc[:60].isna().all()
    assert not np.isnan(result["US_2Y"].iloc[60])


def test_bond_yield_change_value_reindexes_to_daily_when_given_daily_index():
    n = 65
    two_yr = [1.0 + 0.01 * i for i in range(n)]
    yield_curve = _monthly_frame({"2Y": two_yr}, periods=n)
    daily_index = pd.date_range(yield_curve.index[0], yield_curve.index[-1], freq="D")

    result = bond_yield_change_value(
        yield_curve, maturity_map={"US_2Y": "2Y"}, lookback_months=60, daily_index=daily_index
    )

    assert result.index.equals(daily_index)


# ---------------------------------------------------------------------------
# fx_ppp_value_feature
# ---------------------------------------------------------------------------

def test_fx_ppp_value_feature_subtracts_inflation_differential():
    # Flat FX price (nominal component == 0) isolates the CPI adjustment: a
    # country with HIGHER inflation than the US should show up as a LOWER
    # (more negative) PPP value than a country with equal inflation.
    n = 12
    flat_price = [1.0] * n
    close = _monthly_frame({"EURUSD": flat_price, "GBPUSD": flat_price}, periods=n)

    us_cpi = [100.0] * n
    eur_cpi = [100.0] * n  # no inflation vs. US
    gbp_cpi = [100.0 * 1.05 ** i for i in range(n)]  # inflating faster than US
    cpi = _monthly_frame({"US": us_cpi, "EUR": eur_cpi, "GBP": gbp_cpi}, periods=n)

    result = fx_ppp_value_feature(
        close, cpi,
        country_map={"EURUSD": "EUR", "GBPUSD": "GBP"},
        avg_end_months=2, avg_window_months=2, cpi_lookback_months=3,
    )

    last = result.iloc[-1]
    assert last["EURUSD"] > last["GBPUSD"]
    assert np.isclose(last["EURUSD"], 0.0)


def test_fx_ppp_value_feature_reindex_ffill_is_bounded():
    # SwissFranc's CPI (CHF) is available Jan-Aug 2020, then goes missing for
    # good (mirrors GBP/CAD/CHF/MXN's real documented FRED lag). The daily
    # reindex must bridge a short gap but must NOT replay the last known PPP
    # value indefinitely once the source has been gone a long time.
    daily_index = pd.date_range("2020-01-01", periods=365, freq="D")
    monthly_dates = pd.date_range("2020-01-31", periods=12, freq="ME")

    flat_daily = pd.Series(1.0, index=daily_index)
    close = pd.DataFrame({"SwissFranc": flat_daily})

    us_cpi = [100.0] * 12
    chf_cpi = [100.0] * 8 + [np.nan] * 4  # goes missing from Sep 2020 onward
    cpi = pd.DataFrame({"US": us_cpi, "CHF": chf_cpi}, index=monthly_dates)

    result = fx_ppp_value_feature(
        close, cpi,
        country_map={"SwissFranc": "CHF"},
        avg_end_months=2, avg_window_months=2, cpi_lookback_months=3,
        reindex_ffill_limit=35,
    )

    # Last real monthly value is at 2020-08-31 (value 0.0, flat price/flat CPI).
    assert np.isclose(result["SwissFranc"].loc["2020-08-31"], 0.0)
    # 15 days later (well within the 35-day bounded ffill) - still filled.
    assert np.isclose(result["SwissFranc"].loc["2020-09-15"], 0.0)
    # ~40 days later - beyond the bounded ffill limit, must revert to NaN,
    # not silently keep replaying the last known PPP value.
    assert pd.isna(result["SwissFranc"].loc["2020-10-10"])


def test_fx_ppp_value_feature_signature_has_bounded_reindex_param():
    params = inspect.signature(fx_ppp_value_feature).parameters
    assert "reindex_ffill_limit" in params
    assert params["reindex_ffill_limit"].default is not None


# ---------------------------------------------------------------------------
# build_value_panel
# ---------------------------------------------------------------------------

def test_build_value_panel_overwrites_bond_and_fx_columns():
    n = 70
    default_prices = [100 + i for i in range(n)]
    close = _monthly_frame(
        {"Gold": default_prices, "US_2Y": default_prices, "EURUSD": default_prices}, periods=n
    )

    yield_dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    two_yr = [1.0 + 0.01 * i for i in range(n)]
    yield_curve = pd.DataFrame({"2Y": two_yr}, index=yield_dates)

    cpi_dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    # EUR inflating faster than the US so the PPP adjustment is non-zero -
    # otherwise fx_ppp_value_feature degenerates to the same formula as the
    # plain default and this test can't distinguish "overwritten" from
    # "coincidentally identical."
    cpi = pd.DataFrame({"US": [100.0] * n, "EUR": [100.0 * 1.01 ** i for i in range(n)]}, index=cpi_dates)

    panel = build_value_panel(
        close, yield_curve, cpi,
        bond_map={"US_2Y": "2Y"},
        fx_map={"EURUSD": "EUR"},
    )

    default_only = negative_5yr_return_value(close)
    bond_only = bond_yield_change_value(yield_curve, {"US_2Y": "2Y"}, daily_index=close.index)
    fx_only = fx_ppp_value_feature(close, cpi, {"EURUSD": "EUR"})

    # Gold was never overwritten - matches the plain default construction.
    pd.testing.assert_series_equal(panel["Gold"], default_only["Gold"], check_names=False)
    # US_2Y and EURUSD were overwritten - do NOT match the generic default construction.
    assert not panel["US_2Y"].equals(default_only["US_2Y"])
    assert not panel["EURUSD"].equals(default_only["EURUSD"])
    # ...and DO match their special-case constructions.
    pd.testing.assert_series_equal(panel["US_2Y"], bond_only["US_2Y"], check_names=False)
    pd.testing.assert_series_equal(panel["EURUSD"], fx_only["EURUSD"], check_names=False)


# ---------------------------------------------------------------------------
# value_signal
# ---------------------------------------------------------------------------

def test_value_signal_ranks_within_sector_not_across():
    n = 70
    flat = [100.0] * n
    fell = [100.0] * 60 + [100.0 * 0.90 ** i for i in range(1, n - 60 + 1)]  # cheap now
    rose = [100.0] * 60 + [100.0 * 1.10 ** i for i in range(1, n - 60 + 1)]  # expensive now
    close = _monthly_frame({"A": fell, "B": flat, "C": rose, "D": flat}, periods=n)

    yield_dates = close.index
    yield_curve = pd.DataFrame({"2Y": [1.0] * n}, index=yield_dates)
    cpi = pd.DataFrame({"US": [100.0] * n}, index=yield_dates)

    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}

    signal = value_signal(close, yield_curve, cpi, sectors, bond_map={}, fx_map={})
    last = signal.iloc[-1]

    assert last["A"] > last["B"]  # A fell -> cheap -> higher value rank
    assert last["D"] > last["C"]  # C rose -> expensive -> lower value rank than flat D


def test_value_signal_has_no_vol_scaling_parameter():
    params = inspect.signature(value_signal).parameters
    assert "vol" not in params
    assert "target_vol" not in params


def test_bond_and_fx_maps_cover_the_documented_assets():
    # Regression guard on the exact lookup tables the reconstruction plan called for.
    assert BOND_YIELD_MATURITY_MAP == {
        "US_2Y": "2Y",
        "US_5Y": "5Y",
        "US_10Y": "10Y",
        "US_30Y": "30Y",
        "UltraBond": "30Y",
    }
    assert FX_CPI_COUNTRY_MAP == {
        "EURUSD": "EUR",
        "JPYUSD": "JPY",
        "GBPUSD": "GBP",
        "AUDUSD": "AUD",
        "CADUSD": "CAD",
        "SwissFranc": "CHF",
        "MexicanPeso": "MXN",
    }
