import numpy as np
import pandas as pd

from signals.seasonality import (
    SEASONALITY_HALF_MONTH_ASSETS,
    TSMOM_SEASONAL_WINDOWS,
    TSMOM_SEASONAL_AMPLITUDE,
    half_month_direction,
    half_month_signal,
    same_month_average_return,
    same_month_signal,
    build_all_seasonality_signals,
    seasonal_weight_multiplier,
    tsmom_seasonal_signal,
)


def test_half_month_direction_boundary_at_day_15_16():
    index = pd.to_datetime(["2020-01-01", "2020-01-15", "2020-01-16", "2020-01-31"])
    result = half_month_direction(index)
    assert list(result.values) == [1.0, 1.0, -1.0, -1.0]


def test_half_month_direction_handles_february_and_leap_year_month_ends():
    index = pd.to_datetime(["2020-02-28", "2020-02-29", "2021-02-28"])
    result = half_month_direction(index)
    # All well past day 15 -> second half regardless of month length.
    assert (result.values == -1.0).all()


def test_half_month_signal_scoped_to_named_assets_only():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame(100.0, index=dates, columns=["Corn", "SP500"])
    vol = pd.DataFrame(0.2, index=dates, columns=["Corn", "SP500"])

    signal = half_month_signal(close, vol)

    assert "Corn" in signal.columns
    assert "SP500" not in signal.columns  # not in SEASONALITY_HALF_MONTH_ASSETS


def test_half_month_signal_drops_assets_not_present_in_close():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.DataFrame(100.0, index=dates, columns=["Corn"])  # missing the other 6
    vol = pd.DataFrame(0.2, index=dates, columns=["Corn"])

    signal = half_month_signal(close, vol)

    assert list(signal.columns) == ["Corn"]


def test_half_month_signal_is_vol_targeted():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame(100.0, index=dates, columns=["Corn"])
    vol = pd.DataFrame(0.25, index=dates, columns=["Corn"])

    signal = half_month_signal(close, vol, assets=["Corn"], target_vol=0.40)

    # target_vol * sign(direction) / vol; all 5 January days are first-half (+1).
    expected = 0.40 * 1.0 / 0.25
    assert np.allclose(signal["Corn"].to_numpy(), expected)


def test_half_month_signal_nan_where_close_is_nan():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.DataFrame({"Corn": [100.0, np.nan, 101.0, 102.0, np.nan]}, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=["Corn"])

    signal = half_month_signal(close, vol, assets=["Corn"])

    assert signal["Corn"].isna().tolist() == [False, True, False, False, True]


def test_default_half_month_assets_list_unchanged():
    # Regression guard: this exact 7-name scope is a WORKFLOW.md Phase 11b
    # decision (the 9 commodities Li et al. found significant, minus Soy Oil/
    # Soymeal which aren't pulled), not something to silently drift.
    assert SEASONALITY_HALF_MONTH_ASSETS == [
        "Corn", "KC_Wheat", "Wheat", "FeederCattle", "LeanHogs", "Silver", "Soybeans",
    ]


def test_same_month_average_return_uses_only_prior_same_month_occurrences():
    # January return grows 0.01, 0.02, 0.03, ... every year; all other months flat.
    n_years = 5
    returns = []
    for year in range(n_years):
        returns.append(0.01 * (year + 1))  # January
        returns.extend([0.0] * 11)  # Feb..Dec
    dates = pd.date_range("2015-01-31", periods=len(returns), freq="ME")
    monthly_returns = pd.DataFrame({"A": returns}, index=dates)

    result = same_month_average_return(monthly_returns, min_years=2, max_years=20)

    # January of year index 3 (0-indexed, the 4th January = 2018-01-31): prior
    # Januaries are years 0,1,2 -> returns 0.01, 0.02, 0.03 -> mean 0.02.
    jan_2018 = pd.Timestamp("2018-01-31")
    assert np.isclose(result.loc[jan_2018, "A"], np.mean([0.01, 0.02, 0.03]))

    # The very first January (no prior occurrence) and second January (only
    # 1 prior, below min_years=2) must be NaN.
    assert pd.isna(result.loc[pd.Timestamp("2015-01-31"), "A"])
    assert pd.isna(result.loc[pd.Timestamp("2016-01-31"), "A"])

    # Non-January months are never touched by the January-only rolling calc -
    # every other month should be NaN throughout (no cross-month leakage) or 0
    # (their own same-month history is a constant 0, once enough years exist).
    feb_2018 = pd.Timestamp("2018-02-28")
    assert np.isclose(result.loc[feb_2018, "A"], 0.0)


def test_same_month_average_return_respects_max_years_window():
    # 25 years of Januaries: 0.01, 0.02, ..., 0.25. With max_years=20, the
    # trailing window at the last January should only average the most
    # recent 20 PRIOR Januaries, not all 24.
    returns = []
    for year in range(25):
        returns.append(0.01 * (year + 1))
        returns.extend([0.0] * 11)
    dates = pd.date_range("1995-01-31", periods=len(returns), freq="ME")
    monthly_returns = pd.DataFrame({"A": returns}, index=dates)

    result = same_month_average_return(monthly_returns, min_years=2, max_years=20)

    last_jan = dates[dates.month == 1][-1]
    # Prior Januaries available: years 0..23 (returns 0.01..0.24), most
    # recent 20 of those are years 4..23 (0.05..0.24).
    expected = np.mean([0.01 * (y + 1) for y in range(4, 24)])
    assert np.isclose(result.loc[last_jan, "A"], expected)


def _dec_jan_diverging_close(n_years=6):
    dates = pd.date_range("2015-01-31", periods=n_years * 12, freq="ME")

    def a_ret(d):
        if d.month == 1:
            return 0.10
        if d.month == 12:
            return -0.10
        return 0.0

    def b_ret(d):
        if d.month == 1:
            return -0.10
        if d.month == 12:
            return 0.10
        return 0.0

    a_prices = 100 * np.cumprod([1.0 + a_ret(d) for d in dates])
    b_prices = 100 * np.cumprod([1.0 + b_ret(d) for d in dates])
    close = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)
    sectors = {"S": ["A", "B"]}
    return close, sectors


def test_same_month_signal_decembers_row_reflects_januarys_history_not_decembers():
    # Regression test for a real bug found and fixed while building this
    # signal: the raw same-month score for January is computed using only
    # data through the prior December, but backtest.engine's universal
    # "form at month-end t, trade month t+1" convention means that score
    # must be LABELED at December's row (not January's) to actually trade
    # during January. Asset A has a strong January history but a weak
    # December history (and vice-versa for B) - if December's row picked up
    # A's/B's DECEMBER history instead of their JANUARY history, the ranking
    # here would flip.
    close, sectors = _dec_jan_diverging_close()

    signal = same_month_signal(close, sectors, min_years=2, max_years=20)

    dec_date = pd.Timestamp("2019-12-31")
    assert signal.loc[dec_date, "A"] > signal.loc[dec_date, "B"]


def test_same_month_signal_is_sector_scoped_rank_not_full_universe():
    close, sectors = _dec_jan_diverging_close()
    # Add a second sector that should not influence Sector "S"'s ranking.
    close["C"] = 100.0
    close["D"] = 200.0
    sectors = dict(sectors, S2=["C", "D"])

    signal = same_month_signal(close, sectors, min_years=2, max_years=20)

    dec_date = pd.Timestamp("2019-12-31")
    # A 2-member sector rank-demean always produces +/-0.5 regardless of
    # magnitude - confirms C/D's presence elsewhere didn't leak in.
    assert np.isclose(abs(signal.loc[dec_date, "A"]), 0.5)


def test_build_all_seasonality_signals_returns_both_specs_no_headline_pick():
    dates = pd.date_range("2015-01-01", periods=400, freq="D")
    assets = SEASONALITY_HALF_MONTH_ASSETS + ["SP500"]
    close = pd.DataFrame(100.0, index=dates, columns=assets)
    vol = pd.DataFrame(0.2, index=dates, columns=assets)
    sectors = {
        "Grains": ["Corn", "Wheat", "KC_Wheat", "Soybeans"],
        "Livestock": ["FeederCattle", "LeanHogs"],
        "PreciousMetals": ["Silver"],
        "EquityIndex": ["SP500"],
    }

    result = build_all_seasonality_signals(close, vol, sectors)

    assert set(result.keys()) == {"half_month", "same_month"}
    assert set(result["half_month"].columns) == set(SEASONALITY_HALF_MONTH_ASSETS)


def test_seasonal_weight_multiplier_peaks_at_window_center():
    # Corn's window is Jun-Aug; its center is ~Jul 17.
    index = pd.to_datetime(["2020-07-17"])
    result = seasonal_weight_multiplier(index, assets=["Corn"])
    assert np.isclose(result.loc[index[0], "Corn"], 1.0 + TSMOM_SEASONAL_AMPLITUDE, atol=1e-2)


def test_seasonal_weight_multiplier_is_exactly_one_at_window_edges_and_outside():
    index = pd.to_datetime(["2020-06-01", "2020-08-31", "2020-09-01", "2020-01-01"])
    result = seasonal_weight_multiplier(index, assets=["Corn"])
    assert np.allclose(result["Corn"].to_numpy(), 1.0, atol=1e-2)


def test_seasonal_weight_multiplier_handles_year_end_wraparound():
    # Natural Gas's window is Nov-Mar (wraps); center is ~Jan 15-16.
    index = pd.to_datetime(["2020-01-15", "2020-07-01"])
    result = seasonal_weight_multiplier(index, assets=["Natural Gas"])
    assert result.loc[pd.Timestamp("2020-01-15"), "Natural Gas"] > 1.4
    assert np.isclose(result.loc[pd.Timestamp("2020-07-01"), "Natural Gas"], 1.0, atol=1e-2)


def test_seasonal_weight_multiplier_is_continuous_not_a_step_function():
    # Values just inside vs. just outside a window boundary should be close
    # to each other, not jump discontinuously (CLAUDE.md Rule 5 / WORKFLOW.md
    # Phase 11c's own "continuous, not binary" requirement).
    index = pd.date_range("2020-05-30", "2020-06-03", freq="D")
    result = seasonal_weight_multiplier(index, assets=["Corn"])
    diffs = result["Corn"].diff().dropna().abs()
    assert (diffs < 0.05).all()


def test_seasonal_weight_multiplier_neutral_for_assets_without_a_window():
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    result = seasonal_weight_multiplier(index, assets=["SP500"])
    assert (result["SP500"] == 1.0).all()


def test_tsmom_seasonal_signal_only_modifies_scoped_assets():
    dates = pd.date_range("2018-01-01", periods=700, freq="D")
    assets = ["Corn", "SP500"]
    rng = np.random.RandomState(0)
    prices = 100 * np.cumprod(1 + 0.001 * rng.standard_normal((len(dates), len(assets))), axis=0)
    close = pd.DataFrame(prices, index=dates, columns=assets)
    vol = pd.DataFrame(0.2, index=dates, columns=assets)

    base = tsmom_seasonal_signal(close, vol, windows={"Corn": (6, 8)})
    from signals.momentum import tsmom_signal as _plain_tsmom
    plain = _plain_tsmom(close, vol)

    # SP500 has no window - must be untouched.
    pd.testing.assert_series_equal(base["SP500"], plain["SP500"], check_names=False)
    # Corn must differ somewhere (the seasonal window actually does something).
    assert not base["Corn"].equals(plain["Corn"])


def test_tsmom_seasonal_signal_never_flips_sign_relative_to_plain_tsmom():
    dates = pd.date_range("2018-01-01", periods=700, freq="D")
    close = pd.DataFrame({"Corn": 100 * np.cumprod(1 + 0.002 * np.sin(np.linspace(0, 20, len(dates))))}, index=dates)
    vol = pd.DataFrame({"Corn": 0.2}, index=dates)

    seasonal = tsmom_seasonal_signal(close, vol, windows={"Corn": (6, 8)})
    from signals.momentum import tsmom_signal as _plain_tsmom
    plain = _plain_tsmom(close, vol)

    both_valid = seasonal["Corn"].notna() & plain["Corn"].notna() & (plain["Corn"] != 0)
    assert (np.sign(seasonal["Corn"][both_valid]) == np.sign(plain["Corn"][both_valid])).all()


def test_tsmom_seasonal_windows_match_phase_11c_medium_confidence_or_higher():
    # Regression guard: this exact 7-asset window table is a WORKFLOW.md
    # Phase 11c decision (fixed a priori, not to drift silently).
    assert set(TSMOM_SEASONAL_WINDOWS.keys()) == {
        "Natural Gas", "HeatingOil", "RBOB", "Corn", "Soybeans", "Wheat", "KC_Wheat",
    }
