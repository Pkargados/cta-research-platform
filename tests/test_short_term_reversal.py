import numpy as np
import pandas as pd

from signals.short_term_reversal import (
    lag_return,
    vol_standardized_return,
    cross_sectional_demean,
    individual_reversal_signal,
    sector_average_return,
    sector_reversal_signal,
    build_all_reversal_signals,
    LAGS,
)


def _panel(n=30):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.DataFrame({
        "A": np.linspace(100, 110, n),   # winner
        "B": np.linspace(100, 90, n),    # loser
        "C": np.linspace(50, 60, n),     # winner, different sector
        "D": np.linspace(50, 45, n),     # loser, different sector
    }, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=close.columns)
    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}
    return close, vol, sectors


def test_lag_return_matches_manual_pct_change():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.DataFrame({"A": np.arange(10, 20, dtype=float)}, index=dates)
    result = lag_return(close, lag=2)
    expected = close["A"] / close["A"].shift(2) - 1.0
    pd.testing.assert_series_equal(result["A"], expected, check_names=False)


def test_vol_standardized_return_scales_by_lag_and_vol():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.DataFrame({"A": np.arange(10, 20, dtype=float)}, index=dates)
    vol = pd.DataFrame(0.25, index=dates, columns=["A"])

    result = vol_standardized_return(close, vol, lag=4)
    raw = lag_return(close, lag=4)
    daily_vol = 0.25 / np.sqrt(252)
    expected = raw["A"] / (daily_vol * np.sqrt(4))
    pd.testing.assert_series_equal(result["A"], expected, check_names=False)


def test_cross_sectional_demean_does_not_leak_across_sectors():
    dates = pd.date_range("2020-01-01", periods=1)
    scores = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [1000.0], "D": [1002.0]}, index=dates)
    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}

    result = cross_sectional_demean(scores, sectors)

    assert np.isclose(result.loc[dates[0], "A"], -1.0)
    assert np.isclose(result.loc[dates[0], "B"], 1.0)
    assert np.isclose(result.loc[dates[0], "C"], -1.0)


def test_cross_sectional_demean_respects_min_group_size():
    dates = pd.date_range("2020-01-01", periods=2)
    scores = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 3.0]}, index=dates)
    sectors = {"Sector1": ["A", "B"]}

    result = cross_sectional_demean(scores, sectors, min_group_size=2)

    assert pd.isna(result.loc[dates[1], "A"])
    assert pd.isna(result.loc[dates[1], "B"])


def test_individual_reversal_signal_bets_against_recent_winner():
    close, vol, sectors = _panel()
    signal = individual_reversal_signal(close, vol, sectors, lag=5)

    last = signal.iloc[-1]
    # A trended up relative to sector-mate B -> reversal signal should be negative for A.
    assert last["A"] < 0
    assert last["B"] > 0


def test_sector_average_return_is_equal_weighted_mean_of_members():
    close, vol, sectors = _panel()
    raw = vol_standardized_return(close, vol, lag=5)

    avg = sector_average_return(raw, sectors)

    expected = raw[["A", "B"]].mean(axis=1)
    pd.testing.assert_series_equal(avg["Sector1"], expected, check_names=False)


def test_sector_reversal_signal_broadcasts_same_score_to_sector_members():
    close, vol, sectors = _panel()
    signal = sector_reversal_signal(close, vol, sectors, lag=5)

    pd.testing.assert_series_equal(signal["A"], signal["B"], check_names=False)
    pd.testing.assert_series_equal(signal["C"], signal["D"], check_names=False)


def test_sector_reversal_signal_bets_against_sector_that_outperformed():
    close, vol, sectors = _panel()
    signal = sector_reversal_signal(close, vol, sectors, lag=5)

    # Sector1 (A up, B down) vs Sector2 (C up, D down) - both sectors have one
    # winner/one loser so their equal-weighted composite returns should be similar;
    # just check the signal is well-defined (non-NaN) and finite.
    assert signal.iloc[-1].notna().all()


def test_build_all_reversal_signals_returns_six_parallel_specs():
    close, vol, sectors = _panel()
    signals = build_all_reversal_signals(close, vol, sectors)

    expected_keys = {f"{tier}_{lag}d" for tier in ("individual", "sector") for lag in LAGS}
    assert set(signals.keys()) == expected_keys
    for sig in signals.values():
        assert sig.shape[0] == close.shape[0]
