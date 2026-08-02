import numpy as np
import pandas as pd
import pytest

from data.term_structure import (
    real_spread_carry, proxy_carry, build_carry_panel, ICE_PROXY_ASSETS,
    DAYS_PER_MONTH, ANNUALIZATION_DAYS,
    build_databento_only_continuous_curve, build_databento_only_carry,
    DATABENTO_ONLY_SPREAD_SCALE,
)


def test_real_spread_carry_matches_formula():
    dates = pd.date_range("2020-01-06", periods=5, freq="D")
    front_symbols = pd.Series("Near1", index=dates)

    spread_close = 2.0
    near_outright_close = 100.0
    spreads = pd.DataFrame({
        "date": dates, "asset": "TestAsset",
        "near_contract_symbol": "Near1", "far_contract_symbol": "Far1",
        "close": spread_close,
        "near_expiry_year": 2020, "near_expiry_code": "F",  # Jan
        "far_expiry_year": 2020, "far_expiry_code": "H",    # Mar (2 months later)
    })
    outrights = pd.DataFrame({
        "date": dates, "asset": "TestAsset", "contract_symbol": "Near1", "close": near_outright_close,
    })

    result = real_spread_carry("TestAsset", front_symbols, spreads, outrights)

    days = 2 * DAYS_PER_MONTH
    expected = (spread_close / near_outright_close) * (ANNUALIZATION_DAYS / days)
    assert np.allclose(result.dropna(), expected)
    assert result.index.equals(dates)


def test_real_spread_carry_no_matching_spread_is_nan():
    dates = pd.date_range("2020-01-06", periods=3, freq="D")
    front_symbols = pd.Series("Near1", index=dates)
    empty_spreads = pd.DataFrame(columns=[
        "date", "asset", "near_contract_symbol", "far_contract_symbol", "close",
        "near_expiry_year", "near_expiry_code", "far_expiry_year", "far_expiry_code",
    ])
    outrights = pd.DataFrame({"date": dates, "asset": "TestAsset", "contract_symbol": "Near1", "close": 100.0})

    result = real_spread_carry("TestAsset", front_symbols, empty_spreads, outrights)
    assert result.isna().all()
    assert result.index.equals(dates)


def test_real_spread_carry_picks_nearest_far_leg_when_multiple_quoted():
    dates = pd.date_range("2020-01-06", periods=1, freq="D")
    front_symbols = pd.Series("Near1", index=dates)

    # Two far legs quoted against the same front on the same day - the NEAREST
    # (smallest month distance) should be selected, per the recipe.
    spreads = pd.DataFrame({
        "date": [dates[0], dates[0]], "asset": ["TestAsset", "TestAsset"],
        "near_contract_symbol": ["Near1", "Near1"], "far_contract_symbol": ["Far_near", "Far_far"],
        "close": [1.0, 5.0],
        "near_expiry_year": [2020, 2020], "near_expiry_code": ["F", "F"],
        "far_expiry_year": [2020, 2020], "far_expiry_code": ["G", "Z"],  # +1mo vs +11mo
    })
    outrights = pd.DataFrame({"date": [dates[0]], "asset": ["TestAsset"], "contract_symbol": ["Near1"], "close": [100.0]})

    result = real_spread_carry("TestAsset", front_symbols, spreads, outrights)
    days = 1 * DAYS_PER_MONTH
    expected = (1.0 / 100.0) * (ANNUALIZATION_DAYS / days)  # uses the close=1.0 (nearest) leg
    assert np.isclose(result.iloc[0], expected)


def test_proxy_carry_matches_formula_via_chain_walk():
    dates = pd.date_range("2020-01-06", periods=3, freq="D")
    front_symbols = pd.Series("Near1", index=dates)

    near_close, far_close = 100.0, 97.0
    outrights = pd.DataFrame([
        {"date": d, "asset": "SoftAsset", "contract_symbol": "Near1", "expiry_year": 2020, "expiry_code": "F", "close": near_close}
        for d in dates
    ] + [
        {"date": d, "asset": "SoftAsset", "contract_symbol": "Far1", "expiry_year": 2020, "expiry_code": "N", "close": far_close}
        for d in dates  # N = July, 6 months after Jan
    ])

    result = proxy_carry("SoftAsset", front_symbols, outrights)

    days = 6 * DAYS_PER_MONTH
    expected = ((near_close - far_close) / near_close) * (ANNUALIZATION_DAYS / days)
    assert np.allclose(result.dropna(), expected)


def test_proxy_carry_no_data_for_asset_is_all_nan():
    dates = pd.date_range("2020-01-06", periods=3, freq="D")
    front_symbols = pd.Series("Near1", index=dates)
    outrights = pd.DataFrame(columns=["date", "asset", "contract_symbol", "expiry_year", "expiry_code", "close"])

    result = proxy_carry("MissingAsset", front_symbols, outrights)
    assert result.isna().all()


def test_ice_proxy_assets_list_is_the_five_documented_softs():
    assert set(ICE_PROXY_ASSETS) == {"Coffee", "Sugar", "Cocoa", "Cotton", "OrangeJuice"}


def _synthetic_databento_only_outrights():
    """Two contracts, A1 front for the first half then a sustained volume
    crossover to B1 for the second half - enough for assign_front_contract's
    default confirmation_days=2 to roll partway through, giving
    build_continuous_curve real roll/back-adjustment work to do. 20 days, both
    contracts trading every day (real rows throughout, never dropping out) so
    neither contract's own last-observed date falls within the default
    backstop_days=5 of day 0 - a shorter window made the backstop rule (not the
    volume-crossover rule this test means to exercise) fire immediately on day 0,
    since a contract's "last seen" date was trivially close to every date in a
    too-short synthetic series."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    rows = []
    for i, d in enumerate(dates):
        vol_a, vol_b = (100, 10) if i < 10 else (10, 100)
        rows.append({
            "date": d, "asset": "TestDBAsset", "contract_symbol": "A1",
            "expiry_year": 2020, "expiry_code": "F",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + i, "volume": vol_a,
        })
        rows.append({
            "date": d, "asset": "TestDBAsset", "contract_symbol": "B1",
            "expiry_year": 2020, "expiry_code": "H",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 105.0 + i, "volume": vol_b,
        })
    return pd.DataFrame(rows)


def test_build_databento_only_continuous_curve_filters_by_asset_and_reuses_roll_logic(monkeypatch):
    outrights = _synthetic_databento_only_outrights()
    other_asset = pd.DataFrame([{
        "date": outrights["date"].iloc[0], "asset": "OtherAsset", "contract_symbol": "X1",
        "expiry_year": 2020, "expiry_code": "F", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
    }])
    combined = pd.concat([outrights, other_asset], ignore_index=True)
    monkeypatch.setattr("data.term_structure.load_outrights", lambda data_dir=None: combined)

    curve = build_databento_only_continuous_curve("TestDBAsset")
    assert set(curve["front_contract_symbol"].unique()) <= {"A1", "B1"}
    assert curve["front_contract_symbol"].iloc[0] == "A1"
    assert curve["front_contract_symbol"].iloc[-1] == "B1"
    assert "adj_close" in curve.columns
    assert curve["is_roll_date"].any()


def test_build_databento_only_continuous_curve_raises_for_unknown_asset(monkeypatch):
    monkeypatch.setattr("data.term_structure.load_outrights", lambda data_dir=None: _synthetic_databento_only_outrights())
    with pytest.raises(ValueError):
        build_databento_only_continuous_curve("NoSuchAsset")


def test_build_databento_only_carry_reuses_real_spread_carry_formula(monkeypatch):
    outrights = _synthetic_databento_only_outrights()
    monkeypatch.setattr("data.term_structure.load_outrights", lambda data_dir=None: outrights)

    # Real calendar spread quoted against whichever contract is front on the
    # LAST date (B1, per the sustained volume crossover above).
    last_date = outrights["date"].max()
    spreads = pd.DataFrame({
        "date": [last_date], "asset": ["TestDBAsset"],
        "near_contract_symbol": ["B1"], "far_contract_symbol": ["C1"],
        "close": [2.0],
        "near_expiry_year": [2020], "near_expiry_code": ["H"],
        "far_expiry_year": [2020], "far_expiry_code": ["N"],  # H->N = 4 months
    })
    monkeypatch.setattr("data.term_structure.load_real_spreads", lambda data_dir=None: spreads)

    carry = build_databento_only_carry("TestDBAsset")
    near_outright_close = outrights.loc[
        (outrights["contract_symbol"] == "B1") & (outrights["date"] == last_date), "close"
    ].iloc[0]
    days = 4 * DAYS_PER_MONTH
    expected = (2.0 / near_outright_close) * (ANNUALIZATION_DAYS / days)
    assert np.isclose(carry.loc[last_date], expected)


def test_build_databento_only_carry_rescales_sofr_spread_close():
    """SOFR's quoted calendar-spread close is on a different scale than its own
    outright close - confirmed live (2026-07-13 SR3Z27-SR3Z28: spread quoted -8.5,
    but the two outrights' own closes differ by exactly -0.085, i.e. 1/100th).
    DATABENTO_ONLY_SPREAD_SCALE applies that correction before computing carry -
    this pins the correction factor itself, not just that build_databento_only_carry
    runs without error."""
    assert DATABENTO_ONLY_SPREAD_SCALE["SOFR"] == 0.01


def test_build_databento_only_carry_applies_scale_before_formula(monkeypatch):
    outrights = _synthetic_databento_only_outrights()
    outrights = outrights.assign(asset="SOFR")
    monkeypatch.setattr("data.term_structure.load_outrights", lambda data_dir=None: outrights)

    last_date = outrights["date"].max()
    near_outright_close = outrights.loc[
        (outrights["contract_symbol"] == "B1") & (outrights["date"] == last_date), "close"
    ].iloc[0]
    spreads = pd.DataFrame({
        "date": [last_date], "asset": ["SOFR"],
        "near_contract_symbol": ["B1"], "far_contract_symbol": ["C1"],
        "close": [850.0],  # quoted units - scaled by 0.01 before the formula sees it
        "near_expiry_year": [2020], "near_expiry_code": ["H"],
        "far_expiry_year": [2020], "far_expiry_code": ["N"],  # H->N = 4 months
    })
    monkeypatch.setattr("data.term_structure.load_real_spreads", lambda data_dir=None: spreads)

    carry = build_databento_only_carry("SOFR")
    days = 4 * DAYS_PER_MONTH
    expected = ((850.0 * 0.01) / near_outright_close) * (ANNUALIZATION_DAYS / days)
    assert np.isclose(carry.loc[last_date], expected)
