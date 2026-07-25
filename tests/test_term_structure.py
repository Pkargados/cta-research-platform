import numpy as np
import pandas as pd

from data.term_structure import (
    real_spread_carry, proxy_carry, build_carry_panel, ICE_PROXY_ASSETS,
    DAYS_PER_MONTH, ANNUALIZATION_DAYS,
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
