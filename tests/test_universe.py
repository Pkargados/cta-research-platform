import pytest
import pandas as pd

from data.universe import (
    get_liquid_universe,
    ICE_SOFTS_DATA_BLOCKED,
    compress_for_family,
    CLUSTER_REDUNDANT_ALL,
    CLUSTER_REDUNDANT_TREND_ONLY,
)


def _volume_panel():
    dates = pd.date_range("2024-06-01", periods=60, freq="D")
    return pd.DataFrame({
        "Liquid": [5000] * 60,
        "Thin": [500] * 60,
        "Borderline": [1000] * 60,
    }, index=dates)


def test_get_liquid_universe_splits_by_adv_threshold():
    volume = _volume_panel()
    included, excluded = get_liquid_universe(volume, window_start="2024-06-01", threshold=1000)
    assert "Liquid" in included
    assert "Thin" in excluded
    assert "Borderline" in included  # >= threshold, not strictly >


def test_get_liquid_universe_only_considers_window_start_onward():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    volume = pd.DataFrame({"A": 0}, index=dates)
    # Only real (high) volume AFTER the window_start cutoff - an asset that was
    # thin historically but genuinely liquid recently should still be included.
    volume.loc[dates >= "2024-06-01", "A"] = 5000

    included, excluded = get_liquid_universe(volume, window_start="2024-06-01", threshold=1000)
    assert "A" in included


def test_get_liquid_universe_excludes_when_recent_window_is_thin_despite_old_history():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    volume = pd.DataFrame({"A": 5000}, index=dates)
    volume.loc[dates >= "2024-06-01", "A"] = 100  # went quiet recently

    included, excluded = get_liquid_universe(volume, window_start="2024-06-01", threshold=1000)
    assert "A" in excluded


def test_get_liquid_universe_returns_disjoint_complete_partition():
    volume = _volume_panel()
    included, excluded = get_liquid_universe(volume, window_start="2024-06-01", threshold=1000)
    assert set(included).isdisjoint(set(excluded))
    assert set(included) | set(excluded) == set(volume.columns)


def test_get_liquid_universe_excludes_ice_softs_data_block_even_with_high_adv():
    dates = pd.date_range("2024-06-01", periods=60, freq="D")
    volume = pd.DataFrame({
        "Liquid": [5000] * 60,
        "Coffee": [5000] * 60,  # high ADV, but a known Databento stuck-front artifact
    }, index=dates)
    included, excluded = get_liquid_universe(volume, window_start="2024-06-01", threshold=1000)
    assert "Liquid" in included
    assert "Coffee" in excluded
    assert set(ICE_SOFTS_DATA_BLOCKED) == {"Coffee", "Cotton", "OrangeJuice"}


def _full_universe():
    return ["WTI Crude", "Brent", "Corn", "Wheat", "KC_Wheat", "SP500", "Dow", "US_10Y", "US_5Y", "UltraBond"]


def test_compress_for_family_rank_drops_only_near_duplicates():
    result = compress_for_family(_full_universe(), "rank")
    for a in CLUSTER_REDUNDANT_ALL:
        assert a not in result
    # 0.85-0.95 directional-only pairs are kept for a rank-based family.
    assert "Brent" in result
    assert "KC_Wheat" in result


def test_compress_for_family_trend_drops_near_duplicates_and_directional_pairs():
    result = compress_for_family(_full_universe(), "trend")
    for a in CLUSTER_REDUNDANT_ALL:
        assert a not in result
    for a in CLUSTER_REDUNDANT_TREND_ONLY:
        assert a not in result
    # The more liquid twin of each directional-only pair survives.
    assert "WTI Crude" in result
    assert "Wheat" in result


def test_compress_for_family_rejects_unknown_family():
    with pytest.raises(ValueError):
        compress_for_family(_full_universe(), "not_a_family")


def test_compress_for_family_only_drops_names_present_in_input():
    # A shorter input universe that never had the redundant names to begin with
    # should pass through unchanged, not error.
    result = compress_for_family(["Corn", "Soybeans"], "trend")
    assert result == ["Corn", "Soybeans"]


def test_compress_for_family_trend_universe_is_subset_of_rank_universe():
    full = _full_universe()
    trend = set(compress_for_family(full, "trend"))
    rank = set(compress_for_family(full, "rank"))
    assert trend.issubset(rank)
