import numpy as np
import pandas as pd

from signals.transforms import (
    binary_signal,
    continuous_signal,
    rank_signal,
    vol_targeted_sign_signal,
    cross_sectional_rank,
)


def test_binary_signal_is_pure_sign():
    momentum = pd.DataFrame({"A": [1.0, -2.0, 0.0]})
    result = binary_signal(momentum)
    assert result["A"].tolist() == [1.0, -1.0, 0.0]


def test_continuous_signal_scales_by_inverse_vol():
    momentum = pd.DataFrame({"A": [10.0]})
    vol = pd.DataFrame({"A": [2.0]})
    result = continuous_signal(momentum, vol)
    assert result["A"].iloc[0] == 5.0


def test_rank_signal_centered_at_zero():
    momentum = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
    result = rank_signal(momentum)
    # 3 assets: ranks 1/3, 2/3, 3/3 (pct=True) minus 0.5
    assert np.isclose(result.loc[0, "A"], 1 / 3 - 0.5)
    assert np.isclose(result.loc[0, "C"], 1.0 - 0.5)


def test_vol_targeted_sign_signal_uses_sign_not_magnitude():
    raw = pd.DataFrame({"A": [100.0], "B": [-0.001]})
    vol = pd.DataFrame({"A": [0.2], "B": [0.2]})
    result = vol_targeted_sign_signal(raw, vol, target_vol=0.40)
    # Same vol, opposite sign raw signal -> equal magnitude, opposite sign positions
    assert np.isclose(result["A"].iloc[0], 0.40 / 0.2)
    assert np.isclose(result["B"].iloc[0], -0.40 / 0.2)


def test_vol_targeted_sign_signal_default_target_vol_is_neutral():
    raw = pd.DataFrame({"A": [5.0]})
    vol = pd.DataFrame({"A": [1.0]})
    result = vol_targeted_sign_signal(raw, vol)
    assert result["A"].iloc[0] == 1.0


def test_vol_targeted_sign_signal_propagates_nan():
    raw = pd.DataFrame({"A": [np.nan]})
    vol = pd.DataFrame({"A": [0.2]})
    result = vol_targeted_sign_signal(raw, vol, target_vol=0.4)
    assert pd.isna(result["A"].iloc[0])


def test_cross_sectional_rank_demeans_within_sector_not_across():
    dates = pd.date_range("2020-01-01", periods=1)
    scores = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [100.0], "D": [200.0]}, index=dates)
    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}

    result = cross_sectional_rank(scores, sectors)

    # Within Sector1: A has lower score than B -> negative demeaned rank for A
    assert result.loc[dates[0], "A"] < 0
    assert result.loc[dates[0], "B"] > 0
    # Sector2's huge scores don't leak into Sector1's ranks
    assert abs(result.loc[dates[0], "A"]) < 1.0


def test_cross_sectional_rank_respects_min_group_size():
    dates = pd.date_range("2020-01-01", periods=2)
    scores = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 3.0]}, index=dates)
    sectors = {"Sector1": ["A", "B"]}

    result = cross_sectional_rank(scores, sectors, min_group_size=2)

    # Date 0: both present -> real values
    assert result.loc[dates[0], "A"] != 0 or result.loc[dates[0], "B"] != 0
    # Date 1: only B present (group size 1 < min_group_size=2) -> NaN for both
    assert pd.isna(result.loc[dates[1], "A"])
    assert pd.isna(result.loc[dates[1], "B"])


def test_cross_sectional_rank_drops_assets_outside_sectors():
    dates = pd.date_range("2020-01-01", periods=1)
    scores = pd.DataFrame({"A": [1.0], "B": [2.0], "Unmapped": [5.0]}, index=dates)
    sectors = {"Sector1": ["A", "B"]}

    result = cross_sectional_rank(scores, sectors)

    assert "Unmapped" not in result.columns
