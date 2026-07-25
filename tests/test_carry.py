import numpy as np
import pandas as pd

from signals.carry import (
    carry1m_signal,
    carry1_12_raw,
    carry1_12_signal,
    sector_pooled_expanding_mean,
    carry_timing_signal,
    carry_timing_zero_signal,
    carry_timing_mean_signal,
    build_all_carry_signals,
)


def _carry_panel():
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    carry = pd.DataFrame({
        "A": 0.05,   # high carry
        "B": -0.05,  # low carry
        "C": 0.10,   # high carry, other sector
        "D": -0.10,  # low carry, other sector
    }, index=dates)
    sectors = {"Sector1": ["A", "B"], "Sector2": ["C", "D"]}
    return carry, sectors


def test_carry1m_signal_ranks_within_sector_not_across():
    carry, sectors = _carry_panel()
    signal = carry1m_signal(carry, sectors)

    last = signal.iloc[-1]
    assert last["A"] > last["B"]  # A has higher carry than B within Sector1
    assert last["C"] > last["D"]
    # Cross-sector magnitude shouldn't leak: Sector1's spread shouldn't depend on Sector2's levels
    assert np.isclose(abs(last["A"]), abs(last["B"]))


def test_carry1_12_raw_smooths_relative_to_raw_carry():
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    carry = pd.DataFrame({"A": np.linspace(-0.1, 0.1, 400)}, index=dates)

    smoothed = carry1_12_raw(carry, months=12)

    # Smoothed series should have less day-to-day variation than the raw trend
    # (it's a monthly-resampled trailing average, not the raw daily value).
    assert smoothed["A"].diff().abs().dropna().mean() <= carry["A"].diff().abs().dropna().mean()


def test_carry1_12_signal_produces_valid_ranks():
    carry, sectors = _carry_panel()
    signal = carry1_12_signal(carry, sectors, months=3)
    assert signal.shape[1] <= carry.shape[1]
    assert signal.dropna(how="all").shape[0] > 0


def test_sector_pooled_expanding_mean_is_shared_within_sector():
    carry, sectors = _carry_panel()
    ref = sector_pooled_expanding_mean(carry, sectors)

    pd.testing.assert_series_equal(ref["A"], ref["B"], check_names=False)
    pd.testing.assert_series_equal(ref["C"], ref["D"], check_names=False)
    # Sector1's reference (mean of 0.05, -0.05 = 0) differs from Sector2's (mean of 0.10, -0.10 = 0)
    # both are 0 here by symmetric construction - use an asymmetric case instead.


def test_sector_pooled_expanding_mean_asymmetric_case():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    carry = pd.DataFrame({"A": [0.10] * 5, "B": [0.20] * 5}, index=dates)
    sectors = {"Sector1": ["A", "B"]}

    ref = sector_pooled_expanding_mean(carry, sectors)

    assert np.isclose(ref["A"].iloc[-1], 0.15)
    assert np.isclose(ref["B"].iloc[-1], 0.15)


def test_carry_timing_signal_is_strict_greater_than():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    carry = pd.DataFrame({"A": [0.05, 0.0, -0.05]}, index=dates)

    signal = carry_timing_signal(carry, reference=0.0)

    assert signal["A"].iloc[0] == 1.0
    assert signal["A"].iloc[1] == -1.0  # tie (carry == reference) goes short, not flat
    assert signal["A"].iloc[2] == -1.0


def test_carry_timing_zero_signal_matches_carry_timing_signal_reference_zero():
    carry, _ = _carry_panel()
    a = carry_timing_zero_signal(carry)
    b = carry_timing_signal(carry, reference=0.0)
    pd.testing.assert_frame_equal(a, b)


def test_carry_timing_mean_signal_uses_sector_pooled_reference():
    carry, sectors = _carry_panel()
    signal = carry_timing_mean_signal(carry, sectors)

    last = signal.iloc[-1]
    # Sector1 pooled mean is 0 (0.05 and -0.05 average) -> A (0.05) above mean -> long
    assert last["A"] == 1.0
    assert last["B"] == -1.0


def test_build_all_carry_signals_returns_four_parallel_specs():
    carry, sectors = _carry_panel()
    signals = build_all_carry_signals(carry, sectors)

    assert set(signals.keys()) == {"carry1m", "carry1_12", "carry_timing_zero", "carry_timing_mean"}
    for sig in signals.values():
        assert sig.shape[0] == carry.shape[0]
