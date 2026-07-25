import numpy as np
import pandas as pd

from signals.breakout import (
    donchian_state_machine,
    breakout_direction,
    breakout_signal,
    system1_signal,
    system2_signal,
    SYSTEM_1,
    SYSTEM_2,
)


def test_new_high_from_flat_triggers_long_entry():
    # Flat range, then a clean breakout above the trailing window.
    prices = [100.0] * 25 + [200.0]
    close = pd.Series(prices)

    state = donchian_state_machine(close, entry_window=20, exit_window=10)

    assert state.iloc[-1] == 1.0


def test_new_low_from_flat_triggers_short_entry():
    prices = [100.0] * 25 + [1.0]
    close = pd.Series(prices)

    state = donchian_state_machine(close, entry_window=20, exit_window=10)

    assert state.iloc[-1] == -1.0


def test_long_position_exits_on_new_exit_window_low_not_entry_window_low():
    # Build a long position, then a moderate pullback that breaks the tighter
    # exit-window low but not the wider entry-window low.
    prices = [100.0] * 25 + [200.0]  # triggers long entry at index 25
    prices += [150.0] * 8            # holds long (no new exit-window low yet)
    prices += [90.0]                 # new 10-day low (exit_window=10) -> flatten
    close = pd.Series(prices)

    state = donchian_state_machine(close, entry_window=20, exit_window=10)

    entry_idx = 25
    assert state.iloc[entry_idx] == 1.0
    assert state.iloc[-1] == 0.0


def test_position_never_flips_directly_without_passing_through_flat():
    # A single-day price crash right after a long entry should flatten, not flip
    # straight to short, since exit_window < entry_window guarantees the exit
    # channel triggers before a fresh short entry could be evaluated.
    prices = [100.0] * 25 + [200.0, 0.01]
    close = pd.Series(prices)

    state = donchian_state_machine(close, entry_window=20, exit_window=10)

    assert state.iloc[-2] == 1.0
    assert state.iloc[-1] == 0.0  # flattens, doesn't jump to -1 in one step


def test_warmup_period_is_nan():
    close = pd.Series(np.linspace(100, 110, 15))
    state = donchian_state_machine(close, entry_window=20, exit_window=10)
    assert state.isna().all()


def test_breakout_direction_applies_per_asset():
    dates = pd.RangeIndex(30)
    close = pd.DataFrame({
        "Up": [100.0] * 25 + [200.0] * 5,
        "Down": [100.0] * 25 + [1.0] * 5,
    }, index=dates)

    direction = breakout_direction(close, entry_window=20, exit_window=10)

    assert direction["Up"].iloc[-1] == 1.0
    assert direction["Down"].iloc[-1] == -1.0


def test_breakout_signal_scales_by_target_vol_over_vol():
    dates = pd.RangeIndex(30)
    close = pd.DataFrame({"A": [100.0] * 25 + [200.0] * 5}, index=dates)
    vol = pd.DataFrame(0.25, index=dates, columns=["A"])

    signal = breakout_signal(close, vol, entry_window=20, exit_window=10, target_vol=0.40)

    assert np.isclose(signal["A"].iloc[-1], 0.40 / 0.25)


def test_system1_and_system2_use_documented_windows():
    assert SYSTEM_1 == {"entry": 20, "exit": 10}
    assert SYSTEM_2 == {"entry": 55, "exit": 20}

    dates = pd.RangeIndex(80)
    close = pd.DataFrame({"A": [100.0] * 60 + [200.0] * 20}, index=dates)
    vol = pd.DataFrame(0.2, index=dates, columns=["A"])

    s1 = system1_signal(close, vol)
    s2 = system2_signal(close, vol)

    # System 1 (shorter entry window) should react to the breakout before System 2.
    first_nonzero_s1 = s1["A"].abs().gt(0).idxmax()
    first_nonzero_s2 = s2["A"].abs().gt(0).idxmax()
    assert first_nonzero_s1 <= first_nonzero_s2
