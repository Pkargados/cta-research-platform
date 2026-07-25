"""
signals/breakout.py — Donchian-channel breakout, authentic dual-channel Turtle
Trading systems (Dennis/Parker, 1983 — a trading-practice convention, not an
academic paper; no PDF anchors this the way momentum/carry/XSMOM/value have one).

Two historical systems, both reported as parallel specs (Turtles traded both
simultaneously as a blended book — no "headline pick"), per
`references/turtle_breakout_implementation_recipe.md`:
- System 1: 20-day entry channel / 10-day exit channel
- System 2: 55-day entry channel / 20-day exit channel

Genuinely path-dependent (long/flat/short state machine — which threshold applies
depends on the position currently held), implemented as a per-asset walk-forward,
same pattern as `data.continuous_curve.assign_front_contract`. A single vectorized
`sign(close - rolling_max)` expression can't express "stay long until price
crosses the EXIT channel," which is the entire point of the dual-channel design
(the exit channel is deliberately looser than the entry channel, so a position
isn't stopped out by the same noise that would block a fresh entry).

Channels built off the BACK-ADJUSTED curve (`data.continuous_curve.
load_continuous_backadjusted`) — the OPPOSITE choice from momentum's Yang-Zhang
input (which needs the raw curve). A raw roll-date price jump could spuriously
register as a "new N-day high," a hazard this signal is far more exposed to than
momentum (whose 12-month average dilutes a single-day artifact).

Direction updates DAILY — non-negotiable (monthly formation, momentum's
convention, would delay reacting to a breakout until month-end, destroying the
entire reason to build this signal). Position SIZE re-derivation cadence
(daily vs. coarser) is a separate, measured question left to the research driver
(`research/breakout.py`), not assumed here — this module always emits a daily
direction signal; the caller decides how often to actually resize it via
`backtest.engine.backtest_signal(..., frequency=...)`.

Scope note: "authentic Turtle" means the entry/exit price-TRIGGER logic, not the
original 1983 ATR-based money-management system — this project's existing
vol-targeting (`signals.transforms.vol_targeted_sign_signal`, `target_vol=0.40`,
same as momentum) supersedes that for sizing (CLAUDE.md Rule 7: re-derive
mechanics per family, don't port a whole external system wholesale).
"""

import numpy as np
import pandas as pd

from signals.transforms import vol_targeted_sign_signal

SYSTEM_1 = {"entry": 20, "exit": 10}
SYSTEM_2 = {"entry": 55, "exit": 20}

DEFAULT_TARGET_VOL = 0.40


def _rolling_extremes(close: pd.Series, window: int):
    """Rolling max/min of `close` over `window` days, EXCLUDING today (a breakout
    is a break of the prior range, not today's own bar comparing to itself) —
    shifted by 1 before the rolling window is applied."""
    prior = close.shift(1)
    return prior.rolling(window, min_periods=window).max(), prior.rolling(window, min_periods=window).min()


def donchian_state_machine(close: pd.Series, entry_window: int, exit_window: int) -> pd.Series:
    """Per-asset walk-forward: long (+1) / flat (0) / short (-1) state, updated
    daily. Genuinely path-dependent - implemented as an explicit loop (same
    pattern as `assign_front_contract`), not vectorized, since "which threshold
    applies" depends on the state carried in from the prior day.

    Entry: from FLAT, go long on a new `entry_window`-day high, short on a new
    `entry_window`-day low.
    Exit: from LONG, flatten on a new `exit_window`-day LOW (the looser exit
    channel); from SHORT, flatten on a new `exit_window`-day HIGH. A position
    never flips directly long-to-short or short-to-long in one step - it always
    passes through flat first (the exit channel triggers before a new entry could
    even be evaluated on the same bar, since exit_window < entry_window in both
    Turtle systems).

    First `entry_window` days are NaN (not enough history to compute either
    channel) - the state machine only starts once init_idx has a real
    entry-window high/low to compare against.
    """
    entry_high, entry_low = _rolling_extremes(close, entry_window)
    exit_high, exit_low = _rolling_extremes(close, exit_window)

    n = len(close)
    state = np.full(n, np.nan)
    current = 0.0
    started = False

    for i in range(n):
        price = close.iloc[i]
        eh, el = entry_high.iloc[i], entry_low.iloc[i]
        xh, xl = exit_high.iloc[i], exit_low.iloc[i]

        if pd.isna(eh) or pd.isna(el) or pd.isna(price):
            state[i] = np.nan
            continue

        started = True

        if current == 0.0:
            if price > eh:
                current = 1.0
            elif price < el:
                current = -1.0
        elif current == 1.0:
            if not pd.isna(xl) and price < xl:
                current = 0.0
        elif current == -1.0:
            if not pd.isna(xh) and price > xh:
                current = 0.0

        state[i] = current if started else np.nan

    return pd.Series(state, index=close.index)


def breakout_direction(close: pd.DataFrame, entry_window: int, exit_window: int) -> pd.DataFrame:
    """`donchian_state_machine`, applied per-asset across the (T x N) back-adjusted
    close panel. Returns a DataFrame of {-1, 0, 1, NaN}."""
    return pd.DataFrame(
        {asset: donchian_state_machine(close[asset], entry_window, exit_window) for asset in close.columns},
        index=close.index,
    )


def breakout_signal(
    close: pd.DataFrame, vol: pd.DataFrame, entry_window: int, exit_window: int, target_vol: float = DEFAULT_TARGET_VOL
) -> pd.DataFrame:
    """Direction (state machine) * vol-targeted sizing, one system's full signal.

    `direction` is already in {-1, 0, 1}, so `vol_targeted_sign_signal`'s own
    `sign()` is a no-op on it (sign(0) = 0 correctly produces a flat position,
    not a divide-by-zero or spurious +1/-1) - reusing the same generic transform
    every other signal family uses, not a bespoke breakout-specific sizing rule.
    """
    direction = breakout_direction(close, entry_window, exit_window)
    return vol_targeted_sign_signal(direction, vol, target_vol=target_vol)


def system1_signal(close: pd.DataFrame, vol: pd.DataFrame, target_vol: float = DEFAULT_TARGET_VOL) -> pd.DataFrame:
    return breakout_signal(close, vol, SYSTEM_1["entry"], SYSTEM_1["exit"], target_vol)


def system2_signal(close: pd.DataFrame, vol: pd.DataFrame, target_vol: float = DEFAULT_TARGET_VOL) -> pd.DataFrame:
    return breakout_signal(close, vol, SYSTEM_2["entry"], SYSTEM_2["exit"], target_vol)
