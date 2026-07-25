"""
signals/crossover.py — Moving-average crossovers (golden/death cross family).

On `feature_engineering.ipynb`'s own feature wishlist ("Moving Average Features:
50, 100, 200") but never built pre-rebuild — not invented here. Three pairs, all
reported as parallel Books, no headline pick (same discipline as breakout's two
systems): 50/100, 50/200 (the textbook golden-cross/death-cross pairing, the one
with real trading-practice precedent), 100/200.

Fully vectorized `sign(fast_SMA - slow_SMA)` — unlike breakout, no per-asset
state-machine walk-forward is needed here: there's no separate, looser "exit"
threshold the way Donchian's dual-channel design has one, so a plain vectorized
sign flip on the two SMAs' difference IS the whole signal.

SMA, not EMA — the golden-cross/death-cross definition is specifically SMA-based,
not a generic "fast MA vs slow MA" concept. Back-adjusted curve (same roll-jump-
artifact reasoning as breakout: a raw roll-date jump could spuriously register as
a crossover). Daily frequency, immediate flip (no confirmation filter — simpler
than breakout's dual-channel design, since there's no separate exit channel to
delay a reversal). Vol estimator is Yang-Zhang directly (the project-wide
decision was already in place by the time this was built).

Real bug found and fixed while building this: a strict `min_periods == window`
rolling mean makes any asset with a sparser trading calendar than the panel's
union index (Corn, Soybeans, Wheat, Cocoa, Coffee, Cotton, Sugar — the same
sparse-calendar assets already documented in `data.volatility`'s module
docstring) go 100% NaN across their ENTIRE history, not just reduced coverage —
checked directly, 0 of ~5000 dates. Fixed by reusing `MIN_FRAC = 0.7`, the same
already-validated tolerance `data.volatility.yang_zhang_volatility`'s own
`min_frac` and `signals.breakout`'s effective full-window requirement both rest
on, rather than guessing a fresh number.
"""

import pandas as pd

from signals.transforms import vol_targeted_sign_signal

PAIRS = {
    "50_100": (50, 100),
    "50_200": (50, 200),  # golden cross / death cross
    "100_200": (100, 200),
}

MIN_FRAC = 0.7
DEFAULT_TARGET_VOL = 0.40


def _sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    min_periods = max(1, int(window * MIN_FRAC))
    return close.rolling(window, min_periods=min_periods).mean()


def crossover_raw(close: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    """fast_SMA - slow_SMA, sparse-calendar-tolerant (see module docstring)."""
    return _sma(close, fast_window) - _sma(close, slow_window)


def crossover_signal(
    close: pd.DataFrame, vol: pd.DataFrame, fast_window: int, slow_window: int, target_vol: float = DEFAULT_TARGET_VOL
) -> pd.DataFrame:
    raw = crossover_raw(close, fast_window, slow_window)
    return vol_targeted_sign_signal(raw, vol, target_vol=target_vol)


def crossover_pair_signal(close: pd.DataFrame, vol: pd.DataFrame, pair: str, target_vol: float = DEFAULT_TARGET_VOL) -> pd.DataFrame:
    """One of `PAIRS`' three named pairs ("50_100", "50_200", "100_200")."""
    fast_window, slow_window = PAIRS[pair]
    return crossover_signal(close, vol, fast_window, slow_window, target_vol=target_vol)


def all_pair_signals(close: pd.DataFrame, vol: pd.DataFrame, target_vol: float = DEFAULT_TARGET_VOL) -> dict:
    """One signal per pair in `PAIRS` — all three reported as parallel Books, no
    auto-picked winner (CLAUDE.md's own reporting convention for this signal)."""
    return {name: crossover_pair_signal(close, vol, name, target_vol=target_vol) for name in PAIRS}
