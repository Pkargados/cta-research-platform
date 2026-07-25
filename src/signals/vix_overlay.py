"""
signals/vix_overlay.py — VIX-conditional position-size multiplier for short-term
reversal (Nagel 2011, "Evaporating Liquidity").

Nagel finds short-term reversal's expected return is strongly, positively
predictable by the VIX level — reversal profits are a compensation for liquidity
provision, and that compensation is richest exactly when funding/liquidity
conditions are stressed (VIX high). This module is ONLY the sizing multiplier
(scale the position up when VIX is high, down when it's low) — the regression
evidence for WHETHER this relationship is real (Newey-West HAC-adjusted t-stats,
individual vs. sector tier) lives in `research/short_term_reversal.py`, not here.

Real bug found and fixed live while building this (CLAUDE.md's short-term-
reversal row): the VIX-adjusted sizing multiplier was first applied to the RAW
signal, BEFORE `backtest.engine`'s own gross-exposure normalization
(`normalized_positions`'s `positions.div(gross_exposure, axis=0)`). A uniform
daily multiplier (the same scalar applied to every asset's position that day)
factors out of both the position sum (numerator of gross_exposure) and the
normalization denominator IDENTICALLY — so applying it pre-normalization is a
no-op, silently. "Simple" and "VIX-adjusted" Sharpe came back bit-for-bit
identical in every cell, which is what exposed the bug. Fixed by applying the
multiplier to the ALREADY-NORMALIZED position array instead
(`apply_size_multiplier` below is meant to be called AFTER
`backtest.engine.normalized_positions`, never before).
"""

import numpy as np
import pandas as pd


def vix_size_multiplier(vix: pd.Series, lookback: int = 252, min_periods: int = 60) -> pd.Series:
    """VIX level, expressed as a size multiplier relative to its own trailing
    rolling average: `vix_t / rolling_mean(vix, lookback)_{t-1}`. Shifted by one day
    before use (CLAUDE.md Rule 3) — the multiplier applied on date t must only use
    information knowable through t-1, not t's own contemporaneous VIX close.

    A ratio around its own trailing history (not the raw VIX level) keeps this
    comparable across VIX's very different historical regimes (e.g. the ~10-90
    range pre-2008 vs. the 2020 COVID spike above 80) - a fixed absolute
    threshold would need re-calibrating for either era, while "VIX relative to
    its own recent history" doesn't.
    """
    rolling_mean = vix.rolling(lookback, min_periods=min_periods).mean()
    multiplier = vix / rolling_mean.shift(1)
    return multiplier


def apply_size_multiplier(positions: pd.DataFrame, multiplier: pd.Series) -> pd.DataFrame:
    """Scale an ALREADY gross-exposure-normalized position array by a per-date
    multiplier (broadcast across all assets that day).

    Must be called AFTER `backtest.engine.normalized_positions` — applying this
    multiplier to the raw, pre-normalization signal is the exact bug documented in
    this module's docstring: a uniform daily scalar cancels out of gross-exposure
    normalization identically, silently making the whole overlay a no-op.
    """
    aligned_multiplier = multiplier.reindex(positions.index)
    return positions.mul(aligned_multiplier, axis=0)
