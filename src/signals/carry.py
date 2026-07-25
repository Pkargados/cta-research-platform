"""
signals/carry.py — Carry (roll yield), rebuilt to match Koijen-Moskowitz-Pedersen-
Vrugt 2018 "Carry" (`references/Carry.pdf`, read directly) exactly.

Formula (unchanged from the original build, `data.term_structure`'s own docstring,
LiveCattle prototype): Carry = (F_near - F_far) / F_near × 365 / (days to F_far
expiry). That module already produces the real per-asset carry panel (37 assets
from real exchange-quoted calendar spreads, the 4 ICE softs present in this
project's universe via a back-differenced proxy, `is_proxy` flagged) — this module
consumes that panel, it does not recompute carry itself.

Four parallel specs, no headline pick (same discipline as every other signal family):

- **carry1m** — the paper's "current carry": rank-weighted cross-section of the
  current-month carry level (Eq. 19: `rank(C_i) - (N_t+1)/2`), sector-scoped rather
  than the paper's own single global cross-section (Rule 7: this project's universe
  spans FX/rates/commodities/equities, so "high carry vs. the full 42-asset universe"
  has no clean interpretation — see `signals.transforms.cross_sectional_rank`'s own
  docstring, which was extracted specifically for this construction). Monthly
  rebalancing (the paper's own stated cadence, p.207) — the single biggest driver
  of this rebuild's ~120x-to-~1x annualized turnover collapse versus the original,
  non-paper-matching first attempt.
- **carry1-12** — the paper's trailing-12-month moving average of the RAW carry
  level (footnote 14), computed BEFORE ranking, to smooth seasonal components
  (e.g. equity dividend-payment months, commodity harvest seasons) that a single
  month's carry can otherwise pick up as noise. Reuses `backtest.engine.
  holding_period_positions` directly — that function's own construction (month-end
  resample, rolling `holding_months`-month mean, reindexed forward) turns out to
  be exactly this smoothing operation, not a new implementation.
- **carry_timing (reference=0)** — the paper's Eq. before Table VI: `2×I(C_i>0)-1`,
  a simple ±1 direction, no vol-scaling (the paper's own construction has none —
  a deliberate, logged exception to CLAUDE.md Rule 5's general binary-underperforms
  finding, made specifically to reproduce this paper's exact published methodology).
- **carry_timing (reference=sector mean)** — the paper's alternative cutoff, the
  security's own historical mean carry up to that point in time (Table VI: "C̄ =
  ... the average carry across all securities... up to that point in time"),
  adapted to this project's sector-scoped design: one pooled, expanding-in-time
  threshold per SECTOR (not per individual asset — the paper's single global
  cross-section has no sector concept to begin with; sector-pooling is this
  project's own adaptation, not a literal reading of the paper).
"""

import numpy as np
import pandas as pd

from backtest.engine import holding_period_positions
from signals.transforms import cross_sectional_rank

CARRY1_12_MONTHS = 12


def carry1m_signal(carry_panel: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Rank-weighted cross-sectional carry signal, current-month carry level."""
    return cross_sectional_rank(carry_panel, sectors)


def carry1_12_raw(carry_panel: pd.DataFrame, months: int = CARRY1_12_MONTHS) -> pd.DataFrame:
    """Trailing `months`-month moving average of the RAW carry level, before
    ranking (footnote 14 of the paper) — reuses `holding_period_positions` as-is,
    since a rolling mean of month-end-resampled values, reindexed forward, is
    exactly that smoothing operation."""
    return holding_period_positions(carry_panel, holding_months=months)


def carry1_12_signal(carry_panel: pd.DataFrame, sectors: dict, months: int = CARRY1_12_MONTHS) -> pd.DataFrame:
    """Rank-weighted cross-sectional carry signal, carry1-12 (smoothed) level."""
    smoothed = carry1_12_raw(carry_panel, months)
    return cross_sectional_rank(smoothed, sectors)


def sector_pooled_expanding_mean(carry_panel: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """One shared, expanding-in-time reference carry per sector: the sector's own
    cross-sectional average carry on each date, averaged again over time from the
    start of the sample through that date (Eq./Table VI's "average carry... up to
    that point in time," adapted to a sector-pooled rather than per-asset
    reference — see module docstring). Broadcast to every member asset."""
    result = pd.DataFrame(np.nan, index=carry_panel.index, columns=carry_panel.columns)
    for sector, members in sectors.items():
        present = [m for m in members if m in carry_panel.columns]
        if not present:
            continue
        sector_avg = carry_panel[present].mean(axis=1)
        expanding_mean = sector_avg.expanding(min_periods=1).mean()
        for m in present:
            result[m] = expanding_mean
    return result


def carry_timing_signal(carry_panel: pd.DataFrame, reference) -> pd.DataFrame:
    """±1 direction: long if carry exceeds `reference`, short otherwise (paper's
    own `2×I(C_i - C̄ > 0) - 1` — a strict ">" test, so a tie goes short, not flat).
    No vol-scaling — the paper's own construction has none (CLAUDE.md Rule 5
    exception, logged).

    `reference` : scalar (e.g. 0) or a (T×N) DataFrame aligned to `carry_panel`
    (e.g. `sector_pooled_expanding_mean`'s output).
    """
    indicator = (carry_panel - reference) > 0
    return indicator.astype(float) * 2.0 - 1.0


def carry_timing_zero_signal(carry_panel: pd.DataFrame) -> pd.DataFrame:
    return carry_timing_signal(carry_panel, reference=0.0)


def carry_timing_mean_signal(carry_panel: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    reference = sector_pooled_expanding_mean(carry_panel, sectors)
    return carry_timing_signal(carry_panel, reference=reference)


def build_all_carry_signals(carry_panel: pd.DataFrame, sectors: dict) -> dict:
    """All 4 parallel Books' worth of raw signals, no headline pick."""
    return {
        "carry1m": carry1m_signal(carry_panel, sectors),
        "carry1_12": carry1_12_signal(carry_panel, sectors),
        "carry_timing_zero": carry_timing_zero_signal(carry_panel),
        "carry_timing_mean": carry_timing_mean_signal(carry_panel, sectors),
    }
