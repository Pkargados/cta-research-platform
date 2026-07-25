"""
backtest/costs.py — Turnover-based transaction cost drag.

Added 2026-07-21 (WORKFLOW.md Phase 6, pulled forward as a deliberate exception -
see that section for the full rationale) so any signal's reported Sharpe can be
shown net-of-cost, not just gross, before a full portfolio-construction/optimizer
layer exists to net costs out itself (that's Phase 7's turnover-*penalized
optimization* - a different, later thing from the cost *estimate* here).

Pure functions, no optimizer dependency, same convention as every other
signal/backtest module in this project (CLAUDE.md Architecture section).

Cost assumption: liquidity-tiered by each asset's own trailing ADV (reusing
data.universe.get_liquid_universe's ADV calculation, not inventing new numbers),
NOT a real measured cost - searched for a citable institutional bps-by-asset-class
table before assuming one (2026-07-21) and found none; Moskowitz-Ooi-Pedersen (2012)
itself is entirely gross-of-cost. The tier boundaries and bps values below are a
labeled placeholder, directionally supported by anecdotal evidence (a GSCI-tracking
commodity basket costs >100bp/year vs. <10bp/year for S&P 500 futures) but not a
measured fact - recalibrate against real broker cost sheets or fill data if that
ever becomes available, same "label it, don't fake it" discipline as carry
(CLAUDE.md Rule 4).
"""

import pandas as pd

DEFAULT_TIER_BPS = (1.0, 3.0, 6.0, 10.0)  # very liquid -> thin, one-way cost in bps


def turnover(positions: pd.DataFrame) -> pd.DataFrame:
    """|Δposition| per asset per day - how much got traded, not just held.

    Pass the same FINAL position array used to multiply against returns (already
    shift(1)'d and gross-exposure-normalized where applicable) so this reflects the
    actual weight change the book would trade, not raw signal noise.
    """
    return positions.diff().abs()


def transaction_cost_drag(positions: pd.DataFrame, cost_bps: pd.Series) -> pd.Series:
    """Daily portfolio return drag from trading: turnover priced in each asset's own
    assumed one-way cost (bps), summed across the book.

    `cost_bps` : pd.Series indexed by asset (same columns as `positions`), one-way
    cost in basis points (e.g. 3.0 = 3bp = 0.0003 of notional per unit turnover).
    """
    cost_frac = cost_bps / 10_000
    return (turnover(positions) * cost_frac).sum(axis=1)


def liquidity_tiered_cost_bps(
    volume: pd.DataFrame,
    window_start: str = "2024-07-14",
    tier_bps: tuple = DEFAULT_TIER_BPS,
) -> pd.Series:
    """Assign each asset a one-way cost assumption (bps) by which ADV quartile it
    falls into within the given universe - highest-ADV quartile gets the cheapest
    tier, lowest-ADV quartile the most expensive. See module docstring: these bps
    values are a labeled placeholder, not a measured cost.

    `volume` : (T x N) DataFrame, same shape as data.continuous_curve's volume panel.
    `tier_bps` : 4 values, cheapest tier first (highest ADV -> lowest cost).
    """
    adv = volume.loc[window_start:].mean()
    tier = pd.qcut(adv, q=len(tier_bps), labels=False, duplicates="drop")
    # pd.qcut labels 0 = lowest-ADV bucket; that bucket should get the most expensive
    # (last) bps tier, so index into the reversed list rather than tier_bps directly.
    cost_by_tier = list(reversed(tier_bps))
    return tier.apply(lambda t: cost_by_tier[int(t)])
