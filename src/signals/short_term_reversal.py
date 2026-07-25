"""
signals/short_term_reversal.py — Cross-sectional short-term reversal, Lehmann (1990)
mechanics with a sector-scoped peer group (`references/Fads_Martingales_and_Market_
Efficiency Short Term Reversal vol1.pdf`, read directly), a VIX-conditional sizing
overlay (Nagel 2011, "Evaporating Liquidity" — `references/Evaporating Liquidity Short
Term Reversal vol 2.pdf`), and design choices cross-checked against Blitz-van der
Grient-Honarvar (2023) "Reversing the Trend of Short-Term Reversal"
(`references/Reversing the Trend Short Term Reversal vol3.pdf`).

The first genuinely cross-sectional signal in this project — an asset's score depends
on its return relative to its own SECTOR peer group (`data.sectors.SECTORS`), not the
full 42-name universe (comparing AUDUSD against Corn has no clean interpretation,
unlike Lehmann's single-equity-market cross-section) and not on its own history alone
(CLAUDE.md Rule 7: re-derived, not ported from momentum/breakout/crossover's
time-series shape).

Lehmann's own construction (Eq. 1 of the recipe): weight_i = -(R_i - R̄) — go long
recent losers, short recent winners, relative to the peer-group mean return. This
module reproduces that exactly via `cross_sectional_demean` (raw demean, NOT rank —
unlike `signals.transforms.cross_sectional_rank`, built later for carry/XSMOM/value;
this module's demean predates that extraction and uses a different, un-ranked,
explicitly sign-flipped convention, so it is not a thin wrapper over that function).

Two tiers, mirroring Nagel's own individual-stock vs. industry-portfolio comparison
(his industry-level reversal strategy is unconditionally near-zero but still earns a
real, VIX-conditional premium):
- individual: vol-standardized lag-day return, demeaned WITHIN each sector, sign-flipped.
- sector: sector-level equal-weighted composite of the same vol-standardized returns,
  demeaned ACROSS sectors, sign-flipped, then broadcast back to every member asset.

Three parallel lags (1/5/10 days), no headline pick — same discipline as momentum/
breakout/crossover's own multi-spec reporting. Confirmed against Blitz et al. (2023):
shorter lookbacks (they test 5/10/15 trading days vs. a 1-month baseline) strengthen
the reversal effect, consistent with including 1d/5d as parallel specs here rather
than only a slower one.
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252
LAGS = (1, 5, 10)


def lag_return(close: pd.DataFrame, lag: int) -> pd.DataFrame:
    return close / close.shift(lag) - 1.0


def vol_standardized_return(close: pd.DataFrame, vol: pd.DataFrame, lag: int) -> pd.DataFrame:
    """`lag`-day return divided by the `lag`-day-scale vol implied by an ANNUALIZED
    vol estimate (`vol / sqrt(252) * sqrt(lag)`) — puts assets with very different
    volatility levels on a comparable scale before cross-sectional demeaning, the
    same reasoning `signals.transforms.vol_targeted_sign_signal` gives for scaling by
    vol generally, applied here to the raw return itself rather than to position size.
    """
    daily_vol = vol / np.sqrt(ANNUALIZATION)
    lag_vol = daily_vol * np.sqrt(lag)
    return lag_return(close, lag) / lag_vol


def cross_sectional_demean(scores: pd.DataFrame, sectors: dict, min_group_size: int = 2) -> pd.DataFrame:
    """Raw (non-rank) demean within each sector: `score_i - mean(score)` over that
    sector's present members on a given date. Deliberately NOT
    `signals.transforms.cross_sectional_rank` — that function insulates against
    outliers via ranking, which is the right choice for carry/XSMOM/value's slower-
    moving, sector-pooled scores, but this module predates that extraction and uses
    Lehmann's own raw-magnitude construction directly (Eq. 1 of the recipe). No sign
    flip here — callers apply that themselves, since the sign convention differs by
    tier (individual vs. sector) in this module and isn't a fixed property of the
    demean step itself.

    A sector with fewer than `min_group_size` members present on a date produces NaN
    for every member that date (`data.sectors`'s own documented IndustrialMetals-has-
    only-Copper case) — no peer group, no meaningful demean.
    """
    result = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns, dtype=float)
    for sector, members in sectors.items():
        present_members = [m for m in members if m in scores.columns]
        if len(present_members) < min_group_size:
            continue
        group = scores[present_members]
        valid_count = group.notna().sum(axis=1)
        demeaned = group.sub(group.mean(axis=1), axis=0)
        demeaned = demeaned.where(valid_count.ge(min_group_size), other=np.nan)
        result[present_members] = demeaned
    covered = [c for c in scores.columns if any(c in members for members in sectors.values())]
    return result[covered]


def individual_reversal_signal(close: pd.DataFrame, vol: pd.DataFrame, sectors: dict, lag: int) -> pd.DataFrame:
    """Individual-asset tier: bet against each asset's own recent return relative to
    its sector peers (Lehmann 1990's direct construction, sector-scoped)."""
    raw = vol_standardized_return(close, vol, lag)
    demeaned = cross_sectional_demean(raw, sectors)
    return -demeaned


def sector_average_return(vol_standardized_returns: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Equal-weighted composite of each sector's present members' vol-standardized
    returns — one column per sector, Nagel's own industry-portfolio construction."""
    result = {}
    for sector, members in sectors.items():
        present = [m for m in members if m in vol_standardized_returns.columns]
        if not present:
            continue
        result[sector] = vol_standardized_returns[present].mean(axis=1)
    return pd.DataFrame(result)


def sector_reversal_signal(close: pd.DataFrame, vol: pd.DataFrame, sectors: dict, lag: int) -> pd.DataFrame:
    """Sector-level tier: bet against each SECTOR's recent composite return relative
    to the other sectors, broadcast to every member asset (Nagel's industry-reversal
    strategy, applied at this project's sector granularity rather than GICS
    industries)."""
    raw = vol_standardized_return(close, vol, lag)
    sector_avg = sector_average_return(raw, sectors)
    if sector_avg.shape[1] < 2:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    demeaned_sectors = sector_avg.sub(sector_avg.mean(axis=1), axis=0)
    sector_score = -demeaned_sectors

    broadcast = {}
    for sector, members in sectors.items():
        if sector not in sector_score.columns:
            continue
        for m in members:
            if m in close.columns:
                broadcast[m] = sector_score[sector]
    return pd.DataFrame(broadcast)


def build_all_reversal_signals(close: pd.DataFrame, vol: pd.DataFrame, sectors: dict, lags=LAGS) -> dict:
    """All 6 parallel Books' worth of raw signals: {tier}_{lag}d -> (T x N) DataFrame.
    No headline pick, matching this signal family's own logged discipline."""
    signals = {}
    for lag in lags:
        signals[f"individual_{lag}d"] = individual_reversal_signal(close, vol, sectors, lag)
        signals[f"sector_{lag}d"] = sector_reversal_signal(close, vol, sectors, lag)
    return signals
