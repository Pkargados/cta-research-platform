"""
signals/transforms.py — Shared, reusable signal transforms.

Two generations of transforms live here:

1. Legacy pre-rebuild transforms (`binary_signal`, `continuous_signal`,
   `rank_signal`) — the original feature_engineering.ipynb comparison that led to
   CLAUDE.md Rule 5's finding ("binary momentum signals are a disaster" — binary
   signals underperform continuous, vol-scaled ones). Kept only so
   `signal_lib.py`'s backward-compat import surface still resolves (it re-exports
   these names even though only `load_price_data`/`get_liquid_universe` are still
   actually consumed, by the untouched `Time_Series_Models.ipynb`).
2. Current transforms used by every rebuilt signal family:
   - `vol_targeted_sign_signal` — the actual TSMOM position-sizing rule
     (Moskowitz-Ooi-Pedersen 2012 Eq. 5): `target_vol * sign(x) / vol`. Generic
     over any per-asset time-series signal (momentum, breakout, crossover), not
     momentum-specific — each caller passes its own raw signal and vol estimate.
   - `cross_sectional_rank` — sector-scoped rank-demean (`rank(S_i) - mean_rank`),
     the construction shared by carry's Eq. 19, XSMOM's Eq. 1, and value (all three
     AQR/KMPV papers use an algebraically identical rank-demean). Carry's own
     `cross_sectional_carry_signal` is a thin wrapper over this (CLAUDE.md Rule 6:
     extract once a second consumer exists — XSMOM was that second consumer).

Pure functions, no optimizer dependency (CLAUDE.md Architecture section).
"""

import numpy as np
import pandas as pd


def binary_signal(momentum: pd.DataFrame) -> pd.DataFrame:
    """+1/-1/0 sign of a raw momentum/return DataFrame — no vol-scaling, no
    magnitude information. The transform CLAUDE.md Rule 5 documents as
    underperforming continuous, vol-scaled signals in this universe."""
    return np.sign(momentum)


def continuous_signal(momentum: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Momentum scaled by inverse volatility, magnitude preserved (not just sign) —
    the continuous alternative to `binary_signal` compared in
    feature_engineering.ipynb. Not the same construction as
    `vol_targeted_sign_signal` (which discards magnitude and rescales to a target
    vol) — this preserves the raw signal's relative magnitude across assets/dates,
    only using vol to make that magnitude comparable across differently-volatile
    instruments."""
    return momentum / vol


def rank_signal(momentum: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of momentum across the full universe on each
    date, centered at zero (`rank(pct=True) - 0.5`) — the cross-sectional-momentum
    style transform `signals.momentum.build_momentum_features`'s skip-month
    convention was originally paired with. Not sector-scoped (unlike
    `cross_sectional_rank` below) — this is the legacy, pre-sector-grouping
    version."""
    pct_rank = momentum.rank(axis=1, pct=True)
    return pct_rank - 0.5


def vol_targeted_sign_signal(raw_signal: pd.DataFrame, vol: pd.DataFrame, target_vol: float = 1.0) -> pd.DataFrame:
    """`target_vol * sign(raw_signal) / vol` — Moskowitz-Ooi-Pedersen (2012) Eq. 5's
    exact position-sizing rule, generalized to any per-asset time-series signal
    (momentum, breakout, crossover), not just TSMOM.

    `target_vol` defaults to a neutral 1.0 since this is a generic transform reused
    across signal families with different target-vol conventions
    (`research/momentum.py` passes the paper's own 0.40). Dropping this constant
    entirely would be fine for a gross-exposure-normalized POOLED backtest (Sharpe
    is scale-invariant there — it cancels in `backtest.engine.normalized_positions`'
    own per-date renormalization), but NOT for an unnormalized per-asset diagnostic
    view, where it directly sets each asset's implied leverage (CLAUDE.md's
    momentum row: SwissFranc's ~9% annualized vol implied 11-26x leverage without
    this constant, corrupting one real high-vol day into a mathematically
    impossible <-100% return). Always pass the real target_vol explicitly rather
    than relying on the neutral default for anything but a scale-invariant
    (already-normalized) consumer.

    NaN-safe: `sign(NaN)` and `NaN / vol` both propagate NaN rather than raising —
    a missing raw_signal or vol value produces a NaN position for that
    (date, asset), not a crash or a silently-wrong zero.
    """
    return target_vol * np.sign(raw_signal) / vol


def cross_sectional_rank(scores: pd.DataFrame, sectors: dict, min_group_size: int = 2) -> pd.DataFrame:
    """Sector-scoped rank-demean: `rank(S_i) - mean_rank` within each sector, per
    date — the construction shared algebraically by carry (Koijen-Moskowitz-
    Pedersen-Vrugt 2018 Eq. 19: `rank(C_i) - (N_t+1)/2`) and XSMOM (Asness-
    Moskowitz-Pedersen 2013 Eq. 1), extracted here once XSMOM became a second
    consumer of the same rank-demean shape (CLAUDE.md Rule 6).

    Ranking (rather than raw-magnitude demeaning) insulates weights from outliers —
    the paper's own stated reason (Koijen et al., footnote to Eq. 19): a raw
    cross-sectional demean lets whichever asset has the single most extreme score
    dominate its sector's whole trade, the same failure mode short-term reversal's
    build already learned to avoid once (`signals.short_term_reversal.
    cross_sectional_demean`, a DIFFERENT, non-rank, sign-flipped construction built
    before this shared function existed — not redirected to this one, since
    reversal's bet direction and demean convention are genuinely different, see
    that module).

    `sectors` : dict[str, list[str]] — sector name -> member asset list (e.g.
    `data.sectors.SECTORS`, restricted to the active universe via
    `data.sectors.sectors_for_universe`).

    `min_group_size` : a sector with fewer than this many assets WITH a real
    (non-NaN) score on a given date produces NaN for every member that date —
    ranking against a peer group of size 0-1 has no meaningful interpretation
    (`data.sectors`'s own documented IndustrialMetals-has-only-Copper case).

    Assets not present in any `sectors` group are not touched (their column, if
    present in `scores`, is dropped from the result) — this function only ever
    scores assets it can assign a real peer group to.
    """
    result = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns, dtype=float)

    for sector, members in sectors.items():
        present_members = [m for m in members if m in scores.columns]
        if len(present_members) < min_group_size:
            continue

        group = scores[present_members]
        valid_count = group.notna().sum(axis=1)
        rank = group.rank(axis=1)
        mean_rank = (valid_count + 1) / 2.0
        demeaned = rank.sub(mean_rank, axis=0)
        demeaned = demeaned.where(valid_count.ge(min_group_size), other=np.nan)

        result[present_members] = demeaned

    return result[[c for c in scores.columns if any(c in members for members in sectors.values())]]
