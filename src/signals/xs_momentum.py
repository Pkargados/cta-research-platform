"""
signals/xs_momentum.py — Cross-sectional momentum (XSMOM), Asness-Moskowitz-
Pedersen 2013 "Value and Momentum Everywhere" (`references/Value and Momentum
Everywhere.pdf`, read directly).

Per direct instruction — a deliberate scope expansion, not one of this project's
original six signal families. Added as this project's second consumer of the
rank-weighted cross-sectional construction (Eq. 1 of the paper: `rank(S_i) -
mean_rank(S)`, scaled to $1 long/$1 short), reused here via `signals.transforms.
cross_sectional_rank` — the same shared function carry's `carry1m_signal` wraps
(CLAUDE.md Rule 6: extract once a second consumer exists).

MOM2-12 (the paper's own momentum measure, Section I.B): the past 12-month
cumulative return, skipping the most recent month — computed on month-end-
resampled BACK-ADJUSTED close (the paper's own convention is monthly data
throughout; skipping the most recent month is "standard in the momentum
literature... to avoid the 1-month reversal in stock returns" — matched here even
though the paper itself notes this isn't strictly necessary for liquid futures,
"to maintain uniformity across asset classes," the same reason the paper gives).

No vol-scaling (the paper's own headline construction has none). Rank-weighted
WITHIN SECTOR rather than across the paper's full cross-section — this project's
universe spans FX/rates/commodities/equities, so comparing AUDUSD's momentum
against Corn's has no clean interpretation, the same reasoning `cross_sectional_
rank`'s own docstring gives.

ONE Book, not a multi-horizon grid — the paper is explicit that it deliberately
avoids testing several lookbacks "to minimize the pernicious effects of data
snooping" (Section I.B footnote), unlike TSMOM/breakout/crossover/reversal's own
multi-spec Book counts.
"""

import numpy as np
import pandas as pd

from signals.transforms import cross_sectional_rank

HEADLINE_LOOKBACK_MONTHS = 12
HEADLINE_SKIP_MONTHS = 1


def mom2_12(close: pd.DataFrame, lookback_months: int = HEADLINE_LOOKBACK_MONTHS, skip_months: int = HEADLINE_SKIP_MONTHS) -> pd.DataFrame:
    """MOM2-12: month-end resampled, skip the most recent month, then take the
    trailing `lookback_months`-month cumulative return."""
    monthly = close.resample("ME").last()
    base = monthly.shift(skip_months)
    raw = base / base.shift(lookback_months) - 1.0
    return raw.reindex(close.index).ffill()


def xs_momentum_signal(close: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """The paper's Eq. 1 rank-weighted cross-sectional momentum signal, sector-
    scoped, no vol-scaling. Monthly rebalancing is a backtest-layer choice
    (`backtest.engine`'s own monthly resampling), not baked in here."""
    raw = mom2_12(close)
    return cross_sectional_rank(raw, sectors)
