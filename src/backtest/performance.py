"""
backtest/performance.py — Performance statistics for a backtested return series.

Moved from signal_lib.py (2026-07-21) - see cleanup.md section 2. Extended 2026-07-21
with Sortino/Calmar/Win Rate (dashboard Strategy Performance page, WORKFLOW.md's
"dashboard follow-on" task 3) - a standard CTA tearsheet needs more than
Sharpe/Max DD alone, and this is the second consumer (research/momentum.py's own
train/validation/test comparison table being the first), so this is a real
extension, not a guess at future needs.

min_obs guard added at the same time: some assets (the ICE softs - Cocoa, Coffee,
Cotton, Sugar - term-structure data only starts 2024-07-14, see WORKFLOW.md's known
Databento gaps) have ZERO valid observations in the train period (<= 2019-12-31),
which crashed the original ann_return calc via a 252/len(returns) division by zero.
Same threshold (20) every research/*.py driver script's own former MIN_OBS_FOR_SHARPE
constant used (consolidated into this module's MIN_OBS 2026-07-23 - see simple_sharpe
below), generalized here to the whole stats dict rather than just Sharpe, since every
stat below is equally undefined/noisy with too few observations, not just Sharpe
specifically.

simple_sharpe added 2026-07-23, consolidating a private `_sharpe` helper that had been
copy-pasted identically into both research/tune_all_books.py and backtest/cpcv.py -
deliberately a SEPARATE function from performance_stats above, not a redirect to it:
performance_stats' Sharpe is geometric (compounding annualized return / annualized
vol), the right convention for a standalone signal's own return series; simple_sharpe
is the plain arithmetic mean/std*sqrt(N) convention portfolio.book.Book._compute_pnl
already uses inline for Book-level PnL - matching that existing convention for
Book/Allocator-level diagnostics (tune_all_books.py's grid search, backtest.cpcv's
CSCV selection statistic) rather than silently mixing two different Sharpe
definitions across the same PnL series.
"""

import numpy as np
import pandas as pd

MIN_OBS = 20
_METRICS = ["Ann Return", "Ann Vol", "Sharpe", "Sortino", "Calmar", "Max DD", "Win Rate"]


def simple_sharpe(returns: pd.Series, periods_per_year: int = 252, min_obs: int = MIN_OBS) -> float:
    """Arithmetic-mean/std Sharpe (not compounding) - see module docstring for why
    this is a deliberately separate convention from performance_stats' geometric one."""
    r = returns.dropna()
    if len(r) < min_obs or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def performance_stats(returns, min_obs: int = MIN_OBS):
    returns = returns.dropna()
    if len(returns) < min_obs:
        return pd.Series({m: np.nan for m in _METRICS})

    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = ann_return / downside_dev if downside_dev and downside_dev > 0 else np.nan

    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    win_rate = (returns > 0).mean()

    return pd.Series({
        "Ann Return": ann_return,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_dd,
        "Win Rate": win_rate,
    })
