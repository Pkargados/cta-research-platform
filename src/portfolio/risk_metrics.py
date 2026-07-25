"""
portfolio/risk_metrics.py — Historical (empirical) VaR and Expected Shortfall.

First risk-calculation pass (Phase 8), per direct instruction: start with the
COMBINED Allocator-level portfolio only (not per-Book), historical/empirical
method (not parametric-Gaussian — CTA return distributions aren't normal: trend
tends toward positive skew, carry/reversal toward negative skew and fat tails, so
assuming normality here would be exactly the kind of unvalidated shortcut this
project has avoided everywhere else), 95% confidence.

Sign convention: both VaR and ES are reported as NEGATIVE numbers (a loss),
matching this project's own `max_dd` convention elsewhere (`backtest.performance`,
`portfolio.book`) rather than reporting a positive "loss magnitude" — one fewer
sign flip to get wrong when reading a table next to Max DD.

Point-in-time window choice — expanding, not rolling: `portfolio.book.Book.run()`
produces one PnL value per REBALANCE date (monthly here, `COV_FREQ="ME"`), not
daily — already the exact periodicity trap `research/portfolio.py`'s own
`_period_stats` docstring warns about elsewhere in this project. The Allocator's
combined pnl therefore has only ~120-150 monthly observations total. A fixed
ROLLING window long enough to be meaningful (36-60 months) would leave a 95% tail
estimate just 2-3 effective observations wide — too noisy to trust, and the kind
of thing this project labels rather than hides. An EXPANDING window (all history
to date, gated by `min_periods`) is the more defensible choice for a short,
monthly-cadence series — the same "rolling OR expanding window" allowance
CLAUDE.md's Rule 2 already makes for exactly this situation. `min_periods=24` (2
years) is a labeled default, not a validated one: even at 24 observations, a 5%
tail is only ~1.2 expected observations — genuinely thin, stated here plainly
rather than implied to be more precise than it is. More history mitigates this as
the series grows; it does not eliminate the small-sample noise for early dates.
"""

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical VaR: the (1 - confidence) quantile of the return distribution
    (e.g. the 5th percentile at 95% confidence) — a NEGATIVE number (see module
    docstring). No distributional assumption; purely the observed empirical
    quantile of whatever sample is passed in.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    return float(returns.quantile(1.0 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): the mean of returns at or beyond the VaR
    threshold — the average loss IN the tail, not just the threshold itself.
    Always <= VaR in loss terms (more negative), since it averages the tail
    rather than reporting its boundary.
    """
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def expanding_var(returns: pd.Series, confidence: float = 0.95, min_periods: int = 24) -> pd.Series:
    """Point-in-time expanding-window historical VaR — at each date, uses ONLY
    `returns` observed up to and including that date (no look-ahead), gated by
    `min_periods`. See module docstring for why expanding, not rolling, and why
    `min_periods=24` is a labeled default, not a validated one.
    """
    returns = returns.dropna()
    out = {}
    for i, date in enumerate(returns.index):
        if i + 1 < min_periods:
            out[date] = np.nan
            continue
        out[date] = historical_var(returns.iloc[: i + 1], confidence)
    return pd.Series(out, index=returns.index)


def expanding_expected_shortfall(returns: pd.Series, confidence: float = 0.95, min_periods: int = 24) -> pd.Series:
    """Point-in-time expanding-window historical Expected Shortfall — same
    windowing discipline as `expanding_var`.
    """
    returns = returns.dropna()
    out = {}
    for i, date in enumerate(returns.index):
        if i + 1 < min_periods:
            out[date] = np.nan
            continue
        out[date] = expected_shortfall(returns.iloc[: i + 1], confidence)
    return pd.Series(out, index=returns.index)


def expanding_var_and_es(returns: pd.Series, confidence: float = 0.95, min_periods: int = 24) -> pd.DataFrame:
    """Both `expanding_var` and `expanding_expected_shortfall` in a single pass.

    Added 2026-07-23: every real caller (`research/portfolio.py`, `dashboard/
    pages/13_portfolio_performance.py`) already wants both series on the same
    `returns`/`confidence`/`min_periods`, computed independently via the two
    functions above — which each re-run their own expanding quantile
    computation at every date, so calling both duplicates the VaR quantile
    calculation for every date in the series (ES's own `expected_shortfall`
    call internally recomputes `historical_var` to find the tail threshold).
    This function computes each date's quantile ONCE and reuses it directly
    for the tail mean. `expanding_var`/`expanding_expected_shortfall` are
    UNCHANGED and still the right choice for a caller that only wants one of
    the two series on its own.

    Returns a DataFrame with columns "var" and "es", same index as `returns`.
    """
    returns = returns.dropna()
    var_out, es_out = {}, {}
    for i, date in enumerate(returns.index):
        if i + 1 < min_periods:
            var_out[date], es_out[date] = np.nan, np.nan
            continue
        window = returns.iloc[: i + 1]
        var = historical_var(window, confidence)
        tail = window[window <= var]
        var_out[date] = var
        es_out[date] = float(tail.mean()) if len(tail) > 0 else var
    return pd.DataFrame({"var": pd.Series(var_out), "es": pd.Series(es_out)}, index=returns.index)
