"""
portfolio/correlation.py — Rolling and EWMA pairwise correlation between two
return series (two Books' PnL, or two standalone signals' strategy returns).

First pass at "dynamic correlation measurement for signals" (per direct
instruction, 2026-07-23): AQR's "integrate" edge for Value + XSMOM (`references/
AQR - Portfolio Construction Matters.pdf`, see CLAUDE.md's "Mix vs. integrate
(Value + XSMOM)" row) depends on the two being negatively correlated. Whether
that correlation is stable or decaying/flipping over time is exactly what a
static full-sample correlation number can't show. Deliberately simple (plain
rolling-window and EWMA Pearson correlation) rather than DCC-GARCH — the
existing full DCC-GJR discussion in `cleanup.md` section 3 already flags that a
multivariate GARCH fit is an empirical question, not a default, and with only a
handful of Books/signals to compare (not the 38-asset case that discussion was
about), a simple estimator is the right place to start; escalate to DCC-GARCH
(see `src/dcc_garch`, evaluated 2026-07-23) only if this proves insufficient.

Pure functions, no optimizer dependency, same convention as `portfolio/
covariance.py` and every signal module in this project.
"""

import numpy as np
import pandas as pd


def rolling_correlation(return_a: pd.Series, return_b: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Rolling Pearson correlation between two return series, aligned on their
    shared (inner-joined) index. `min_periods` defaults to `window` — no
    partial-window estimate — unless given explicitly."""
    aligned = pd.concat([return_a, return_b], axis=1, join="inner").dropna()
    if min_periods is None:
        min_periods = window
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return a.rolling(window, min_periods=min_periods).corr(b)


def ewma_correlation(return_a: pd.Series, return_b: pd.Series, halflife: float, min_periods: int = 2) -> pd.Series:
    """EWMA (RiskMetrics-style) correlation: EWMA cross-product / sqrt(EWMA
    auto-products), on RAW (not demeaned) returns — matches `portfolio.book.
    Book._apply_vol_target`'s own EWMA-of-realized-PnL convention (not a
    demeaned textbook-Pearson EWMA), for consistency with how this project
    already treats near-zero-mean strategy returns. `halflife` -> `alpha` via
    the identical formula `Book.run()` already uses
    (`1 - exp(-ln(2)/halflife)`), not re-derived differently."""
    aligned = pd.concat([return_a, return_b], axis=1, join="inner").dropna()
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    cov = (a * b).ewm(alpha=alpha, min_periods=min_periods).mean()
    var_a = (a * a).ewm(alpha=alpha, min_periods=min_periods).mean()
    var_b = (b * b).ewm(alpha=alpha, min_periods=min_periods).mean()
    return cov / np.sqrt(var_a * var_b)


def correlation_summary(series: pd.Series) -> pd.Series:
    """Descriptive summary of a rolling/EWMA correlation series — n, mean, std,
    min/max, and % of observations negative (the AQR-relevant question: does
    the negative correlation that makes 'integrate' attractive actually hold up
    over time, or does it spend meaningful time positive)."""
    s = series.dropna()
    if len(s) == 0:
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "pct_negative": np.nan})
    return pd.Series({
        "n": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
        "pct_negative": (s < 0).mean() * 100,
    })
