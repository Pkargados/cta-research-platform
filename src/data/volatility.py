"""
data/volatility.py — Yang-Zhang OHLC volatility estimator.

Ported verbatim (2026-07-21) from notebooks/volatility.ipynb cell 10 - same formula,
same safe_log guard for non-positive prices, just relocated so it's importable from
research/*.py instead of living only inside a notebook (CLAUDE.md: "new research work
happens in research/*.py... not new notebooks").

One difference from the notebook, not a bug: the notebook computed an annualized
version (cell 13) but persisted the non-annualized one to
Data/yang_zhang_features.parquet (cell 16) - already documented as intentional in
DATA_SCHEMA.md §2 ("non-annualized... annualized versions computed on the fly...
not persisted separately"), not a new finding. This module just defaults the other
way (`annualize=True`) since its own callers want annualized vol directly.

Used against Data/continuous_futures.parquet's RAW OHLC (research/momentum.py), not
the back-adjusted series and not the old raw yfinance panel. Three distinct, real
issues found live 2026-07-21 - the first explanation reached for was wrong for most of
the assets it was blamed on, and that's recorded here deliberately rather than quietly
fixed, per this project's own "log real findings including when wrong" convention:

1. Computing this against the BACK-ADJUSTED curve makes some old segments non-
   positive, which this log-based estimator can't handle. VERIFIED (queried
   Data/continuous_futures.parquet directly) to actually affect only HeatingOil,
   RBOB, Brent, WTI Crude, and Oats - energy contracts, where individual roll gaps
   are large in dollar terms relative to the low absolute price levels early in the
   1965-2010-ish history, not the broad "backwardation-prone commodities" (grains,
   meats, softs) originally guessed by analogy to Keynes' "normal backwardation" -
   that guess was checked against adj_close directly and found false (zero negative
   rows for any of those assets, ever). Raw prices avoid this - they stay positive
   everywhere except the one genuine case (WTI's real 2020 print), which safe_log
   below still guards.
2. Separately - and this is what was actually driving the grains/meats/softs group's
   near-100% NaN, not point 1 - the 42-asset panel's date index is the UNION of
   every asset's own trading calendar, so any single asset with a sparser calendar
   than the panel as a whole (e.g. Corn: 849 of 5015 panel dates aren't real CBOT
   trading days - mostly scattered single-day exchange-holiday mismatches, 779
   separate gaps averaging 1.09 days long, not one missing block) shows up as
   scattered NaN cells. A default `rolling(window).var()` (min_periods=window) turns
   almost every window NaN the moment even one such day falls inside it - which
   happens in nearly every window once gaps are this dense. `min_frac` below exists
   for this.
3. The overnight term specifically (`r_o`) is structurally sparser than the other two
   components even after fixing point 2: it needs BOTH today's open AND yesterday's
   close to be valid (roughly double the missingness rate of a single-day term like
   r_c/rs), plus it's roll-masked on top. Applying the same min_periods to all three
   components under-serves r_o - verified live on Cocoa: at its most recent date,
   r_c and rs each had 49/63 valid values (comfortably over a 44 min_periods) but
   r_o had only 34/63, nulling the entire estimate through simple addition even
   though the other two components were fine. `overnight_min_frac` below is r_o's
   own, separately configured (lower) tolerance.

See references/momentum_implementation_recipe.md and WORKFLOW.md for the full trail.
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252


def _safe_log(df: pd.DataFrame) -> pd.DataFrame:
    """Non-positive prices -> NaN before taking logs, rather than raising or
    silently producing complex/inf values. Back-adjusted price series can go
    non-positive in old segments (the accumulated roll-gap shift can push a low
    absolute price level below zero, not just WTI's real 2020 negative print) -
    this guard lets those cells propagate as NaN instead of corrupting the whole
    rolling window they fall into."""
    return np.log(df.where(df > 0))


def yang_zhang_volatility(
    open_prices: pd.DataFrame,
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    window: int = 63,
    annualize: bool = True,
    roll_mask: pd.DataFrame = None,
    min_frac: float = 0.7,
    overnight_min_frac: float = 0.3,
) -> pd.DataFrame:
    """Yang-Zhang volatility estimator (Yang & Zhang, 2000).

    Combines overnight, open-to-close, and Rogers-Satchell range variance
    components, rolled over `window` trading days. Returns annualized volatility
    by default (`annualize=True`); pass False for the raw daily-scale estimate.

    `min_frac` (fraction of `window` required present for the open-to-close and
    Rogers-Satchell components, min_periods=int(window*min_frac)) exists for
    scattered MISSING data - a single asset's own trading calendar being sparser
    than the 42-asset panel's union date index (module docstring point 2). Default
    0.7 tolerates real-world exchange-calendar gap density comfortably while still
    requiring most of the window to be genuine data.

    `overnight_min_frac` is the SAME idea applied to the overnight component alone,
    at its own (lower) default - the overnight term is structurally sparser than the
    other two even under identical raw missingness, since it needs both today's open
    AND yesterday's close to be valid (module docstring point 3). Using min_frac for
    it too would null the whole estimate through simple addition even when the other
    two components are fine - verified live on Cocoa.

    `roll_mask` (bool, same shape/index as the price inputs, True on roll dates) is
    for a different problem - a WRONG value, not a missing one (module docstring
    point 1): on a roll date, the raw curve's "overnight" term (open_t vs.
    close_{t-1}) spans two different contracts, not a real overnight move in one
    held position. When given, that day's overnight return is excluded from the
    rolling variance entirely (masked to NaN, then subject to overnight_min_frac
    like any other missing value) rather than left in to inflate the estimate
    around every roll.
    """
    log_o = _safe_log(open_prices)
    log_h = _safe_log(high_prices)
    log_l = _safe_log(low_prices)
    log_c = _safe_log(close_prices)

    # Overnight returns
    r_o = log_o - log_c.shift(1)
    if roll_mask is not None:
        r_o = r_o.mask(roll_mask)

    # Open-to-close returns
    r_c = log_c - log_o

    # Rogers-Satchell volatility component
    rs = (
        (log_h - log_c) * (log_h - log_o)
        + (log_l - log_c) * (log_l - log_o)
    )

    # Rolling variances - overnight gets its own, more lenient tolerance (see
    # overnight_min_frac docstring above); the other two share min_frac.
    min_periods = max(1, int(window * min_frac))
    overnight_min_periods = max(1, int(window * overnight_min_frac))
    sigma_o2 = r_o.rolling(window, min_periods=overnight_min_periods).var()
    sigma_c2 = r_c.rolling(window, min_periods=min_periods).var()
    sigma_rs = rs.rolling(window, min_periods=min_periods).mean()

    # Yang-Zhang weighting parameter
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Yang-Zhang variance
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs

    yz_vol = np.sqrt(yz_var)
    if annualize:
        yz_vol = yz_vol * np.sqrt(ANNUALIZATION)
    return yz_vol
