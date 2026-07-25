"""
portfolio/covariance.py — Rolling Ledoit-Wolf shrinkage covariance, keyed by real
rebalance dates.

Closes a real gap: `cleanup.md` section 3 already evaluated "Ledoit-Wolf rolling
covariance" as the design decision for this project's optimizer input, but no such
estimator was ever built as a reusable function anywhere in `src/` — `Book` has
always taken a pre-built `cov_dict` as an external constructor argument. The retired
Cross Asset Stat Arb Engine has the real math (`sklearn.covariance.LedoitWolf`, a
rolling window, point-in-time correct: `returns.loc[:date].iloc[-window:]`) but only
ever inlined it directly in two driver scripts (`run_baseline.py`/`run_engine.py`),
never factored into a standalone function — extracted here, not copy-pasted, since
it never existed as a file to copy.

Window/frequency chosen for *this* project's own conventions, not the retired
engine's (120-day/weekly): a 252-trading-day (~1yr) rolling window recomputed at
month-end, consistent with this project's other "about a year" windows (Yang-Zhang's
own multi-horizon set, carry timing's `min_periods`) and its own monthly rebalance
convention for most signal families. A reasonable default, not empirically tuned.

Real trading-day keys, not calendar labels — a correctness detail the retired
engine's own weekly usage mostly avoided by luck (Friday market holidays are rare)
but which matters here: `returns_df.resample(freq).last()`'s own index uses CALENDAR
period-end labels (e.g. every "2024-01-31," whether or not that's an actual trading
day) — a large fraction of month-ends fall on a weekend, so using those labels
directly as dict keys would frequently fail to intersect with any signal's own real
trading-day index (`Book.run()`'s own rebalance-date intersection logic expects an
exact match). Fixed by resampling the INDEX ITSELF (not the returns values) — each
period's ".last()" then genuinely is the last REAL trading day on or before that
period boundary, not a phantom calendar date.
"""

from collections import OrderedDict

import pandas as pd
from sklearn.covariance import LedoitWolf

DEFAULT_WINDOW = 252  # ~1yr trading days
DEFAULT_FREQ = "ME"  # month-end - matches this project's own monthly rebalance convention

# Same tolerance and rationale as data.volatility's min_frac / signals.breakout's
# DEFAULT_MIN_FRAC: this project's multi-decade, multi-asset panel has scattered
# per-asset calendar gaps (different exchanges' holidays, different listing start
# dates), which show up as scattered NaN rows within any rolling window. sklearn's
# LedoitWolf has no native NaN handling at all (raises), unlike this project's own
# rolling-window functions (which tolerate a NaN fraction via min_periods) - so NaN
# rows are dropped from the window before fitting, gated by this same
# already-validated 0.7 tolerance rather than a fresh guess.
DEFAULT_MIN_FRAC = 0.7


def real_period_end_dates(index: pd.DatetimeIndex, freq: str = DEFAULT_FREQ) -> pd.DatetimeIndex:
    """The last REAL date in `index` on or before each period boundary (`freq`) -
    NOT the calendar period-end label itself, which may not be an actual trading day
    (most month-ends aren't business days). See module docstring."""
    return pd.DatetimeIndex(pd.Series(index, index=index).resample(freq).last().dropna().values)


def build_cov_dict(returns_df: pd.DataFrame, window: int = DEFAULT_WINDOW, freq: str = DEFAULT_FREQ,
                    min_frac: float = DEFAULT_MIN_FRAC) -> OrderedDict:
    """Rolling Ledoit-Wolf shrinkage covariance, one matrix per real rebalance date
    (see `real_period_end_dates`), each fit on the trailing `window` trading days up
    to and including that date - point-in-time correct by construction (no future
    data enters any single date's estimate). Dates with fewer than `window` prior
    observations are skipped, not padded or backfilled (real warmup, not fabricated).

    Rows with ANY NaN within the window are dropped before fitting (sklearn's
    LedoitWolf has no native NaN tolerance) - a date is skipped only if fewer than
    `window * min_frac` clean rows remain, not merely because some scattered gap
    exists (see module docstring / DEFAULT_MIN_FRAC).

    Returns an OrderedDict {date: pd.DataFrame(N x N, index/columns = returns_df.
    columns)} - the exact shape `portfolio.book.Book`'s `cov_dict` constructor
    argument expects.
    """
    min_rows = max(2, int(window * min_frac))
    reb_dates = real_period_end_dates(returns_df.index, freq)
    cov_dict = OrderedDict()
    for date in reb_dates:
        window_data = returns_df.loc[:date].iloc[-window:]
        if len(window_data) < window:
            continue
        clean = window_data.dropna(how="any")
        if len(clean) < min_rows:
            continue
        lw = LedoitWolf().fit(clean.values)
        cov_dict[date] = pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns)
    return cov_dict
