"""
regime/interface.py — Regime -> book-action lookup interface.

This is scaffolding, not a regime classifier. The interface itself is what transfers
from the retired Cross Asset Stat Arb Engine's regime_mapping.py (see cleanup.md
section 3): a pure lookup that finds the most recent regime label as-of a date (no
look-ahead) and resolves it to per-book actions via a caller-supplied mapping
function. What does NOT transfer is the content — that engine's specific regimes
(normal/clustered/crowded/crisis/broken) were DCC-GJR correlation-spike-based and
assumed crisis hurts the strategy, which is backwards for trend-following's
documented crisis-alpha property (CLAUDE.md Rule 7).

The regime classifier that actually produces `regime_df` and the `action_fn` decision
table below is a separate research task, not defined here. This project already has
the inputs a macro-driven classifier would want and uses none of them yet — yield
curve, GSCPI, fed funds, trade policy uncertainty, all in Data/, all point-in-time-
correct via macro_point_in_time.py (DATA_SCHEMA.md section 3).

Usage
-----
    regime_lookup = lambda date: get_actions_for_date(
        regime_df, date, my_action_fn, book_names=[b.name for b in books]
    )
    allocator = Allocator(books, regime_lookup=regime_lookup)
"""

import pandas as pd


def get_actions_for_date(regime_df: pd.DataFrame, date, action_fn, book_names, default_multiplier=1.0):
    """
    Look up the most recent regime label on or before `date`, then resolve it to
    per-book actions via `action_fn`.

    No look-ahead: only rows with index <= date are considered. Applied to a book's
    alpha *before* that book's optimizer runs (never post-solve — vol targeting
    silently cancels out any post-solve position scaling).

    Parameters
    ----------
    regime_df  : pd.DataFrame — must have a "regime" column of string labels, and a
                                 DatetimeIndex
    date       : date-like
    action_fn  : callable — regime_label (str) -> {book_name: {"active": bool,
                             "alpha_multiplier": float}}. This is where an actual
                             regime classifier's decision table lives (e.g. a macro
                             growth/inflation-quadrant -> book-action mapping) — not
                             this module's job to define.
    book_names : list[str] — guarantees every book has an entry in the result, even
                              if action_fn's mapping doesn't mention it (e.g. a book
                              added after the mapping was last updated)
    default_multiplier : float — used for the neutral fallback (no regime history
                              yet, or an unrecognized/NaN label — most commonly during
                              a classifier's own warmup period)

    Returns
    -------
    dict — {book_name: {"active": bool, "alpha_multiplier": float}}
    """
    neutral = {name: {"active": True, "alpha_multiplier": default_multiplier} for name in book_names}

    available = regime_df.loc[:date]
    if len(available) == 0:
        return neutral

    label = available["regime"].iloc[-1]
    if not isinstance(label, str) or pd.isna(label):
        return neutral

    actions = action_fn(label)
    return {name: actions.get(name, neutral[name]) for name in book_names}
