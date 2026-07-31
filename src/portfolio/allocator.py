"""
portfolio/allocator.py — Combines Books; applies regime conditioning before the optimizer.

Adapted from the retired Cross Asset Stat Arb Engine's Allocator
(engine/portfolio/allocator.py) — see cleanup.md section 3. The original was already
minimal (run each active book, sum PnL, no regime logic) and transfers essentially
as-is; the one addition here is `regime_lookup`, an optional callable applied to each
book's alpha *before* that book solves its weights.

Design constraints (invariants)
--------------------------------
- Regime conditioning happens before the optimizer runs, never after — vol targeting
  silently cancels out any post-solve position scaling (CLAUDE.md Architecture
  section; this is the one lesson from the retired project's regime layer that
  transfers unconditionally, independent of what the regime classifier measures).
- Books are run independently; the Allocator does not mutate Book state beyond the
  alpha_df swap needed to apply regime conditioning for that run.
- Inactive books (book.is_active == False, or deactivated by regime_lookup for a given
  book) are skipped silently.
- PnL combination uses pandas .add(fill_value=0.0) to handle index mismatches between
  books with different date coverage.
- With a single active book and no regime_lookup, combined_pnl is identical to
  book.run()["pnl"].
- `book_weights` (added 2026-07-30, WORKFLOW.md decision #12): an optional
  {book.name: float} multiplier applied to each book's own "pnl" (and
  "asset_contributions", if present) BEFORE the `.add(fill_value=0.0)`
  combination — same mechanic `research/sleeve_risk_parity.py`'s own
  `combine_static` already validated on real Trend/Carry Book PnL. A book
  not named in `book_weights` defaults to 1.0, so the default `book_weights=
  None` (-> {}) preserves the exact prior equal-sum behavior for every
  existing caller. This is a STATIC weight, decided once by the caller
  (e.g. `portfolio.risk_parity.risk_parity_weights` fit on train-period
  sleeve PnL) — the Allocator itself has no covariance-estimation logic,
  same architectural boundary `regime_lookup` already establishes (content/
  computation lives outside this module, only the injection point does).
"""

import pandas as pd


class Allocator:
    """
    Combines a list of Books, optionally conditioning each on a regime signal
    and/or weighting each Book's PnL contribution.

    Parameters
    ----------
    books         : list[Book]
    regime_lookup : callable | None
        Optional `date -> {book_name: {"active": bool, "alpha_multiplier": float}}`.
        See src/regime/ for the interface this is expected to satisfy — content
        (what regime, how it's classified) is deliberately not this module's concern.
    book_weights  : dict[str, float] | None
        Optional {book.name: weight} multiplier on each book's PnL contribution
        before combining — see module docstring. Missing names default to 1.0.
    """

    def __init__(self, books, regime_lookup=None, book_weights=None):
        self.books = books
        self.regime_lookup = regime_lookup
        self.book_weights = book_weights or {}

    def run(self, returns_df: pd.DataFrame) -> dict:
        """
        Run all active books (regime-conditioned if regime_lookup is set) and combine
        their PnL by weighted addition (weight defaults to 1.0 — see book_weights).

        Returns
        -------
        dict with:
            "pnl"                 : pd.Series — weighted combined PnL across active books
            "book_results"        : dict[str, dict] — {book.name: result} for each active book
            "asset_contributions" : pd.DataFrame | None — weighted combined per-asset
                                     attribution, only if every active book's own result
                                     includes it (see Book._compute_pnl's own docstring)
        """
        book_results = {}

        for book in self.books:
            if not book.is_active:
                continue

            run_book = book
            if self.regime_lookup is not None:
                run_book, keep_active = self._apply_regime(book)
                if not keep_active:
                    continue

            res = run_book.run(returns_df)
            book_results[book.name] = res

        combined_pnl = None
        combined_contributions = None
        for name, res in book_results.items():
            weight = self.book_weights.get(name, 1.0)
            pnl = res["pnl"] * weight
            combined_pnl = pnl.copy() if combined_pnl is None else combined_pnl.add(pnl, fill_value=0.0)

            contributions = res.get("asset_contributions")
            if contributions is not None:
                weighted_contributions = contributions * weight
                combined_contributions = (
                    weighted_contributions.copy() if combined_contributions is None
                    else combined_contributions.add(weighted_contributions, fill_value=0.0)
                )

        return {"pnl": combined_pnl, "book_results": book_results, "asset_contributions": combined_contributions}

    def _apply_regime(self, book):
        """Scale a book's alpha by its regime multiplier for every date in its
        index, applied before that book's optimizer ever sees it. Returns a
        shallow-copied book (original is never mutated) and whether it stays active.
        """
        import copy

        multipliers = pd.Series(
            {date: self.regime_lookup(date).get(book.name, {"active": True, "alpha_multiplier": 1.0})
             for date in book.alpha_df.index}
        )
        active_mask = multipliers.apply(lambda a: a["active"])
        if not active_mask.any():
            return book, False

        scaled_alpha = book.alpha_df.multiply(
            multipliers.apply(lambda a: a["alpha_multiplier"]), axis=0
        )
        scaled_alpha = scaled_alpha.loc[active_mask]

        run_book = copy.copy(book)
        run_book.alpha_df = scaled_alpha
        return run_book, True
