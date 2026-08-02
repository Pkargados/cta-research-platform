"""
research/relative_value_book.py — Pool the Relative Value sleeve into ONE
"Relative Value" Book (WORKFLOW.md §11d, build-order step 5), reusing
`single_strategy_portfolios.build_book`'s calibration directly (GAMMA/KAPPA/
LAMBD/TARGET_VOL_BOOK/MAX_WEIGHT/COV_WINDOW/COV_FREQ/EWMA_HALFLIFE) rather than
re-deriving Book hyperparameters from scratch — CLAUDE.md Rule 6, and the exact
reuse the build handoff called for.

**Per-pair construction, one column of `alpha_df` per pair (7 candidate
columns)**: hedge-ratio method already settled in `research/relative_value.py`
(Kalman won both Corn-Wheat and RBOB-HeatingOil); entry/exit sizing rule
(continuous vs. threshold) picked per pair on VALIDATION NET Sharpe AT WEEKLY
FREQUENCY specifically — reusing `relative_value.py`'s own already-computed
standalone table (not recomputed here), and NET (not gross), since this
sleeve's central problem is turnover/cost, unlike Trend/Carry's own
gross-selected flavor bake-off where cost wasn't the dominant driver at
selection time. This mirrors the exact precedent `single_strategy_portfolios.
py`'s own Trend/Carry flavor bake-off already set (a small number of
economically-motivated constructions per Book, explicitly NOT the same
multiple-comparisons risk the numerical grid-search tuning work flagged) —
picking 1-of-2 constructions for each of 7 pairs is the same shape of decision,
not a new one.

**No separate "daily vs weekly" choice at the Book level**: `portfolio.book.
Book.run()` reads `alpha_df.loc[date]` only at `cov_dict`'s own rebalance
dates (COV_FREQ="W-FRI", matching `single_strategy_portfolios.py`'s
established weekly-cadence convention) — feeding it the raw, un-resampled
daily signal already has the same effect as `backtest.engine.
normalized_positions`'s own weekly resampling (both just read each week's
Friday value). The `research/relative_value.py` daily-vs-weekly comparison
already showed weekly reduces turnover substantially without hurting most
pairs' Sharpe — consistent with, not contradicted by, the Book's own
already-weekly default.

**`_active_columns`'s 90% valid-return-density gate (reused verbatim from
`single_strategy_portfolios.py`, not modified) is expected to exclude some
pairs** — `spread_return` (a diff of two already-gappy legs) is structurally
sparser than a real asset's own return series (checked directly: wti_brent
92%, gold_silver 96%, rbob_heatingoil/crack_spread ~99.6% — all clear the 90%
gate; platinum_palladium 87%, corn_wheat 68%, wheat_kcwheat 50% — all fail
it). This is a genuine cross-commodity trading-calendar sparsity finding
(Corn/Wheat/KC_Wheat/Platinum/Palladium don't trade on exactly the same days),
not a construction bug — reported honestly, the gate is NOT loosened just to
force more pairs in (same "label it, don't force it" discipline as every
other documented coverage gap in this project, e.g. Value's ICE-softs gap).

Run: `python research/relative_value_book.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import relative_value as rv
import single_strategy_portfolios as ssp

from signals.relative_value import ALL_PAIR_NAMES, build_pair_signal, spread_return
from portfolio.risk_metrics import historical_var, expected_shortfall
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe

BOOK_WEEKLY_FREQUENCY = "weekly"


def pick_entry_exit_winners(result: pd.DataFrame) -> dict:
    """Per pair, the entry_exit spec with the higher validation_net Sharpe at
    weekly frequency, from relative_value.py's already-computed standalone
    table. Ties/NaNs default to "continuous" (the house-style default)."""
    winners = {}
    for pair_name in ALL_PAIR_NAMES:
        try:
            sub = result.xs((pair_name, BOOK_WEEKLY_FREQUENCY), level=("pair", "frequency"))
        except KeyError:
            winners[pair_name] = "continuous"
            continue
        sub = sub["validation_net"].dropna()
        winners[pair_name] = sub.idxmax() if len(sub) else "continuous"
    return winners


def build_rv_alpha_and_returns(spreads: dict, entry_exit_winners: dict):
    alpha_cols, return_cols = {}, {}
    for pair_name in ALL_PAIR_NAMES:
        spread = spreads[pair_name]
        alpha_cols[pair_name] = build_pair_signal(spread, entry_exit=entry_exit_winners[pair_name], target_vol=rv.TARGET_VOL_SIGNAL)
        return_cols[pair_name] = spread_return(spread)
    alpha_df = pd.DataFrame(alpha_cols)
    returns_df = pd.DataFrame(return_cols)
    return alpha_df, returns_df


def prepare_rv_book_inputs(standalone=None):
    """One shared preparation step — standalone bake-off, entry/exit winners,
    alpha/returns panels, active-column set, and a per-pair cost_bps series
    (sum of legs' liquidity-tiered cost, same convention as `relative_value.
    py`'s own `pair_cost_bps`) — reused by both this script's `main()` and
    `research/relative_value_gerber.py`'s covariance-estimator comparison, so
    the two never silently drift apart on which pairs/specs feed the Book.

    `standalone` (optional): an already-computed `(result, hedge_winners,
    spreads)` tuple from `rv.main()` — pass this to avoid re-running the full
    standalone bake-off a second time (e.g. `dashboard/pages/26_relative_
    value_performance.py` needs both the standalone table AND the Book on one
    page render). Defaults to None, which runs `rv.main()` itself, unchanged
    behavior for every existing caller."""
    result, hedge_winners, spreads = standalone if standalone is not None else rv.main()

    entry_exit_winners = pick_entry_exit_winners(result)
    print("\n=== Per-pair entry/exit winner (validation net Sharpe, weekly) ===")
    for pair_name, spec in entry_exit_winners.items():
        print(f"  {pair_name:20s} -> {spec}")

    alpha_df, returns_df = build_rv_alpha_and_returns(spreads, entry_exit_winners)
    active = ssp._active_columns(alpha_df, returns_df)
    excluded = [c for c in alpha_df.columns if c not in active]
    print("\nActive pairs (>=90% valid return density): " + str(active))
    print("Excluded (sparse cross-commodity calendar, <90% density): " + str(excluded))

    _, volume, included = rv.load_and_prepare_data()
    from backtest.costs import liquidity_tiered_cost_bps
    cost_by_leg = liquidity_tiered_cost_bps(volume[included], window_start=rv.ADV_WINDOW_START)
    cost_bps = pd.Series({p: rv.pair_cost_bps(cost_by_leg, p) for p in alpha_df.columns})

    return alpha_df, returns_df, active, cost_bps


def main(cov_dict_builder=None, use_net_cost=True, label="Ledoit-Wolf (default)"):
    alpha_df, returns_df, active, cost_bps = prepare_rv_book_inputs()
    cost_bps_arg = cost_bps if use_net_cost else None

    print(f"\n=== Pooled Relative Value Book ({label}) ===")
    book = ssp.build_book("relative_value", alpha_df, returns_df, cov_dict_builder=cov_dict_builder, cost_bps=cost_bps_arg)
    result_book = book.run(returns_df)

    if len(result_book["pnl"]) == 0:
        print("  INSUFFICIENT DATA: fewer than 20 valid rebalance dates - Book could not run.")
        return book, result_book

    for period, series in zip(("train", "validation", "test"), train_validation_test_split(result_book["pnl"])):
        sh = simple_sharpe(series, periods_per_year=ssp.PERIODS_PER_YEAR)
        print(f"  {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")
    print(f"  turnover={result_book.get('turnover', float('nan')):.3f}  max_dd={result_book.get('max_dd', float('nan')):.3f}")
    print(f"  n_rebalance_dates_valid={result_book.get('n_rebalance_dates_valid')}  n_stale_gaps={result_book.get('n_stale_gaps')}")

    var95 = historical_var(result_book["pnl"], confidence=0.95)
    es95 = expected_shortfall(result_book["pnl"], confidence=0.95)
    print(f"  95% VaR: {var95:.3f}  ES: {es95:.3f} (weekly)")

    return book, result_book


if __name__ == "__main__":
    main()
