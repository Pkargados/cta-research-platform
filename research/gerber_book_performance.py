"""
research/gerber_book_performance.py — Does swapping Gerber statistic
covariance into a REAL Book change realized Sharpe/turnover/drawdown, even
though it lost the pure forecast-accuracy (QLIKE) diagnostic in
`research/covariance_estimator_comparison.py`?

Direct follow-up to that diagnostic, per direct instruction, extending past
WORKFLOW.md's own stated gate for the Gerber investigation ("only pursue
live-Book integration if the diagnostic shows a real forecast-accuracy
edge" — it didn't; see that comparison's own result). Forecast-variance
accuracy of ONE reference portfolio (the global minimum-variance portfolio,
scored in isolation) is not the only channel through which a covariance
estimator could still move a REAL Book's realized Sharpe:
- Weight/turnover stability: Gerber's threshold excludes small, noisy moves
  from the concordant/discordant count entirely, so its correlation
  estimate may move less window-to-window than Ledoit-Wolf's - a smoother
  position path, lower net-of-cost turnover drag, independent of forecast
  accuracy.
- `Book`'s actual objective is a mean-variance trade-off
  (alpha - gamma*risk - kappa*turnover), not pure risk minimization -
  Sigma's full off-diagonal structure shapes which alpha gets crowded down
  or levered up, a return-side effect the earlier risk-only diagnostic
  can't see.
- Outlier/regime robustness that a full-sample pooled QLIKE average can
  wash out.

**NET of transaction costs, not just gross** — the first pass through this
script reported gross-only Sharpe, which left an open question (raised
directly): is a lower turnover under Gerber an actual net benefit, or just
a number with no P&L consequence? Fixed by wiring in the same
liquidity-tiered `cost_bps` (`backtest.costs.liquidity_tiered_cost_bps`,
`research/tune_all_books.py`'s own established construction, reused not
re-derived) every other net-of-cost comparison in this project already
uses. `single_strategy_portfolios.build_book` gained a `cost_bps=None`
parameter for this (backward compatible — every existing caller's default
GROSS behavior is unchanged). Both gross and net Sharpe are reported side
by side below, same convention as this project's own gross/net dashboard
toggles.

Three things compared, per direct instruction:
1. **Single-Book swap** — Trend (`tsmom_alone`, compressed universe), Trend's
   own seasonality flavor (`tsmom_seasonal`, WORKFLOW.md Phase 11c — an
   explicit second ask, not assumed to behave like `tsmom_alone`), and Carry
   (`carry_timing_zero`, full universe) — each run under Ledoit-Wolf
   (baseline) and Gerber at all three thresholds (c=0.5/0.7/0.9, the same
   parallel-spec discipline used everywhere else in this project, not just
   the diagnostic's best-looking threshold).
2. **Multi-strategy combination** — this project's ARCHITECTURE already
   supports a different covariance estimator per Book: `Allocator` only
   combines each Book's own post-solve "pnl", it has no opinion on how any
   individual Book computed its own Sigma internally (see `src/portfolio/
   allocator.py`'s own docstring — "does not know or care how any
   individual Book computes its weights"). Three combinations of the
   ADOPTED Trend+Carry mandate (`single_strategy_portfolios.
   build_adopted_books`'s own pairing) via the naive equal-Book-risk
   `Allocator` (same baseline construction `research/multi_strategy_
   seasonality.py` already uses, no risk-parity re-weighting layered on
   top): both Ledoit-Wolf (current mandate), both Gerber c=0.5, and a
   PER-BOOK-BEST mix — each Book independently kept on whichever estimator
   won ITS OWN net validation Sharpe (selected on validation only, test
   touched once for the winning combination — CLAUDE.md Rule 1/2, not
   picked by peeking at test).
3. Gerber's own disclosed per-date NaN caveat (see `portfolio.gerber_
   covariance`'s module docstring) is handled the same way as the first
   pass: any date whose Gerber matrix has a NaN over a Book's own fixed
   active-asset set is dropped before the dict reaches `Book`
   (`_clean_cov_dict` below).

Cached to Data/research/gerber_book_performance_{summary,pnl}.{csv,parquet}
and Data/research/gerber_multi_strategy_summary.csv for the dashboard page.
Run: `python research/gerber_book_performance.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
from portfolio.gerber_covariance import build_gerber_cov_dict
from portfolio.allocator import Allocator
from backtest.costs import liquidity_tiered_cost_bps
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe

GERBER_THRESHOLDS = (0.5, 0.7, 0.9)
MIXED_ESTIMATOR_NAME = "per_book_best"

SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_book_performance_summary.csv"
PNL_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_book_performance_pnl.parquet"
MULTI_STRATEGY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_multi_strategy_summary.csv"


def _clean_cov_dict(cov_dict: dict, active: list) -> tuple:
    """Drop any date whose matrix has a NaN over `active` - Gerber's own
    disclosed per-date caveat; Ledoit-Wolf never needs this (always fully
    dense once a date exists at all). Returns (cleaned_dict, n_dropped)."""
    cleaned = {}
    n_dropped = 0
    for date, cov in cov_dict.items():
        sub = cov.loc[active, active]
        if sub.isna().any().any():
            n_dropped += 1
            continue
        cleaned[date] = cov
    return cleaned, n_dropped


def _gerber_builder(c: float):
    def _build(returns_df, window, freq):
        cov_dict = build_gerber_cov_dict(returns_df, window=window, freq=freq, c=c)
        active = list(returns_df.columns)
        cleaned, n_dropped = _clean_cov_dict(cov_dict, active)
        if n_dropped:
            print(f"    (gerber c={c}: dropped {n_dropped}/{len(cov_dict)} dates with a NaN over the active set)")
        return cleaned
    return _build


def _estimators():
    est = {"ledoit_wolf": None}  # None -> build_book's own default (build_cov_dict)
    for c in GERBER_THRESHOLDS:
        est[f"gerber_c{c:.1f}".replace(".", "")] = _gerber_builder(c)
    return est


def _score(book_label: str, estimator: str, result: dict) -> dict:
    pnl = result.get("pnl", pd.Series(dtype=float))
    gross_pnl = result.get("gross_pnl", pd.Series(dtype=float))
    if len(pnl) == 0:
        return {
            "book": book_label, "estimator": estimator, "n_rebalance_dates_valid": 0,
            "sharpe_train_net": np.nan, "sharpe_validation_net": np.nan, "sharpe_test_net": np.nan,
            "sharpe_train_gross": np.nan, "sharpe_validation_gross": np.nan, "sharpe_test_gross": np.nan,
            "turnover": np.nan, "max_dd": np.nan,
        }
    train, val, test = train_validation_test_split(pnl)
    gtrain, gval, gtest = train_validation_test_split(gross_pnl) if len(gross_pnl) else (pnl, pnl, pnl)
    return {
        "book": book_label, "estimator": estimator,
        "n_rebalance_dates_valid": result.get("n_rebalance_dates_valid"),
        "sharpe_train_net": simple_sharpe(train, periods_per_year=ssp.PERIODS_PER_YEAR),
        "sharpe_validation_net": simple_sharpe(val, periods_per_year=ssp.PERIODS_PER_YEAR),
        "sharpe_test_net": simple_sharpe(test, periods_per_year=ssp.PERIODS_PER_YEAR),
        "sharpe_train_gross": simple_sharpe(gtrain, periods_per_year=ssp.PERIODS_PER_YEAR),
        "sharpe_validation_gross": simple_sharpe(gval, periods_per_year=ssp.PERIODS_PER_YEAR),
        "sharpe_test_gross": simple_sharpe(gtest, periods_per_year=ssp.PERIODS_PER_YEAR),
        "turnover": result.get("turnover", np.nan),
        "max_dd": result.get("max_dd", np.nan),
    }


def run_single_book_comparisons(returns, cost_bps, book_specs):
    """`book_specs`: {book_label: alpha_df}. Returns (summary_df, pnl_dict,
    books_by_label_by_estimator) - the last needed by the multi-strategy
    pass below so Books aren't rebuilt a second time."""
    rows, pnl_series, books = [], {}, {}
    for book_label, alpha_df in book_specs.items():
        print(f"\n=== {book_label} ===")
        books[book_label] = {}
        for est_name, builder in _estimators().items():
            book = ssp.build_book(book_label, alpha_df, returns, vol_estimator="garch",
                                   cov_dict_builder=builder, cost_bps=cost_bps)
            result = book.run(returns)
            score = _score(book_label, est_name, result)
            rows.append(score)
            pnl_series[f"{book_label}__{est_name}"] = result.get("pnl", pd.Series(dtype=float))
            books[book_label][est_name] = (book, result)
            print(
                f"  {est_name:14s} n={score['n_rebalance_dates_valid']:>4}  "
                f"train(net/gross)={score['sharpe_train_net']:.3f}/{score['sharpe_train_gross']:.3f}  "
                f"val(net/gross)={score['sharpe_validation_net']:.3f}/{score['sharpe_validation_gross']:.3f}  "
                f"test(net/gross)={score['sharpe_test_net']:.3f}/{score['sharpe_test_gross']:.3f}  "
                f"turnover={score['turnover']:.3f}  max_dd={score['max_dd']:.3f}"
            )
    return pd.DataFrame(rows), pnl_series, books


def _best_estimator_by_validation(summary: pd.DataFrame, book_label: str) -> str:
    sub = summary[summary["book"] == book_label].dropna(subset=["sharpe_validation_net"])
    return sub.loc[sub["sharpe_validation_net"].idxmax(), "estimator"]


def run_multi_strategy_combinations(returns, summary, books):
    """Combines the ADOPTED Trend(tsmom_alone)+Carry(carry_timing_zero)
    mandate via the naive equal-Book-risk Allocator (`research/
    multi_strategy_seasonality.py`'s own baseline construction), under three
    covariance-estimator pairings. Demonstrates directly that a different
    estimator per Book is architecturally supported (Allocator only ever
    touches each Book's already-solved "pnl")."""
    trend_label = "trend_" + ssp.TREND_FLAVOR
    carry_label = "carry_" + ssp.CARRY_FLAVOR

    trend_best = _best_estimator_by_validation(summary, trend_label)
    carry_best = _best_estimator_by_validation(summary, carry_label)
    print(f"\nPer-Book best (by NET validation Sharpe): {trend_label} -> {trend_best}, {carry_label} -> {carry_best}")

    combos = {
        "both_ledoit_wolf (current mandate)": ("ledoit_wolf", "ledoit_wolf"),
        "both_gerber_c05": ("gerber_c05", "gerber_c05"),
        f"{MIXED_ESTIMATOR_NAME} ({trend_best} + {carry_best})": (trend_best, carry_best),
    }

    rows = []
    for combo_label, (trend_est, carry_est) in combos.items():
        trend_book, _ = books[trend_label][trend_est]
        carry_book, _ = books[carry_label][carry_est]
        allocator = Allocator([trend_book, carry_book])
        combined = allocator.run(returns)
        pnl = combined["pnl"]
        train, val, test = train_validation_test_split(pnl)
        row = {
            "combo": combo_label, "trend_estimator": trend_est, "carry_estimator": carry_est,
            "sharpe_train": simple_sharpe(train, periods_per_year=ssp.PERIODS_PER_YEAR),
            "sharpe_validation": simple_sharpe(val, periods_per_year=ssp.PERIODS_PER_YEAR),
            "sharpe_test": simple_sharpe(test, periods_per_year=ssp.PERIODS_PER_YEAR),
            "n_periods": int(len(pnl.dropna())),
        }
        rows.append(row)
        print(f"  {combo_label:42s} train={row['sharpe_train']:.3f}  val={row['sharpe_validation']:.3f}  test={row['sharpe_test']:.3f}")

    return pd.DataFrame(rows)


def main():
    print("Loading data and building alpha for Trend (tsmom_alone, tsmom_seasonal) and Carry (carry_timing_zero)...")
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = ssp.build_vol(raw)
    carry_panel, _ = ssp.build_carry_panel(included)

    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=ssp.ADV_WINDOW_START)

    trend_universe = ssp.compress_for_family(included, "trend")
    trend_flavors = ssp.build_trend_flavors(close[trend_universe], vol[trend_universe], returns[trend_universe])

    carry_flavors = ssp.build_all_carry_signals(carry_panel, sectors)

    book_specs = {
        "trend_" + ssp.TREND_FLAVOR: trend_flavors[ssp.TREND_FLAVOR],
        "trend_tsmom_seasonal": trend_flavors["tsmom_seasonal"],
        "carry_" + ssp.CARRY_FLAVOR: carry_flavors[ssp.CARRY_FLAVOR],
    }

    summary, pnl_series, books = run_single_book_comparisons(returns, cost_bps, book_specs)
    print("\n" + summary.to_string(index=False))

    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CACHE_PATH, index=False)
    pd.DataFrame(pnl_series).to_parquet(PNL_CACHE_PATH)
    print(f"\nSaved summary to {SUMMARY_CACHE_PATH}")
    print(f"Saved PnL series to {PNL_CACHE_PATH}")

    print("\n=== Multi-strategy (Trend + Carry, naive equal-Book-risk Allocator) ===")
    multi_summary = run_multi_strategy_combinations(returns, summary, books)
    multi_summary.to_csv(MULTI_STRATEGY_CACHE_PATH, index=False)
    print(f"\nSaved multi-strategy summary to {MULTI_STRATEGY_CACHE_PATH}")


if __name__ == "__main__":
    main()
