"""
research/multi_strategy_seasonality.py — does adding the economic-driver
same_month Book (WORKFLOW.md decision #13's follow-up — 7-name universe,
Natural Gas/HeatingOil/RBOB/Corn/Soybeans/Wheat/KC_Wheat, weekly + GARCH) to
the existing Trend/Carry Two-Book mandate help, hurt, or do nothing?

Three combinations, per direct instruction, no headline pick:
  A. Trend + Carry + Seasonality
  B. Trend + Seasonality
  C. Trend + Carry (reproduced here directly, for a clean side-by-side —
     identical construction to `single_strategy_portfolios.py`'s own ADOPTED
     section)

All three use the naive equal-Book-risk Allocator combination (`.add(fill_value
=0.0)`, no `book_weights`) — the same baseline `single_strategy_portfolios.py`
reports before any risk-parity weighting (decision #12's own follow-up, not
re-applied here). Trend and Carry are the already-ADOPTED constructions
(`single_strategy_portfolios.build_adopted_books()` — Trend compressed,
Carry reverted to uncompressed); Seasonality is the economic-driver same_month
Book (`seasonality_single_strategy.build_economic_seasonality_book()`).

Run: `python research/multi_strategy_seasonality.py` from the repo root.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from single_strategy_portfolios import build_adopted_books, PERIODS_PER_YEAR
from seasonality_single_strategy import build_economic_seasonality_book
from portfolio.allocator import Allocator
from portfolio.risk_metrics import historical_var, expected_shortfall
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe


def report_combo(books, returns, label):
    print(f"\n=== {label} ===")
    print(f"  Books: {[b.name for b in books]}")
    allocator = Allocator(books)
    combined = allocator.run(returns)
    pnl = combined["pnl"]

    sharpes = {}
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(pnl)):
        sh = simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR)
        sharpes[period] = sh
        print(f"  {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")

    var95 = historical_var(pnl, confidence=0.95)
    es95 = expected_shortfall(pnl, confidence=0.95)
    print(f"  95% VaR: {var95:.3f}  ES: {es95:.3f} (weekly)")
    return {"var95": var95, "es95": es95, **sharpes}


def main():
    returns_ts, trend_book, carry_book = build_adopted_books()
    _returns_season, seasonality_book = build_economic_seasonality_book()

    # returns_ts (the full ADV-filtered universe Trend/Carry are built from)
    # already contains all 7 of the seasonality Book's own economic-driver
    # names - none of them were touched by any compression - so it's a valid
    # shared returns frame for every combination below.
    combo_a = report_combo([trend_book, carry_book, seasonality_book], returns_ts, "A: Trend + Carry + Seasonality")
    combo_b = report_combo([trend_book, seasonality_book], returns_ts, "B: Trend + Seasonality")
    combo_c = report_combo([trend_book, carry_book], returns_ts, "C: Trend + Carry")

    print("\n=== Comparison (test Sharpe / 95% VaR), reported as found ===")
    for name, combo in (("A: Trend+Carry+Season", combo_a), ("B: Trend+Season", combo_b), ("C: Trend+Carry", combo_c)):
        print(f"  {name:24s} train={combo['train']:.3f}  validation={combo['validation']:.3f}  test={combo['test']:.3f}  VaR95={combo['var95']:.3f}")


if __name__ == "__main__":
    main()
