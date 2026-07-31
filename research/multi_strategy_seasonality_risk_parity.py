"""
research/multi_strategy_seasonality_risk_parity.py — direct follow-up to
`research/multi_strategy_seasonality.py`'s own finding: every one of its three
NAIVE equal-Book-risk combinations (Trend+Carry+Seasonality, Trend+Seasonality,
Trend+Carry) underperformed standalone Trend alone (test Sharpe 1.600) in
EVERY period. Caught directly: the naive Allocator forces equal risk regardless
of each Book's own quality, so blending Trend (strong) with a materially
weaker Book (Carry test 0.293, Seasonality test 0.454) at a forced 50/50 (or
33/33/33) risk split dilutes Trend's own edge rather than adding it -
exactly the same failure mode already documented for Trend+Carry alone in
WORKFLOW.md decision #12, which is why risk-parity weighting was built in the
first place. This script applies that SAME already-validated tool
(`portfolio.risk_parity`, `research/sleeve_risk_parity.py`'s own pattern,
generalized here to n=3) to the three combinations instead of re-deriving a
new method - the real question this answers: does a PROPERLY risk-weighted
blend recover value over standalone Trend, or does Trend alone actually win
regardless of how the other Books are weighted?

Discipline unchanged from `sleeve_risk_parity.py`: weights fit ONCE on TRAIN
sleeve PnL only (CLAUDE.md Rule 1/2), applied as a fixed static weight across
all three periods - not walk-forward refit.

Run: `python research/multi_strategy_seasonality_risk_parity.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from single_strategy_portfolios import build_adopted_books, PERIODS_PER_YEAR
from seasonality_single_strategy import build_economic_seasonality_book
from portfolio.allocator import Allocator
from portfolio.risk_parity import risk_parity_weights, risk_contributions
from portfolio.sleeve_covariance import rolling_covariance, ewma_covariance
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe

EWMA_HALFLIFE = 87  # weekly-observation units, matches ssp.EWMA_HALFLIFE's own rescale


def sleeve_pnl_frame(results: dict) -> pd.DataFrame:
    return pd.concat({name: r["pnl"] for name, r in results.items()}, axis=1)


def combine_static(weights: dict, sleeve_returns: pd.DataFrame) -> pd.Series:
    combined = None
    for name, w in weights.items():
        contrib = sleeve_returns[name].dropna() * w
        combined = contrib.copy() if combined is None else combined.add(contrib, fill_value=0.0)
    return combined.sort_index()


def report_periods(label, pnl):
    row = {"label": label}
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(pnl)):
        clean = series.dropna()
        row[period] = simple_sharpe(clean, periods_per_year=PERIODS_PER_YEAR)
    print(f"  {label:45s} train={row['train']:+.3f}  validation={row['validation']:+.3f}  test={row['test']:+.3f}")
    return row


def risk_parity_combo(names, results, books, returns, label):
    sleeve_returns = sleeve_pnl_frame({n: results[n] for n in names})
    train_sleeve = sleeve_returns.loc[:TRAIN_END]
    n = len(names)

    print(f"\n=== {label} ===")
    naive_pnl = Allocator([books[nm] for nm in names]).run(returns)["pnl"]
    report_periods(f"naive (equal risk, w=[{','.join(['1']*n)}])", naive_pnl)

    for est_name, cov_fn in (
        ("rolling", lambda: rolling_covariance(train_sleeve, window=len(train_sleeve.dropna(how="any")))),
        ("ewma", lambda: ewma_covariance(train_sleeve, halflife=EWMA_HALFLIFE)),
    ):
        cov = cov_fn()
        # risk_parity_weights' log-barrier objective can fail to converge
        # (ABNORMAL termination) when Sigma's own values are this small
        # (~1e-4 to 1e-5, weekly Book PnL variance) - its tight ftol/gtol
        # (tuned for the 2-sleeve Trend/Carry case, tests/test_risk_parity.py)
        # can't resolve the gradient balance at that scale. Weights are
        # scale-invariant AFTER normalization (Sigma -> c*Sigma solves for
        # w/sqrt(c), which normalizes to the identical result) - a safe,
        # local rescale, not a change to the shared, already-tested function.
        w = risk_parity_weights(cov.values * 1e4)
        rc = risk_contributions(w, cov.values)
        weights_str = ", ".join(f"{nm}={w[i]:.2f}" for i, nm in enumerate(names))
        risk_share_str = ", ".join(f"{nm}={rc[i]/rc.sum():.2f}" for i, nm in enumerate(names))
        print(f"  risk-parity weights ({est_name}): {weights_str}  |  risk shares: {risk_share_str}")
        rp_pnl = combine_static({nm: n * w[i] for i, nm in enumerate(names)}, sleeve_returns)
        report_periods(f"risk-parity ({est_name})", rp_pnl)


def main():
    returns, trend_book, carry_book = build_adopted_books()
    _r, season_book = build_economic_seasonality_book()
    books = {"trend": trend_book, "carry": carry_book, "seasonality": season_book}
    results = {name: b.run(returns) for name, b in books.items()}

    print("=== Standalone (for reference) ===")
    for name in ("trend", "carry", "seasonality"):
        report_periods(name, results[name]["pnl"])

    risk_parity_combo(["trend", "carry", "seasonality"], results, books, returns, "A: Trend + Carry + Seasonality")
    risk_parity_combo(["trend", "seasonality"], results, books, returns, "B: Trend + Seasonality")
    risk_parity_combo(["trend", "carry"], results, books, returns, "C: Trend + Carry")

    print(
        "\nStandalone Trend alone (test=1.600) is the bar every combination above needs "
        "to clear to be worth adding anything to Trend at all - reported as found, not "
        "assumed to clear it just because risk-parity (rather than naive) weighting was used."
    )


if __name__ == "__main__":
    main()
