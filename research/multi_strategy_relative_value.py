"""
research/multi_strategy_relative_value.py — does adding the Relative Value
Book to the existing Trend Book help, hurt, or do nothing?

Two Books, naive equal-Book-risk Allocator combination (`.add(fill_value=0.0)`,
no `book_weights` — same baseline convention as `research/
multi_strategy_seasonality.py`): Trend is the already-ADOPTED construction
(`single_strategy_portfolios.build_adopted_books()` — compressed universe,
`tsmom_alone`, GARCH vol-targeted); Relative Value is the pooled Book from
`research/relative_value_book.py` (Ledoit-Wolf covariance, 4 of 7 pairs active,
weekly cadence, net-of-cost).

`returns_ts` (Trend's own full ADV-filtered-universe returns) and RV's own
synthetic per-pair spread-return panel are joined column-wise before being
passed to the Allocator — no column-name collision (real asset names vs.
lowercase pair names), and `Book.run()` only ever indexes its own
`alpha_df.columns` out of whatever `returns_df` it's given, so a joined
superset is safe to share across both Books.

**Non-naive combination** (added per direct instruction, following the exact
precedent `research/sleeve_risk_parity.py` already set for Trend/Carry —
WORKFLOW.md decision #12): risk-parity weighting across 3 sleeve-covariance
estimators (rolling full-train-sample, EWMA, DCC-GARCH), fit ONCE on TRAIN
sleeve PnL only (CLAUDE.md Rule 1/2), applied as a FIXED weight across all
three periods — not re-fit walk-forward. Weights are rescaled by n_sleeves=2
(`2*w_trend`, `2*w_rv`) so risk-parity sits on the same total gross-weight
budget as naive ([1, 1]), keeping reported vol levels comparable, not just
Sharpe (which is scale-invariant to this rescaling anyway).

**RV leverage grid** (added per direct instruction, motivated by RV's own
much lower max drawdown than Trend's — -5.8% vs. -15.0%, standalone): a
pre-committed grid of RV-only weight multipliers (Trend fixed at 1.0),
selected on VALIDATION Sharpe, test touched once for the winner — same
"small number of economically-motivated grid points on ONE combination, no
Bonferroni/FDR correction needed" discipline as the Single Strategy
Portfolios flavor bake-offs, NOT the same multiple-comparisons risk
`tune_all_books.py`'s 19-Book x 25-50-point grid search already found
overfits. Full grid reported regardless of pattern (Rule 1/2).

Run: `python research/multi_strategy_relative_value.py` from the repo root.
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
import relative_value_book as rvb
from portfolio.allocator import Allocator
from portfolio.risk_parity import risk_parity_weights, risk_contributions
from portfolio.sleeve_covariance import rolling_covariance, ewma_covariance, dcc_garch_covariance
from portfolio.risk_metrics import historical_var, expected_shortfall
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe

EWMA_HALFLIFE = 87  # weekly-observation units, matches ssp.EWMA_HALFLIFE's own rescale for weekly cadence
LEVERAGE_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)  # RV weight; Trend fixed at 1.0 - pre-committed before running


def report_book(name, result, periods_per_year):
    pnl = result["pnl"]
    if len(pnl) == 0:
        print(f"  {name}: INSUFFICIENT DATA")
        return
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(pnl)):
        sh = simple_sharpe(series, periods_per_year=periods_per_year)
        print(f"  {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")
    print(f"  turnover={result.get('turnover', float('nan')):.3f}  max_dd={result.get('max_dd', float('nan')):.3f}")
    var95 = historical_var(pnl, confidence=0.95)
    es95 = expected_shortfall(pnl, confidence=0.95)
    print(f"  95% VaR: {var95:.3f}  ES: {es95:.3f} (weekly)")


def pnl_stats(pnl, periods_per_year=PERIODS_PER_YEAR):
    """Sharpe, annualized vol, max drawdown for one pnl series - the same
    three numbers Book._compute_pnl reports for a single Book, computed here
    for an arbitrary COMBINED pnl series (Allocator.run() doesn't compute
    vol/max_dd itself, only "pnl")."""
    clean = pnl.dropna()
    if len(clean) < 2:
        return {"sharpe": np.nan, "ann_vol": np.nan, "max_dd": np.nan, "n": len(clean)}
    sharpe = simple_sharpe(clean, periods_per_year=periods_per_year)
    ann_vol = float(clean.std() * np.sqrt(periods_per_year))
    cumret = (1 + clean).cumprod()
    max_dd = float(((cumret - cumret.cummax()) / cumret.cummax()).min())
    return {"sharpe": sharpe, "ann_vol": ann_vol, "max_dd": max_dd, "n": len(clean)}


def report_periods_with_risk(label, pnl, periods_per_year=PERIODS_PER_YEAR):
    print(f"\n  {label}:")
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(pnl)):
        s = pnl_stats(series, periods_per_year)
        print(f"    {period:10s}: Sharpe={s['sharpe']:.3f}  ann_vol={s['ann_vol']:.3f}  max_dd={s['max_dd']:.3f}  (n={s['n']})")


def combine_static(weights: dict, sleeve_returns) -> "pd.Series":
    """Fixed-weight combination of sleeve PnL - same `.add(fill_value=0.0)`
    mechanic Allocator.run() uses, reused from sleeve_risk_parity.py's own
    convention (a date only one sleeve has a value for still contributes
    that sleeve's own weighted PnL, not NaN)."""
    combined = None
    for name, w in weights.items():
        contrib = sleeve_returns[name].dropna() * w
        combined = contrib.copy() if combined is None else combined.add(contrib, fill_value=0.0)
    return combined.sort_index()


def build_rv_book():
    alpha_df, returns_rv, active, cost_bps = rvb.prepare_rv_book_inputs()
    rv_book = rvb.ssp.build_book("relative_value", alpha_df, returns_rv, cost_bps=cost_bps)
    return rv_book, returns_rv


def main():
    returns_ts, trend_book, carry_book = build_adopted_books()
    rv_book, returns_rv = build_rv_book()

    print("\n=== Single Strategy Portfolio: Relative Value (pooled Book, Ledoit-Wolf, net-of-cost) ===")
    rv_result = rv_book.run(returns_rv)
    report_book("relative_value", rv_result, PERIODS_PER_YEAR)

    print("\n=== Single Strategy Portfolio: Trend (adopted construction, for reference) ===")
    trend_result = trend_book.run(returns_ts)
    report_book("trend", trend_result, PERIODS_PER_YEAR)

    combined_returns = returns_ts.join(returns_rv, how="outer")

    print("\n=== Multi-Strategy Portfolio: Trend + Relative Value (naive equal-Book-risk) ===")
    allocator = Allocator([trend_book, rv_book])
    combined = allocator.run(combined_returns)
    report_book("trend_plus_rv", combined, PERIODS_PER_YEAR)

    print("\n=== Multi-Strategy Portfolio: Trend + Carry + Relative Value (naive equal-Book-risk), for context ===")
    allocator3 = Allocator([trend_book, carry_book, rv_book])
    combined3 = allocator3.run(combined_returns)
    report_book("trend_plus_carry_plus_rv", combined3, PERIODS_PER_YEAR)

    # ------------------------------------------------------------------
    # Non-naive combination: risk parity across 3 sleeve-covariance estimators
    # ------------------------------------------------------------------
    sleeve_returns = pd.concat({"trend": trend_result["pnl"], "relative_value": rv_result["pnl"]}, axis=1)
    train_sleeve_returns = sleeve_returns.loc[:TRAIN_END]
    n_train_clean = len(train_sleeve_returns.dropna(how="any"))
    print(f"\n=== Risk-parity combination (fit on TRAIN only, n={n_train_clean} joint weekly obs) ===")

    cov_simple = rolling_covariance(train_sleeve_returns, window=n_train_clean)
    w_simple = risk_parity_weights(cov_simple.values)
    rc_simple = risk_contributions(w_simple, cov_simple.values)
    print(f"  rolling (full-train cov): trend={w_simple[0]:.3f} rv={w_simple[1]:.3f}  "
          f"risk shares: trend={rc_simple[0]/rc_simple.sum():.3f} rv={rc_simple[1]/rc_simple.sum():.3f}")
    rp_simple_pnl = combine_static({"trend": 2 * w_simple[0], "relative_value": 2 * w_simple[1]}, sleeve_returns)

    cov_ewma = ewma_covariance(train_sleeve_returns, halflife=EWMA_HALFLIFE)
    w_ewma = risk_parity_weights(cov_ewma.values)
    print(f"  EWMA (halflife={EWMA_HALFLIFE} obs): trend={w_ewma[0]:.3f} rv={w_ewma[1]:.3f}")
    rp_ewma_pnl = combine_static({"trend": 2 * w_ewma[0], "relative_value": 2 * w_ewma[1]}, sleeve_returns)

    try:
        dcc_result = dcc_garch_covariance(train_sleeve_returns.dropna(how="any"))
        print(f"  DCC-GARCH converged: {dcc_result['converged']}")
        w_dcc = risk_parity_weights(dcc_result["cov"].values)
        print(f"  DCC-GARCH: trend={w_dcc[0]:.3f} rv={w_dcc[1]:.3f}")
        rp_dcc_pnl = combine_static({"trend": 2 * w_dcc[0], "relative_value": 2 * w_dcc[1]}, sleeve_returns)
    except Exception as e:
        print(f"  DCC-GARCH unavailable ({e}) - skipping")
        w_dcc, rp_dcc_pnl = None, None

    report_periods_with_risk(f"naive (w=[1, 1])", combined["pnl"])
    report_periods_with_risk(f"rp_rolling (w=[{w_simple[0]:.2f}, {w_simple[1]:.2f}] x2)", rp_simple_pnl)
    report_periods_with_risk(f"rp_ewma (w=[{w_ewma[0]:.2f}, {w_ewma[1]:.2f}] x2)", rp_ewma_pnl)
    if rp_dcc_pnl is not None:
        report_periods_with_risk(f"rp_dcc_garch (w=[{w_dcc[0]:.2f}, {w_dcc[1]:.2f}] x2)", rp_dcc_pnl)

    # ------------------------------------------------------------------
    # RV leverage grid: Trend fixed at 1.0, RV weight swept over a
    # pre-committed grid, validation-selected, test touched once for the winner
    # ------------------------------------------------------------------
    print(f"\n=== RV leverage grid (Trend weight=1.0 fixed, RV weight swept), validation-selected ===")
    grid_rows = []
    for k in LEVERAGE_GRID:
        pnl_k = combine_static({"trend": 1.0, "relative_value": k}, sleeve_returns)
        train_k, val_k, test_k = train_validation_test_split(pnl_k)
        row = {
            "rv_weight": k,
            "train_sharpe": pnl_stats(train_k)["sharpe"], "train_maxdd": pnl_stats(train_k)["max_dd"],
            "val_sharpe": pnl_stats(val_k)["sharpe"], "val_maxdd": pnl_stats(val_k)["max_dd"],
            "test_sharpe": pnl_stats(test_k)["sharpe"], "test_maxdd": pnl_stats(test_k)["max_dd"],
            "pnl": pnl_k,
        }
        grid_rows.append(row)
        print(f"  rv_weight={k:.1f}  train={row['train_sharpe']:.3f} (dd={row['train_maxdd']:.3f})  "
              f"val={row['val_sharpe']:.3f} (dd={row['val_maxdd']:.3f})  "
              f"test={row['test_sharpe']:.3f} (dd={row['test_maxdd']:.3f})")

    valid_rows = [r for r in grid_rows if not np.isnan(r["val_sharpe"])]
    winner = max(valid_rows, key=lambda r: r["val_sharpe"])
    print(f"\n  Winner by validation Sharpe: rv_weight={winner['rv_weight']:.1f} (val={winner['val_sharpe']:.3f})")
    print(f"  Winner's test Sharpe (touched once): {winner['test_sharpe']:.3f} (max_dd={winner['test_maxdd']:.3f})")

    print("\n=== Comparison (test Sharpe), reported as found ===")
    comparisons = [
        ("Trend alone", trend_result["pnl"]),
        ("Relative Value alone", rv_result["pnl"]),
        ("Trend + RV (naive)", combined["pnl"]),
        ("Trend + Carry + RV (naive)", combined3["pnl"]),
        ("Trend + RV (rp_rolling)", rp_simple_pnl),
        ("Trend + RV (rp_ewma)", rp_ewma_pnl),
    ]
    if rp_dcc_pnl is not None:
        comparisons.append(("Trend + RV (rp_dcc_garch)", rp_dcc_pnl))
    comparisons.append((f"Trend + RV (leverage grid winner, rv_weight={winner['rv_weight']:.1f})", winner["pnl"]))

    for label, pnl in comparisons:
        pnl = pnl.dropna()
        if len(pnl) == 0:
            print(f"  {label:44s} INSUFFICIENT DATA")
            continue
        _, _, test = train_validation_test_split(pnl)
        print(f"  {label:44s} test Sharpe={simple_sharpe(test, periods_per_year=PERIODS_PER_YEAR):.3f}")

    return trend_result, rv_result, combined, combined3, grid_rows


if __name__ == "__main__":
    main()
