"""
research/sleeve_risk_parity.py — Naive equal-weight vs. risk-parity
combination of the two decided Single Strategy Portfolios (Trend =
tsmom_alone, Carry = carry_timing_zero), per direct instruction following
from the Multi-Strategy Portfolio page's own finding: `portfolio.allocator.
Allocator` just sums each Book's PnL with no risk-weighting at all, so
Carry (weak/negative) drags down Trend (strong) under that naive scheme.
Real CTA practice combines sleeves by risk, not return, to avoid overfitting
on so little sleeve-level history (~140-250 weekly observations) — see
WORKFLOW.md decision #12 for the write-up of this script's result.

Reuses `single_strategy_portfolios.py`'s exact winning constructions (both
Books GARCH vol-targeted — decisions #10/#11), called directly outside
Streamlit, same pattern `research/book_vol_targeting_estimator.py` already
established (`import single_strategy_portfolios as ssp`, not the cached
dashboard wrapper).

Two sleeve-level covariance estimators (`portfolio.sleeve_covariance`):
rolling/EWMA plain sample covariance (cheap baseline) and DCC-GARCH
(Engle 2002, via the already-validated local `dcc_garch` package) —
cross-checked against each other, not trusted blindly, given how few
sleeve-level observations DCC's own correlation-recursion parameters are
fit on.

Discipline (CLAUDE.md Rule 1/2): `risk_parity.risk_parity_weights` is fit
ONCE, on the TRAIN period's weekly sleeve PnL only, then applied as a FIXED
static weight to combine Trend/Carry PnL across train/validation/test —
NOT re-fit walk-forward period by period. A walk-forward refit is a
reasonable next step but a bigger scope change than this first exercise;
flagged here, not silently done.

Risk-parity weights (which sum to 1 by construction — see risk_parity.py)
are rescaled by n_sleeves=2 before combining, so "naive" ([1, 1], each Book
already at its own target_vol) and "risk parity" ([2*w_trend, 2*w_carry])
sit on the same total gross-weight budget — this keeps reported vol levels
directly comparable across the three combinations, not just Sharpe (Sharpe
itself is invariant to this rescaling choice, since it only depends on the
w_trend/w_carry RATIO, not their absolute scale).

Run: `python research/sleeve_risk_parity.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Gotcha (hit twice this session): research/ must be inserted BEFORE src/, so
# src/ ends up FIRST in sys.path resolution order (index 0) - otherwise
# research/portfolio.py (the old 6-Book pilot script) shadows the src/
# portfolio/ package and every `from portfolio...` import breaks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio.allocator import Allocator
from portfolio.risk_parity import risk_parity_weights, risk_contributions
from portfolio.sleeve_covariance import rolling_covariance, ewma_covariance, dcc_garch_covariance
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe

import single_strategy_portfolios as ssp

PERIODS_PER_YEAR = 52  # weekly Book cadence, matches single_strategy_portfolios.py
TREND_FLAVOR = "tsmom_alone"
CARRY_FLAVOR = "carry_timing_zero"
EWMA_HALFLIFE = 87  # in weekly-observation units, matches ssp.EWMA_HALFLIFE's own rescale for weekly cadence


def build_sleeve_books():
    """Rebuilds exactly the two Books single_strategy_portfolios.py's bake-off
    selected - not re-run through the bake-off itself, just the winning
    (GARCH vol-targeted) construction each, same as
    research/book_vol_targeting_estimator.py's own build_selected_books_pnl."""
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = ssp.build_vol(raw)
    carry_panel, _ = ssp.build_carry_panel(included)

    trend_flavors = ssp.build_trend_flavors(close, vol, returns)
    trend_book = ssp.build_book("trend_" + TREND_FLAVOR, trend_flavors[TREND_FLAVOR], returns, vol_estimator="garch")
    trend_result = trend_book.run(returns)

    carry_flavors = ssp.build_all_carry_signals(carry_panel, sectors)
    carry_book = ssp.build_book("carry_" + CARRY_FLAVOR, carry_flavors[CARRY_FLAVOR], returns, vol_estimator="garch")
    carry_result = carry_book.run(returns)

    return returns, trend_book, trend_result, carry_book, carry_result


def sleeve_pnl_frame(trend_result, carry_result) -> pd.DataFrame:
    """(T x 2) weekly sleeve PnL, outer-joined on rebalance date (a Book's
    own valid rebalance dates need not exactly coincide with the other's)."""
    return pd.concat({"trend": trend_result["pnl"], "carry": carry_result["pnl"]}, axis=1)


def combine_static(weights: dict, sleeve_returns: pd.DataFrame) -> pd.Series:
    """Fixed-weight combination of sleeve PnL, `.add(fill_value=0.0)` same
    mechanic as `Allocator.run`'s own combination (a date only one sleeve has
    a value for still contributes that sleeve's own weighted PnL, not NaN)."""
    combined = None
    for name, w in weights.items():
        contrib = sleeve_returns[name].dropna() * w
        combined = contrib.copy() if combined is None else combined.add(contrib, fill_value=0.0)
    return combined.sort_index()


def report_periods(label: str, pnl: pd.Series):
    print(f"\n  {label}:")
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(pnl)):
        clean = series.dropna()
        sharpe = simple_sharpe(clean, periods_per_year=PERIODS_PER_YEAR)
        vol = float(clean.std() * np.sqrt(PERIODS_PER_YEAR)) if len(clean) > 1 else np.nan
        print(f"    {period:10s}: Sharpe={sharpe:.3f}  ann_vol={vol:.3f}  (n={len(clean)})")


def main():
    print("Building Trend/Carry Books (GARCH vol-targeted, decisions #10/#11)...")
    returns, trend_book, trend_result, carry_book, carry_result = build_sleeve_books()
    sleeve_returns = sleeve_pnl_frame(trend_result, carry_result)
    print(f"Sleeve PnL: trend n={trend_result['pnl'].notna().sum()}, carry n={carry_result['pnl'].notna().sum()}, "
          f"joint n={len(sleeve_returns.dropna(how='any'))}")

    train_sleeve_returns = sleeve_returns.loc[:TRAIN_END]
    print(f"\nFitting risk-parity weights on TRAIN only (through {TRAIN_END}), n={len(train_sleeve_returns.dropna(how='any'))} obs")

    # --- Naive combination (Allocator's own equal-sum, w=[1, 1]) ---
    allocator = Allocator([trend_book, carry_book])
    naive_pnl = allocator.run(returns)["pnl"]

    # --- Risk parity, rolling/EWMA sleeve covariance (full-train-sample) ---
    n_train_clean = len(train_sleeve_returns.dropna(how="any"))
    cov_simple = rolling_covariance(train_sleeve_returns, window=n_train_clean)
    w_simple = risk_parity_weights(cov_simple.values)
    rc_simple = risk_contributions(w_simple, cov_simple.values)
    print(f"\nRisk-parity weights (rolling, full-train-sample cov): "
          f"trend={w_simple[0]:.3f} carry={w_simple[1]:.3f}  "
          f"risk shares: trend={rc_simple[0]/rc_simple.sum():.3f} carry={rc_simple[1]/rc_simple.sum():.3f}")
    rp_simple_pnl = combine_static({"trend": 2 * w_simple[0], "carry": 2 * w_simple[1]}, sleeve_returns)

    cov_ewma = ewma_covariance(train_sleeve_returns, halflife=EWMA_HALFLIFE)
    w_ewma = risk_parity_weights(cov_ewma.values)
    print(f"Risk-parity weights (EWMA, halflife={EWMA_HALFLIFE} obs, train): "
          f"trend={w_ewma[0]:.3f} carry={w_ewma[1]:.3f}")
    rp_ewma_pnl = combine_static({"trend": 2 * w_ewma[0], "carry": 2 * w_ewma[1]}, sleeve_returns)

    # --- Risk parity, DCC-GARCH sleeve covariance (fit on train only) ---
    dcc_result = dcc_garch_covariance(train_sleeve_returns.dropna(how="any"))
    print(f"DCC-GARCH fit converged: {dcc_result['converged']}")
    w_dcc = risk_parity_weights(dcc_result["cov"].values)
    print(f"Risk-parity weights (DCC-GARCH, train): trend={w_dcc[0]:.3f} carry={w_dcc[1]:.3f}")
    rp_dcc_pnl = combine_static({"trend": 2 * w_dcc[0], "carry": 2 * w_dcc[1]}, sleeve_returns)

    print("\n=== Standalone sleeve results (for reference) ===")
    report_periods("Trend (tsmom_alone)", trend_result["pnl"])
    report_periods("Carry (carry_timing_zero)", carry_result["pnl"])

    print("\n=== Combined Trend+Carry: naive vs. risk-parity ===")
    report_periods("naive (Allocator equal-sum, w=[1, 1])", naive_pnl)
    report_periods(f"rp_rolling (full-train cov, w=[{w_simple[0]:.2f}, {w_simple[1]:.2f}] x2)", rp_simple_pnl)
    report_periods(f"rp_ewma (halflife={EWMA_HALFLIFE}, w=[{w_ewma[0]:.2f}, {w_ewma[1]:.2f}] x2)", rp_ewma_pnl)
    report_periods(f"rp_dcc_garch (w=[{w_dcc[0]:.2f}, {w_dcc[1]:.2f}] x2)", rp_dcc_pnl)


if __name__ == "__main__":
    main()
