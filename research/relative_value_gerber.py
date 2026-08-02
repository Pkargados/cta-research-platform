"""
research/relative_value_gerber.py — Gerber statistic vs. Ledoit-Wolf covariance
on the pooled Relative Value Book (WORKFLOW.md §11d build-order step 6, and the
original motivating question behind this whole build session — see WORKFLOW.md
Phase 7's Gerber section for the revised RV prior: "an RV book with multiple
spreads needs a full spread-to-spread covariance matrix... and the classic
RV/stat-arb tail risk is a cross-spread correlation spike during a
deleveraging shock, exactly the kind of fast, magnitude-driven event a
threshold-based estimator is structurally slow to register" — Ledoit-Wolf was
EXPECTED to win here, for a sharper reason than the original Trend/Carry
framing. This script actually runs it, rather than assuming the prior holds.

Reuses `research.gerber_book_performance`'s own `_gerber_builder`/
`_clean_cov_dict` helpers (the exact `cov_dict_builder` swap-in pattern already
validated for Trend/Carry/XSMOM/Value/Seasonality/Integrated — CLAUDE.md
Rule 6) and `relative_value_book.prepare_rv_book_inputs` (the same alpha/
returns/cost panel `relative_value_book.py`'s own Ledoit-Wolf run uses, so this
comparison is apples-to-apples against that already-reported baseline, not a
separately-rebuilt panel).

NET of transaction costs throughout (`relative_value_book.py`'s own per-pair
liquidity-tiered cost_bps), same discipline as every other net-of-cost
covariance-estimator comparison in this project.

Run: `python research/relative_value_gerber.py` from the repo root.
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
from relative_value_book import prepare_rv_book_inputs
from gerber_book_performance import _gerber_builder
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe

GERBER_THRESHOLDS = (0.5, 0.7, 0.9)


def _estimators():
    est = {"ledoit_wolf": None}
    for c in GERBER_THRESHOLDS:
        est[f"gerber_c{c:.1f}".replace(".", "")] = _gerber_builder(c)
    return est


def _score(estimator_name, alpha_df, returns_df, cost_bps, builder):
    book = ssp.build_book("relative_value", alpha_df, returns_df, cov_dict_builder=builder, cost_bps=cost_bps)
    result = book.run(returns_df)
    if len(result.get("pnl", [])) == 0:
        return {"estimator": estimator_name, "n_rebalance_dates_valid": 0}

    row = {"estimator": estimator_name, "n_rebalance_dates_valid": result["n_rebalance_dates_valid"],
           "turnover": result.get("turnover", np.nan), "max_dd": result.get("max_dd", np.nan)}
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(result["pnl"])):
        row[f"sharpe_{period}"] = simple_sharpe(series, periods_per_year=ssp.PERIODS_PER_YEAR)
    return row


def main():
    alpha_df, returns_df, active, cost_bps = prepare_rv_book_inputs()

    print("\n=== Gerber vs. Ledoit-Wolf covariance, pooled Relative Value Book, net-of-cost ===")
    rows = []
    for name, builder in _estimators().items():
        row = _score(name, alpha_df, returns_df, cost_bps, builder)
        rows.append(row)
        print(f"  {name:15s} {row}")

    result = pd.DataFrame(rows).set_index("estimator")
    print("\n" + result.round(3).to_string())

    baseline = result.loc["ledoit_wolf"]
    print(f"\nLedoit-Wolf baseline (matching relative_value_book.py's own reported run): "
          f"train={baseline['sharpe_train']:.3f} validation={baseline['sharpe_validation']:.3f} test={baseline['sharpe_test']:.3f}")

    return result


if __name__ == "__main__":
    main()
