"""
research/gerber_integrated_value_xsmom.py — Integrated Value+XSMOM as a
proper Single Strategy Portfolio (weekly Book + GARCH vol-targeting, same
treatment Trend/Carry/same_month already got), then Gerber vs. Ledoit-Wolf
on it.

"Integrated" here is AQR's own definition (Fitzgibbons-Hecht-McQuinn-Serban
2017, `references/AQR - Portfolio Construction Matters.pdf`, read directly
before this script was written): average each asset's Value and Momentum
RANK into one composite score first (Exhibit 1/2's own worked example —
literally `(Value Rank + Momentum Rank) / 2`, no re-ranking step), then
build ONE portfolio from that composite. Already mapped onto this codebase
in `research/value_momentum_combine.py`: `signals.combine.combine_alphas(
[value_signal, xs_momentum_signal], method="equal")` — both component
signals already emit `cross_sectional_rank`-centered scores internally
(confirmed against `Value and Momentum Everywhere`'s own Eq. 1, also read
directly), so averaging them directly *is* AQR's worked example, not an
approximation of it.

Distinguished on purpose from `Value and Momentum Everywhere`'s own
"COMBO" (Eq. 3: `0.5*r_VALUE + 0.5*r_MOMENTUM`, a 50/50 blend of two
ALREADY-BUILT portfolios' returns) — that is structurally AQR's "mix", not
"integrate" (rank isn't linear, so blending two ranked portfolios' returns
is not the same as ranking a blended score), despite the similar name.
This script builds AQR's actual "integrate", matching `value_momentum_
combine.py`'s own mapping.

What's new here vs. `value_momentum_combine.py`: that script only ever fed
Integrated through the lightweight `backtest_signal` path (monthly, no
optimizer) — Integrated was never itself promoted to a real `Book` (weekly
cadence, GARCH vol-targeting, covariance-aware optimizer), the way Trend,
Carry, and same_month all were. This script does that promotion directly
(`single_strategy_portfolios.build_book`, same calibration as every other
Single Strategy Portfolio in this project), then runs the same Ledoit-Wolf
vs. Gerber (c=0.5/0.7/0.9) comparison as `research/gerber_book_
performance.py` and `research/gerber_xsmom_value_seasonality.py` — reusing
their `run_single_book_comparisons` helper directly (CLAUDE.md Rule 6).

Same universe/signal construction as `value_momentum_combine.py`
(`data.universe.get_liquid_universe`, ADV-filtered) - value_alpha and
xsmom_alpha are the identical objects that comparison already validated,
not re-derived.

Cached to Data/research/gerber_integrated_value_xsmom_summary.csv.
Run: `python research/gerber_integrated_value_xsmom.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.macro import load_yield_curve, load_cpi
from signals.value import value_signal
from signals.xs_momentum import xs_momentum_signal
from signals.combine import combine_alphas
from backtest.costs import liquidity_tiered_cost_bps
from gerber_book_performance import run_single_book_comparisons

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000

SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_integrated_value_xsmom_summary.csv"


def main():
    print("Loading data and building Value, XSMOM, and the AQR-integrated composite...")
    adj = ssp.load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets (excluded: {excluded})")
    close = adj["close"][included]
    volume = adj["volume"][included]
    sectors = sectors_for_universe(included)
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    value_alpha = value_signal(close, yield_curve, cpi, sectors)
    xsmom_alpha = xs_momentum_signal(close, sectors)
    integrated_alpha = combine_alphas([value_alpha, xsmom_alpha], method="equal")

    summary, _pnl, _books = run_single_book_comparisons(
        returns, cost_bps, {"integrated_value_xsmom": integrated_alpha},
    )
    print("\n" + summary.to_string(index=False))

    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CACHE_PATH, index=False)
    print(f"\nSaved summary to {SUMMARY_CACHE_PATH}")


if __name__ == "__main__":
    main()
