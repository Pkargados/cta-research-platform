"""
research/value_momentum_combine.py — Mix vs. integrate (Value + XSMOM), per AQR's
"Portfolio Construction Matters" (Fitzgibbons, Hecht, McQuinn, Serban 2017,
`references/AQR - Portfolio Construction Matters.pdf`, read directly).

AQR's own definitions (confirmed against the PDF): "mix" = separately build the
top-value and top-momentum portfolios, then combine the two already-built
portfolios; "integrate" = average each asset's value and momentum score into ONE
composite score first, then build a single portfolio from that composite.

Mapping onto this codebase:
- integrate = `signals.combine.combine_alphas([value_signal, xsmom_signal],
  method="equal")` (both already output within-sector centered ranks — literally
  AQR's own "Value Rank"/"Momentum Rank" quantity), fed through the same
  lightweight `backtest_signal` path every standalone signal uses.
- mix = one `portfolio.book.Book` per signal, combined via `portfolio.allocator.
  Allocator`'s post-solve PnL sum — literally AQR's "combine two already-built
  portfolios."

Two flagged asymmetries in this comparison (per CLAUDE.md, not smoothed over):
Mixed runs through the full covariance-aware optimizer (vol targeting, position
inertia) while Integrated is just a rank-averaged signal at unit gross exposure —
a confound AQR's own equities study doesn't have; and Mixed is reported gross-only
(LAMBD=0.0, no cost_bps deduction), not net-comparable to Integrated/standalone's
net rows.

Book rebalancing is WEEKLY (the cadence this script was switched to, per CLAUDE.md
— monthly Book rebalancing left too few validation observations to be useful for
the hyperparameter-tuning work that followed this script). Integrated/standalone
numbers are unaffected by that switch (they never go through Book).

Reproduces CLAUDE.md's documented (train/validation/test, gross Sharpe): Value
standalone -0.01/+0.13/-0.69; XSMOM standalone -0.34/-1.39/+0.04; Integrated
-0.24/-1.19/-0.72; Mixed (weekly) -0.34/-0.75/-0.17.

Run: `python research/value_momentum_combine.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.macro import load_yield_curve, load_cpi
from signals.value import value_signal
from signals.xs_momentum import xs_momentum_signal
from signals.combine import combine_alphas
from backtest.engine import backtest_signal
from backtest.performance import simple_sharpe
from backtest.splits import TRAIN_END, train_validation_test_split
from portfolio.covariance import build_cov_dict
from portfolio.book import Book
from portfolio.allocator import Allocator

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
STANDALONE_FREQUENCY = "monthly"

# Book calibration — same as research/portfolio.py's first 6-Book pass, reused
# verbatim (CLAUDE.md: "calibration reused... not re-derived").
GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252
COV_FREQ = "W-FRI"
PERIODS_PER_YEAR = 52
EWMA_HALFLIFE = 87  # rescaled from the monthly-cadence value of 20 (20 * 52/12 ~= 87)
TARGET_VOL_BOOK = 0.10
MAX_WEIGHT = 0.30


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    close = adj["close"][included]
    sectors = sectors_for_universe(included)
    return close, sectors


def standalone_sharpe(signal, returns):
    strategy_returns = backtest_signal(signal, returns, frequency=STANDALONE_FREQUENCY)
    train, val, test = train_validation_test_split(strategy_returns)
    return simple_sharpe(train), simple_sharpe(val), simple_sharpe(test)


def _active_columns(alpha_df, returns_df, min_valid_frac=0.90):
    has_alpha = alpha_df.notna().any()
    returns_valid_frac = returns_df.notna().mean()
    return [c for c in alpha_df.columns if has_alpha.get(c, False) and returns_valid_frac.get(c, 0.0) >= min_valid_frac]


def build_book(name, alpha_df, returns):
    active = _active_columns(alpha_df, returns)
    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:TRAIN_END].stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    cov_dict = build_cov_dict(returns[active], window=COV_WINDOW, freq=COV_FREQ)
    return Book(
        name=name, alpha_df=alpha_scaled, cov_dict=cov_dict,
        gamma=GAMMA, kappa=KAPPA, lambd=LAMBD, max_weight=MAX_WEIGHT,
        target_vol=TARGET_VOL_BOOK, ewma_halflife=EWMA_HALFLIFE,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=PERIODS_PER_YEAR, dollar_neutral=False,
    )


def main():
    close, sectors = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    value_alpha = value_signal(close, yield_curve, cpi, sectors)
    xsmom_alpha = xs_momentum_signal(close, sectors)

    print("\n--- Standalone (monthly, gross) ---")
    for name, alpha in [("value", value_alpha), ("xs_momentum", xsmom_alpha)]:
        t, v, s = standalone_sharpe(alpha, returns)
        print(f"{name}: train={t:.3f} validation={v:.3f} test={s:.3f}")

    print("\n--- Integrated (rank-averaged composite, monthly, gross) ---")
    integrated_alpha = combine_alphas([value_alpha, xsmom_alpha], method="equal")
    t, v, s = standalone_sharpe(integrated_alpha, returns)
    print(f"integrated: train={t:.3f} validation={v:.3f} test={s:.3f}")

    print("\n--- Mixed (Book per signal, weekly, gross-only) ---")
    value_book = build_book("value", value_alpha, returns)
    xsmom_book = build_book("xs_momentum", xsmom_alpha, returns)
    allocator = Allocator([value_book, xsmom_book])
    combined = allocator.run(returns)
    for name, res in combined["book_results"].items():
        print(f"{name}: book-level sharpe={res.get('sharpe')}")

    mixed_pnl = combined["pnl"]
    m_train, m_val, m_test = train_validation_test_split(mixed_pnl)
    print(f"mixed: train={simple_sharpe(m_train, periods_per_year=PERIODS_PER_YEAR):.3f} "
          f"validation={simple_sharpe(m_val, periods_per_year=PERIODS_PER_YEAR):.3f} "
          f"test={simple_sharpe(m_test, periods_per_year=PERIODS_PER_YEAR):.3f}")

    print(
        "\nDocumented (train/validation/test, gross): Value -0.01/+0.13/-0.69; "
        "XSMOM -0.34/-1.39/+0.04; Integrated -0.24/-1.19/-0.72; Mixed (weekly) "
        "-0.34/-0.75/-0.17."
    )


if __name__ == "__main__":
    main()
