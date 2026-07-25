"""
research/portfolio.py — First `portfolio.book.Book`/`portfolio.allocator.Allocator`
exercise: one representative Book per signal family (6 total — momentum 12mo,
breakout System 1, crossover 50/200, short-term reversal individual/5d, carry
timing mean, XSMOM), explicitly NOT the full 19+-spec roster each family's own
Book-count decision calls for (that's a later scale-up, not attempted here).

Monthly rebalancing (`COV_FREQ="ME"`), the cadence this first pass used before
the later monthly->weekly switch made specifically for
`value_momentum_combine.py`/`tune_book_hyperparameters.py`'s own validation-
sample-size problem (not repeated here — this script reproduces the ORIGINAL
6-Book pass, not the later ones).

Reproduces CLAUDE.md's documented per-Book Sharpe (momentum_12mo 0.31,
breakout_system1 0.30, crossover_50_200 0.34, reversal_individual_5d 0.39,
carry_timing_mean -0.17, xs_momentum -0.04) and combined Optimizer/Allocator
Sharpe (train 0.46, validation -1.21, test 0.09), plus the risk-metrics pass
(95% VaR/ES on the combined portfolio, full-sample -10.2%/-15.6% monthly).

Run: `python research/portfolio.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from data.term_structure import build_carry_panel
from signals.momentum import tsmom_signal
from signals.breakout import system1_signal
from signals.crossover import crossover_pair_signal
from signals.short_term_reversal import individual_reversal_signal
from signals.carry import carry_timing_mean_signal
from signals.xs_momentum import xs_momentum_signal
from portfolio.covariance import build_cov_dict
from portfolio.book import Book
from portfolio.allocator import Allocator
from portfolio.risk_metrics import historical_var, expected_shortfall, expanding_var_and_es
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
TARGET_VOL_SIGNAL = 0.40

# Book calibration — dimensional-sanity, not backtested/tuned (CLAUDE.md's
# Portfolio-construction row: gamma sized to keep the risk term non-negligible
# against kappa at this universe's actual variance scale).
GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252
COV_FREQ = "ME"
PERIODS_PER_YEAR = 12
EWMA_HALFLIFE = 20
TARGET_VOL_BOOK = 0.10
MAX_WEIGHT = 0.30


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    sectors = sectors_for_universe(included)
    return adj, raw, included, sectors


def build_vol(raw):
    return yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )


def build_six_alphas(adj, raw, included, sectors, vol):
    close = adj["close"]
    carry_panel, _ = build_carry_panel(included)
    return {
        "momentum_12mo": tsmom_signal(close, vol, lookback_months=12, target_vol=TARGET_VOL_SIGNAL),
        "breakout_system1": system1_signal(close, vol, target_vol=TARGET_VOL_SIGNAL),
        "crossover_50_200": crossover_pair_signal(close, vol, "50_200", target_vol=TARGET_VOL_SIGNAL),
        "reversal_individual_5d": individual_reversal_signal(close, vol, sectors, lag=5),
        "carry_timing_mean": carry_timing_mean_signal(carry_panel, sectors),
        "xs_momentum": xs_momentum_signal(close, sectors),
    }


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
    adj, raw, included, sectors = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)

    alphas = build_six_alphas(adj, raw, included, sectors, vol)
    print(f"\nBooks: {list(alphas.keys())}")

    books = []
    print("\n--- Per-Book results ---")
    for name, alpha_df in alphas.items():
        book = build_book(name, alpha_df, returns)
        result = book.run(returns)
        books.append(book)
        print(f"{name}: sharpe={result['sharpe']}, max_dd={result['max_dd']}, "
              f"turnover={result['turnover']}, n_rebalance_dates_valid={result['n_rebalance_dates_valid']}, "
              f"n_stale_gaps={result['n_stale_gaps']}")

    allocator = Allocator(books)
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]

    print("\n--- Combined Allocator PnL, train/validation/test Sharpe (monthly PnL, simple_sharpe @ 12/yr) ---")
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(combined_pnl)):
        print(f"{period}: Sharpe={simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR):.3f} (n={len(series.dropna())})")

    print(
        "\nDocumented per-Book Sharpe: momentum_12mo 0.31, breakout_system1 0.30, "
        "crossover_50_200 0.34, reversal_individual_5d 0.39, carry_timing_mean -0.17, "
        "xs_momentum -0.04. Documented combined Sharpe: train 0.46, validation -1.21, test 0.09."
    )

    print("\n--- Risk metrics (combined portfolio, 95% confidence, monthly) ---")
    var95 = historical_var(combined_pnl, confidence=0.95)
    es95 = expected_shortfall(combined_pnl, confidence=0.95)
    print(f"Full-sample VaR: {var95:.3f}  ES: {es95:.3f}  (documented: VaR -0.102, ES -0.156)")

    expanding = expanding_var_and_es(combined_pnl, confidence=0.95, min_periods=24)
    print(f"Expanding VaR/ES computed for {len(expanding.dropna())} of {len(combined_pnl.dropna())} dates "
          f"(min_periods=24 warmup).")


if __name__ == "__main__":
    main()
