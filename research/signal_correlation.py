"""
research/signal_correlation.py — Dynamic correlation between Value and XSMOM's
own strategy returns, per direct instruction following from `research/
value_momentum_combine.py`: AQR's "integrate" edge depends on value/XSMOM being
negatively correlated, and a static full-sample number can't show whether that
holds up over time.

Deliberately plain rolling/EWMA Pearson correlation (`portfolio.correlation`),
not DCC-GARCH — see that module's own docstring for why a simple estimator is
the right place to start with only two series to compare.

Two views, matching CLAUDE.md's documented result:
1. Daily standalone strategy returns (`backtest_signal`'s own daily-marked output
   even under monthly rebalancing — positions are forward-filled onto the daily
   returns index regardless of resample cadence). Documented: full-sample -0.31,
   rolling/EWMA means ~-0.34, negative ~86% of the time, swings to +0.46 at times.
2. Book/Allocator weekly PnL (reusing the same weekly-cadence Book construction
   `research/value_momentum_combine.py` builds). Documented: full-sample -0.40,
   rolling/EWMA negative 100% of the time, tighter std (0.08-0.14 vs. 0.21-0.23
   daily/monthly).

Run: `python research/signal_correlation.py` from the repo root.
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
from backtest.engine import backtest_signal
from backtest.splits import TRAIN_END
from portfolio.covariance import build_cov_dict
from portfolio.book import Book
from portfolio.allocator import Allocator
from portfolio.correlation import rolling_correlation, ewma_correlation, correlation_summary

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000

DAILY_ROLLING_WINDOW = 63
DAILY_EWMA_HALFLIFE = 60
WEEKLY_ROLLING_WINDOW = 13
WEEKLY_EWMA_HALFLIFE = 12

GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252
COV_FREQ = "W-FRI"
PERIODS_PER_YEAR = 52
EWMA_HALFLIFE_BOOK = 87
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
        target_vol=TARGET_VOL_BOOK, ewma_halflife=EWMA_HALFLIFE_BOOK,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=PERIODS_PER_YEAR, dollar_neutral=False,
    )


def report(label, series_a, series_b, window, halflife):
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    full_sample = float(aligned["a"].corr(aligned["b"]))
    rolling = rolling_correlation(series_a, series_b, window=window)
    ewma = ewma_correlation(series_a, series_b, halflife=halflife)

    print(f"\n--- {label} ---")
    print(f"Full-sample correlation: {full_sample:.3f}")
    print("Rolling summary:")
    print(correlation_summary(rolling).round(3).to_string())
    print("EWMA summary:")
    print(correlation_summary(ewma).round(3).to_string())


def main():
    close, sectors = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    value_alpha = value_signal(close, yield_curve, cpi, sectors)
    xsmom_alpha = xs_momentum_signal(close, sectors)

    print("\n=== 1. Daily standalone strategy returns ===")
    value_returns = backtest_signal(value_alpha, returns, frequency="monthly")
    xsmom_returns = backtest_signal(xsmom_alpha, returns, frequency="monthly")
    report("Daily standalone (value vs. xs_momentum)", value_returns, xsmom_returns,
           DAILY_ROLLING_WINDOW, DAILY_EWMA_HALFLIFE)
    print("Documented: full-sample -0.31, rolling/EWMA means ~-0.34, negative ~86% of the time, swings to +0.46.")

    print("\n=== 2. Book/Allocator weekly PnL ===")
    value_book = build_book("value", value_alpha, returns)
    xsmom_book = build_book("xs_momentum", xsmom_alpha, returns)
    allocator = Allocator([value_book, xsmom_book])
    combined = allocator.run(returns)
    value_pnl = combined["book_results"]["value"]["pnl"]
    xsmom_pnl = combined["book_results"]["xs_momentum"]["pnl"]
    report("Book weekly PnL (value vs. xs_momentum)", value_pnl, xsmom_pnl,
           WEEKLY_ROLLING_WINDOW, WEEKLY_EWMA_HALFLIFE)
    print("Documented: full-sample -0.40, rolling/EWMA negative 100% of the time, tighter std (0.08-0.14).")


if __name__ == "__main__":
    main()
