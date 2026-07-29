"""
research/trend_correlation.py — Correlation between the three trend-family
signals (TSMOM, Donchian breakout, MA crossover), per direct instruction
following from `next_steps.md` Phase 3: the roadmap proposes collapsing
TSMOM/crossover/breakout into a single "trend ensemble" IF their position/
return correlations are high — that hasn't been checked anywhere in this
project yet, only Value vs. XSMOM has (`research/signal_correlation.py`,
full-sample -0.31).

Same three representative specs already used in the first `Book`/`Allocator`
pilot (`research/portfolio.py`) — momentum_12mo, breakout_system1,
crossover_50_200 — reused here rather than re-picked, so this isn't
accidentally also a "different spec choice" comparison.

Two views:
1. Daily standalone strategy returns, plain rolling/EWMA Pearson
   (`portfolio.correlation`) — same cheap first cut as
   `research/signal_correlation.py`.
2. DCC-GARCH (Engle 2002) conditional correlation path, fit on the same
   three daily return series via the author's own `dcc_garch` package
   (`../DCC Garch Rompolis/src`, local sibling project, not vendored into
   this repo — WORKFLOW.md Phase 7 already scoped DCC-GARCH as the next
   escalation for "between signals/Books" correlation once a simple
   estimator's read needs a regime-conditional check). Reports full-sample
   summary stats on the R_t path per pair, plus two named stress windows
   (2020 COVID, 2022 inflation/rate shock) against the full-sample mean —
   the real question a trend-ensemble decision needs answered: do these
   signals still diversify each other exactly when it matters, not just on
   average.

Unlike `data.garch_volatility` (per-asset PRICE-return vol estimation, where
one asset's tiny native return scale broke the package's fixed x100 internal
scaling), these are STRATEGY-level vol-targeted daily returns, already in a
normal range (~similar magnitude to equity returns) — no per-asset rescale
needed here.

Run: `python research/trend_correlation.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "DCC Garch Rompolis" / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.volatility import yang_zhang_volatility
from signals.momentum import tsmom_signal
from signals.breakout import system1_signal
from signals.crossover import crossover_pair_signal
from backtest.engine import backtest_signal
from portfolio.correlation import rolling_correlation, ewma_correlation, correlation_summary
from dcc_garch.garch.gjr_garch import fit_multivariate_gjr
from dcc_garch.dcc.optimizer import fit as fit_dcc

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
TARGET_VOL_SIGNAL = 0.40  # matches research/portfolio.py's own trend-Book calibration

DAILY_ROLLING_WINDOW = 63
DAILY_EWMA_HALFLIFE = 60

# Named stress windows for the DCC-GARCH conditional-correlation read.
STRESS_WINDOWS = {
    "2020 COVID crash": ("2020-02-15", "2020-04-30"),
    "2022 inflation/rate shock": ("2022-01-01", "2022-12-31"),
}


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    return adj, raw


def build_vol(raw):
    return yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )


def build_trend_returns(close, returns, vol):
    """Each family's own established natural cadence (CLAUDE.md's own
    per-family frequency choices), not a uniform re-sample forced on all
    three - matches research/portfolio.py's exact spec choice."""
    momentum_signal = tsmom_signal(close, vol, lookback_months=12, target_vol=TARGET_VOL_SIGNAL)
    breakout_signal = system1_signal(close, vol, target_vol=TARGET_VOL_SIGNAL)
    crossover_signal = crossover_pair_signal(close, vol, "50_200", target_vol=TARGET_VOL_SIGNAL)

    return {
        "momentum_12mo": backtest_signal(momentum_signal, returns, frequency="monthly", holding_months=1),
        "breakout_system1": backtest_signal(breakout_signal, returns, frequency="daily"),
        "crossover_50_200": backtest_signal(crossover_signal, returns, frequency="daily"),
    }


def pairwise_report(name_a, series_a, name_b, series_b):
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    full_sample = float(aligned["a"].corr(aligned["b"]))
    rolling = rolling_correlation(series_a, series_b, window=DAILY_ROLLING_WINDOW)
    ewma = ewma_correlation(series_a, series_b, halflife=DAILY_EWMA_HALFLIFE)

    print(f"\n--- {name_a} vs. {name_b} ---")
    print(f"Full-sample correlation: {full_sample:.3f}  (n={len(aligned)})")
    print("Rolling summary:")
    print(correlation_summary(rolling).round(3).to_string())
    print("EWMA summary:")
    print(correlation_summary(ewma).round(3).to_string())


def dcc_report(returns_dict):
    names = list(returns_dict.keys())
    aligned = pd.concat([returns_dict[n].rename(n) for n in names], axis=1, join="inner").dropna()
    print(f"\n=== 2. DCC-GARCH conditional correlation (n={len(aligned)} shared daily obs) ===")

    fit = fit_multivariate_gjr(aligned.values)
    dcc = fit_dcc(fit["Z"], fit["sigmas"], model="DCC")
    print(f"DCC params (a, b): {dcc['params'].round(4)}  converged={dcc['converged']}")

    R = dcc["R"]  # (T, N, N)
    dates = aligned.index
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    for i, j in pairs:
        path = pd.Series(R[:, i, j], index=dates)
        print(f"\n--- {names[i]} vs. {names[j]} (DCC-GARCH conditional correlation) ---")
        print(correlation_summary(path).round(3).to_string())
        for label, (start, end) in STRESS_WINDOWS.items():
            window_mean = path.loc[start:end].mean()
            print(f"{label} ({start} to {end}) mean: {window_mean:.3f}" if pd.notna(window_mean)
                  else f"{label}: no data in window")


def main():
    adj, raw = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)

    trend_returns = build_trend_returns(close, returns, vol)
    print(f"\nTrend signals: {list(trend_returns.keys())}")

    print("\n=== 1. Daily standalone strategy returns - rolling/EWMA Pearson ===")
    names = list(trend_returns.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise_report(names[i], trend_returns[names[i]], names[j], trend_returns[names[j]])

    dcc_report(trend_returns)


if __name__ == "__main__":
    main()
