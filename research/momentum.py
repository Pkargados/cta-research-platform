"""
research/momentum.py — TSMOM driver: vol-estimator comparison, headline spec,
Table-2-style lookback x holding grid, per-asset Sharpe, net-of-cost sanity check.

Reproduces CLAUDE.md's documented "Time-series momentum" result: Yang-Zhang beats
EWMA on TRAIN evidence (0.239 vs. 0.200), headline (k=12mo, h=1mo, target_vol=0.40)
train/validation/test Sharpe 0.239/0.475/0.402, and the Transaction-costs row's
net-of-cost sanity check (test Sharpe 0.402 gross -> ~0.363 net). The headline spec
is fixed a priori (CLAUDE.md Rule 1/2) — the grid below is a train-period
robustness view only, never used to pick k/h.

Run: `python research/momentum.py` from the repo root.
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
from data.volatility import yang_zhang_volatility
from data.ewma_volatility import ewma_volatility
from signals.momentum import tsmom_signal, momentum_grid_signals, GRID_MONTHS, HEADLINE_LOOKBACK_MONTHS, HEADLINE_TARGET_VOL
from backtest.engine import backtest_signal, backtest_signal_per_asset
from backtest.performance import performance_stats, simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
HOLDING_GRID = GRID_MONTHS  # same 8-point grid used for lookback (paper's Table 2)


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")

    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    return adj, raw, included


def build_vol_estimators(adj, raw):
    """Yang-Zhang off the RAW curve (data.volatility's own documented input choice
    — the back-adjusted curve goes non-positive in old energy segments), EWMA off
    the back-adjusted curve's own daily returns (data.ewma_volatility's
    documented convention)."""
    yang_zhang = yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )
    adj_returns = adj["close"].pct_change(fill_method=None)
    ewma = ewma_volatility(adj_returns)
    return {"yang_zhang": yang_zhang, "ewma": ewma}


def compare_vol_estimators(close, returns, vol_estimators):
    """Train-period pooled Sharpe for the headline spec under each vol estimator —
    the winner is picked by TRAIN evidence only (CLAUDE.md Rule 1/2), never
    validation/test."""
    rows = []
    for name, vol in vol_estimators.items():
        signal = tsmom_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=HEADLINE_TARGET_VOL)
        strategy_returns = backtest_signal(signal, returns, frequency="monthly", holding_months=1)
        train, validation, test = train_validation_test_split(strategy_returns)
        rows.append({
            "vol_estimator": name,
            "train_sharpe": simple_sharpe(train),
            "validation_sharpe": simple_sharpe(validation),
            "test_sharpe": simple_sharpe(test),
        })
    return pd.DataFrame(rows).set_index("vol_estimator")


def headline_result(close, returns, vol):
    signal = tsmom_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=HEADLINE_TARGET_VOL)
    strategy_returns = backtest_signal(signal, returns, frequency="monthly", holding_months=1)
    train, validation, test = train_validation_test_split(strategy_returns)
    return {
        "train": performance_stats(train), "validation": performance_stats(validation), "test": performance_stats(test),
    }, strategy_returns


def lookback_holding_grid(close, returns, vol):
    """Full 8x8 lookback x holding grid, TRAIN period only — descriptive robustness
    view, matching the paper's own Table 2. Never used to pick the headline spec."""
    grid_signals = momentum_grid_signals(close, vol, lookback_months_grid=GRID_MONTHS, target_vol=HEADLINE_TARGET_VOL)
    rows = []
    for k, signal in grid_signals.items():
        for h in HOLDING_GRID:
            strategy_returns = backtest_signal(signal, returns, frequency="monthly", holding_months=h)
            train, _, _ = train_validation_test_split(strategy_returns)
            rows.append({"lookback_months": k, "holding_months": h, "train_sharpe": simple_sharpe(train)})
    return pd.DataFrame(rows).pivot(index="lookback_months", columns="holding_months", values="train_sharpe")


def per_asset_sharpe(close, returns, vol):
    """Moskowitz-Ooi-Pedersen Figure 2 style: per-instrument Sharpe, headline spec,
    full sample - evaluated BEFORE any cross-asset pooling."""
    signal = tsmom_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=HEADLINE_TARGET_VOL)
    per_asset_returns = backtest_signal_per_asset(signal, returns, frequency="monthly", holding_months=1)
    return per_asset_returns.apply(simple_sharpe).sort_values(ascending=False)


def net_of_cost_check(close, returns, vol, volume):
    """CLAUDE.md's Transaction-costs row: test-period Sharpe should drop from
    0.402 gross to roughly 0.363 net (~13% relative haircut)."""
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)
    signal = tsmom_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=HEADLINE_TARGET_VOL)
    gross = backtest_signal(signal, returns, frequency="monthly", holding_months=1)
    net = backtest_signal(signal, returns, frequency="monthly", holding_months=1, cost_bps=cost_bps)
    _, _, gross_test = train_validation_test_split(gross)
    _, _, net_test = train_validation_test_split(net)
    return simple_sharpe(gross_test), simple_sharpe(net_test)


def main():
    adj, raw, included = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)

    vol_estimators = build_vol_estimators(adj, raw)

    print("\n--- Vol estimator comparison (headline spec, k=12mo, target_vol=0.40) ---")
    comparison = compare_vol_estimators(close, returns, vol_estimators)
    print(comparison.to_string())
    winner = comparison["train_sharpe"].idxmax()
    print(f"\nTrain-evidence winner: {winner} (documented: yang_zhang, 0.239 vs. ewma's 0.200)")

    winning_vol = vol_estimators[winner]

    print(f"\n--- Headline result ({winner}) ---")
    stats, strategy_returns = headline_result(close, returns, winning_vol)
    for period, s in stats.items():
        print(f"\n{period}:")
        print(s.to_string())

    print("\n--- Lookback x holding grid (TRAIN Sharpe, descriptive only) ---")
    grid = lookback_holding_grid(close, returns, winning_vol)
    print(grid.round(3).to_string())

    print("\n--- Per-asset Sharpe (headline spec, full sample) ---")
    print(per_asset_sharpe(close, returns, winning_vol).round(3).to_string())

    print("\n--- Net-of-cost sanity check (test period) ---")
    gross_sharpe, net_sharpe = net_of_cost_check(close, returns, winning_vol, adj["volume"])
    print(f"Gross: {gross_sharpe:.3f}  Net: {net_sharpe:.3f}  (documented: 0.402 gross -> ~0.363 net)")


if __name__ == "__main__":
    main()
