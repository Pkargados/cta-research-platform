"""
research/breakout.py — Turtle breakout (System 1/System 2) driver: pooled
gross/net Sharpe, turnover measurement, and the daily-vol-resizing-cadence
question CLAUDE.md's "Breakout (Donchian)" row documents as tested and ruled
out (the real turnover driver is pooling many independently-triggered regimes
under one daily gross-exposure-normalized book, not resizing cadence).

Reproduces: pooled Sharpe weak-to-negative gross and worse net-of-cost; measured
turnover ~50-60x annualized at daily resizing cadence.

Run: `python research/breakout.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.volatility import yang_zhang_volatility
from signals.breakout import system1_signal, system2_signal, DEFAULT_TARGET_VOL
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import performance_stats, simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
RESIZE_FREQUENCIES = ("daily", "weekly", "monthly")

SYSTEMS = {"system1": system1_signal, "system2": system2_signal}


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


def annualized_turnover(signal, frequency, holding_months=1):
    positions = normalized_positions(signal, frequency, holding_months)
    daily_book_turnover = turnover_fn(positions).sum(axis=1)
    return float(daily_book_turnover.mean() * 252)


def evaluate_system(signal, returns, cost_bps):
    rows = []
    for freq in RESIZE_FREQUENCIES:
        gross = backtest_signal(signal, returns, frequency=freq)
        net = backtest_signal(signal, returns, frequency=freq, cost_bps=cost_bps)
        gross_train, gross_val, gross_test = train_validation_test_split(gross)
        net_train, net_val, net_test = train_validation_test_split(net)
        rows.append({
            "resize_frequency": freq,
            "annualized_turnover": annualized_turnover(signal, freq),
            "gross_train": simple_sharpe(gross_train), "gross_validation": simple_sharpe(gross_val), "gross_test": simple_sharpe(gross_test),
            "net_train": simple_sharpe(net_train), "net_validation": simple_sharpe(net_val), "net_test": simple_sharpe(net_test),
        })
    return pd.DataFrame(rows).set_index("resize_frequency")


def main():
    adj, raw = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=ADV_WINDOW_START)

    for name, builder in SYSTEMS.items():
        print(f"\n=== {name} ===")
        signal = builder(close, vol, target_vol=DEFAULT_TARGET_VOL)
        result = evaluate_system(signal, returns, cost_bps)
        print(result.round(3).to_string())

    print(
        "\nDocumented finding: turnover ~50-60x annualized at daily resizing; "
        "resizing cadence alone doesn't drive it down materially (pooling many "
        "independently-triggered regimes under one daily gross-exposure-normalized "
        "book is the real driver, a portfolio-construction-layer issue, not a "
        "signal-level one). Pooled Sharpe weak-to-negative gross, worse net."
    )


if __name__ == "__main__":
    main()
