"""
research/crossover.py — Moving-average crossover driver: all three pairs
(50/100, 50/200, 100/200), gross/net Sharpe, no auto-picked winner.

Reproduces CLAUDE.md's documented mixed result: 50/200 (golden cross) is the
most consistent (positive throughout); 50/100 fades badly out-of-sample;
100/200 sign-flips sharply in validation. Measured annualized turnover ~5-8x,
much lower than breakout's ~50-60x, as expected for a slower trend-confirmation
signal.

Run: `python research/crossover.py` from the repo root.
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
from signals.crossover import all_pair_signals, PAIRS, DEFAULT_TARGET_VOL
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
FREQUENCY = "daily"  # immediate flip, no confirmation filter (signals.crossover's own convention)


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


def annualized_turnover(signal):
    positions = normalized_positions(signal, FREQUENCY)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate_pairs(signals, returns, cost_bps):
    rows = []
    for name, signal in signals.items():
        gross = backtest_signal(signal, returns, frequency=FREQUENCY)
        net = backtest_signal(signal, returns, frequency=FREQUENCY, cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "pair": name,
            "annualized_turnover": annualized_turnover(signal),
            "train_gross": simple_sharpe(g_train), "train_net": simple_sharpe(n_train),
            "validation_gross": simple_sharpe(g_val), "validation_net": simple_sharpe(n_val),
            "test_gross": simple_sharpe(g_test), "test_net": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("pair")


def main():
    adj, raw = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=ADV_WINDOW_START)

    print(f"\nPairs: {PAIRS}")
    signals = all_pair_signals(close, vol, target_vol=DEFAULT_TARGET_VOL)

    print("\n--- Gross/net Sharpe by pair, no auto-picked winner ---")
    result = evaluate_pairs(signals, returns, cost_bps)
    print(result.round(3).to_string())

    print(
        "\nDocumented: 50/200 (golden cross) most consistent (positive throughout); "
        "50/100 fades in test; 100/200 sign-flips in validation. Turnover ~5-8x."
    )


if __name__ == "__main__":
    main()
