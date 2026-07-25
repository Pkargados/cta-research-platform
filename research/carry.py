"""
research/carry.py — Carry driver: 4 paper-matched specs (carry1m, carry1-12,
carry_timing zero/mean), monthly rebalancing, gross/net Sharpe.

Reproduces CLAUDE.md's documented, genuinely mixed result (train/validation/test
Sharpe): carry1m -0.01/-0.58/-0.68; carry1-12 +0.33/-0.36/-0.06; carry_timing
(ref=0) -0.30/-0.98/+0.33; carry_timing (ref=mean) +0.02/-0.56/+0.26. Turnover
collapsed to ~0.7-3.3x annualized once rebalancing matched the paper's monthly
cadence — net-of-cost should be nearly identical to gross for every spec.

Run: `python research/carry.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.term_structure import build_carry_panel
from signals.carry import build_all_carry_signals
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
FREQUENCY = "monthly"


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    close = adj["close"][included]
    sectors = sectors_for_universe(included)
    return close, adj["volume"][included], included, sectors


def annualized_turnover(signal):
    positions = normalized_positions(signal, FREQUENCY)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate_all_specs(signals, returns, cost_bps):
    rows = []
    for name, signal in signals.items():
        gross = backtest_signal(signal, returns, frequency=FREQUENCY)
        net = backtest_signal(signal, returns, frequency=FREQUENCY, cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "spec": name,
            "annualized_turnover": annualized_turnover(signal),
            "train_gross": simple_sharpe(g_train), "train_net": simple_sharpe(n_train),
            "validation_gross": simple_sharpe(g_val), "validation_net": simple_sharpe(n_val),
            "test_gross": simple_sharpe(g_test), "test_net": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("spec")


def main():
    close, volume, included, sectors = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)

    carry_panel, is_proxy = build_carry_panel(included)
    print(f"\nCarry panel: {carry_panel.shape[1]} assets, {int(is_proxy.sum())} proxy (ICE softs)")
    print(f"Carry panel date range: {carry_panel.index.min()} to {carry_panel.index.max()}")

    signals = build_all_carry_signals(carry_panel, sectors)
    print(f"\nSpecs: {list(signals.keys())}")

    print("\n--- Gross/net Sharpe, all 4 specs (monthly rebalancing) ---")
    result = evaluate_all_specs(signals, returns[carry_panel.columns], cost_bps)
    print(result.round(3).to_string())

    print(
        "\nDocumented (train/validation/test gross Sharpe): carry1m -0.01/-0.58/-0.68; "
        "carry1_12 +0.33/-0.36/-0.06; carry_timing_zero -0.30/-0.98/+0.33; "
        "carry_timing_mean +0.02/-0.56/+0.26. Turnover ~0.7-3.3x annualized."
    )


if __name__ == "__main__":
    main()
