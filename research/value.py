"""
research/value.py — Value driver: ONE Book (negative-5yr-return default +
bond yield-change + FX PPP special cases), monthly rebalancing, gross/net Sharpe.

Reproduces CLAUDE.md's documented weak/negative result: train/validation/test
Sharpe -0.01/+0.13/-0.69 gross, -0.03/+0.09/-0.71 net. Turnover ~2.7x annualized.
Coffee/Cocoa/Sugar/Cotton are expected 100% NaN — real price history only starts
2023-2024, short of the 60-month (5yr) lookback (a labeled data-coverage gap, not
a bug).

Run: `python research/value.py` from the repo root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.continuous_curve import load_continuous_backadjusted
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.macro import load_yield_curve, load_cpi
from signals.value import value_signal
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import performance_stats, simple_sharpe
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
    return close, adj["volume"][included], sectors


def annualized_turnover(signal):
    positions = normalized_positions(signal, FREQUENCY)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def main():
    close, volume, sectors = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    signal = value_signal(close, yield_curve, cpi, sectors)

    coverage = signal.notna().mean().sort_values()
    print("\n--- Value score coverage (fraction of dates non-NaN) ---")
    print(coverage.round(3).to_string())
    print("Documented: Coffee/Cocoa/Sugar/Cotton ~0% (real history starts 2023-2024, short of the 5yr lookback)")

    print(f"\nAnnualized turnover: {annualized_turnover(signal):.3f} (documented ~2.7x)")

    gross = backtest_signal(signal, returns, frequency=FREQUENCY)
    net = backtest_signal(signal, returns, frequency=FREQUENCY, cost_bps=cost_bps)

    print("\n--- Gross ---")
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(gross)):
        print(f"{period}: Sharpe={simple_sharpe(series):.3f}")
        print(performance_stats(series).to_string())

    print("\n--- Net ---")
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(net)):
        print(f"{period}: Sharpe={simple_sharpe(series):.3f}")

    print(
        "\nDocumented (train/validation/test): gross -0.01/+0.13/-0.69, net -0.03/+0.09/-0.71."
    )


if __name__ == "__main__":
    main()
