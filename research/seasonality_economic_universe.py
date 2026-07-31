"""
research/seasonality_economic_universe.py — does restricting same_month to
assets with a real, physical/economic seasonal demand driver (WORKFLOW.md
Phase 11c's own conviction table — Natural Gas/HeatingOil/RBOB/Corn/Soybeans/
Wheat/KC_Wheat, `signals.seasonality.SEASONALITY_ECONOMIC_DRIVER_ASSETS`)
change the result, versus running it on the full ADV-filtered universe?

This is a hypothesis-driven restriction, fixed from a documented physical-
driver theory BEFORE looking at same_month's performance on these names
specifically — a different kind of universe decision from CLAUDE.md Rule 1's
concern (never edit the universe after observing backtest performance), and a
deliberate departure from KLN/Li et al.'s own construction (rank across the
full commodity universe, no driver theory involved) toward a specifically-
motivated exploration, not a reproduction of their methodology. See
`signals/seasonality.py`'s own constant docstring for the full reasoning.

Plain-Sharpe comparison only (matching how same_month's full-universe result
was first read before any Single Strategy Portfolio treatment) — escalating to
the weekly/GARCH Book treatment is a natural follow-up if this looks
promising, not built here.

Run: `python research/seasonality_economic_universe.py` from the repo root.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import seasonality as season_research
from data.sectors import sectors_for_universe
from signals.seasonality import SEASONALITY_ECONOMIC_DRIVER_ASSETS, same_month_signal
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import performance_stats, simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

FREQUENCY = "monthly"


def annualized_turnover(signal):
    positions = normalized_positions(signal, FREQUENCY)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate(signal, returns, cost_bps, label):
    signal_returns = returns[signal.columns]
    signal_cost_bps = cost_bps[signal.columns]
    gross = backtest_signal(signal, signal_returns, frequency=FREQUENCY)
    net = backtest_signal(signal, signal_returns, frequency=FREQUENCY, cost_bps=signal_cost_bps)
    turn = annualized_turnover(signal)

    print(f"\n=== {label} ===")
    print(f"  universe ({len(signal.columns)}): {sorted(signal.columns.tolist())}")
    print(f"  annualized turnover: {turn:.2f}")
    for period, g, n in zip(("train", "validation", "test"), train_validation_test_split(gross), train_validation_test_split(net)):
        print(f"  {period}: gross Sharpe={simple_sharpe(g):.3f}  net Sharpe={simple_sharpe(n):.3f}  (n={len(g.dropna())})")
    return gross, net


def main():
    close, volume, sectors, vol = season_research.load_and_prepare_data(family="rank")
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=season_research.ADV_WINDOW_START)

    full_signal = same_month_signal(close, sectors)
    evaluate(full_signal, returns, cost_bps, "same_month, FULL universe (rank-compressed, decision #13)")

    economic_universe = [a for a in SEASONALITY_ECONOMIC_DRIVER_ASSETS if a in close.columns]
    economic_sectors = sectors_for_universe(economic_universe)
    economic_close = close[economic_universe]
    economic_signal = same_month_signal(economic_close, economic_sectors)
    evaluate(economic_signal, returns, cost_bps, "same_month, ECONOMIC-DRIVER universe (7 names, hypothesis-driven)")

    print(
        "\nReported as found, either direction - a hypothesis-driven restriction "
        "doesn't guarantee an improvement, only a more economically defensible test."
    )


if __name__ == "__main__":
    main()
