"""
research/seasonality.py — Phase 11b driver: two parallel, unrelated seasonal
effects, no headline pick between them (see `signals/seasonality.py`'s own
module docstring for the full correction record — the half-month/RRT plan in
the original WORKFLOW.md Phase 11b write-up did not match either source
paper's actual tradeable construction; both specs below are what the papers
actually support).

- half_month (Milonas 1991, this project's own trading-rule interpretation):
  daily rebalancing, 7-name scope, vol-targeted +-1 direction.
- same_month (Keloharju-Linnainmaa-Nyberg 2014/2016, replicated for
  commodities by Li et al. 2023 Section 3.1): monthly rebalancing,
  ADV-filtered liquid universe, rank-weighted cross-sectional.

Run: `python research/seasonality.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe, compress_for_family
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from signals.seasonality import (
    SEASONALITY_HALF_MONTH_ASSETS,
    DEFAULT_TARGET_VOL,
    half_month_signal,
    same_month_signal,
)
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import performance_stats, simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63

FREQUENCIES = {"half_month": "daily", "same_month": "monthly"}


def load_and_prepare_data(family=None):
    """`family=None` (default) preserves the original ADV-only (layer A)
    universe exactly — dashboard/pages/23_seasonality_performance.py calls
    this with no args, so it is unaffected by the option below.
    `family="rank"` additionally applies `compress_for_family`'s layer B/C
    compression (next_steps.md Phase 2 — see research/universe_compression.py),
    appropriate for same_month (a cross-sectional rank signal). half_month is
    NOT re-scoped by this option — it's dropped from further work (deeply
    negative net-of-cost, not a paper-validated construction to begin with),
    kept only as the historical Phase 11b record."""
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    if family is not None:
        compressed = compress_for_family(included, family)
        print(f"Compressed ({family}): {len(compressed)} of {len(included)} assets (dropped {sorted(set(included) - set(compressed))})")
        included = compressed

    close = adj["close"][included]
    volume = adj["volume"][included]
    sectors = sectors_for_universe(included)
    vol = yang_zhang_volatility(
        raw["open"][included], raw["high"][included], raw["low"][included], raw["close"][included],
        window=YZ_WINDOW, roll_mask=raw["is_roll_date"][included],
    )
    return close, volume, sectors, vol


def build_signals(close, vol, sectors):
    """Both parallel specs, no headline pick. half_month is scoped to whichever
    of SEASONALITY_HALF_MONTH_ASSETS survive the ADV liquidity floor (all 7
    are expected to, but this doesn't assume it)."""
    half_month_assets = [a for a in SEASONALITY_HALF_MONTH_ASSETS if a in close.columns]
    return {
        "half_month": half_month_signal(close, vol, assets=half_month_assets, target_vol=DEFAULT_TARGET_VOL),
        "same_month": same_month_signal(close, sectors),
    }


def annualized_turnover(signal, frequency):
    positions = normalized_positions(signal, frequency)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate(signals, returns, cost_bps):
    rows = []
    for name, signal in signals.items():
        freq = FREQUENCIES[name]
        signal_returns = returns[signal.columns]
        signal_cost_bps = cost_bps[signal.columns]
        gross = backtest_signal(signal, signal_returns, frequency=freq)
        net = backtest_signal(signal, signal_returns, frequency=freq, cost_bps=signal_cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "spec": name,
            "annualized_turnover": annualized_turnover(signal, freq),
            "train_gross": simple_sharpe(g_train), "train_net": simple_sharpe(n_train),
            "validation_gross": simple_sharpe(g_val), "validation_net": simple_sharpe(n_val),
            "test_gross": simple_sharpe(g_test), "test_net": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("spec")


def main():
    close, volume, sectors, vol = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)

    signals = build_signals(close, vol, sectors)
    print(f"\nhalf_month asset scope: {sorted(signals['half_month'].columns.tolist())}")

    print("\n--- Gross/net Sharpe by spec, no auto-picked winner ---")
    result = evaluate(signals, returns, cost_bps)
    print(result.round(3).to_string())

    print(
        "\nBoth specs are expected, per WORKFLOW.md 11a's synthesis, to come back "
        "null/negative (Li et al.'s own finding: seasonality 'almost completely "
        "disappeared' since 1990) — reported honestly either way."
    )


if __name__ == "__main__":
    main()
