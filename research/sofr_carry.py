"""
research/sofr_carry.py — SOFR futures calendar-spread carry (WORKFLOW.md §11e,
first of the three fixed-income-RV extensions, built in the stated order).

SOFR (CME 3-month SOFR, `SR3`) has real Databento outright + calendar-spread
data (2018-05 to 2026-07, spreads frozen at `SPREAD_DATA_END` like every other
real-quote asset) but was excluded from every prior signal because it has no
yfinance-based continuous price series — `data.term_structure.
build_databento_only_continuous_curve`/`build_databento_only_carry` (new,
2026-08-01) reuse the SAME roll-detection/back-adjustment/real-spread-carry
machinery every other asset's continuous curve and carry already use, just
sourced directly from `term_structure.parquet`'s own outrights instead of the
yfinance-backed panel (CLAUDE.md Rule 6).

**Real bug found and fixed before this backtest ran, not after**: SOFR's
quoted calendar-spread `close` is on a DIFFERENT scale than its own outright
close — checked directly (2026-07-13, SR3Z27-SR3Z28: spread quoted -8.5, but
the two outrights' own closes differed by exactly -0.085, i.e. 1/100th).
SOFR's spreads are quoted directly in basis points of rate, not the same
"100-rate" price-point units the outrights use — every other real-quote
asset's spread/outright closes already share a scale (their carry values land
in a sane 0.03%-1.4% annualized range with no correction needed). Fixed via
`DATABENTO_ONLY_SPREAD_SCALE["SOFR"] = 0.01` in `data.term_structure` —
without it, SOFR's carry came back with a nonsensical mean of -55%
annualized (min -890%); with it, mean -0.55%, comparable in magnitude to
US_2Y/US_10Y.

Two standalone (single-asset, no cross-sectional peers) constructions, no
headline pick: `carry_timing_zero` (the paper's own ±1 direction, reused
as-is — already per-asset/elementwise, no sector needed) and a version timed
against SOFR's OWN expanding mean (reuses `sector_pooled_expanding_mean` with
a single-member "sector" — algebraically identical to an individual asset's
own expanding mean, not a new construction). Then a POOLED test: SOFR added
as a 6th member of a LOCAL COPY of the existing 5-name Rates sector — not the
shared `data.sectors.SECTORS` dict, since every other caller of
`sectors_for_universe` scopes to the yfinance-based liquid universe, which
SOFR was never part of, so this stays scoped to this script and doesn't
touch shared state — re-running `carry1m`/`carry1_12`/`carry_timing_zero`/
`carry_timing_mean` on the combined 6-name cross-section and reporting
SOFR's own per-asset contribution via `backtest_signal_per_asset`.

Run: `python research/sofr_carry.py` from the repo root.
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
from data.term_structure import (
    build_carry_panel, build_databento_only_continuous_curve, build_databento_only_carry,
)
from signals.carry import (
    carry_timing_zero_signal, carry_timing_signal, sector_pooled_expanding_mean,
    build_all_carry_signals,
)
from backtest.engine import backtest_signal, backtest_signal_per_asset, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
FREQUENCY = "monthly"


def _one_col(series: pd.Series, col: str) -> pd.DataFrame:
    return series.to_frame(col)


def load_rates_universe():
    adj = load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    close = adj["close"][included]
    volume = adj["volume"][included]
    sectors = sectors_for_universe(included)
    return close, volume, included, sectors


def evaluate_standalone(signal: pd.Series, returns: pd.Series, cost_bps_value: float, label: str) -> dict:
    signal_df, ret_df = _one_col(signal, "sofr"), _one_col(returns, "sofr")
    cost_series = pd.Series({"sofr": cost_bps_value})
    gross = backtest_signal(signal_df, ret_df, frequency=FREQUENCY)
    net = backtest_signal(signal_df, ret_df, frequency=FREQUENCY, cost_bps=cost_series)
    g_tr, g_va, g_te = train_validation_test_split(gross)
    n_tr, n_va, n_te = train_validation_test_split(net)
    positions = normalized_positions(signal_df, FREQUENCY)
    annualized_turnover = float(turnover_fn(positions).sum(axis=1).mean() * 252)
    return {
        "spec": label, "annualized_turnover": annualized_turnover,
        "train_gross": simple_sharpe(g_tr), "train_net": simple_sharpe(n_tr),
        "validation_gross": simple_sharpe(g_va), "validation_net": simple_sharpe(n_va),
        "test_gross": simple_sharpe(g_te), "test_net": simple_sharpe(n_te),
    }


def main():
    print("=== SOFR futures calendar-spread carry ===")

    curve = build_databento_only_continuous_curve("SOFR")
    sofr_returns = curve["adj_close"].pct_change(fill_method=None)
    sofr_volume = curve["volume"]
    sofr_carry = build_databento_only_carry("SOFR")

    print(f"Continuous curve: {curve.index.min().date()} to {curve.index.max().date()}, "
          f"{int(curve['is_roll_date'].sum())} roll dates")
    print(f"Carry: {sofr_carry.notna().sum()} of {len(sofr_carry)} obs valid "
          f"({sofr_carry.dropna().index.min().date()} to {sofr_carry.dropna().index.max().date()}), "
          f"mean={sofr_carry.mean():.4f} std={sofr_carry.std():.4f}")

    close, volume, included, sectors = load_rates_universe()
    combined_volume = volume.copy()
    combined_volume["SOFR"] = sofr_volume.reindex(combined_volume.index)
    cost_bps = liquidity_tiered_cost_bps(combined_volume, window_start=ADV_WINDOW_START)
    sofr_cost = float(cost_bps["SOFR"])
    print(f"SOFR liquidity-tiered one-way cost: {sofr_cost:.2f}bp")

    print("\n=== Standalone (no cross-sectional peers) ===")
    rows = []

    timing_zero = carry_timing_zero_signal(_one_col(sofr_carry, "SOFR"))["SOFR"]
    rows.append(evaluate_standalone(timing_zero, sofr_returns, sofr_cost, "carry_timing_zero"))

    solo_sectors = {"SOFR_solo": ["SOFR"]}
    expanding_ref = sector_pooled_expanding_mean(_one_col(sofr_carry, "SOFR"), solo_sectors)["SOFR"]
    timing_mean = carry_timing_signal(_one_col(sofr_carry, "SOFR"), reference=_one_col(expanding_ref, "SOFR"))["SOFR"]
    rows.append(evaluate_standalone(timing_mean, sofr_returns, sofr_cost, "carry_timing_expanding_mean"))

    result = pd.DataFrame(rows).set_index("spec")
    print(result.round(3).to_string())

    print("\n=== Pooled: SOFR added as a 6th Rates-sector member ===")
    print("(local test copy of the sector map - data.sectors.SECTORS itself is unchanged)")
    carry_panel, is_proxy = build_carry_panel(included)
    pooled_carry = carry_panel.copy()
    pooled_carry["SOFR"] = sofr_carry.reindex(pooled_carry.index)

    pooled_returns = close.pct_change(fill_method=None).copy()
    pooled_returns["SOFR"] = sofr_returns.reindex(pooled_returns.index)

    pooled_sectors = {k: list(v) for k, v in sectors.items()}
    pooled_sectors["Rates"] = pooled_sectors.get("Rates", []) + ["SOFR"]
    rates_members = pooled_sectors["Rates"]

    pooled_signals = build_all_carry_signals(pooled_carry, pooled_sectors)
    pooled_cost_bps = cost_bps.reindex(pooled_carry.columns)

    for spec_name, signal in pooled_signals.items():
        per_asset_gross = backtest_signal_per_asset(
            signal[rates_members], pooled_returns[rates_members], frequency=FREQUENCY
        )
        per_asset_net = backtest_signal_per_asset(
            signal[rates_members], pooled_returns[rates_members], frequency=FREQUENCY,
            cost_bps=pooled_cost_bps[rates_members],
        )
        g_tr, g_va, g_te = train_validation_test_split(per_asset_gross["SOFR"])
        n_tr, n_va, n_te = train_validation_test_split(per_asset_net["SOFR"])
        print(
            f"  {spec_name:22s} SOFR gross/net Sharpe: "
            f"train={simple_sharpe(g_tr):.3f}/{simple_sharpe(n_tr):.3f}  "
            f"val={simple_sharpe(g_va):.3f}/{simple_sharpe(n_va):.3f}  "
            f"test={simple_sharpe(g_te):.3f}/{simple_sharpe(n_te):.3f}"
        )

    return result


if __name__ == "__main__":
    main()
