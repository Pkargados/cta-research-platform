"""
research/gerber_sector_breakdown.py — Per-sector performance breakdown
(train/validation/test Sharpe, turnover, max drawdown) for the four
non-Trend Single Strategy Portfolios (XSMOM, Value, Integrated
Value+XSMOM, Carry) under Ledoit-Wolf vs. Gerber (c=0.5) — direct
follow-up to `research/gerber_book_performance.py`, `gerber_xsmom_
value_seasonality.py`, and `gerber_integrated_value_xsmom.py`'s own
aggregate (whole-Book) results, raised directly: does either estimator's
edge (or lack of one) concentrate in a specific sector, or is it spread
evenly across the book?

Sectors: coarse 4-group roll-up (Commodities/Equities/Rates/FX) of this
project's own 9-group `data.sectors.SECTORS` taxonomy (Energy/
PreciousMetals/IndustrialMetals/Grains/Softs/Livestock -> Commodities;
EquityIndex -> Equities; Rates -> Rates; FX -> FX) - not a new taxonomy,
just a coarser view of the existing one.

Per-sector PnL is an EXACT decomposition of the Book's own net PnL, not
approximated: `Book.run()` already returns `asset_contributions` (T x N,
exact elementwise gross P&L per asset, `asset_contributions.sum(axis=1) ==
gross_pnl`, see `portfolio/book.py`'s own docstring) - summing a sector's
member columns gives that sector's exact gross contribution. Net requires
one further step this project's own `asset_contributions` docstring
explicitly declines to do at the whole-Book level ("not cleanly
attributable to one asset without an extra modeling choice") - here it CAN
be done exactly, not modeled, because `LAMBD=0.0` in every Book's
calibration (`single_strategy_portfolios.py`), so there is no turnover-
penalty term to allocate, and `backtest.costs.transaction_cost_drag`'s own
per-date cost is itself `(turnover(positions) * cost_bps / 10_000).sum(
axis=1)` - i.e. already a per-asset quantity before that final sum. Summing
that same per-asset cost within a sector, before its own final sum, is the
identical operation `Book._compute_pnl` uses to build `real_cost_s`, just
stopped one step earlier. A sector's net PnL therefore sums EXACTLY to the
whole Book's own net PnL across sectors, by construction (verified via
`main()`'s own sanity check).

Single representative Gerber threshold (c=0.5, the paper's own primary
spec), not all three - a 4-signal x 4-sector x 3-period x 2-estimator table
is already dense; further thresholds are a straightforward re-run if wanted.

Turnover reported in the SAME units as every other Book result this
session (`Book.run()`'s own raw per-period mean |Δw|, NOT annualized -
`result["turnover"]` is never multiplied by `periods_per_year` anywhere in
this project), for direct comparability with the whole-Book turnover
figures already reported.

Cached to Data/research/gerber_sector_breakdown.csv.
Run: `python research/gerber_sector_breakdown.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
from data.macro import load_yield_curve, load_cpi
from signals.value import value_signal
from signals.xs_momentum import xs_momentum_signal
from signals.combine import combine_alphas
from data.sectors import SECTORS, sectors_for_universe
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe
from gerber_book_performance import _gerber_builder

GERBER_C = 0.5
COARSE_GROUPS = {
    "Commodities": ["Energy", "PreciousMetals", "IndustrialMetals", "Grains", "Softs", "Livestock"],
    "Equities": ["EquityIndex"],
    "Rates": ["Rates"],
    "FX": ["FX"],
}

SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_sector_breakdown.csv"


def _coarse_sector_map(assets: list) -> dict:
    present = set(assets)
    coarse = {}
    for group, fine_names in COARSE_GROUPS.items():
        members = [a for fn in fine_names for a in SECTORS.get(fn, []) if a in present]
        if members:
            coarse[group] = members
    return coarse


def _sector_breakdown(result: dict, cost_bps: pd.Series, periods_per_year: int) -> pd.DataFrame:
    weights = result["weights"]
    asset_contrib = result["asset_contributions"]
    assets = list(asset_contrib.columns)
    sector_map = _coarse_sector_map(assets)

    cost_bps_aligned = cost_bps.reindex(assets).fillna(0.0)
    per_asset_turnover = turnover_fn(weights).reindex(columns=assets).fillna(0.0)
    per_asset_cost = per_asset_turnover * (cost_bps_aligned / 10_000)

    # Sanity check: sector nets must sum exactly to the whole Book's own net
    # PnL (LAMBD=0.0 -> no penalty term left unallocated).
    whole_book_net = (asset_contrib.sum(axis=1) - per_asset_cost.sum(axis=1))
    assert np.allclose(whole_book_net.fillna(0.0).values, result["pnl"].reindex(whole_book_net.index).fillna(0.0).values, atol=1e-9), \
        "Sector decomposition does not sum back to the whole-Book net PnL - investigate before trusting this table."

    rows = []
    for sector, present in sector_map.items():
        sector_pnl = asset_contrib[present].sum(axis=1) - per_asset_cost[present].sum(axis=1)
        train, val, test = train_validation_test_split(sector_pnl)
        cumret = (1 + sector_pnl.fillna(0.0)).cumprod()
        running_max = cumret.cummax()
        max_dd = float(((cumret - running_max) / running_max).min()) if len(cumret) else np.nan
        rows.append({
            "sector": sector, "n_assets": len(present),
            "sharpe_train": simple_sharpe(train, periods_per_year=periods_per_year),
            "sharpe_validation": simple_sharpe(val, periods_per_year=periods_per_year),
            "sharpe_test": simple_sharpe(test, periods_per_year=periods_per_year),
            "turnover": float(per_asset_turnover[present].sum(axis=1).mean()),
            "max_dd": max_dd,
        })
    return pd.DataFrame(rows)


def main():
    print("Loading data and building alpha for XSMOM, Value, Integrated, and Carry...")
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=ssp.ADV_WINDOW_START)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    value_alpha = value_signal(close, yield_curve, cpi, sectors)
    xsmom_alpha = xs_momentum_signal(close, sectors)
    integrated_alpha = combine_alphas([value_alpha, xsmom_alpha], method="equal")

    carry_panel, _ = ssp.build_carry_panel(included)
    carry_flavors = ssp.build_all_carry_signals(carry_panel, sectors)
    carry_alpha = carry_flavors[ssp.CARRY_FLAVOR]

    signals = {
        "xsmom": xsmom_alpha, "value": value_alpha,
        "integrated_value_xsmom": integrated_alpha, "carry_" + ssp.CARRY_FLAVOR: carry_alpha,
    }
    estimators = {"ledoit_wolf": None, f"gerber_c{GERBER_C:.1f}".replace(".", ""): _gerber_builder(GERBER_C)}

    all_rows = []
    for signal_name, alpha_df in signals.items():
        for est_name, builder in estimators.items():
            print(f"\n=== {signal_name} / {est_name} ===")
            book = ssp.build_book(signal_name, alpha_df, returns, vol_estimator="garch",
                                   cov_dict_builder=builder, cost_bps=cost_bps)
            result = book.run(returns)
            if len(result.get("pnl", pd.Series(dtype=float))) == 0:
                print("  insufficient valid rebalance dates - skipped")
                continue
            breakdown = _sector_breakdown(result, cost_bps, ssp.PERIODS_PER_YEAR)
            breakdown.insert(0, "estimator", est_name)
            breakdown.insert(0, "signal", signal_name)
            print(breakdown.drop(columns=["signal", "estimator"]).round(3).to_string(index=False))
            all_rows.append(breakdown)

    full = pd.concat(all_rows, ignore_index=True)
    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(SUMMARY_CACHE_PATH, index=False)
    print(f"\nSaved sector breakdown to {SUMMARY_CACHE_PATH}")


if __name__ == "__main__":
    main()
