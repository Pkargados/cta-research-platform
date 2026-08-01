"""
research/gerber_xsmom_value_seasonality.py — Gerber statistic vs. Ledoit-Wolf
on three more Single-Book strategies: XSMOM, Value, and same-calendar-month
seasonality (economic-driver 7-name universe). Direct follow-up to
`research/gerber_book_performance.py`'s Trend/Carry result, testing the
hypothesis raised directly in that discussion: Gerber should tend to help
slower-cadence, cross-sectional, rank-based strategies whose covariance
matrix mostly does diversification bookkeeping rather than needing to react
fast to a magnitude-driven regime shift (Carry's own profile) — XSMOM and
Value are exactly that shape (monthly rebalance, rank-averaged within
sector, no per-asset fast-reaction mechanism), and same_month is as well,
once promoted to a Book (see `research/seasonality_single_strategy.py`).

Universe: XSMOM and Value use their own existing standalone universe
(`data.universe.get_liquid_universe`, ADV-filtered, no "economic rationale"
analogue exists for these two generic cross-sectional signals). Same_month
uses the ECONOMIC-DRIVER 7-name universe specifically (`signals.seasonality.
SEASONALITY_ECONOMIC_DRIVER_ASSETS` — Natural Gas/HeatingOil/RBOB/Corn/
Soybeans/Wheat/KC_Wheat), reusing `seasonality_single_strategy.
build_economic_seasonality_book`'s exact construction — this is the ALREADY
-PROMOTED Single Strategy Portfolio version (CLAUDE.md's Seasonality row: a
genuine test-Sharpe sign flip vs. the full universe), not the deprecated
full-universe one.

Each Book built via `single_strategy_portfolios.build_book` directly (same
GAMMA/KAPPA/MAX_WEIGHT/EWMA_HALFLIFE/`vol_estimator="garch"` calibration as
every other Single Strategy Portfolio in this project — nothing about a
Book's OWN parameters is touched), under Ledoit-Wolf (baseline) and Gerber
at all three thresholds (c=0.5/0.7/0.9), NET of the same liquidity-tiered
transaction costs (`backtest.costs.liquidity_tiered_cost_bps`) as the
Trend/Carry pass. Reuses `gerber_book_performance`'s own `_gerber_builder`/
`_clean_cov_dict`/`_score` helpers directly (CLAUDE.md Rule 6), not
reimplemented.

Cached to Data/research/gerber_xsmom_value_seasonality_summary.csv.
Run: `python research/gerber_xsmom_value_seasonality.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
import seasonality as season_research
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.macro import load_yield_curve, load_cpi
from signals.xs_momentum import xs_momentum_signal
from signals.value import value_signal
from signals.seasonality import same_month_signal, SEASONALITY_ECONOMIC_DRIVER_ASSETS
from backtest.costs import liquidity_tiered_cost_bps
from gerber_book_performance import run_single_book_comparisons

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000

SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "gerber_xsmom_value_seasonality_summary.csv"


def _load_xsmom_value_inputs():
    adj = ssp.load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"XSMOM/Value universe: {len(included)} of {len(adj['volume'].columns)} assets (excluded: {excluded})")
    close = adj["close"][included]
    volume = adj["volume"][included]
    sectors = sectors_for_universe(included)
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=ADV_WINDOW_START)
    return close, returns, sectors, cost_bps


def main():
    print("Loading data and building alpha for XSMOM, Value, and same_month (economic-driver)...")
    close, returns, sectors, cost_bps = _load_xsmom_value_inputs()
    yield_curve, cpi = load_yield_curve(), load_cpi()

    xsmom_alpha = xs_momentum_signal(close, sectors)
    value_alpha = value_signal(close, yield_curve, cpi, sectors)

    season_close, _season_volume, _season_sectors, _season_vol = season_research.load_and_prepare_data(family="rank")
    season_returns = season_close.pct_change(fill_method=None)
    economic_universe = [a for a in SEASONALITY_ECONOMIC_DRIVER_ASSETS if a in season_close.columns]
    economic_sectors = sectors_for_universe(economic_universe)
    same_month_alpha = same_month_signal(season_close[economic_universe], economic_sectors)
    season_cost_bps = liquidity_tiered_cost_bps(_season_volume[economic_universe], window_start=ssp.ADV_WINDOW_START)

    # XSMOM/Value share one returns panel + cost_bps (same universe);
    # same_month uses its own (different universe, different returns panel) -
    # run_single_book_comparisons takes ONE returns/cost_bps pair, so these
    # two groups are run as two separate calls, not force-merged.
    summary_xv, _pnl_xv, _books_xv = run_single_book_comparisons(
        returns, cost_bps, {"xsmom": xsmom_alpha, "value": value_alpha},
    )
    summary_season, _pnl_season, _books_season = run_single_book_comparisons(
        season_returns, season_cost_bps, {"same_month_economic": same_month_alpha},
    )

    summary = pd.concat([summary_xv, summary_season], ignore_index=True)
    print("\n" + summary.to_string(index=False))

    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CACHE_PATH, index=False)
    print(f"\nSaved summary to {SUMMARY_CACHE_PATH}")


if __name__ == "__main__":
    main()
