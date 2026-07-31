"""
research/seasonality_single_strategy.py — same_month as a proper Single
Strategy Portfolio, following the exact process already used for Trend/Carry
(`research/single_strategy_portfolios.py`, WORKFLOW.md decision #10): weekly
Book rebalancing, GARCH vol-targeting (decision #11) — reusing that script's
own `build_book()`/calibration directly, not re-derived.

Two differences from the Trend/Carry treatment, both because same_month has
exactly one construction per universe (no natural alternate flavors, like
XSMOM/Value's own single-Book treatment):
- No flavor bake-off — nothing to select between.
- Two universes, run side by side, no headline pick between them:
  - FULL: `compress_for_family(included, "rank")` (next_steps.md Phase 2
    layers B/C — see `research/universe_compression.py`), applied via
    `research.seasonality.load_and_prepare_data(family="rank")`.
  - ECONOMIC-DRIVER: `signals.seasonality.SEASONALITY_ECONOMIC_DRIVER_ASSETS`
    (7 names with a real physical seasonal demand driver, fixed from Phase
    11c's own conviction table — see `research/seasonality_economic_universe.py`,
    which found this restriction reshapes the plain-Sharpe risk profile
    materially: worse train, much less damaging validation, better net test).

half_month is NOT carried into this step — dropped from further work per the
same session's decision (deeply negative net-of-cost, not a paper-validated
construction to begin with).

Run: `python research/seasonality_single_strategy.py` from the repo root.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import seasonality as season_research
from data.sectors import sectors_for_universe
from signals.seasonality import same_month_signal, SEASONALITY_ECONOMIC_DRIVER_ASSETS
from single_strategy_portfolios import build_book
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe

PERIODS_PER_YEAR = 52  # weekly, matches single_strategy_portfolios.py's own Book cadence


def build_economic_seasonality_book():
    """The economic-driver same_month Book alone, built directly (no bake-off,
    no comparison print) - for reuse by downstream multi-strategy combination
    scripts (CLAUDE.md Rule 6), e.g. `research/multi_strategy_seasonality.py`.
    Returns (returns, book) - `returns` covers the full rank-compressed
    universe, a superset of the 7-name economic-driver universe, safe to
    reuse directly for any Allocator combination."""
    close, volume, sectors, vol = season_research.load_and_prepare_data(family="rank")
    returns = close.pct_change(fill_method=None)
    economic_universe = [a for a in SEASONALITY_ECONOMIC_DRIVER_ASSETS if a in close.columns]
    economic_sectors = sectors_for_universe(economic_universe)
    economic_close = close[economic_universe]
    signal = same_month_signal(economic_close, economic_sectors)
    book = build_book("same_month_economic", signal, returns, vol_estimator="garch")
    return returns, book


def run_same_month_book(close, sectors, returns, name, label):
    signal = same_month_signal(close, sectors)
    print(f"\n=== {label} ===")
    print(f"  universe ({len(signal.columns)}): {sorted(signal.columns.tolist())}")

    book = build_book(name, signal, returns, vol_estimator="garch")
    result = book.run(returns)

    print(f"  valid rebalance dates: {result.get('n_rebalance_dates_valid')}")
    sharpes = {}
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(result["pnl"])):
        sh = simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR)
        sharpes[period] = sh
        print(f"  {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")
    print(f"  turnover={result.get('turnover', float('nan')):.3f}  max_dd={result.get('max_dd', float('nan')):.3f}")
    return {"turnover": result.get("turnover", float("nan")), "max_dd": result.get("max_dd", float("nan")), **sharpes}


def main():
    close, volume, sectors, vol = season_research.load_and_prepare_data(family="rank")
    returns = close.pct_change(fill_method=None)

    full = run_same_month_book(close, sectors, returns, "same_month_full", "same_month, FULL rank-compressed universe")

    economic_universe = [a for a in SEASONALITY_ECONOMIC_DRIVER_ASSETS if a in close.columns]
    economic_sectors = sectors_for_universe(economic_universe)
    economic_close = close[economic_universe]
    economic = run_same_month_book(economic_close, economic_sectors, returns, "same_month_economic", "same_month, ECONOMIC-DRIVER universe (7 names)")

    print("\n=== FULL vs. ECONOMIC-DRIVER comparison, reported as found ===")
    for period in ("train", "validation", "test"):
        print(f"  {period:10s} full={full[period]:.3f}  economic={economic[period]:.3f}")
    print(f"  {'turnover':10s} full={full['turnover']:.3f}  economic={economic['turnover']:.3f}")
    print(f"  {'max_dd':10s} full={full['max_dd']:.3f}  economic={economic['max_dd']:.3f}")

    print(
        "\nExpectation set going in (per the Trend/Carry precedent, same_month's own "
        "standalone weak/mixed result, and the economic-driver plain-Sharpe read): a "
        "plausible outcome for either universe looks like Value/XSMOM's already-parked "
        "profile, not Trend/Carry's - reported honestly either way."
    )


if __name__ == "__main__":
    main()
