"""
research/rates_carry_book.py — Single-strategy Book: US_10Y, US_30Y, SOFR
carry1m (WORKFLOW.md §11e follow-up, 2026-08-01). These are the only 3 of the
6-member Rates carry1m cross-section (the 5 existing Rates futures plus SOFR)
with net-of-cost Sharpe positive in all three periods (train/validation/test)
- found via the per-asset investigation earlier this session. This Book asks
a different, more decision-relevant question: does a REAL covariance-aware
optimizer (`portfolio.book.Book` via `single_strategy_portfolios.build_book`
- Ledoit-Wolf covariance, vol targeting, position inertia, the same
calibration every other Book in this project reuses, CLAUDE.md Rule 6)
produce a genuinely good COMBINED portfolio out of just these 3, not just 3
individually-decent per-asset Sharpe numbers standing next to each other.

`carry1m` is re-ranked within a LOCAL 3-member "Rates_3" sector, not
inherited from the 6-member group these 3 were selected from - a
single-strategy portfolio "of these 3 contracts" should reflect their own
carry cross-section, not a stale rank computed against a peer group that's
no longer in scope. Worth flagging up front: with only 3 members,
`cross_sectional_rank`'s score is `rank(C_i) - (N+1)/2` = -1/0/+1 - the
middle-ranked asset always scores exactly 0 (flat) on any date all three are
present. This construction is structurally a rotating long-best/short-worst
pair trade among the 3, not a diversified 3-way allocation - relevant context
for interpreting the result, not a bug.

Run: `python research/rates_carry_book.py` from the repo root.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sofr_carry as sc
import single_strategy_portfolios as ssp

from data.term_structure import build_carry_panel
from signals.carry import carry1m_signal
from backtest.costs import liquidity_tiered_cost_bps
from backtest.splits import train_validation_test_split
from backtest.performance import simple_sharpe
from portfolio.risk_metrics import historical_var, expected_shortfall

RATES_3 = ["US_10Y", "US_30Y", "SOFR"]
SOFR_MIN_VALID_FRAC = 0.40  # just below SOFR's own ~44.7% return density, to admit it


def load_inputs():
    close, volume, included, sectors = sc.load_rates_universe()
    sofr_curve = sc.build_databento_only_continuous_curve("SOFR")
    sofr_carry = sc.build_databento_only_carry("SOFR")
    carry_panel, is_proxy = build_carry_panel(included)

    combined_carry = carry_panel.copy()
    combined_carry["SOFR"] = sofr_carry.reindex(combined_carry.index)

    combined_returns = close.pct_change(fill_method=None).copy()
    combined_returns["SOFR"] = sofr_curve["adj_close"].pct_change(fill_method=None).reindex(combined_returns.index)

    combined_volume = volume.copy()
    combined_volume["SOFR"] = sofr_curve["volume"].reindex(combined_volume.index)
    cost_bps = liquidity_tiered_cost_bps(combined_volume, window_start=sc.ADV_WINDOW_START)

    return combined_carry, combined_returns, cost_bps


def main():
    print(f"=== Single-strategy Book: {RATES_3}, carry1m, own local 3-member cross-section ===")
    combined_carry, combined_returns, cost_bps = load_inputs()

    print("Return density:", {a: f"{combined_returns[a].notna().mean():.1%}" for a in RATES_3})

    sectors_local = {"Rates_3": RATES_3}
    alpha_df = carry1m_signal(combined_carry[RATES_3], sectors_local)
    returns_df = combined_returns[RATES_3]

    active = ssp._active_columns(alpha_df, returns_df)
    print(f"Active columns (>=90% return density gate): {active}")

    book_gross = ssp.build_book("rates_carry_3", alpha_df, returns_df, cost_bps=None)
    result_gross = book_gross.run(returns_df)
    book_net = ssp.build_book("rates_carry_3", alpha_df, returns_df, cost_bps=cost_bps[RATES_3])
    result_net = book_net.run(returns_df)

    if len(result_net.get("pnl", [])) == 0:
        print("INSUFFICIENT DATA: fewer than 20 valid rebalance dates.")
        return book_gross, result_gross, book_net, result_net

    for label, result in [("gross", result_gross), ("net", result_net)]:
        pnl = result["pnl"]
        train, val, test = train_validation_test_split(pnl)
        print(f"\n--- {label} ---")
        for name, series in zip(("train", "validation", "test"), (train, val, test)):
            sh = simple_sharpe(series, periods_per_year=ssp.PERIODS_PER_YEAR)
            print(f"  {name}: Sharpe={sh:.3f} (n={len(series.dropna())})")
        print(f"  turnover={result.get('turnover', float('nan')):.4f}  max_dd={result.get('max_dd', float('nan')):.4f}")
        print(f"  n_rebalance_dates_valid={result.get('n_rebalance_dates_valid')}  n_stale_gaps={result.get('n_stale_gaps')}")

    var95 = historical_var(result_net["pnl"], confidence=0.95)
    es95 = expected_shortfall(result_net["pnl"], confidence=0.95)
    print(f"\n95% VaR: {var95:.4f}  ES: {es95:.4f} (weekly, net)")

    if "asset_contributions" in result_net:
        present = [a for a in RATES_3 if a in result_net["asset_contributions"].columns]
        contrib = result_net["asset_contributions"][present].sum(axis=0)
        print("\nPer-asset cumulative gross contribution (net Book, active columns only):")
        print(contrib.round(4).to_string())

    print(f"\n=== Relaxed gate: force SOFR in (min_valid_frac={SOFR_MIN_VALID_FRAC}), same precedent as the "
          "RV sleeve's own sparse-pair check - not a claim this is a good threshold generally ===")
    active_relaxed = ssp._active_columns(alpha_df, returns_df, min_valid_frac=SOFR_MIN_VALID_FRAC)
    print(f"Active columns at relaxed gate: {active_relaxed}")
    book_net_relaxed = ssp.build_book(
        "rates_carry_3_relaxed", alpha_df, returns_df, cost_bps=cost_bps[RATES_3], min_valid_frac=SOFR_MIN_VALID_FRAC
    )
    result_relaxed = book_net_relaxed.run(returns_df)
    if len(result_relaxed.get("pnl", [])) == 0:
        print("INSUFFICIENT DATA even at the relaxed gate: fewer than 20 valid rebalance dates.")
    else:
        pnl = result_relaxed["pnl"]
        train, val, test = train_validation_test_split(pnl)
        for name, series in zip(("train", "validation", "test"), (train, val, test)):
            sh = simple_sharpe(series, periods_per_year=ssp.PERIODS_PER_YEAR)
            print(f"  {name}: Sharpe={sh:.3f} (n={len(series.dropna())})")
        print(f"  turnover={result_relaxed.get('turnover', float('nan')):.4f}  max_dd={result_relaxed.get('max_dd', float('nan')):.4f}")
        print(f"  n_rebalance_dates_valid={result_relaxed.get('n_rebalance_dates_valid')}  n_stale_gaps={result_relaxed.get('n_stale_gaps')}")

    return book_gross, result_gross, book_net, result_net


if __name__ == "__main__":
    main()
