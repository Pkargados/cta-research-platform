"""
research/universe_compression.py — next_steps.md Phase 2, layers B (economic
redundancy) and C (economic coverage), the reproducible analysis behind
`data.universe.compress_for_family`.

Checked directly before building this: layer A (the ADV liquidity floor,
`get_liquid_universe`) is the ONLY layer that has ever run for any Book in this
project, including the already-published Trend/Carry Single Strategy Portfolios
(`research/single_strategy_portfolios.py`) — B and C were written into
`next_steps.md`'s roadmap but never executed. This script does that work.

Methodology (judgment-informed, not a blind mechanical cutoff — see
`data.universe`'s own comment block for the full reasoning and citations):
train-period (`backtest.splits.TRAIN_END`, strictly before any validation/test
data — CLAUDE.md Rule 1's own concern, one level up: a portfolio-level universe
edit must not be justified by having seen any Book's backtest result) pairwise
return correlation within each `data.sectors.SECTORS` cluster.

  >=0.95 (near-duplicate): drop the less-liquid twin, for every family.
  0.85-0.95 (redundant for a directional bet only): drop for "trend" (a
    per-asset time-series signal), keep for "rank" (a cross-sectional rank
    signal — a highly-correlated member still contributes its own estimate).
  <0.85: no action, normal cluster co-movement.

Tie-break within a near-duplicate pair uses ADV (structural, performance-blind),
never Sharpe.

Run: `python research/universe_compression.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from data.universe import (
    get_liquid_universe,
    compress_for_family,
    CLUSTER_REDUNDANT_ALL,
    CLUSTER_REDUNDANT_TREND_ONLY,
)
from data.sectors import sectors_for_universe
from backtest.splits import TRAIN_END

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
NEAR_DUPLICATE = 0.95
DIRECTIONAL_REDUNDANT = 0.85


def load_universe_and_returns():
    adj = load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    close = adj["close"][included]
    volume = adj["volume"][included]
    returns = close.pct_change(fill_method=None)
    sectors = sectors_for_universe(included)
    return included, sectors, returns.loc[:TRAIN_END], volume


def flagged_pairs(train_returns, sectors):
    """All within-sector pairs at or above DIRECTIONAL_REDUNDANT, tiered."""
    rows = []
    for sector, members in sectors.items():
        if len(members) < 2:
            continue
        sub = train_returns[members]
        corr = sub.corr()
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                c = corr.loc[a, b]
                if pd.isna(c) or c < DIRECTIONAL_REDUNDANT:
                    continue
                tier = "near-duplicate (>=0.95)" if c >= NEAR_DUPLICATE else "directional-only (0.85-0.95)"
                rows.append({"sector": sector, "pair": f"{a} / {b}", "train_corr": round(float(c), 3), "tier": tier})
    return pd.DataFrame(rows)


def print_sector_correlations(train_returns, sectors):
    for sector, members in sectors.items():
        if len(members) < 2:
            print(f"=== {sector} ({len(members)} member, no peer to correlate against) ===\n")
            continue
        sub = train_returns[members].dropna(how="all")
        corr = sub.corr()
        print(f"=== {sector} ({len(members)} members) ===")
        print(corr.round(2).to_string())
        print()


def coverage_check(sectors, compressed):
    compressed_set = set(compressed)
    print("=== Coverage (C) check ===")
    ok = True
    for sector, members in sectors.items():
        remaining = [m for m in members if m in compressed_set]
        status = "OK" if len(remaining) >= 2 or len(members) < 2 else "LOST COVERAGE"
        if status == "LOST COVERAGE":
            ok = False
        print(f"  {sector}: {len(members)} -> {len(remaining)} members [{status}]")
    assert ok, "A sector dropped below 2 members after compression - coverage (C) violated."
    print("  All sectors retain >= 2 members (or were already singleton) - coverage intact.\n")


def main():
    included, sectors, train_returns, volume = load_universe_and_returns()
    print(f"Universe: {len(included)} assets (ADV-filtered, layer A)\n")

    print_sector_correlations(train_returns, sectors)

    flagged = flagged_pairs(train_returns, sectors)
    print("=== Flagged pairs (train-period correlation >= 0.85) ===")
    print(flagged.to_string(index=False) if not flagged.empty else "  (none)")
    print()

    print("=== ADV tie-breaks (structural, not performance) ===")
    for a, b in [("WTI Crude", "Brent"), ("Wheat", "KC_Wheat"), ("US_10Y", "US_5Y")]:
        adv_a = volume[a].loc[ADV_WINDOW_START:].mean()
        adv_b = volume[b].loc[ADV_WINDOW_START:].mean()
        print(f"  {a}: {adv_a:,.0f}  vs  {b}: {adv_b:,.0f}  -> keep {a if adv_a > adv_b else b}")
    print()

    print(f"CLUSTER_REDUNDANT_ALL (dropped for every family): {CLUSTER_REDUNDANT_ALL}")
    print(f"CLUSTER_REDUNDANT_TREND_ONLY (dropped for trend only): {CLUSTER_REDUNDANT_TREND_ONLY}\n")

    trend_universe = compress_for_family(included, "trend")
    rank_universe = compress_for_family(included, "rank")
    print(f"trend universe: {len(trend_universe)} of {len(included)} assets (dropped {sorted(set(included) - set(trend_universe))})")
    print(f"rank universe:  {len(rank_universe)} of {len(included)} assets (dropped {sorted(set(included) - set(rank_universe))})\n")

    coverage_check(sectors, trend_universe)
    coverage_check(sectors, rank_universe)


if __name__ == "__main__":
    main()
