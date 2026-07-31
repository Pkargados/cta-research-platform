"""
data/universe.py — Asset universe filtering.

Moved from signal_lib.py (2026-07-21) — see cleanup.md section 2.
"""

# Found 2026-07-28 investigating a volume discrepancy between Data/volume.parquet
# (yfinance) and data.continuous_curve's own volume field: for these 3 of the 5
# ICE softs, term_structure.parquet's historical backfill is missing rows for
# the contracts genuinely trading in 2023-2024 - the only contract-symbol present
# on those early dates is a single far-dated one (e.g. Coffee's KCN26.NYB, a
# July-2026 contract, present back to 2023-08-01 with volume=0).
# continuous_curve.assign_front_contract() didn't malfunction - it correctly
# picked "whichever contract has real data," there just wasn't a genuine
# alternative to roll into, so it stayed parked on that one thin contract for
# 13-28 months (Coffee ~13mo, Cotton ~19mo, OrangeJuice ~28mo) before the daily
# forward-capture job started populating real near-term contracts. This matches
# Data/asset_trusted_since.csv's own documented reason ("budget-trimmed backfill
# scope, not a technical limit; full history to 2018-12-23 available but
# unpurchased") - the gap just extends past those assets' own trusted_since
# cutoffs, so the existing trusted-era mask doesn't fully cover it. Price still
# tracks the real commodity reasonably (checked directly against Data/close.parquet
# - no roll-jump artifacts, ratio back-adjustment is working correctly), but
# volume is genuinely near-zero because that far-dated contract-month wasn't
# actively trading in the real world at those historical dates - not
# representative of genuine tradable liquidity. Sugar and Cocoa (the other 2 of
# the 5 ICE softs) show no such defect (frequent rolls, plausible volume
# throughout) and are NOT included here.
#
# Applied independently of the ADV floor below (not folded into continuous_curve's
# own volume number), so these 3 stay excluded for the real, documented reason
# even if the ADV threshold changes or continuous_curve's reported volume for
# them happens to cross it. In every already-published momentum/breakout/
# crossover/portfolio result, all 3 already failed the ADV floor anyway (this
# doesn't change any prior result, only the documented reason). Purchase-
# contingent, not a permanent liquidity judgment on these commodities - remove
# from this list once the fuller Databento ICE history (available back to
# 2018-12-23, confirmed, just not yet purchased) is bought and this stuck-front
# pattern is reverified as gone.
ICE_SOFTS_DATA_BLOCKED = ["Coffee", "Cotton", "OrangeJuice"]


def get_liquid_universe(volume, window_start="2024-07-14", threshold=1000):
    """ADV liquidity floor (CLAUDE.md Rule 1) - assets with < `threshold` contracts/day
    average volume over the trailing window are excluded. Threshold decided from
    liquidity alone, independent of any signal's performance.

    Also excludes ICE_SOFTS_DATA_BLOCKED regardless of their computed ADV - see
    that constant's docstring for why their volume figure isn't trustworthy
    enough to screen on in the first place."""
    adv = volume.loc[window_start:].mean()
    included = adv[adv >= threshold].index.tolist()
    excluded = adv[adv < threshold].index.tolist()

    newly_blocked = [a for a in ICE_SOFTS_DATA_BLOCKED if a in included]
    included = [a for a in included if a not in ICE_SOFTS_DATA_BLOCKED]
    excluded = excluded + newly_blocked
    return included, excluded


# next_steps.md Phase 2, layers B (economic redundancy) and C (economic coverage),
# applied on top of get_liquid_universe's layer-A output - built 2026-07-31 after
# checking directly that this step had never actually run for ANY Book, including
# the already-published Trend/Carry Single Strategy Portfolios (only layer A ever
# ran there). Decided from TRAIN-period (<=backtest.splits.TRAIN_END) pairwise
# return correlation within each data.sectors.SECTORS cluster, plus ADV for
# tie-breaking within a near-duplicate pair - both performance-blind, structural
# facts about the instruments themselves, never any signal's Sharpe (CLAUDE.md
# Rule 1's own concern, one level up: this is a portfolio-level universe edit, and
# it must not be justified by having seen any Book's backtest result either).
# Full correlation tables and the coverage check are in
# research/universe_compression.py, the reproducible source of this list.
#
# Rule applied (judgment-informed, not a blind mechanical cutoff):
#   >=0.95 train-period correlation ("near-duplicate"): drop the less-liquid twin,
#     for EVERY family, rank-based or time-series - at this level even a rank/
#     seasonal signal can't extract meaningfully distinct information from the
#     "extra" member.
#     - US_30Y vs UltraBond: 0.98 (this project's own Value-row docs already treat
#       these as interchangeable - "UltraBond mapped to 30Y as the closest
#       available point"). ADV tie-break unnecessary; UltraBond is the newer,
#       less-established of the two - drop UltraBond, keep US_30Y.
#     - SP500 vs Dow: 0.97 (next_steps.md's own literal example - "ES, NQ, YM,
#       RTY... retain ES and NQ while dropping YM unless it adds something
#       meaningful"). Drop Dow.
#     - US_5Y vs US_10Y: 0.965 - caught by re-running the actual analysis script
#       after this list's first draft only counted 30Y/UltraBond and SP500/Dow;
#       applying the stated >=0.95 rule consistently (not selectively) surfaced
#       this third pair. ADV: US_10Y ~1.4x US_5Y's - keep US_10Y (also the more
#       standard global benchmark maturity), drop US_5Y.
#   0.85-0.95 ("redundant specifically for a directional bet"): drop for a
#     per-asset TIME-SERIES signal (Trend - two nearly-identical directional bets
#     multiply exposure to one factor), but keep for a CROSS-SECTIONAL RANK signal
#     (same_month, carry - a highly-correlated member still contributes its own
#     seasonal/carry estimate; 11c's own seasonal-window table already treats
#     WTI/Brent/RBOB/HeatingOil as economically distinct despite high price
#     correlation).
#     - Wheat vs KC_Wheat: 0.93 ("literally the same commodity, different region/
#       protein" - Phase 11d's own language). ADV: Wheat ~2.4x KC_Wheat's - keep
#       Wheat, drop KC_Wheat for Trend only.
#     - WTI vs Brent: 0.92 (already an 11d RV spread candidate - same complex,
#       different regional benchmark). ADV: WTI ~11x Brent's - keep WTI, drop
#       Brent for Trend only.
#   <0.85: no action (Gold/Silver 0.80, LiveCattle/FeederCattle 0.81, and every
#     other pair in the universe) - normal cluster co-movement, not redundancy.
#
# Coverage (C) check, done: every sector retains >=2 members after either
# compression (Energy 4, EquityIndex 3, Rates 4, Grains 3) - no macro driver
# category is lost. IndustrialMetals (Copper, 1 member) and Softs' separate,
# unrelated train-period data gap (Sugar/Cocoa have zero valid data before
# TRAIN_END - discovered while computing these correlations, not fixed here) are
# pre-existing, documented conditions, untouched by this change.
CLUSTER_REDUNDANT_ALL = ["Dow", "UltraBond", "US_5Y"]
CLUSTER_REDUNDANT_TREND_ONLY = ["Brent", "KC_Wheat"]


def compress_for_family(included, family):
    """next_steps.md Phase 2 layers B/C, applied on top of get_liquid_universe's
    layer-A output. `family`: "trend" (per-asset time-series signals - TSMOM,
    crossover, breakout) or "rank" (cross-sectional rank signals - same_month,
    carry, XSMOM, value). See this module's own comment block above and
    research/universe_compression.py for the correlation analysis and reasoning."""
    if family == "trend":
        excluded = set(CLUSTER_REDUNDANT_ALL) | set(CLUSTER_REDUNDANT_TREND_ONLY)
    elif family == "rank":
        excluded = set(CLUSTER_REDUNDANT_ALL)
    else:
        raise ValueError("family must be 'trend' or 'rank'")
    return [a for a in included if a not in excluded]
