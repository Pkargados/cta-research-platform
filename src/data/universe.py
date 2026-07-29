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
