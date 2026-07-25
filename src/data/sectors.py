"""
data/sectors.py — Sector groupings for the 42-asset universe.

Copied from DATA_SCHEMA.md's own classification table (section 1), not invented here.
Built 2026-07-21 as the peer-group scope for cross-sectional signals - starting with
short-term reversal (references/short_term_reversal_implementation_recipe.md).
Comparing an asset's return against the FULL 42-name universe has no clean
interpretation (AUDUSD vs. Corn has no common risk factor tying them together, unlike
Lehmann/Nagel's single-equity-market cross-section) - cross-sectional demeaning is
scoped to economically comparable groups instead.

SOFR and Lumber are deliberately absent, not an oversight:
- Lumber is "its own category" per DATA_SCHEMA.md - no peer group to compare it
  against in this universe.
- SOFR has no OHLCV series in the core panel this module's consumers read from (it's
  term-structure-only, see CLAUDE.md's current-state table) - never reaches a signal
  built off `data.continuous_curve`, so it's excluded here for the same reason.

IndustrialMetals has exactly one member (Copper) - not filtered out here (kept for
fidelity to the DATA_SCHEMA.md table), but a single-member group has no peer to
demean against, so callers using `min_group_size >= 2` (the intended default, see
signals.short_term_reversal) will correctly produce NaN for Copper under this
grouping, not a zero-information false signal. Not a bug - the same kind of expected,
documented gap as Lumber's exclusion.
"""

SECTORS = {
    "Energy": ["WTI Crude", "Brent", "Natural Gas", "RBOB", "HeatingOil"],
    "PreciousMetals": ["Gold", "Silver", "Platinum", "Palladium"],
    "IndustrialMetals": ["Copper"],
    "Grains": ["Corn", "Soybeans", "Wheat", "KC_Wheat", "Rice", "Oats"],
    "Softs": ["Coffee", "Sugar", "Cocoa", "Cotton", "OrangeJuice"],
    "Livestock": ["LiveCattle", "LeanHogs", "FeederCattle"],
    "EquityIndex": ["SP500", "Nasdaq100", "Dow", "Russell2000"],
    "Rates": ["US_2Y", "US_5Y", "US_10Y", "US_30Y", "UltraBond"],
    "FX": ["EURUSD", "JPYUSD", "GBPUSD", "AUDUSD", "CADUSD", "SwissFranc", "MexicanPeso"],
}


def asset_to_sector() -> dict:
    """Flat asset -> sector name lookup, derived from SECTORS (not maintained twice)."""
    return {asset: sector for sector, assets in SECTORS.items() for asset in assets}


def sectors_for_universe(assets) -> dict:
    """SECTORS restricted to a given asset universe (e.g. the ADV-filtered universe) -
    each sector's member list filtered to assets actually present. Sectors that end up
    with zero members after filtering are dropped entirely; sectors are NOT
    re-populated with assets outside `assets` (a caller passing an ADV-filtered
    universe gets ADV-filtered groups, consistent with every other signal's universe
    scoping in this project)."""
    present = set(assets)
    filtered = {
        sector: [a for a in members if a in present]
        for sector, members in SECTORS.items()
    }
    return {sector: members for sector, members in filtered.items() if members}
