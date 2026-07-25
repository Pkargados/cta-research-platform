"""
data/trusted_since.py — Trusted-era / legacy-era split for the core OHLCV panel.

Built 2026-07-21, alongside Data/asset_trusted_since.csv (see DATA_SCHEMA.md section 1
for the full evidence trail). The core panel (open/high/low/close.parquet, yfinance)
was found to have a real, material OHLC-consistency problem (3.84% of cells), and its
continuous-series construction rule was never confirmed (raw splice vs. back-adjusted -
DATA_SCHEMA.md's still-open caveat). Decision: don't delete the pre-cutoff history,
mask it out of anything used for signal calibration, and keep it only as an optional
out-of-sample robustness check.

Scope, stated plainly so this isn't overclaimed by a future caller: this module only
removes the pre-cutoff legacy region, where no independent source exists to check
against. It does NOT cross-validate or correct the post-cutoff portion against
term_structure.parquet's real Databento data - that Tier-1 fix, and the full
roll-splice continuous-series construction, are separate, not-yet-built follow-ons
(see DATA_SCHEMA.md section 1 / WORKFLOW.md Phase 0). Data from an asset's
trusted_since date forward is "cross-checkable," not yet "cross-checked."
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"


def load_trusted_since(data_dir=None) -> pd.Series:
    """Per-asset date from which term_structure.parquet has genuine Databento
    coverage - see Data/asset_trusted_since.csv for the full reasoning per asset
    (standard 2010-06-06 GLBX.MDP3 start vs. genuine product-inception dates vs. the
    ICE budget-trimmed backfill scope vs. KC_Wheat's unexplained exception).

    Returns a pd.Series indexed by asset name.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(data_dir / "asset_trusted_since.csv", parse_dates=["trusted_since"])
    return df.set_index("asset")["trusted_since"]


def restrict_to_trusted_era(df: pd.DataFrame, trusted_since: pd.Series) -> pd.DataFrame:
    """Mask each asset's pre-trusted_since values to NaN, per column.

    Deliberately per-column, not a single date-index slice: each asset has its own
    cutoff (2010-06-06 for most, later for KC_Wheat/Russell2000/SOFR/Lumber/the ICE
    softs). A single slice would either truncate early-starting assets down to the
    latest asset's cutoff, or leak a late-starting asset's pre-cutoff data. The full
    date index is preserved; only the affected cells become NaN.

    Columns not present in `trusted_since` are left untouched (not expected in
    practice - Data/asset_trusted_since.csv already covers the full 42-asset
    universe, a superset of the 41-asset core panel).
    """
    trusted_since = pd.to_datetime(trusted_since)
    result = df.copy()
    for asset in result.columns:
        if asset not in trusted_since.index:
            continue
        cutoff = trusted_since.loc[asset]
        result.loc[result.index < cutoff, asset] = float("nan")
    return result
