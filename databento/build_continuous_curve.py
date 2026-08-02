"""
Thin driver that builds `Data/continuous_futures.parquet` from
`Data/term_structure.parquet`'s real per-contract outright rows.

All the actual construction logic (contract-chain ordering, volume-crossover
front-contract assignment with a confirmation-day buffer, ratio back-
adjustment) lives in `src/data/continuous_curve.py` as importable, tested
library code, consumed project-wide via `data.continuous_curve.
load_continuous_raw`/`load_continuous_backadjusted`. This script's only job
is grouping `term_structure.parquet` by asset, calling
`build_continuous_curve()` per asset, and writing the combined result to the
expected output path.
"""

import sys
from pathlib import Path

import pandas as pd

# Same cp1252/non-ASCII-path guard as databento_transform.py -- cheap and
# consistent with every research/*.py script's own convention, even though
# this module's own print statements don't currently include a full path.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.continuous_curve import build_continuous_curve  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUTPUT_COLUMNS = [
    "date", "asset", "front_contract_symbol", "raw_open", "raw_high", "raw_low", "raw_close",
    "volume", "is_roll_date", "adj_open", "adj_high", "adj_low", "adj_close",
]


def build_all(term_structure: pd.DataFrame, confirmation_days: int = 2, backstop_days: int = 5) -> pd.DataFrame:
    """One `build_continuous_curve()` call per asset in `term_structure`,
    concatenated with an `asset` column. Assets whose own construction
    raises (e.g. too few real rows to form a usable chain) are logged and
    skipped, not allowed to abort the whole run."""
    frames = []
    for asset, asset_df in term_structure.groupby("asset"):
        try:
            curve = build_continuous_curve(asset_df, confirmation_days=confirmation_days, backstop_days=backstop_days)
        except Exception as exc:
            print(f"  {asset}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        curve = curve.reset_index().rename(columns={"index": "date"})
        curve.insert(1, "asset", asset)
        frames.append(curve[OUTPUT_COLUMNS])
        print(f"  {asset}: {len(curve)} rows, {int(curve['is_roll_date'].sum())} roll dates")

    if not frames:
        raise RuntimeError("No asset produced a usable continuous curve - aborting without touching the output file.")
    return pd.concat(frames, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)


def run():
    term_structure = pd.read_parquet(DATA_DIR / "term_structure.parquet")
    print(f"Loaded term_structure.parquet: {len(term_structure)} rows, {term_structure['asset'].nunique()} assets")

    combined = build_all(term_structure)

    tmp_path = DATA_DIR / "continuous_futures.parquet.tmp"
    combined.to_parquet(tmp_path, index=False)
    import os
    os.replace(tmp_path, DATA_DIR / "continuous_futures.parquet")

    print(f"\nWrote continuous_futures.parquet: {len(combined)} rows, {combined['asset'].nunique()} assets")


if __name__ == "__main__":
    run()
