"""
One-time historical backfill of individual futures contract-month prices via Databento,
to extend Data/term_structure.parquet (jobs/capture_term_structure.py) further back
than yfinance's forward-capture-only limitation allows.

Run once (not scheduled) - this is a paid, budgeted pull, not a recurring job. See
WORKFLOW.md Phase 4 for the cost breakdown and known gaps (ICE softs only got a 2-year
window, not full history; open interest not included at all).

Requires DATABENTO_API_KEY set in the environment.

Pipeline per asset:
  1. Pull the `definition` schema over the full window to get instrument_id -> raw
     contract metadata (maturity year/month, instrument_class). This is what lets us
     filter out spread/combo instruments (butterflies, calendar spreads, crack-spread
     combos) that come bundled into a parent-symbol query alongside genuine outright
     contracts, and gives an authoritative 4-digit maturity year (Databento's resolved
     `symbol` column uses an ambiguous single-digit year, e.g. "CLZ6").
  2. Pull the `ohlcv-1d` schema over the same window.
  3. Join on instrument_id, keep only instrument_class == "F" (outright futures).
  4. Normalize contract_symbol to the same format capture_term_structure.py uses
     ({ROOT}{MONTH_CODE}{YY}.{EXCHANGE}) so the two sources dedupe correctly against
     each other on (date, contract_symbol) if their date ranges ever overlap.
"""

import sys
from pathlib import Path

import databento as db
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ARCHIVE_PATH = DATA_DIR / "term_structure.parquet"
MANIFEST_PATH = DATA_DIR / "term_structure_manifest.csv"

MONTH_CODE_TO_NUM = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}
NUM_TO_MONTH_CODE = {v: k for k, v in MONTH_CODE_TO_NUM.items()}

# (databento_root, dataset, exchange_code, history_start) - exchange_code matches the
# convention already used in capture_term_structure.py / DATA_SCHEMA.md.
GLBX_START = "2010-06-06"   # GLBX.MDP3 dataset availability floor
IFUS_START = "2024-07-14"   # trimmed to 2yr for budget - see WORKFLOW.md Phase 4 known gaps

UNIVERSE = {
    "WTI Crude": ("CL", "GLBX.MDP3", "NYM", GLBX_START), "Brent": ("BZ", "GLBX.MDP3", "NYM", GLBX_START),
    "Natural Gas": ("NG", "GLBX.MDP3", "NYM", GLBX_START), "RBOB": ("RB", "GLBX.MDP3", "NYM", GLBX_START),
    "HeatingOil": ("HO", "GLBX.MDP3", "NYM", GLBX_START),
    "Gold": ("GC", "GLBX.MDP3", "CMX", GLBX_START), "Silver": ("SI", "GLBX.MDP3", "CMX", GLBX_START),
    "Platinum": ("PL", "GLBX.MDP3", "NYM", GLBX_START), "Palladium": ("PA", "GLBX.MDP3", "NYM", GLBX_START),
    "Copper": ("HG", "GLBX.MDP3", "CMX", GLBX_START),
    "Corn": ("ZC", "GLBX.MDP3", "CBT", GLBX_START), "Soybeans": ("ZS", "GLBX.MDP3", "CBT", GLBX_START),
    "Wheat": ("ZW", "GLBX.MDP3", "CBT", GLBX_START), "KC_Wheat": ("KE", "GLBX.MDP3", "CBT", GLBX_START),
    "Rice": ("ZR", "GLBX.MDP3", "CBT", GLBX_START), "Oats": ("ZO", "GLBX.MDP3", "CBT", GLBX_START),
    "Coffee": ("KC", "IFUS.IMPACT", "NYB", IFUS_START), "Sugar": ("SB", "IFUS.IMPACT", "NYB", IFUS_START),
    "Cocoa": ("CC", "IFUS.IMPACT", "NYB", IFUS_START), "Cotton": ("CT", "IFUS.IMPACT", "NYB", IFUS_START),
    "OrangeJuice": ("OJ", "IFUS.IMPACT", "NYB", IFUS_START),
    "LiveCattle": ("LE", "GLBX.MDP3", "CME", GLBX_START), "LeanHogs": ("HE", "GLBX.MDP3", "CME", GLBX_START),
    "FeederCattle": ("GF", "GLBX.MDP3", "CME", GLBX_START),
    "SP500": ("ES", "GLBX.MDP3", "CME", GLBX_START), "Nasdaq100": ("NQ", "GLBX.MDP3", "CME", GLBX_START),
    "Dow": ("YM", "GLBX.MDP3", "CBT", GLBX_START), "Russell2000": ("RTY", "GLBX.MDP3", "CME", GLBX_START),
    "US_2Y": ("ZT", "GLBX.MDP3", "CBT", GLBX_START), "US_5Y": ("ZF", "GLBX.MDP3", "CBT", GLBX_START),
    "US_10Y": ("ZN", "GLBX.MDP3", "CBT", GLBX_START), "US_30Y": ("ZB", "GLBX.MDP3", "CBT", GLBX_START),
    "UltraBond": ("UB", "GLBX.MDP3", "CBT", GLBX_START),
    "EURUSD": ("6E", "GLBX.MDP3", "CME", GLBX_START), "JPYUSD": ("6J", "GLBX.MDP3", "CME", GLBX_START),
    "GBPUSD": ("6B", "GLBX.MDP3", "CME", GLBX_START), "AUDUSD": ("6A", "GLBX.MDP3", "CME", GLBX_START),
    "CADUSD": ("6C", "GLBX.MDP3", "CME", GLBX_START),
}

END_DATE = "2026-07-14"


def fetch_asset(client, root, dataset, exchange, start):
    defn = client.timeseries.get_range(
        dataset=dataset, symbols=f"{root}.FUT", stype_in="parent",
        schema="definition", start=start, end=END_DATE,
    ).to_df()

    outright = (
        defn[defn["instrument_class"] == "F"]
        [["instrument_id", "raw_symbol", "maturity_year", "maturity_month"]]
        .drop_duplicates(subset="instrument_id")
        .set_index("instrument_id")
    )

    ohlcv = client.timeseries.get_range(
        dataset=dataset, symbols=f"{root}.FUT", stype_in="parent",
        schema="ohlcv-1d", start=start, end=END_DATE,
    ).to_df()

    ohlcv = ohlcv.join(outright, on="instrument_id", how="inner")

    ohlcv["expiry_code"] = ohlcv["maturity_month"].map(NUM_TO_MONTH_CODE)
    ohlcv["expiry_year"] = ohlcv["maturity_year"].astype(int)
    ohlcv["contract_symbol"] = (
        root + ohlcv["expiry_code"] + (ohlcv["expiry_year"] % 100).astype(str).str.zfill(2) + "." + exchange
    )

    result = ohlcv.reset_index()[
        ["ts_event", "contract_symbol", "expiry_code", "expiry_year", "open", "high", "low", "close", "volume"]
    ].rename(columns={"ts_event": "date"})
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result["root"] = root
    result["exchange"] = exchange

    return result


def main() -> int:
    client = db.Historical()
    manifest_rows = []
    all_rows = []

    for asset, (root, dataset, exchange, start) in UNIVERSE.items():
        try:
            df = fetch_asset(client, root, dataset, exchange, start)
            df["asset"] = asset
            all_rows.append(df)
            manifest_rows.append({
                "asset": asset, "dataset": dataset, "start": start, "end": END_DATE,
                "rows": len(df), "contracts": df["contract_symbol"].nunique(), "status": "OK",
            })
            print(f"OK   {asset}: {len(df)} rows, {df['contract_symbol'].nunique()} contracts")
        except Exception as exc:
            manifest_rows.append({
                "asset": asset, "dataset": dataset, "start": start, "end": END_DATE,
                "rows": 0, "contracts": 0, "status": f"FAILED: {exc}",
            })
            print(f"FAIL {asset}: {exc}")

    if not all_rows:
        print("Nothing pulled - aborting without touching the archive.")
        return 1

    new_data = pd.concat(all_rows, ignore_index=True)
    new_data = new_data[
        ["date", "asset", "contract_symbol", "root", "exchange", "expiry_code", "expiry_year",
         "open", "high", "low", "close", "volume"]
    ]

    if ARCHIVE_PATH.exists():
        existing = pd.read_parquet(ARCHIVE_PATH)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (
        combined
        .drop_duplicates(subset=["date", "contract_symbol"], keep="last")
        .sort_values(["asset", "contract_symbol", "date"])
        .reset_index(drop=True)
    )
    combined.to_parquet(ARCHIVE_PATH, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["run_date"] = pd.Timestamp.now().isoformat()
    manifest_df["source"] = "databento_backfill"
    if MANIFEST_PATH.exists():
        manifest_df = pd.concat([pd.read_csv(MANIFEST_PATH), manifest_df], ignore_index=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    n_ok = sum(1 for r in manifest_rows if r["status"] == "OK")
    print(f"\nBackfill complete: {n_ok}/{len(UNIVERSE)} assets OK. Archive now {len(combined)} rows.")
    return 0 if n_ok == len(UNIVERSE) else 1


if __name__ == "__main__":
    sys.exit(main())
