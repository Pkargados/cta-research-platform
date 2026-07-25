"""
Submits Databento batch jobs for the historical backfill: all 33 CME-cleared
assets (full 2010-06-06->present history) + 5 ICE-cleared softs
(2024-07-14->present, a deliberate budget-trimmed window). Two schemas per
asset (ohlcv-1d, definition) since the transform stage needs both.

This only submits jobs - it does not wait for them or download anything. Job
IDs are saved to Data/databento_jobs.csv immediately so nothing is lost track
of, since submission is fast but processing happens server-side and can take
a while.

Run retry_databento_jobs.py separately once jobs show state=done, to download
completed jobs and resubmit any that failed or expired.
"""

import sys
from pathlib import Path

import databento as db
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
JOBS_PATH = DATA_DIR / "databento_jobs.csv"
END_DATE = "2026-07-14"

GLBX_START = "2010-06-06"
IFUS_START = "2024-07-14"

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
    # Added 2026-07-17: SOFR (short-rate gap), Lumber/LBR (commodity gap, new
    # physically-settled contract only - LBS discontinued ~2022, no forward-capture
    # path), SwissFranc/MexicanPeso (FX gaps). Jobs submitted directly, not via a
    # main() re-run of this file - see Data/databento_jobs.csv for job IDs.
    "SOFR": ("SR3", "GLBX.MDP3", "CME", GLBX_START), "Lumber": ("LBR", "GLBX.MDP3", "CME", GLBX_START),
    "SwissFranc": ("6S", "GLBX.MDP3", "CME", GLBX_START), "MexicanPeso": ("6M", "GLBX.MDP3", "CME", GLBX_START),
}

SCHEMAS = ["ohlcv-1d", "definition"]


def main() -> int:
    client = db.Historical()
    rows = []

    for asset, (root, dataset, exchange, start) in UNIVERSE.items():
        for schema in SCHEMAS:
            try:
                job = client.batch.submit_job(
                    dataset=dataset, symbols=f"{root}.FUT", stype_in="parent",
                    schema=schema, start=start, end=END_DATE, encoding="csv",
                )
                rows.append({
                    "asset": asset, "root": root, "dataset": dataset, "exchange": exchange,
                    "schema": schema, "start": start, "end": END_DATE,
                    "job_id": job["id"], "state": job["state"], "submitted": True,
                })
                print(f"OK   {asset} {schema}: {job['id']}", flush=True)
            except Exception as exc:
                rows.append({
                    "asset": asset, "root": root, "dataset": dataset, "exchange": exchange,
                    "schema": schema, "start": start, "end": END_DATE,
                    "job_id": None, "state": f"SUBMIT_FAILED: {exc}", "submitted": False,
                })
                print(f"FAIL {asset} {schema}: {exc}", flush=True)

        # Save after every asset (both schemas), not just at the end
        pd.DataFrame(rows).to_csv(JOBS_PATH, index=False)

    df = pd.DataFrame(rows)
    n_ok = df["submitted"].sum()
    print(f"\n{n_ok}/{len(df)} jobs submitted successfully. Saved to {JOBS_PATH.name}", flush=True)
    return 0 if n_ok == len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
