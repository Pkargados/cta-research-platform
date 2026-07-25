"""
databento/retry_databento_jobs.py — Checks every job in `Data/
databento_jobs.csv` (submitted by `submit_databento_jobs.py`) against the
Databento API, downloads any that are `done` but not yet present in
`Data/databento_raw/`, and resubmits any that `failed`/`expired`.

Exists to make "did every submitted job actually make it into the raw
archive" a repeatable, re-runnable check instead of the one-off manual pass
WORKFLOW.md Phase 4 describes doing once by hand (closing a 32-job gap where
several assets' jobs were submitted but never downloaded/verified).

Download method — bypasses the server-side zip-bundle endpoint
(`client.batch.download()`'s default path) entirely, downloading each job's
individual files instead via `filename_to_download` and rebuilding the zip
locally. This is not a fallback for a special case — it's the ONLY method
this script uses, because the zip-bundle endpoint is the exact thing that
produced two real, confirmed corrupted/truncated archives this project
already hit (Copper: a `504 gateway timeout` mid-transfer, reproduced twice;
Nasdaq100: a wrong-content zip predating this discovery) — see
`databento/DATA_QUALITY_REPORT.md`'s "Asset 11 update" and "Assets 30-31"
sections for the full incident and recovery record this script generalizes.
Every downloaded file is SHA256-verified against Databento's own reported
hash before being written into the rebuilt zip, matching that same recovery.

Requires `DATABENTO_API_KEY` in the environment and network access — not
runnable in this reconstruction session (no live credentials here); written
and reviewed to the same standard as `backfill_databento.py`/
`submit_databento_jobs.py` (also API-dependent, also untouched by the
2026-07-24 incident, also never executed in a sandboxed session), not
independently tested end-to-end.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import databento as db
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
JOBS_PATH = DATA_DIR / "databento_jobs.csv"
RAW_DIR = DATA_DIR / "databento_raw"

# schema -> the "_Definition"/"_OHLCV" suffix build_continuous_curve.py's own
# sibling scripts (transform_databento.py) expect on the local zip filename.
SCHEMA_SUFFIX = {"definition": "Definition", "ohlcv-1d": "OHLCV"}

TERMINAL_DONE = {"done"}
TERMINAL_FAILED = {"expired", "cancelled"}  # eligible for resubmission, per direct instruction


def _zip_filename(row) -> str:
    if row["dataset"] == "GLBX.MDP3":
        prefix = f"CME_Globex_MDP3.0_{row['root']}_FUT"
    else:
        prefix = f"ICE_Futures_US_{row['root']}"
    return f"{prefix}_{SCHEMA_SUFFIX[row['schema']]}.zip"


def _download_job_as_zip(client, job_id: str, dest_path: Path) -> int:
    """Download every file in a completed job individually
    (`filename_to_download`), SHA256-verify each against Databento's own
    reported hash, and bundle them into a local zip (`ZIP_STORED` — the
    files are already zstd-compressed, no reason to compress twice).
    Returns the number of files written. Raises on any hash mismatch rather
    than writing a silently-corrupt archive."""
    files = client.batch.list_files(job_id)
    tmp_path = dest_path.with_suffix(".zip.tmp")
    n_written = 0
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            filename = f["filename"]
            content = client.batch.download(job_id, filename_to_download=filename)
            expected_hash = f.get("hash")
            if expected_hash:
                actual_hash = hashlib.sha256(content).hexdigest()
                # Databento reports hashes prefixed with the algorithm name
                # (e.g. "sha256:...") in some SDK versions - compare against
                # whichever form is present rather than assuming one.
                if actual_hash not in expected_hash:
                    raise ValueError(f"hash mismatch for {filename}: expected {expected_hash}, got sha256:{actual_hash}")
            zf.writestr(filename, content)
            n_written += 1
    tmp_path.replace(dest_path)
    return n_written


def check_and_download(client, jobs_df: pd.DataFrame) -> pd.DataFrame:
    updated_rows = []
    for _, row in jobs_df.iterrows():
        row = row.to_dict()
        job_id = row.get("job_id")
        if not row.get("submitted") or not job_id:
            updated_rows.append(row)
            continue

        dest = RAW_DIR / _zip_filename(row)
        if dest.exists():
            row["state"] = "done"
            updated_rows.append(row)
            continue

        try:
            jobs = client.batch.list_jobs(states=["received", "queued", "processing", "done", "expired"])
            match = next((j for j in jobs if j["id"] == job_id), None)
            state = match["state"] if match else "unknown"
        except Exception as exc:
            print(f"  {row['asset']} {row['schema']}: status check failed ({exc})")
            updated_rows.append(row)
            continue

        row["state"] = state
        if state in TERMINAL_DONE:
            try:
                n = _download_job_as_zip(client, job_id, dest)
                print(f"  {row['asset']} {row['schema']}: downloaded {n} files -> {dest.name}")
            except Exception as exc:
                print(f"  {row['asset']} {row['schema']}: download FAILED ({exc})")
        elif state in TERMINAL_FAILED:
            print(f"  {row['asset']} {row['schema']}: {state}, will resubmit")
        else:
            print(f"  {row['asset']} {row['schema']}: still {state}")

        updated_rows.append(row)
    return pd.DataFrame(updated_rows)


def resubmit_failed(client, jobs_df: pd.DataFrame) -> pd.DataFrame:
    rows = jobs_df.to_dict("records")
    for row in rows:
        if row.get("state") not in TERMINAL_FAILED:
            continue
        try:
            job = client.batch.submit_job(
                dataset=row["dataset"], symbols=f"{row['root']}.FUT", stype_in="parent",
                schema=row["schema"], start=row["start"], end=row["end"], encoding="csv",
            )
            print(f"  RESUBMITTED {row['asset']} {row['schema']}: {job['id']}")
            row["job_id"], row["state"], row["submitted"] = job["id"], job["state"], True
        except Exception as exc:
            print(f"  RESUBMIT FAILED {row['asset']} {row['schema']}: {exc}")
    return pd.DataFrame(rows)


def run():
    if not JOBS_PATH.exists():
        print(f"{JOBS_PATH} not found - nothing to retry. Run submit_databento_jobs.py first.")
        return 1

    jobs_df = pd.read_csv(JOBS_PATH)
    client = db.Historical()

    print("--- Checking job status / downloading completed jobs ---")
    jobs_df = check_and_download(client, jobs_df)

    n_failed = (jobs_df["state"].isin(TERMINAL_FAILED)).sum()
    if n_failed:
        print(f"\n--- Resubmitting {n_failed} failed/expired job(s) ---")
        jobs_df = resubmit_failed(client, jobs_df)

    jobs_df.to_csv(JOBS_PATH, index=False)

    n_done = (jobs_df["state"] == "done").sum()
    print(f"\n{n_done}/{len(jobs_df)} jobs done. Updated {JOBS_PATH.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
