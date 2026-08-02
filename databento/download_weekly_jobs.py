"""
Polls/downloads this week's Databento batch jobs (submitted by
submit_weekly_jobs.py), writing to the FIXED per-root zip filenames
databento_transform.py already expects -- overwriting each week is what
lets the transform step run completely unmodified. Before overwriting, the
previous zip is archived to Data/databento_raw_weekly_archive/<week_id>/
(audit trail + a cheap idempotency aid: if a later pipeline step fails,
this week's exact raw input is still recoverable without re-paying for the
API call).

Download method matches the recovered retry_databento_jobs.py exactly:
bypasses the server-side zip-bundle endpoint (produced corrupted archives
in practice, per that script's own docstring) -- downloads each file
individually via filename_to_download, SHA256-verifies against Databento's
own reported hash, and rebuilds the zip locally.
"""

import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
JOBS_PATH = DATA_DIR / "databento_weekly_jobs.csv"
RAW_DIR = DATA_DIR / "databento_raw"
ARCHIVE_DIR = DATA_DIR / "databento_raw_weekly_archive"

SCHEMA_SUFFIX = {"definition": "Definition", "ohlcv-1d": "OHLCV"}
TERMINAL_DONE = {"done"}
# The recovered original script's TERMINAL_FAILED included "cancelled", but
# the installed SDK's JobState enum (databento 0.81.0) only defines
# queued/processing/done/expired -- confirmed live, not assumed. "cancelled"
# is no longer a real state to check for.
TERMINAL_FAILED = {"expired"}  # eligible for resubmission


def _zip_filename(row) -> str:
    return f"CME_Globex_MDP3.0_{row['root']}_FUT_{SCHEMA_SUFFIX[row['schema']]}.zip"


def _archive_existing(dest_path: Path, week_id: str):
    if not dest_path.exists():
        return
    archive_subdir = ARCHIVE_DIR / week_id
    archive_subdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest_path, archive_subdir / dest_path.name)


def _download_job_as_zip(client, job_id: str, dest_path: Path) -> int:
    """Download every file in a completed job individually, SHA256-verify
    each against Databento's own reported hash, and bundle into a local zip
    (ZIP_STORED -- already zstd-compressed, no reason to compress twice).
    Raises on any hash mismatch rather than writing a silently-corrupt
    archive.

    The recovered original script assumed `client.batch.download(...,
    filename_to_download=...)` returns raw bytes directly - confirmed live
    (not assumed) that the installed SDK (databento 0.81.0) instead writes
    the file to `{output_dir}/{job_id}/{filename}` and returns
    `list[Path]`. Downloads into a scratch temp dir (cleaned up
    automatically) and reads the bytes back from there."""
    files = client.batch.list_files(job_id)
    tmp_path = dest_path.with_suffix(".zip.tmp")
    n_written = 0
    with tempfile.TemporaryDirectory() as scratch_dir:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for f in files:
                filename = f["filename"]
                downloaded = client.batch.download(job_id, output_dir=scratch_dir, filename_to_download=filename)
                content = downloaded[0].read_bytes()
                expected_hash = f.get("hash")
                if expected_hash:
                    actual_hash = hashlib.sha256(content).hexdigest()
                    if actual_hash not in expected_hash:
                        raise ValueError(f"hash mismatch for {filename}: expected {expected_hash}, got sha256:{actual_hash}")
                zf.writestr(filename, content)
                n_written += 1
    tmp_path.replace(dest_path)
    return n_written


def _save(jobs_df: pd.DataFrame, week_id: str, updated_df: pd.DataFrame):
    other_weeks = jobs_df[jobs_df["week_id"] != week_id]
    pd.concat([other_weeks, updated_df], ignore_index=True).to_csv(JOBS_PATH, index=False)


def poll_and_download(week_id: str) -> bool:
    """Returns True iff every job for `week_id` is downloaded and present on
    disk. Downloads whatever's newly `done`, resubmits whatever's
    `expired`/`cancelled`, leaves in-progress jobs alone. Safe to call
    repeatedly (idempotent) -- a row already marked "downloaded" from a
    prior invocation this week is skipped, not re-downloaded."""
    if not JOBS_PATH.exists():
        raise RuntimeError(f"{JOBS_PATH} not found - submit_weekly_jobs.py hasn't run for week {week_id}")

    import databento as db
    client = db.Historical()

    jobs_df = pd.read_csv(JOBS_PATH)
    week_jobs = jobs_df[jobs_df["week_id"] == week_id].copy()
    if week_jobs.empty:
        raise RuntimeError(f"No jobs recorded for week {week_id} in {JOBS_PATH}")

    all_jobs = client.batch.list_jobs(states=["queued", "processing", "done", "expired"])
    jobs_by_id = {j["id"]: j for j in all_jobs}

    updated_rows = []
    for _, row in week_jobs.iterrows():
        row = row.to_dict()
        job_id = row.get("job_id")

        if not row.get("submitted") or not job_id:
            updated_rows.append(row)
            continue
        if row.get("state") == "downloaded":
            updated_rows.append(row)
            continue

        match = jobs_by_id.get(job_id)
        state = match["state"] if match else "unknown"
        row["state"] = state
        dest = RAW_DIR / _zip_filename(row)

        if state in TERMINAL_DONE:
            try:
                _archive_existing(dest, week_id)
                n = _download_job_as_zip(client, job_id, dest)
                row["state"] = "downloaded"
                print(f"  {row['asset']} {row['schema']}: downloaded {n} files -> {dest.name}")
            except Exception as exc:
                print(f"  {row['asset']} {row['schema']}: download FAILED ({exc})")
        elif state in TERMINAL_FAILED:
            print(f"  {row['asset']} {row['schema']}: {state}, resubmitting")
            try:
                new_job = client.batch.submit_job(
                    dataset=row["dataset"], symbols=f"{row['root']}.FUT", stype_in="parent",
                    schema=row["schema"], start=row["start"], end=row["end"], encoding="csv",
                )
                row["job_id"], row["state"], row["submitted"] = new_job["id"], new_job["state"], True
                print(f"  RESUBMITTED {row['asset']} {row['schema']}: {new_job['id']}")
            except Exception as exc:
                print(f"  RESUBMIT FAILED {row['asset']} {row['schema']}: {exc}")
        else:
            print(f"  {row['asset']} {row['schema']}: still {state}")

        updated_rows.append(row)

    updated_df = pd.DataFrame(updated_rows)
    _save(jobs_df, week_id, updated_df)

    n_done = (updated_df["state"] == "downloaded").sum()
    print(f"\n{n_done}/{len(updated_df)} jobs downloaded for week {week_id}.")
    return n_done == len(updated_df)


if __name__ == "__main__":
    wk = sys.argv[1] if len(sys.argv) > 1 else None
    if not wk:
        print("Usage: python download_weekly_jobs.py <week_id>")
        sys.exit(1)
    sys.exit(0 if poll_and_download(wk) else 1)
