"""
Submits weekly incremental Databento batch jobs for the CME-only (GLBX.MDP3)
universe -- 37 roots x 2 schemas (ohlcv-1d, definition), parent symbology
(stype_in="parent", symbols=f"{root}.FUT"). Adapted from the original
one-time historical backfill script (databento/submit_databento_jobs.py,
recovered from git history at 49332c8^ -- deliberately untracked 2026-07-25,
not lost) for a small, recurring incremental window instead of full history.

ICE (IFUS.IMPACT) roots are explicitly out of scope -- those 5 softs stay on
their existing/manual cadence, per direct instruction (CLAUDE.md's CME-only
automation scope).

This module does not decide the fetch window on its own -- compute_window()
is a pure function (no network call), so the date math is testable in
isolation. Only submit_jobs() touches the live, paid API.
"""

import datetime as dt
import re
import sys
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.databento_transform import UNIVERSE, ICE_ROOTS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
JOBS_PATH = DATA_DIR / "databento_weekly_jobs.csv"

SCHEMAS = ["ohlcv-1d", "definition"]
DATASET = "GLBX.MDP3"

# The 37 CME-listed assets: everything in UNIVERSE whose root isn't one of
# the 5 ICE (IFUS.IMPACT) roots databento_transform.py already identified --
# single source of truth, not re-derived here.
CME_UNIVERSE = {asset: root for asset, (root, _exch) in UNIVERSE.items() if root not in ICE_ROOTS}

# Historical backfill's own frozen end date (submit_databento_jobs.py,
# recovered) -- the natural fallback start if no weekly cycle has ever
# completed yet.
BACKFILL_END_DATE = "2026-07-14"

# Small overlap so a missed/late run never leaves a silent gap -- the
# transform stage's dedup-on-write (keep="last") makes re-pulling a couple
# of overlap days harmless.
OVERLAP_DAYS = 2

# Found live (first real 74-job submission, 2026-08-02): submitting all 74
# jobs back-to-back with no delay hits Databento's rate limit after ~10
# assets (20 requests) -- every subsequent request 429s until the window
# clears. A modest per-request throttle avoids tripping the limit in the
# first place; the retry-with-backoff below is a safety net for whatever
# slips through anyway (rate limits can change).
SUBMIT_THROTTLE_SECONDS = 1.0
MAX_SUBMIT_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 30
_RETRY_AFTER_RE = re.compile(r"Retry in (\d+)s")


def compute_window(last_successful_end_date: str = None, today: dt.date = None) -> tuple:
    """Pure date-window computation, no network call -- testable in
    isolation. `last_successful_end_date` (a "YYYY-MM-DD" string, or None if
    no weekly cycle has ever completed) comes from the pipeline checkpoint;
    None means start from the historical backfill's own frozen end date
    instead of guessing a lookback window."""
    today = today or dt.date.today()
    if last_successful_end_date:
        last_end = dt.date.fromisoformat(last_successful_end_date)
        start = last_end - dt.timedelta(days=OVERLAP_DAYS)
    else:
        start = dt.date.fromisoformat(BACKFILL_END_DATE)
    return start.isoformat(), today.isoformat()


def _save(rows: list):
    """Rewrites Data/databento_weekly_jobs.csv with every other week's rows
    untouched plus this week's rows-so-far -- called after every asset (not
    just at the end), so a partial-submit failure doesn't lose track of jobs
    already paid for."""
    new_df = pd.DataFrame(rows)
    week_id = new_df["week_id"].iloc[0]
    if JOBS_PATH.exists():
        existing = pd.read_csv(JOBS_PATH)
        existing = existing[existing["week_id"] != week_id]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(JOBS_PATH, index=False)


def _submit_one(client, dataset: str, root: str, schema: str, start: str, end: str):
    """Submits one job, retrying on 429 with backoff (parses Databento's own
    "Retry in Ns" hint when present, falls back to a fixed delay otherwise).
    Raises the last exception if every retry is exhausted."""
    last_exc = None
    for attempt in range(MAX_SUBMIT_RETRIES):
        try:
            return client.batch.submit_job(
                dataset=dataset, symbols=f"{root}.FUT", stype_in="parent",
                schema=schema, start=start, end=end, encoding="csv",
            )
        except Exception as exc:
            last_exc = exc
            if "429" not in str(exc) and "Too Many Requests" not in str(exc):
                raise
            m = _RETRY_AFTER_RE.search(str(exc))
            delay = int(m.group(1)) + 1 if m else DEFAULT_RETRY_BACKOFF_SECONDS
            print(f"    429, retrying in {delay}s (attempt {attempt + 1}/{MAX_SUBMIT_RETRIES})", flush=True)
            time.sleep(delay)
    raise last_exc


def submit_jobs(start: str, end: str, week_id: str) -> pd.DataFrame:
    """Submits one batch job per (asset, schema) across the 37-asset CME
    universe -- 74 jobs. Saves after every asset, same discipline as the
    recovered original script. Throttled (SUBMIT_THROTTLE_SECONDS between
    requests) with 429 retry-with-backoff as a safety net -- found live
    that submitting all 74 back-to-back trips Databento's rate limit after
    ~10 assets."""
    import databento as db
    client = db.Historical()
    rows = []

    for asset, root in CME_UNIVERSE.items():
        for schema in SCHEMAS:
            try:
                job = _submit_one(client, DATASET, root, schema, start, end)
                rows.append({
                    "week_id": week_id, "asset": asset, "root": root, "dataset": DATASET,
                    "schema": schema, "start": start, "end": end,
                    "job_id": job["id"], "state": job["state"], "submitted": True,
                })
                print(f"OK   {asset} {schema}: {job['id']}", flush=True)
            except Exception as exc:
                rows.append({
                    "week_id": week_id, "asset": asset, "root": root, "dataset": DATASET,
                    "schema": schema, "start": start, "end": end,
                    "job_id": None, "state": f"SUBMIT_FAILED: {exc}", "submitted": False,
                })
                print(f"FAIL {asset} {schema}: {exc}", flush=True)
            time.sleep(SUBMIT_THROTTLE_SECONDS)

        _save(rows)

    df = pd.DataFrame(rows)
    n_ok = df["submitted"].sum()
    print(f"\n{n_ok}/{len(df)} jobs submitted successfully for week {week_id}.", flush=True)
    return df


def resubmit_failed(week_id: str) -> int:
    """Retries every row for `week_id` still marked submitted=False (a
    submission that exhausted its retries during the original submit_jobs()
    call). Returns the number successfully resubmitted. Safe to call
    repeatedly -- rows that succeed are updated in place; rows still failing
    stay marked submitted=False for the next attempt."""
    import databento as db
    if not JOBS_PATH.exists():
        return 0
    jobs_df = pd.read_csv(JOBS_PATH)
    week_jobs = jobs_df[jobs_df["week_id"] == week_id]
    to_retry = week_jobs[~week_jobs["submitted"]]
    if to_retry.empty:
        return 0

    client = db.Historical()
    n_ok = 0
    updated = []
    for _, row in to_retry.iterrows():
        row = row.to_dict()
        try:
            job = _submit_one(client, row["dataset"], row["root"], row["schema"], row["start"], row["end"])
            row["job_id"], row["state"], row["submitted"] = job["id"], job["state"], True
            print(f"RESUBMITTED {row['asset']} {row['schema']}: {job['id']}", flush=True)
            n_ok += 1
        except Exception as exc:
            row["state"] = f"SUBMIT_FAILED: {exc}"
            print(f"STILL FAILING {row['asset']} {row['schema']}: {exc}", flush=True)
        updated.append(row)
        time.sleep(SUBMIT_THROTTLE_SECONDS)

    updated_df = pd.DataFrame(updated)
    other = jobs_df[~((jobs_df["week_id"] == week_id) & (~jobs_df["submitted"]))]
    pd.concat([other, updated_df], ignore_index=True).to_csv(JOBS_PATH, index=False)
    return n_ok


if __name__ == "__main__":
    s, e = compute_window()
    wk = dt.date.today().isoformat()
    print(f"Submitting weekly jobs: {s} -> {e} (week_id={wk})")
    submit_jobs(s, e, wk)
