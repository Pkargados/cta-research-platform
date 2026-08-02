"""
jobs/weekly_databento_pipeline.py -- the single orchestrator for the weekly
CME-only Databento refresh -> transform -> continuous curve -> Trend/RV/
Multi-Strategy backtest -> written summary pipeline (CLAUDE.md's CME-only
automation scope; the 5 ICE softs stay on their existing/manual cadence,
unaffected).

Idempotent and checkpoint-driven (Data/databento_weekly_checkpoint.json),
not a single long-blocking run -- Databento batch jobs process server-side
with unknown turnaround, so this script is designed to be invoked
repeatedly (Task Scheduler: every 30 min, Friday 6PM through Saturday noon)
and pick up wherever the last invocation left off:
  - no-ops immediately if this week's cycle is already complete
  - submits this week's jobs exactly once
  - polls/downloads on every invocation until all jobs are done
  - runs transform -> continuous curve -> backtest+report exactly once,
    only after every job is downloaded

Non-zero exit code on any step failure (Task Scheduler needs this to flag a
failed run, not silently succeed) -- the checkpoint records which step was
in progress and the last error, so a re-invocation retries from there
rather than from scratch, and a human can diagnose a stuck run without
re-reading the whole log.
"""

import datetime as dt
import json
import sys
import traceback
from pathlib import Path

# This repo's own path contains non-ASCII characters (Greek) -- on Windows'
# default console codepage, prints containing a path built from __file__
# crash with UnicodeEncodeError (found live in databento_transform.py's own
# final status print). This guard covers the whole process, including every
# module this orchestrator imports.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "databento"))
sys.path.insert(0, str(_ROOT / "src"))

DATA_DIR = _ROOT / "Data"
CHECKPOINT_PATH = DATA_DIR / "databento_weekly_checkpoint.json"


def current_week_id(today: dt.date = None) -> str:
    """The most recent Friday <= today, as an ISO date string -- the weekly
    cycle identifier. Matches this project's own Book rebalancing
    convention (COV_FREQ="W-FRI" in research/single_strategy_portfolios.py)
    rather than an arbitrary choice."""
    today = today or dt.date.today()
    friday = today - dt.timedelta(days=(today.weekday() - 4) % 7)
    return friday.isoformat()


def _default_checkpoint() -> dict:
    return {
        "week_id": None, "cycle_status": "not_started",
        "start_date": None, "end_date": None,
        "last_successful_end_date": None,
        "last_updated": None, "last_error": None,
    }


def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return _default_checkpoint()


def _save_checkpoint(cp: dict):
    cp["last_updated"] = dt.datetime.now().isoformat(timespec="seconds")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    tmp.replace(CHECKPOINT_PATH)


def _log(msg: str):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def run() -> int:
    week_id = current_week_id()
    cp = _load_checkpoint()

    if cp.get("week_id") != week_id:
        if cp.get("cycle_status") not in (None, "not_started", "complete"):
            _log(f"WARNING: prior cycle for week {cp.get('week_id')} was left in state "
                 f"'{cp.get('cycle_status')}' (not complete) - starting a fresh cycle for {week_id} "
                 f"anyway. Prior cycle's last error: {cp.get('last_error')}")
        cp = _default_checkpoint()
        cp["week_id"] = week_id
        cp["last_successful_end_date"] = _load_checkpoint().get("last_successful_end_date")

    if cp["cycle_status"] == "complete":
        _log(f"Week {week_id} already complete - nothing to do.")
        return 0

    try:
        import submit_weekly_jobs as swj
        import download_weekly_jobs as dwj

        if cp["cycle_status"] == "not_started":
            start, end = swj.compute_window(cp.get("last_successful_end_date"))
            _log(f"Submitting jobs for week {week_id}: {start} -> {end}")
            swj.submit_jobs(start, end, week_id)
            cp["start_date"], cp["end_date"] = start, end
            cp["cycle_status"] = "submitted"
            _save_checkpoint(cp)

        if cp["cycle_status"] == "submitted":
            n_resubmitted = swj.resubmit_failed(week_id)
            if n_resubmitted:
                _log(f"Resubmitted {n_resubmitted} previously-failed submission(s)")
            _log(f"Polling/downloading jobs for week {week_id}")
            all_done = dwj.poll_and_download(week_id)
            if not all_done:
                _log("Jobs still processing - exiting cleanly, will retry next invocation.")
                _save_checkpoint(cp)
                return 0
            cp["cycle_status"] = "downloaded"
            _save_checkpoint(cp)

        if cp["cycle_status"] == "downloaded":
            from data.databento_transform import run as transform_run
            cme_assets = list(swj.CME_UNIVERSE.keys())
            _log(f"Transforming {len(cme_assets)} CME assets")
            transform_run(assets=cme_assets)
            cp["cycle_status"] = "transformed"
            _save_checkpoint(cp)

        if cp["cycle_status"] == "transformed":
            import build_continuous_curve as bcc
            _log("Rebuilding continuous futures curve")
            bcc.run()
            cp["cycle_status"] = "curve_built"
            _save_checkpoint(cp)

        if cp["cycle_status"] == "curve_built":
            import _weekly_summary as summary
            _log("Running Trend/RV/Multi-Strategy backtests and writing report")
            summary.run(week_id=week_id)
            cp["cycle_status"] = "complete"
            cp["last_successful_end_date"] = cp["end_date"]
            _save_checkpoint(cp)

        _log(f"Week {week_id} complete.")
        return 0

    except Exception:
        err = traceback.format_exc()
        _log(f"FAILED at state={cp['cycle_status']}:\n{err}")
        cp["last_error"] = err
        _save_checkpoint(cp)
        return 1


if __name__ == "__main__":
    sys.exit(run())
