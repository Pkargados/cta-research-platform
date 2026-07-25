"""
Off-machine backup of `Data/databento_raw/`'s raw Databento pull (~2.6GB) to
Google Drive, via `rclone` (a `drive.file`-scoped remote named `gdrive_cta`
-> the `CTA_Databento_Raw` folder).

Why this exists: Databento job outputs expire 30 days after
`ts_process_done` — a real, ticking deadline per job, not indefinite
storage — and this project's own local folder is not actually cloud-synced.
`rclone` was chosen over installing the Google Drive desktop client
specifically to avoid adding a persistent background sync process; the
`drive.file` OAuth scope means rclone can only see/manage files IT creates,
never the whole Drive.

Upload + checksum-verify via `rclone check` (not just "upload succeeded" —
an upload can complete with the wrong bytes, this catches that), logs every
file to `Data/databento_archive_manifest.csv`. Deliberately does NOT delete
any local file, ever, as a side effect of uploading — `cleanup_local_after_
verified_upload()` exists but defaults to a dry run and is never called
automatically; freeing local disk space is a separate, explicit, user-gated
action.

Two real bugs already found and fixed running this once (both are why this
rebuild's manifest write and process-liveness discipline look the way they
do, not a guess at good practice):
1. **Save the manifest after EVERY file, not once at the end.** The
   original run was backgrounded with a bare `&` inside a bash call rather
   than `nohup`; when that shell session ended, the still-running upload
   process died with it after 11/45 files had genuinely succeeded — but the
   manifest, written only once at the very end, was left completely empty,
   so those 11 real uploads looked lost from this script's own record even
   though they were sitting correctly on Drive. Same failure shape as an
   earlier `get_range()`-killed-process incident in the Databento backfill
   itself — evidently needed re-learning at this layer too.
2. **A resumed run can race a still-alive prior attempt.** Resuming with
   `nohup`+`disown`, the previous (first) attempt's process turned out to
   still be running; both copies read "pending" near-simultaneously and
   raced on 6 files, each independently uploading and verifying them. Not a
   data-loss bug (each upload was individually genuine and checksum-
   correct), but Google Drive allows multiple objects with an identical
   filename to coexist (unlike a real filesystem), leaving duplicate Drive
   objects and duplicate manifest rows for the same file. Fixed with
   `rclone dedupe --dedupe-mode newest` (safe only because the duplicates
   were byte-identical copies of the same source) plus a local manifest
   de-duplication — this script's own `run()` checks for and refuses to
   proceed if another instance's lock file is already present, rather than
   relying on the operator to remember not to double-launch it.

Requires `rclone` installed and a configured `gdrive_cta` remote (its own
one-time OAuth setup, done outside this script) — not runnable end-to-end in
this reconstruction session (no local rclone/OAuth state here); written and
reviewed to the same standard as the Databento API scripts, not independently
executed.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
RAW_DIR = DATA_DIR / "databento_raw"
MANIFEST_PATH = DATA_DIR / "databento_archive_manifest.csv"
LOCK_PATH = DATA_DIR / ".archive_to_drive.lock"

RCLONE_REMOTE = "gdrive_cta"
DRIVE_FOLDER = "CTA_Databento_Raw"
REMOTE_PATH = f"{RCLONE_REMOTE}:{DRIVE_FOLDER}"


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _acquire_lock():
    if LOCK_PATH.exists():
        raise RuntimeError(
            f"{LOCK_PATH.name} already exists - another archive_to_drive.py run may still be "
            "in progress (this exact race, two live copies uploading the same files "
            "simultaneously, already happened once - see module docstring). Delete the lock "
            "file only after confirming no other instance is actually running."
        )
    LOCK_PATH.write_text("locked")


def _release_lock():
    LOCK_PATH.unlink(missing_ok=True)


def _load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame(columns=["file", "status", "detail", "run_date"])


def _append_manifest_row(row: dict):
    """Read-modify-write the manifest file on EVERY row, not batched at the
    end - see module docstring, this is the direct fix for the first real
    bug found running this."""
    manifest = _load_manifest()
    manifest = manifest[manifest["file"] != row["file"]]  # replace any prior attempt for this file
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest.to_csv(MANIFEST_PATH, index=False)


def upload_and_verify(local_path: Path) -> dict:
    run_date = pd.Timestamp.now().isoformat()
    remote_target = f"{REMOTE_PATH}/{local_path.name}"

    upload = _run(["rclone", "copyto", str(local_path), remote_target])
    if upload.returncode != 0:
        return {"file": local_path.name, "status": "UPLOAD_FAILED", "detail": upload.stderr.strip()[:500], "run_date": run_date}

    check = _run(["rclone", "check", str(local_path), remote_target])
    if check.returncode != 0:
        return {"file": local_path.name, "status": "VERIFY_FAILED", "detail": check.stderr.strip()[:500], "run_date": run_date}

    return {"file": local_path.name, "status": "UPLOADED_VERIFIED", "detail": None, "run_date": run_date}


def run(files: list = None):
    _acquire_lock()
    try:
        targets = files if files is not None else sorted(RAW_DIR.glob("*.zip"))
        manifest = _load_manifest()
        already_verified = set(manifest.loc[manifest["status"] == "UPLOADED_VERIFIED", "file"])

        n_ok = 0
        for path in targets:
            if path.name in already_verified:
                print(f"  {path.name}: already UPLOADED_VERIFIED, skipping")
                n_ok += 1
                continue
            print(f"  {path.name}: uploading...")
            result = upload_and_verify(path)
            _append_manifest_row(result)
            print(f"    -> {result['status']}")
            if result["status"] == "UPLOADED_VERIFIED":
                n_ok += 1

        print(f"\n{n_ok}/{len(targets)} files uploaded and verified. Manifest: {MANIFEST_PATH}")
        return 0 if n_ok == len(targets) else 1
    finally:
        _release_lock()


def cleanup_local_after_verified_upload(dry_run: bool = True):
    """Delete local raw zips whose upload is UPLOADED_VERIFIED in the
    manifest. Defaults to `dry_run=True` (prints what WOULD be deleted, does
    not delete) and is NEVER called from `run()` or `__main__` - freeing
    local disk space is a separate, explicit, user-gated action, not a side
    effect of archiving (module docstring)."""
    manifest = _load_manifest()
    verified = set(manifest.loc[manifest["status"] == "UPLOADED_VERIFIED", "file"])
    for path in sorted(RAW_DIR.glob("*.zip")):
        if path.name not in verified:
            continue
        if dry_run:
            print(f"  [dry run] would delete {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            path.unlink()
            print(f"  deleted {path.name}")


if __name__ == "__main__":
    sys.exit(run())
