"""
Public-deployment-only: populates Data/ from the private
cta-research-platform-data repo before any page reads from it.

Not needed for local development — Data/ already exists on disk there.
This module exists only so the public Streamlit Cloud deployment (which
clones just this repo, and Data/ is deliberately not part of it) has
something to render. Imported once, at the very top of app.py, before any
page's own imports run.

Requires a `GITHUB_DATA_REPO_TOKEN` Streamlit secret: a GitHub fine-grained
Personal Access Token scoped to read-only "Contents" access on the private
cta-research-platform-data repo only. Never logged, never rendered, never
readable by a visitor — it only ever touches this process's own `git clone`
call, run server-side before the app serves any page.

Debugging note (2026-07-28): clicking "Reboot app" in Streamlit Community
Cloud's dashboard does NOT immediately produce a fresh container — the
platform is wake-on-request, so the first hit after a reboot can still land
on the container mid-cycle and show a stale error (this module's own
FileNotFoundError, seen twice, both times because of exactly this). The fix
is to actually navigate into the app (click it in the left sidebar) rather
than trust the "Reboot app" click alone or the logs panel's "Updated app!"
message — that real visit is what forces the full cold start (clone this
repo, install deps, run this module's ensure_data()) to actually finish.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
DATA_REPO = "github.com/Pkargados/cta-research-platform-data.git"

# The full set of top-level Data/ entries the deployed code actually reads
# (enumerated directly from every `DATA_DIR / "..."` reference across
# src/, research/, dashboard/, jobs/) -- not just one marker file. A single-
# file check (previously just term_structure.parquet) can be fooled by a
# stale or partially-synced Data/ left over from an earlier container state:
# that's what actually happened on 2026-07-28 -- continuous_futures.parquet
# was missing while term_structure.parquet was present, so the old check
# skipped re-syncing and every strategy-performance page 404'd on a file
# that was never there. Checking the full set makes a partial sync
# self-healing (a reboot re-clones) instead of silently wrong.
_REQUIRED_FILES = [
    "Yield_Curve_6M_to_30Y.csv", "close.parquet", "continuous_futures.parquet",
    "cpi_level_index.csv", "dashboard_summary", "dashboard_summary_manifest.csv",
    "databento_transform_manifest.csv", "gscpi_data.xls", "high.parquet", "low.parquet",
    "macro_data_manifest.csv", "metadata.csv", "open.parquet",
    "overnight_fed_fund_rates_US.xlsx", "research", "term_structure.parquet",
    "term_structure_averages.parquet", "term_structure_butterflies.parquet",
    "term_structure_condors.parquet", "term_structure_manifest.csv",
    "term_structure_packs.parquet", "term_structure_spreads.parquet",
    "trade_policy_uncertainty_US.csv", "vix_data.csv", "volatility_manifest.csv",
    "yang_zhang_features.parquet",
]


def _missing_files() -> list[str]:
    return [f for f in _REQUIRED_FILES if not (DATA_DIR / f).exists()]


def ensure_data():
    missing = _missing_files()
    if not missing:
        return

    token = st.secrets.get("GITHUB_DATA_REPO_TOKEN")
    if not token:
        st.error(
            "GITHUB_DATA_REPO_TOKEN is not set in Streamlit secrets -- cannot "
            "fetch the data this dashboard needs to render. See dashboard/"
            "_bootstrap_data.py's module docstring."
        )
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        clone_url = f"https://{token}@{DATA_REPO}"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, tmp],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Scrub the token from anything that might otherwise leak it into
            # a stack trace or Streamlit's own error display.
            safe_stderr = result.stderr.replace(token, "***")
            st.error(f"Failed to fetch dashboard data: {safe_stderr}")
            st.stop()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for item in Path(tmp).iterdir():
            if item.name == ".git":
                continue
            dest = DATA_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    still_missing = _missing_files()
    if still_missing:
        st.error(
            "Data sync completed but the private data repo (cta-research-"
            "platform-data) is still missing: " + ", ".join(still_missing) +
            ". Push a current copy of these files to that repo's root, then "
            "reboot this app."
        )
        st.stop()


ensure_data()
