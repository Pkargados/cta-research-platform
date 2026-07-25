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
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
DATA_REPO = "github.com/Pkargados/cta-research-platform-data.git"

# Cheap, specific marker rather than checking every expected file --
# term_structure.parquet is read by nearly every strategy-performance page,
# so its presence is a solid proxy for "the sync already happened."
_MARKER = DATA_DIR / "term_structure.parquet"


def ensure_data():
    if _MARKER.exists():
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


ensure_data()
