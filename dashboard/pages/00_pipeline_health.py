"""Page 00 — Pipeline Health.

Reads: Data/dashboard_summary/pipeline_health.csv
Also drills into the raw manifests (term_structure_manifest.csv,
macro_data_manifest.csv) for per-asset/per-source detail when a job isn't
fully OK — display only, no new computation happens on this page.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import DATA_DIR, page_header, render_key_takeaways, require_summary_files, load_csv, status_badge

page_header("Pipeline Health", "Did last night's scheduled jobs run — and what's the state of the occasional Databento transform?")

require_summary_files("pipeline_health.csv")
health = load_csv("pipeline_health.csv", parse_dates=["last_run"])

n_ok = (health["status"] == "OK").sum()
not_ok = health[health["status"] != "OK"]

if not_ok.empty:
    render_key_takeaways([f"**All {len(health)}/{len(health)} scheduled jobs** are healthy as of the last dashboard summary run."])
else:
    bad_jobs = ", ".join(f"**{r['job']}** ({r['status']})" for _, r in not_ok.iterrows())
    render_key_takeaways([
        f"**{n_ok}/{len(health)} jobs OK.** Needs attention: {bad_jobs}.",
    ])

st.divider()

st.subheader("Scheduled Jobs")
cols = st.columns(len(health))
for col, (_, row) in zip(cols, health.iterrows()):
    with col:
        st.markdown(f"**{row['job']}**")
        st.markdown(status_badge(row["status"]))
        st.caption(row["last_run"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["last_run"]) else "never run")
        st.caption(row["detail"])

st.divider()

disp = health.copy()
disp["last_run"] = disp["last_run"].dt.strftime("%Y-%m-%d %H:%M:%S")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.caption(
    "CTA_DailyDataUpdate has no manifest of its own — its status here is inferred from "
    "close.parquet's file modification time plus metadata.csv's own End-date freshness "
    "check, not a logged run record like the other jobs. A scheduled job is flagged "
    "STALE if its last run is more than 30 hours old (jobs run once daily, "
    "~6:00-6:25PM). Databento Transform is exempt from that check — it's a one-time/"
    "occasional-run utility (`databento/`, not `jobs/`), not a daily job, so its status "
    "reflects the last run's per-asset outcome, not recency."
)

st.divider()

# ------------------------------------------------------------------
# Per-item detail for jobs with granular manifests
# ------------------------------------------------------------------
st.subheader("Per-Item Detail (Latest Run)")

ts_manifest = pd.read_csv(DATA_DIR / "term_structure_manifest.csv", parse_dates=["run_date"])
latest_ts_run = ts_manifest["run_date"].max()
ts_latest = ts_manifest[ts_manifest["run_date"] == latest_ts_run]
ts_bad = ts_latest[ts_latest["status"] != "OK"]

with st.expander(f"Term Structure Capture — {len(ts_latest) - len(ts_bad)}/{len(ts_latest)} assets OK"):
    if ts_bad.empty:
        st.markdown("All assets captured successfully in the latest run.")
    else:
        st.dataframe(ts_bad, use_container_width=True, hide_index=True)

macro_manifest = pd.read_csv(DATA_DIR / "macro_data_manifest.csv", parse_dates=["run_date"])
latest_macro_run = macro_manifest["run_date"].max()
macro_latest_rows = macro_manifest[macro_manifest["run_date"] == latest_macro_run]
macro_bad = macro_latest_rows[macro_latest_rows["status"] != "OK"]

with st.expander(f"Macro Data Update — {len(macro_latest_rows) - len(macro_bad)}/{len(macro_latest_rows)} sources OK"):
    if macro_bad.empty:
        st.markdown("All sources updated successfully in the latest run.")
    else:
        st.dataframe(macro_bad, use_container_width=True, hide_index=True)

db_manifest_path = DATA_DIR / "databento_transform_manifest.csv"
if db_manifest_path.exists():
    db_manifest = pd.read_csv(db_manifest_path, parse_dates=["run_date"])
    latest_db_run = db_manifest["run_date"].max()
    db_latest = db_manifest[db_manifest["run_date"] == latest_db_run]
    db_bad = db_latest[db_latest["status"] != "OK"]

    with st.expander(f"Databento Transform — {len(db_latest) - len(db_bad)}/{len(db_latest)} assets OK (latest run only, not cumulative)"):
        st.dataframe(db_latest, use_container_width=True, hide_index=True)
        st.caption(
            "One row per asset transformed so far — not all 38, only whichever have had "
            "`databento/transform_databento.py` run against them."
        )
