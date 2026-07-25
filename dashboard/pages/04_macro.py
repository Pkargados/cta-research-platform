"""Page 04 — Macro Data.

Reads: Data/dashboard_summary/macro_latest.csv (point-in-time-correct latest
       values, via src/macro_point_in_time.py's accessors — not re-derived
       here). Raw source files read directly, display-only, for the history
       sparkline under each tile.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import DATA_DIR, page_header, render_key_takeaways, require_summary_files, load_csv, CATEGORICAL

page_header("Macro Data", "What's the latest point-in-time-correct macro reading?")

require_summary_files("macro_latest.csv")
macro = load_csv("macro_latest.csv", parse_dates=["as_of_date", "value_date"])

n_ok = (macro["data_status"] == "OK").sum()
render_key_takeaways([
    f"**{n_ok}/{len(macro)} sources** have a value available as of today. Values shown "
    "are point-in-time-correct (each source's confirmed publication lag already "
    "applied) — not simply the latest row in the raw file, which would look ahead.",
])

st.divider()

SOURCE_LABELS = {
    "yield_curve": "Yield Curve (10Y)",
    "fed_funds_rates": "Fed Funds (EFFR)",
    "gscpi": "GSCPI",
    "trade_policy_uncertainty": "Trade Policy Uncertainty",
}

cols = st.columns(4)
for col, (_, row) in zip(cols, macro.iterrows()):
    with col:
        label = SOURCE_LABELS.get(row["source"], row["source"])
        if row["data_status"] == "OK":
            st.metric(label, f"{row['key_value']:,.2f}", help=row["key_label"])
            st.caption(f"as of {row['value_date'].date()}")
        else:
            st.metric(label, "—")
            st.caption(row["data_status"])

st.divider()

st.subheader("Recent History")

HISTORY_LOADERS = {
    "yield_curve": lambda: pd.read_csv(DATA_DIR / "Yield_Curve_6M_to_30Y.csv", parse_dates=["Date"])[["Date", "10Y"]].rename(columns={"10Y": "value"}),
    "trade_policy_uncertainty": lambda: (
        pd.read_csv(DATA_DIR / "trade_policy_uncertainty_US.csv")
        .assign(Date=lambda d: pd.to_datetime(dict(year=d["year"], month=d["month"], day=d["day"])))
        [["Date", "daily_tpu_index"]].rename(columns={"daily_tpu_index": "value"})
    ),
}

source = st.selectbox("Select source", list(SOURCE_LABELS.keys()), format_func=lambda s: SOURCE_LABELS[s])

if source not in HISTORY_LOADERS:
    st.info(f"{SOURCE_LABELS[source]} history chart not available on this page — raw file needs sheet/skiprows parsing (see src/macro_point_in_time.py).")
else:
    hist = HISTORY_LOADERS[source]().sort_values("Date").tail(500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["value"], mode="lines", line=dict(color=CATEGORICAL[0], width=2)))
    fig.update_layout(
        xaxis_title="Date", yaxis_title=SOURCE_LABELS[source],
        height=360, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Full Detail")
disp = macro.copy()
disp["as_of_date"] = disp["as_of_date"].dt.strftime("%Y-%m-%d")
disp["value_date"] = disp["value_date"].dt.strftime("%Y-%m-%d")
st.dataframe(disp, use_container_width=True, hide_index=True)
