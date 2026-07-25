"""Page 02 — OHLCV Coverage.

Reads: Data/dashboard_summary/ohlcv_coverage.csv
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import page_header, render_key_takeaways, require_summary_files, load_csv, CATEGORICAL, blue_gradient_style

page_header("OHLCV Coverage", "Is the core price panel complete, per asset?")

require_summary_files("ohlcv_coverage.csv")
cov = load_csv("ohlcv_coverage.csv", parse_dates=["Start", "End", "LastUpdated"])

worst = cov.sort_values("MissingPct", ascending=False).iloc[0]
render_key_takeaways([
    f"**{len(cov)} assets** tracked, real per-exchange trading calendars used for gap "
    "detection (CME/COMEX/NYMEX/CBOT/ICE differ, so a single business-day check would "
    "over-flag holidays as gaps).",
    f"Widest gap: **{worst['Asset']}** at {worst['MissingPct']:.1%} of expected sessions missing.",
])

st.divider()

st.subheader("Missing Sessions by Asset")
disp = cov.sort_values("MissingPct", ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=disp["Asset"], y=disp["MissingPct"],
    marker_color=CATEGORICAL[0],
))
fig.update_layout(
    xaxis_title=None, yaxis_title="Missing sessions (%)", yaxis_tickformat=".0%",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Full Coverage Table")
table = cov[["Asset", "Ticker", "Start", "End", "Rows", "Calendar", "MissingSessions", "MissingPct", "LastUpdated"]].copy()
table["Start"] = table["Start"].dt.strftime("%Y-%m-%d")
table["End"] = table["End"].dt.strftime("%Y-%m-%d")
table["LastUpdated"] = table["LastUpdated"].dt.strftime("%Y-%m-%d %H:%M")
table = table.sort_values("MissingPct", ascending=False)

styled = blue_gradient_style(table, subset=["MissingPct"]).format({"MissingPct": "{:.1%}"})
st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(
    "A missing session is a date the asset's exchange calendar expected trading but "
    "the price panel has no row for that asset. This can reflect a genuine data gap "
    "or a calendar mismatch (e.g. an early-close day the calendar treats as a full "
    "holiday) — worth spot-checking outliers rather than treating every missing "
    "session as a confirmed defect."
)
