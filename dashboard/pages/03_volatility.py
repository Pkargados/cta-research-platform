"""Page 03 — Yang-Zhang Volatility.

Reads: Data/dashboard_summary/volatility_snapshot.csv
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import page_header, render_key_takeaways, require_summary_files, load_csv, CATEGORICAL, blue_gradient_style, apply_chart_theme

page_header("Yang-Zhang Volatility", "What's current annualized vol, per asset and horizon?")

require_summary_files("volatility_snapshot.csv")
vol = load_csv("volatility_snapshot.csv", parse_dates=["AsOfDate"])

as_of = vol["AsOfDate"].iloc[0]
highest = vol.sort_values("yz_vol_21_ann", ascending=False).iloc[0]
render_key_takeaways([
    f"Snapshot as of **{as_of.date()}** — {len(vol)} assets, 4 horizons (21/63/126/252d), annualized.",
    f"Highest 21d vol: **{highest['Asset']}** at {highest['yz_vol_21_ann']:.1%}.",
])

st.divider()

horizon = st.selectbox("Sort/chart by horizon", ["yz_vol_21_ann", "yz_vol_63_ann", "yz_vol_126_ann", "yz_vol_252_ann"],
                        format_func=lambda c: c.replace("yz_vol_", "").replace("_ann", "") + "-day")

st.subheader("Ranked by Selected Horizon")
disp = vol.sort_values(horizon, ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(x=disp["Asset"], y=disp[horizon], marker_color=CATEGORICAL[0]))
fig.update_layout(
    xaxis_title=None, yaxis_title="Annualized vol", yaxis_tickformat=".0%",
    height=400, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Full Table — All Horizons")
horizon_rename = {"yz_vol_21_ann": "21d", "yz_vol_63_ann": "63d", "yz_vol_126_ann": "126d", "yz_vol_252_ann": "252d"}
horizon_cols = list(horizon_rename.values())
sort_col = horizon_rename[horizon]
table = vol.rename(columns=horizon_rename).drop(columns=["AsOfDate"]).sort_values(sort_col, ascending=False)
styled = blue_gradient_style(table, subset=horizon_cols).format({c: "{:.1%}" for c in horizon_cols})
st.dataframe(styled, use_container_width=True, hide_index=True)
