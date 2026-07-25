"""Page 05 — Continuous Curve.

Computes live at render time off Data/continuous_futures.parquet (small,
already-filtered reads per-asset — same precedent page 01 set), not a
precomputed dashboard_summary/ artifact. Raw vs. ratio-back-adjusted series,
roll dates marked, per data.continuous_curve's own module docstring
(WORKFLOW.md Phase 0).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL

from data.continuous_curve import load_continuous_raw, load_continuous_backadjusted

page_header("Continuous Curve", "Raw vs. ratio-back-adjusted continuous futures series, roll dates marked.")


@st.cache_data(ttl=1200)
def _load():
    raw = load_continuous_raw()
    adj = load_continuous_backadjusted()
    return raw, adj


raw, adj = _load()
assets = sorted(raw["close"].columns)

render_key_takeaways([
    f"**{len(assets)} assets** with a real, Databento-built continuous curve — "
    "volume-crossover roll rule, ratio (proportional) back-adjustment "
    "(`data.continuous_curve`, WORKFLOW.md Phase 0), not Yahoo's opaque splice.",
    "Raw series jumps at every roll (a real price discontinuity between "
    "contracts); back-adjusted removes that jump so momentum/breakout/crossover "
    "signals see clean percentage returns across the whole history.",
])

st.divider()

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    asset = st.selectbox("Asset", assets)
with c2:
    series_view = st.radio("Series", ["Raw", "Back-adjusted", "Both"], horizontal=True, index=2)
with c3:
    frequency = st.selectbox("Chart frequency", ["Daily", "Weekly", "Monthly"], index=0)

raw_close = raw["close"][asset].dropna()
adj_close = adj["close"][asset].dropna()
is_roll = raw["is_roll_date"][asset].reindex(raw_close.index).fillna(False)

min_date, max_date = raw_close.index.min().date(), raw_close.index.max().date()
date_range = st.slider(
    "Date range", min_value=min_date, max_value=max_date, value=(min_date, max_date),
)

mask_raw = (raw_close.index.date >= date_range[0]) & (raw_close.index.date <= date_range[1])
mask_adj = (adj_close.index.date >= date_range[0]) & (adj_close.index.date <= date_range[1])
raw_view = raw_close[mask_raw]
adj_view = adj_close[mask_adj]
roll_view = is_roll[mask_raw]

_RESAMPLE = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}
freq = _RESAMPLE[frequency]
if freq is not None:
    raw_view = raw_view.resample(freq).last().dropna()
    adj_view = adj_view.resample(freq).last().dropna()

n_rolls = int(roll_view.sum())
c1, c2, c3 = st.columns(3)
c1.metric("Observations shown", len(raw_view))
c2.metric("Roll dates in range", n_rolls)
c3.metric("Latest raw close", f"{raw_close.iloc[-1]:,.3f}")

st.subheader(f"{asset} — {series_view}")
fig = go.Figure()
if series_view in ("Raw", "Both"):
    fig.add_trace(go.Scatter(
        x=raw_view.index, y=raw_view.values, mode="lines", name="Raw",
        line=dict(color=CATEGORICAL[0], width=1.5),
    ))
if series_view in ("Back-adjusted", "Both"):
    fig.add_trace(go.Scatter(
        x=adj_view.index, y=adj_view.values, mode="lines", name="Back-adjusted",
        line=dict(color=CATEGORICAL[1], width=1.5),
    ))
if series_view in ("Raw", "Both") and freq is None:
    roll_dates = roll_view[roll_view].index
    if len(roll_dates) > 0:
        fig.add_trace(go.Scatter(
            x=roll_dates, y=raw_view.reindex(roll_dates), mode="markers", name="Roll date",
            marker=dict(color=CATEGORICAL[5], size=7, symbol="diamond"),
        ))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Price",
    height=460, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.caption(
    "Roll dates only marked at Daily frequency — coarser resampling can drop the "
    "exact day a roll occurred, and the raw series' jump is the whole point being "
    "shown at Daily zoom."
)

st.divider()

with st.expander("Roll dates in range"):
    if n_rolls == 0:
        st.markdown("No roll dates in this date range.")
    else:
        roll_dates_df = pd.DataFrame({"Date": roll_view[roll_view].index.strftime("%Y-%m-%d")})
        st.dataframe(roll_dates_df, use_container_width=True, hide_index=True)

st.caption(
    "Roll rule: volume crossover (confirmed over 2 consecutive trading days) with "
    "a 5-trading-day-before-last-observed-date backstop — the documented interim "
    "rule until open interest is purchased (open interest crossover is the "
    "industry-standard method; see `data/continuous_curve.py`'s own module "
    "docstring and DATA_SCHEMA.md section 1)."
)
