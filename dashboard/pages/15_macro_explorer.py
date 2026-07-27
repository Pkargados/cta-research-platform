"""Page 15 — Macro Explorer.

Registry-driven explorer (family selector -> within-family series multiselect
-> 1Y/5Y/10Y/Max/Custom timeframe) across all 6 collected macro sources:
Yield Curve, Fed Funds, GSCPI, Trade Policy Uncertainty, VIX, CPI. Raw source
files read directly, display-only — distinct from page 04's "Macro" QA page
(point-in-time-correctness check against jobs/update_dashboard_summary.py's
precomputed macro_latest.csv, not exploration).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import DATA_DIR, page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

from data.macro import load_yield_curve, load_cpi

page_header("Macro Explorer", "Explore all 6 collected macro sources — extensible registry, not one chart per source.")


@st.cache_data(ttl=1200)
def _load_yield_curve():
    return load_yield_curve()


@st.cache_data(ttl=1200)
def _load_cpi():
    return load_cpi()


@st.cache_data(ttl=1200)
def _load_fed_funds():
    df = pd.read_excel(DATA_DIR / "overnight_fed_fund_rates_US.xlsx", sheet_name="Results")
    df["Effective Date"] = pd.to_datetime(df["Effective Date"])
    wide = df.pivot_table(index="Effective Date", columns="Rate Type", values="Rate (%)", aggfunc="last")
    return wide.sort_index()


@st.cache_data(ttl=1200)
def _load_gscpi():
    df = pd.read_excel(DATA_DIR / "gscpi_data.xls")
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df[["GSCPI"]]


@st.cache_data(ttl=1200)
def _load_tpu():
    df = pd.read_csv(DATA_DIR / "trade_policy_uncertainty_US.csv")
    df["Date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    return df.set_index("Date")[["daily_tpu_index"]].sort_index()


@st.cache_data(ttl=1200)
def _load_vix():
    df = pd.read_csv(DATA_DIR / "vix_data.csv", parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close"]].sort_index()


REGISTRY = {
    "Yield Curve": {"loader": _load_yield_curve, "unit": "%"},
    "Fed Funds": {"loader": _load_fed_funds, "unit": "%"},
    "GSCPI": {"loader": _load_gscpi, "unit": "index"},
    "Trade Policy Uncertainty": {"loader": _load_tpu, "unit": "index"},
    "VIX": {"loader": _load_vix, "unit": "level"},
    "CPI": {"loader": _load_cpi, "unit": "index"},
}

render_key_takeaways([
    f"**{len(REGISTRY)} macro sources** collected: {', '.join(REGISTRY.keys())} — "
    "all in `Data/`, none wired into any signal yet (see CLAUDE.md's "
    "Macro/auxiliary data row: \"collected, unused\").",
    "Raw source files, read directly — not point-in-time-correctness-checked "
    "here (that's page 04's job via `macro_latest.csv`).",
])

st.divider()

c1, c2 = st.columns([1, 2])
with c1:
    family = st.selectbox("Family", list(REGISTRY.keys()))

df = REGISTRY[family]["loader"]()
all_series = list(df.columns)

with c2:
    series_selected = st.multiselect("Series", all_series, default=all_series[: min(3, len(all_series))])

c3, c4 = st.columns([2, 2])
with c3:
    timeframe = st.radio("Timeframe", ["1Y", "5Y", "10Y", "Max", "Custom"], horizontal=True)

data_max_date = df.index.max().date()
data_min_date = df.index.min().date()

if timeframe == "Custom":
    with c4:
        custom_range = st.date_input(
            "Custom range", value=(max(data_min_date, data_max_date - timedelta(days=365)), data_max_date),
            min_value=data_min_date, max_value=data_max_date,
        )
    start_date = custom_range[0] if isinstance(custom_range, tuple) and len(custom_range) == 2 else data_min_date
    end_date = custom_range[1] if isinstance(custom_range, tuple) and len(custom_range) == 2 else data_max_date
else:
    end_date = data_max_date
    _YEARS = {"1Y": 1, "5Y": 5, "10Y": 10}
    start_date = data_min_date if timeframe == "Max" else max(data_min_date, end_date - timedelta(days=365 * _YEARS[timeframe]))

if not series_selected:
    st.info("Select at least one series to plot.")
else:
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    view = df.loc[mask, series_selected]

    st.subheader(f"{family} ({REGISTRY[family]['unit']})")
    fig = go.Figure()
    for i, col in enumerate(series_selected):
        series = view[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=col,
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=1.5),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title=REGISTRY[family]["unit"],
        height=460, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
    )
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.caption(f"{start_date} to {end_date} — {len(view)} rows, {view.notna().any(axis=1).sum()} with at least one real value.")

    with st.expander("Data table"):
        disp = view.reset_index()
        disp.columns = ["Date"] + list(view.columns)
        st.dataframe(disp.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
