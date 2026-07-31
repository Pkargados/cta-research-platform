"""Page 24 — Seasonality Book (Single Strategy Portfolio).

Computes live at render time via `dashboard/_single_strategy_pipeline.py`'s
`load_and_run_seasonality()` — a separate cached entry from Trend/Carry's own
`load_and_run()` (independent pipeline/universe), kept in that same shared
module so visiting this page never redundantly re-triggers a different page's
expensive pipeline (see that module's own docstring for the documented
Streamlit Cloud OOM crash this convention prevents).

Construction: same_month (Keloharju-Linnainmaa-Nyberg 2014/2016, replicated
for commodities by Li et al. 2023 — see the Seasonality page, 23), restricted
to the 7-name economic-driver universe (WORKFLOW.md Phase 11c's own
conviction table — Natural Gas/HeatingOil/RBOB/Corn/Soybeans/Wheat/KC_Wheat, a
hypothesis-driven restriction fixed BEFORE looking at same_month's performance
on these names, not a post-hoc trim), weekly Book rebalancing + GJR-GARCH
vol-targeting — the same Single Strategy Portfolio treatment Trend/Carry got
(WORKFLOW.md decisions #10/#11, extended here 2026-07-31).

Honest framing, not softened: this Book is WEAKER than standalone Trend in
every period (see the Multi-Strategy page's own "Does Seasonality help?"
section) — reported here as a real, documented construction anyway, same
discipline already applied to Value/XSMOM/Carry's own weaker standalone pages.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme, render_attribution_section
from _single_strategy_pipeline import load_and_run_seasonality

from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import simple_sharpe
from data.sectors import asset_to_sector

page_header("Seasonality Book", "Single Strategy Portfolio — same_month, economic-driver universe, GJR-GARCH vol-targeted")

returns, season_result, season_book = load_and_run_seasonality()
pnl = season_result["pnl"]
train, val, test = train_validation_test_split(pnl)
periods_per_year = season_book.periods_per_year

render_key_takeaways([
    "Universe fixed from a physical/economic seasonal-demand theory (WORKFLOW.md Phase "
    "11c's own conviction table), BEFORE looking at same_month's performance on these "
    "specific names — 7 assets: Natural Gas, HeatingOil, RBOB, Corn, Soybeans, Wheat, "
    "KC_Wheat. This is a different, narrower universe from the full same_month Book on "
    "the Seasonality page (23).",
    f"Sharpe — train **{simple_sharpe(train, periods_per_year=periods_per_year):.2f}**, "
    f"validation **{simple_sharpe(val, periods_per_year=periods_per_year):.2f}**, "
    f"test **{simple_sharpe(test, periods_per_year=periods_per_year):.2f}**. Every metric "
    "improves over the full-universe same_month Book (turnover ~8x lower, max DD nearly "
    "halved) but train/validation are still negative.",
    "**Does NOT beat standalone Trend alone in any period** — see the Multi-Strategy "
    "page's \"Does Seasonality help?\" section for the direct comparison. Reported here "
    "as a real, documented construction anyway, same discipline as Value/XSMOM/Carry's "
    "own weaker standalone pages.",
])

st.divider()

st.subheader("Tearsheet")
c1, c2, c3 = st.columns(3)
periods = [("train", "Train", train), ("validation", "Validation", val), ("test", "Test", test)]
for col, (key, label, series) in zip([c1, c2, c3], periods):
    sharpe = simple_sharpe(series, periods_per_year=periods_per_year)
    col.metric(f"{label} Sharpe", f"{sharpe:.3f}" if sharpe == sharpe else "N/A")
    col.caption(f"n={len(series.dropna())}")

c4, c5, c6 = st.columns(3)
c4.metric("Turnover (avg per rebalance)", f"{season_result.get('turnover', float('nan')):.3f}")
c5.metric("Max Drawdown", f"{season_result.get('max_dd', float('nan')):.2%}")
c6.metric("Valid rebalance dates", f"{season_result.get('n_rebalance_dates_valid', 0)}")

st.divider()

st.subheader("Equity Curve")
equity = (1 + pnl.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[3], width=1.8)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(f"Dashed lines mark the train/validation ({TRAIN_END}) and validation/test ({VALIDATION_END}) boundaries. Weekly rebalancing.")

st.divider()

st.subheader("Period Stats")
stats_df = pd.DataFrame({
    label: {
        "Sharpe": simple_sharpe(series, periods_per_year=periods_per_year),
        "Mean (weekly)": series.dropna().mean(),
        "Std (weekly)": series.dropna().std(),
        "N": len(series.dropna()),
    }
    for _, label, series in periods
}).T
st.dataframe(stats_df.round(4), use_container_width=True)

st.divider()

render_attribution_section(season_result["asset_contributions"], asset_to_sector(), key_prefix="seasonality")
