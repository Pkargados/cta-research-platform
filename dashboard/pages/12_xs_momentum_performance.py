"""Page 12 — Cross-Sectional Momentum (XSMOM) Performance.

Computes live at render time, reusing `research/xs_momentum.py` directly.
ONE Book, no spec selector (the paper deliberately avoids a lookback grid
"to minimize the pernicious effects of data snooping") — gross/net toggle only.

Per-asset view, not a pooled book: the tearsheet and equity curve are one
selected asset's own return stream (`backtest.engine.backtest_signal_per_asset`,
sliced to that asset). Copper reads NaN — IndustrialMetals has exactly one
member, no peer group to rank against (`data/sectors.py`'s own documented
gap, shared by every rank-based signal in this project).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

import xs_momentum as xsm_research
from signals.xs_momentum import xs_momentum_signal
from backtest.engine import backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats
from backtest.costs import liquidity_tiered_cost_bps

page_header("Cross-Sectional Momentum (XSMOM)", "Asness-Moskowitz-Pedersen (2013) — rank-weighted, one Book, monthly rebalancing.")


@st.cache_data(ttl=1200)
def _load():
    close, volume, sectors = xsm_research.load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=xsm_research.ADV_WINDOW_START)
    signal = xs_momentum_signal(close, sectors)
    return returns, cost_bps, signal


returns, cost_bps, signal = _load()
turnover = xsm_research.annualized_turnover(signal)

asset = st.selectbox("Asset", sorted(returns.columns.tolist()))
if asset == "Copper":
    st.info("Copper has no sector peer (IndustrialMetals has exactly one member) — every stat below will read N/A.")

gross = backtest_signal_per_asset(signal, returns, frequency="monthly")[asset]
net = backtest_signal_per_asset(signal, returns, frequency="monthly", cost_bps=cost_bps)[asset]
g_stats = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(gross))}

render_key_takeaways([
    "The paper is explicit it deliberately avoids a lookback grid \"to minimize "
    "the pernicious effects of data snooping\" — **ONE Book, no spec selector**.",
    f"Pooled-book turnover, for context: ~{turnover:.1f}x annualized (net ≈ gross).",
    f"**{asset}**'s own train/validation/test Sharpe: "
    f"**{g_stats['train']['Sharpe']:.2f} / {g_stats['validation']['Sharpe']:.2f} / {g_stats['test']['Sharpe']:.2f}** "
    "(gross). Pooled result was weak-to-negative overall — validation spans the 2020 "
    "COVID crash, consistent with the well-documented \"momentum crash\" phenomenon.",
])

st.divider()

gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)
asset_returns = net if gross_net == "Net of cost" else gross
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(asset_returns))}

st.subheader(f"Tearsheet — {asset}, {gross_net}")
c1, c2, c3 = st.columns(3)
for col, key, label in zip([c1, c2, c3], ("train", "validation", "test"), ("Train", "Validation", "Test")):
    s = stats_by_period[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Max DD {s['Max DD']:.2%}")

st.dataframe(
    pd.DataFrame({"Train": stats_by_period["train"], "Validation": stats_by_period["validation"], "Test": stats_by_period["test"]}).T.round(4),
    use_container_width=True,
)

equity = (1 + asset_returns.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[6], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
