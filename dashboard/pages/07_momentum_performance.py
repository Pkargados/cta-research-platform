"""Page 07 — Momentum (TSMOM) Performance.

Computes live at render time, reusing `research/momentum.py` directly — no
duplicated data-prep/backtest logic, no precomputed dashboard_summary/
artifact. Headline spec (k=12mo, target_vol=0.40) is fixed a priori
(CLAUDE.md Rule 1/2) — the lookback×holding grid below is a descriptive
robustness view, never a spec-picker.

Per-asset view, not a pooled book: every number on this page is one selected
asset's own return stream (`backtest.engine.backtest_signal_per_asset`,
sliced to that asset), not the gross-exposure-normalized pooled portfolio.
The vol-estimator winner is still a project-level choice (decided once on
pooled TRAIN evidence, CLAUDE.md Rule 1/2) — everything downstream of that
is asset-specific.
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

import momentum as mom
from signals.momentum import tsmom_signal, momentum_grid_signals
from backtest.engine import backtest_signal_per_asset
from backtest.performance import performance_stats, simple_sharpe
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps

page_header("Time-Series Momentum (TSMOM)", "Moskowitz-Ooi-Pedersen (2012), headline spec k=12mo / h=1mo / target_vol=0.40")


@st.cache_data(ttl=1200)
def _load():
    adj, raw, included = mom.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol_estimators = mom.build_vol_estimators(adj, raw)
    comparison = mom.compare_vol_estimators(close, returns, vol_estimators)
    winner = comparison["train_sharpe"].idxmax()
    return close, returns, adj["volume"], vol_estimators, comparison, winner


close, returns, volume, vol_estimators, vol_comparison, winner = _load()
winning_vol = vol_estimators[winner]

asset = st.selectbox("Asset", sorted(close.columns.tolist()))


@st.cache_data(ttl=1200)
def _per_asset_headline(_winner, _asset):
    signal = tsmom_signal(close, vol_estimators[_winner], lookback_months=mom.HEADLINE_LOOKBACK_MONTHS, target_vol=mom.HEADLINE_TARGET_VOL)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=mom.ADV_WINDOW_START)
    gross = backtest_signal_per_asset(signal, returns, frequency="monthly", holding_months=1)[_asset]
    net = backtest_signal_per_asset(signal, returns, frequency="monthly", holding_months=1, cost_bps=cost_bps)[_asset]
    return gross, net


@st.cache_data(ttl=1200)
def _per_asset_grid(_winner, _asset):
    """Same lookback×holding grid as research/momentum.py's own, but each
    cell is this one asset's own train-period Sharpe instead of the pooled
    book's — descriptive robustness view, never a spec-picker."""
    grid_signals = momentum_grid_signals(close, vol_estimators[_winner], lookback_months_grid=mom.GRID_MONTHS, target_vol=mom.HEADLINE_TARGET_VOL)
    rows = []
    for k, signal in grid_signals.items():
        for h in mom.HOLDING_GRID:
            asset_returns = backtest_signal_per_asset(signal, returns, frequency="monthly", holding_months=h)[_asset]
            train, _, _ = train_validation_test_split(asset_returns)
            rows.append({"lookback_months": k, "holding_months": h, "train_sharpe": simple_sharpe(train)})
    return pd.DataFrame(rows).pivot(index="lookback_months", columns="holding_months", values="train_sharpe")


asset_gross, asset_net = _per_asset_headline(winner, asset)
_, _, gross_test = train_validation_test_split(asset_gross)
_, _, net_test = train_validation_test_split(asset_net)
gross_test_sharpe, net_test_sharpe = simple_sharpe(gross_test), simple_sharpe(net_test)

render_key_takeaways([
    f"Vol estimator: **{winner}** wins on TRAIN evidence (pooled, project-level "
    f"choice, Sharpe {vol_comparison.loc[winner, 'train_sharpe']:.3f} vs. "
    f"{vol_comparison.loc[[i for i in vol_comparison.index if i != winner][0], 'train_sharpe']:.3f}) — "
    "picked before looking at validation/test (CLAUDE.md Rule 1/2).",
    f"**{asset}**'s own gross test Sharpe: **{gross_test_sharpe:.3f}**.",
    f"**{asset}**'s net-of-cost test Sharpe: **{net_test_sharpe:.3f}** vs. gross "
    f"**{gross_test_sharpe:.3f}** (liquidity-tiered ADV cost assumption — a labeled "
    "placeholder, not a measured cost).",
])

st.divider()

gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)
plot_returns = asset_net if gross_net == "Net of cost" else asset_gross
stats = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(plot_returns))}

st.subheader(f"Tearsheet — {asset}, {gross_net}")
c1, c2, c3 = st.columns(3)
periods = [("train", "Train"), ("validation", "Validation"), ("test", "Test")]
for col, (key, label) in zip([c1, c2, c3], periods):
    s = stats[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Ann Vol {s['Ann Vol']:.2%} · Max DD {s['Max DD']:.2%}")

st.dataframe(
    pd.DataFrame({label: stats[key] for key, label in periods}).T.round(4),
    use_container_width=True,
)

st.subheader(f"Equity Curve — {asset}")
equity = (1 + plot_returns.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[0], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(f"Dashed lines mark the train/validation ({TRAIN_END}) and validation/test ({VALIDATION_END}) boundaries.")

st.divider()

st.subheader(f"Lookback × Holding Grid — {asset} (Train Sharpe, descriptive robustness view)")
grid = _per_asset_grid(winner, asset)
fig3 = go.Figure(go.Heatmap(
    z=grid.values, x=[str(c) for c in grid.columns], y=[str(i) for i in grid.index],
    colorscale=[[0, "#e34948"], [0.5, "#f5f0e8"], [1, "#2a78d6"]], zmid=0,
    colorbar=dict(title="Sharpe"),
))
fig3.update_layout(
    xaxis_title="Holding (months)", yaxis_title="Lookback (months)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig3 = apply_chart_theme(fig3)
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    "The paper's own Table 2 grid, reproduced on TRAIN only, for this one asset — "
    "descriptive, never used to pick the headline spec (k=12, h=1 is fixed a priori)."
)
