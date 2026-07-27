"""Page 08 — Breakout (Turtle Trading Rules) Performance.

Computes live at render time, reusing `research/breakout.py` directly. Both
Turtle systems reported as parallel Books, no auto-picked winner — the
Turtles traded both simultaneously as a blended book historically.
Gross/net toggle front and center (this signal's net-of-cost result is
meaningfully worse than gross, the whole point of showing both).
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

import breakout as bo
from backtest.engine import backtest_signal, backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe

page_header("Breakout (Donchian Channel)", "Authentic dual-channel Turtle Trading systems — no ATR money management, vol-targeted sizing instead.")


@st.cache_data(ttl=1200)
def _load():
    adj, raw = bo.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = bo.build_vol(raw)
    from backtest.costs import liquidity_tiered_cost_bps
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=bo.ADV_WINDOW_START)
    return close, returns, vol, cost_bps


close, returns, vol, cost_bps = _load()


@st.cache_data(ttl=1200)
def _signal(system_name):
    return bo.SYSTEMS[system_name](close, vol, target_vol=bo.DEFAULT_TARGET_VOL)


@st.cache_data(ttl=1200)
def _all_systems_summary():
    rows = []
    for name in bo.SYSTEMS:
        signal = _signal(name)
        turnover = bo.annualized_turnover(signal, "daily")
        gross = backtest_signal(signal, returns, frequency="daily")
        net = backtest_signal(signal, returns, frequency="daily", cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "System": name, "Annualized turnover": turnover,
            "Train (gross)": simple_sharpe(g_train), "Validation (gross)": simple_sharpe(g_val), "Test (gross)": simple_sharpe(g_test),
            "Train (net)": simple_sharpe(n_train), "Validation (net)": simple_sharpe(n_val), "Test (net)": simple_sharpe(n_test),
        })
    import pandas as pd
    return pd.DataFrame(rows).set_index("System")


summary = _all_systems_summary()
render_key_takeaways([
    "Both systems reported as parallel Books, **no auto-picked winner** — the "
    "original Turtles traded both simultaneously.",
    f"Turnover is genuinely high (System 1: {summary.loc['system1', 'Annualized turnover']:.1f}x annualized) "
    "— tested and ruled out daily resizing cadence as the cause; the real driver is "
    "pooling many independently-triggered regimes under one daily gross-exposure-"
    "normalized book, a portfolio-construction-layer issue, not a signal-level one.",
    "Pooled Sharpe is weak-to-negative gross and worse net-of-cost, reported honestly "
    "(CLAUDE.md Rule 1/2), not tuned after the fact.",
])

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    system_name = st.selectbox("System", list(bo.SYSTEMS.keys()), format_func=lambda s: {"system1": "System 1 (20d entry / 10d exit)", "system2": "System 2 (55d entry / 20d exit)"}[s])
with c2:
    frequency = st.selectbox("Resize frequency", list(bo.RESIZE_FREQUENCIES), index=0)
with c3:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader("All Systems — Gross/Net Sharpe (daily resize)")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

signal = _signal(system_name)
cb = cost_bps if gross_net == "Net of cost" else None
strategy_returns = backtest_signal(signal, returns, frequency=frequency, cost_bps=cb)
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(strategy_returns))}

st.subheader(f"Tearsheet — {system_name}, {frequency} resize, {gross_net}")
c1, c2, c3 = st.columns(3)
for col, key, label in zip([c1, c2, c3], ("train", "validation", "test"), ("Train", "Validation", "Test")):
    s = stats_by_period[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Max DD {s['Max DD']:.2%}")

equity = (1 + strategy_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[1], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Per-Asset Sharpe (full sample)")
per_asset_returns = backtest_signal_per_asset(signal, returns, frequency=frequency, cost_bps=cb)
per_asset_sharpe = per_asset_returns.apply(simple_sharpe).sort_values(ascending=False)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=per_asset_sharpe.index, y=per_asset_sharpe.values, marker_color=CATEGORICAL[1]))
fig2.update_layout(
    xaxis_title=None, yaxis_title="Sharpe",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig2 = apply_chart_theme(fig2)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
