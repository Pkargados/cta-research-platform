"""Page 11 — Carry Performance.

Computes live at render time, reusing `research/carry.py` directly. Four
paper-matched specs (Koijen-Moskowitz-Pedersen-Vrugt 2018), no headline pick.
Proxy (ICE softs, back-differenced) vs. real-quote assets visually
distinguished throughout — CLAUDE.md Rule 4: never blended unlabeled.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL

import carry as carry_research
from signals.carry import build_all_carry_signals
from backtest.engine import backtest_signal, backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe

page_header("Carry", "Koijen-Moskowitz-Pedersen-Vrugt (2018) — rank-weighted cross-section, monthly rebalancing.")


@st.cache_data(ttl=1200)
def _load():
    close, volume, included, sectors = carry_research.load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    from backtest.costs import liquidity_tiered_cost_bps
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=carry_research.ADV_WINDOW_START)
    carry_panel, is_proxy = carry_research.build_carry_panel(included)
    return returns, sectors, cost_bps, carry_panel, is_proxy


returns, sectors, cost_bps, carry_panel, is_proxy = _load()
proxy_assets = sorted(is_proxy[is_proxy].index.tolist())


@st.cache_data(ttl=1200)
def _all_specs_summary():
    signals = build_all_carry_signals(carry_panel, sectors)
    rows = []
    for name, signal in signals.items():
        turnover = carry_research.annualized_turnover(signal)
        gross = backtest_signal(signal, returns[carry_panel.columns], frequency="monthly")
        net = backtest_signal(signal, returns[carry_panel.columns], frequency="monthly", cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "Spec": name, "Annualized turnover": turnover,
            "Train (gross)": simple_sharpe(g_train), "Validation (gross)": simple_sharpe(g_val), "Test (gross)": simple_sharpe(g_test),
            "Train (net)": simple_sharpe(n_train), "Validation (net)": simple_sharpe(n_val), "Test (net)": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("Spec")


summary = _all_specs_summary()
render_key_takeaways([
    "Four parallel specs (carry1m, carry1-12, carry timing ref=0, carry timing "
    "ref=sector-mean), **no headline pick**.",
    f"**{len(proxy_assets)} ICE softs** ({', '.join(proxy_assets)}) use a back-"
    "differenced PROXY (no real spread data exists for them) — flagged `is_proxy` "
    "end-to-end, never blended unlabeled with the 33 real-quote assets.",
    "Turnover collapsed to ~0.7-3.3x annualized once rebalancing matched the "
    "paper's monthly cadence — genuinely mixed result, no spec robustly positive "
    "across all three periods.",
])

st.divider()

c1, c2 = st.columns(2)
with c1:
    spec = st.selectbox("Spec", ["carry1m", "carry1_12", "carry_timing_zero", "carry_timing_mean"])
with c2:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader("All 4 Specs — Gross/Net Sharpe (monthly)")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

signals = build_all_carry_signals(carry_panel, sectors)
signal = signals[spec]
cb = cost_bps if gross_net == "Net of cost" else None
strategy_returns = backtest_signal(signal, returns[carry_panel.columns], frequency="monthly", cost_bps=cb)
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(strategy_returns))}

st.subheader(f"Tearsheet — {spec}, {gross_net}")
c1, c2, c3 = st.columns(3)
for col, key, label in zip([c1, c2, c3], ("train", "validation", "test"), ("Train", "Validation", "Test")):
    s = stats_by_period[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Max DD {s['Max DD']:.2%}")

equity = (1 + strategy_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[4], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Per-Asset Sharpe (full sample, gross) — proxy assets marked")
per_asset_returns = backtest_signal_per_asset(signal, returns[carry_panel.columns], frequency="monthly")
per_asset_sharpe = per_asset_returns.apply(simple_sharpe).sort_values(ascending=False)
bar_colors = [CATEGORICAL[5] if a in proxy_assets else CATEGORICAL[4] for a in per_asset_sharpe.index]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=per_asset_sharpe.index, y=per_asset_sharpe.values, marker_color=bar_colors))
fig2.update_layout(
    xaxis_title=None, yaxis_title="Sharpe",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
st.caption(f"Orange bars = proxy assets (back-differenced, no real spread quote): {', '.join(proxy_assets)}.")

st.divider()

st.subheader("Carry Level — Real vs. Proxy")
asset = st.selectbox("Asset", sorted(carry_panel.columns))
series = carry_panel[asset].dropna()
color = CATEGORICAL[5] if asset in proxy_assets else CATEGORICAL[4]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color, width=1.3)))
fig3.add_hline(y=0, line_dash="dash", line_color="#898781")
fig3.update_layout(
    xaxis_title="Date", yaxis_title="Annualized carry",
    height=340, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    f"{asset} is a {'PROXY (back-differenced outrights)' if asset in proxy_assets else 'real-quote'} "
    "carry series. Frozen at 2026-07-13 (real spread data's frozen end)."
)
