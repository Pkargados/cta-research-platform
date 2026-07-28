"""Page 11 — Carry Performance.

Computes live at render time, reusing `research/carry.py` directly. Four
paper-matched specs (Koijen-Moskowitz-Pedersen-Vrugt 2018), no headline pick.
Proxy (ICE softs, back-differenced) vs. real-quote assets visually
distinguished throughout — CLAUDE.md Rule 4: never blended unlabeled.

Per-asset view, not a pooled book: every number on this page (all-specs
summary, tearsheet, equity curve, carry-level chart) is one selected asset's
own return stream (`backtest.engine.backtest_signal_per_asset`, sliced to
that asset).
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

import carry as carry_research
from signals.carry import build_all_carry_signals
from backtest.engine import backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe
from backtest.costs import liquidity_tiered_cost_bps

page_header("Carry", "Koijen-Moskowitz-Pedersen-Vrugt (2018) — rank-weighted cross-section, monthly rebalancing.")


@st.cache_data(ttl=1200)
def _load():
    close, volume, included, sectors = carry_research.load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=carry_research.ADV_WINDOW_START)
    carry_panel, is_proxy = carry_research.build_carry_panel(included)
    return returns, sectors, cost_bps, carry_panel, is_proxy


returns, sectors, cost_bps, carry_panel, is_proxy = _load()
proxy_assets = sorted(is_proxy[is_proxy].index.tolist())

asset = st.selectbox("Asset", sorted(carry_panel.columns))
asset_is_proxy = asset in proxy_assets


@st.cache_data(ttl=1200)
def _all_specs_summary(_asset):
    signals = build_all_carry_signals(carry_panel, sectors)
    rows = []
    for name, signal in signals.items():
        turn = carry_research.annualized_turnover(signal)
        gross = backtest_signal_per_asset(signal, returns[carry_panel.columns], frequency="monthly")[_asset]
        net = backtest_signal_per_asset(signal, returns[carry_panel.columns], frequency="monthly", cost_bps=cost_bps)[_asset]
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "Spec": name, "Annualized turnover (pooled book)": turn,
            "Train (gross)": simple_sharpe(g_train), "Validation (gross)": simple_sharpe(g_val), "Test (gross)": simple_sharpe(g_test),
            "Train (net)": simple_sharpe(n_train), "Validation (net)": simple_sharpe(n_val), "Test (net)": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("Spec")


summary = _all_specs_summary(asset)
render_key_takeaways([
    "Four parallel specs (carry1m, carry1-12, carry timing ref=0, carry timing "
    "ref=sector-mean), **no headline pick**.",
    f"**{asset}** uses a **{'PROXY (back-differenced, no real spread quote)' if asset_is_proxy else 'real-quote'}** "
    "carry series — CLAUDE.md Rule 4: never blended unlabeled. "
    + (f"{len(proxy_assets)} ICE softs total use the proxy: {', '.join(proxy_assets)}." if not asset_is_proxy else ""),
    "Pooled-book finding, for context: turnover collapsed to ~0.7-3.3x annualized "
    "once rebalancing matched the paper's monthly cadence — genuinely mixed result, "
    "no spec robustly positive across all three periods.",
])

st.divider()

c1, c2 = st.columns(2)
with c1:
    spec = st.selectbox("Spec", ["carry1m", "carry1_12", "carry_timing_zero", "carry_timing_mean"])
with c2:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader(f"All 4 Specs — {asset} Gross/Net Sharpe (monthly)")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

signals = build_all_carry_signals(carry_panel, sectors)
signal = signals[spec]
cb = cost_bps if gross_net == "Net of cost" else None
asset_returns = backtest_signal_per_asset(signal, returns[carry_panel.columns], frequency="monthly", cost_bps=cb)[asset]
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(asset_returns))}

st.subheader(f"Tearsheet — {asset}, {spec}, {gross_net}")
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
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[4], width=1.5)))
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

st.subheader(f"Carry Level — {asset}")
series = carry_panel[asset].dropna()
color = CATEGORICAL[5] if asset_is_proxy else CATEGORICAL[4]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color, width=1.3)))
fig3.add_hline(y=0, line_dash="dash", line_color="#898781")
fig3.update_layout(
    xaxis_title="Date", yaxis_title="Annualized carry",
    height=340, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig3 = apply_chart_theme(fig3)
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    f"{asset} is a {'PROXY (back-differenced outrights)' if asset_is_proxy else 'real-quote'} "
    "carry series. Frozen at 2026-07-13 (real spread data's frozen end)."
)
