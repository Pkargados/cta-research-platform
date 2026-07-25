"""Page 10 — Short-Term Reversal Performance.

Computes live at render time, reusing `research/short_term_reversal.py`
directly. Tier (individual/sector) and sizing (simple/VIX-adjusted) toggles
on top of the usual lag/gross-net ones, reflecting this signal's two-tier,
cross-sectional structure and Nagel (2011)'s VIX-conditioning finding.
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

import short_term_reversal as str_research
from signals.short_term_reversal import build_all_reversal_signals, LAGS
from signals.vix_overlay import vix_size_multiplier, apply_size_multiplier
from backtest.engine import backtest_signal, backtest_signal_per_asset, normalized_positions
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe

page_header("Short-Term Reversal", "Lehmann (1990) mechanics, sector-scoped peer groups, Nagel (2011) VIX-conditional sizing.")


@st.cache_data(ttl=1200)
def _load():
    adj, raw, sectors = str_research.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = str_research.build_vol(raw)
    from backtest.costs import liquidity_tiered_cost_bps
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=str_research.ADV_WINDOW_START)
    vix = str_research.load_vix()
    return close, returns, vol, sectors, cost_bps, vix


close, returns, vol, sectors, cost_bps, vix = _load()


@st.cache_data(ttl=1200)
def _all_specs_summary():
    signals = build_all_reversal_signals(close, vol, sectors, lags=LAGS)
    rows = []
    for name, signal in signals.items():
        turnover = str_research.annualized_turnover(signal)
        gross = backtest_signal(signal, returns, frequency="daily")
        net = backtest_signal(signal, returns, frequency="daily", cost_bps=cost_bps)
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
    "The first genuinely CROSS-SECTIONAL signal in this project — bets against "
    "an asset's return relative to its own SECTOR peer group, not the full universe.",
    f"Turnover is extremely high ({summary['Annualized turnover'].min():.0f}-"
    f"{summary['Annualized turnover'].max():.0f}x annualized) and net-of-cost Sharpe "
    "is deeply negative across every one of the 6 specs — the first outright-"
    "unprofitable signal family in this project.",
    "VIX-conditioning (Nagel): sector-tier is HAC-significant despite a tiny R², "
    "individual-tier is not — see the regression panel below. Doesn't rescue "
    "net-of-cost profitability either way.",
])

st.divider()

c1, c2, c3, c4 = st.columns(4)
with c1:
    tier = st.selectbox("Tier", ["individual", "sector"])
with c2:
    lag = st.selectbox("Lag (days)", list(LAGS), index=1)
with c3:
    sizing = st.radio("Sizing", ["Simple", "VIX-adjusted"], horizontal=True)
with c4:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader("All 6 Specs — Gross/Net Sharpe (daily)")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

spec_name = f"{tier}_{lag}d"
signals = build_all_reversal_signals(close, vol, sectors, lags=LAGS)
signal = signals[spec_name]

positions = normalized_positions(signal, "daily")
if sizing == "VIX-adjusted":
    multiplier = vix_size_multiplier(vix)
    positions = apply_size_multiplier(positions, multiplier)

raw_strategy_returns = (positions * returns).sum(axis=1).dropna()
if gross_net == "Net of cost":
    from backtest.costs import transaction_cost_drag
    cost_bps_aligned = cost_bps.reindex(positions.columns).fillna(0.0)
    strategy_returns = raw_strategy_returns - transaction_cost_drag(positions, cost_bps_aligned).reindex(raw_strategy_returns.index).fillna(0.0)
else:
    strategy_returns = raw_strategy_returns

stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(strategy_returns))}

st.subheader(f"Tearsheet — {spec_name}, {sizing} sizing, {gross_net}")
c1, c2, c3 = st.columns(3)
for col, key, label in zip([c1, c2, c3], ("train", "validation", "test"), ("Train", "Validation", "Test")):
    s = stats_by_period[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Max DD {s['Max DD']:.2%}")

equity = (1 + strategy_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[3], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Per-Asset Sharpe (full sample, gross)")
per_asset_returns = backtest_signal_per_asset(signal, returns, frequency="daily")
per_asset_sharpe = per_asset_returns.apply(simple_sharpe).sort_values(ascending=False)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=per_asset_sharpe.index, y=per_asset_sharpe.values, marker_color=CATEGORICAL[3]))
fig2.update_layout(
    xaxis_title=None, yaxis_title="Sharpe",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("VIX-Conditioning Regression (Newey-West HAC, maxlags=20)")
st.caption("Daily strategy return regressed on the PRIOR day's VIX level — see research/short_term_reversal.py's own docstring for why lagged, not contemporaneous.")
reg_cols = st.columns(2)
for col, reg_tier in zip(reg_cols, ("individual", "sector")):
    reg_signal = signals[f"{reg_tier}_{lag}d"]
    reg_positions = normalized_positions(reg_signal, "daily")
    reg_returns = (reg_positions * returns).sum(axis=1).dropna()
    reg = str_research.hac_vix_regression(reg_returns, vix)
    with col:
        st.markdown(f"**{reg_tier}**")
        st.metric("t-stat", f"{reg['t']:.2f}")
        st.caption(f"p={reg['p']:.3f} · R²={reg['r_squared']:.4f} · n={reg['n_obs']}")
