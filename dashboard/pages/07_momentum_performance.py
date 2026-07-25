"""Page 07 — Momentum (TSMOM) Performance.

Computes live at render time, reusing `research/momentum.py` directly — no
duplicated data-prep/backtest logic, no precomputed dashboard_summary/
artifact. Headline spec (k=12mo, target_vol=0.40) is fixed a priori
(CLAUDE.md Rule 1/2) — the lookback×holding grid below is a descriptive
robustness view, never a spec-picker.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL

import momentum as mom
from backtest.splits import TRAIN_END, VALIDATION_END
from backtest.performance import performance_stats

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


@st.cache_data(ttl=1200)
def _headline(_winner):
    stats, strategy_returns = mom.headline_result(close, returns, vol_estimators[_winner])
    return stats, strategy_returns


@st.cache_data(ttl=1200)
def _grid(_winner):
    return mom.lookback_holding_grid(close, returns, vol_estimators[_winner])


@st.cache_data(ttl=1200)
def _per_asset(_winner):
    return mom.per_asset_sharpe(close, returns, vol_estimators[_winner])


@st.cache_data(ttl=1200)
def _net_check(_winner):
    return mom.net_of_cost_check(close, returns, vol_estimators[_winner], volume)


stats, strategy_returns = _headline(winner)
gross_test_sharpe, net_test_sharpe = _net_check(winner)

render_key_takeaways([
    f"Vol estimator: **{winner}** wins on TRAIN evidence (Sharpe {vol_comparison.loc[winner, 'train_sharpe']:.3f} "
    f"vs. {vol_comparison.loc[[i for i in vol_comparison.index if i != winner][0], 'train_sharpe']:.3f}) — "
    "picked before looking at validation/test (CLAUDE.md Rule 1/2).",
    f"Headline train/validation/test Sharpe: **{stats['train']['Sharpe']:.3f} / "
    f"{stats['validation']['Sharpe']:.3f} / {stats['test']['Sharpe']:.3f}**.",
    f"Net-of-cost test Sharpe: **{net_test_sharpe:.3f}** vs. gross **{gross_test_sharpe:.3f}** "
    "(liquidity-tiered ADV cost assumption — a labeled placeholder, not a measured cost).",
])

st.divider()

gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader("Tearsheet — Headline Spec")
c1, c2, c3 = st.columns(3)
periods = [("train", "Train"), ("validation", "Validation"), ("test", "Test")]
for col, (key, label) in zip([c1, c2, c3], periods):
    s = stats[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Ann Vol {s['Ann Vol']:.2%} · Max DD {s['Max DD']:.2%}")

st.subheader("Equity Curve")
plot_returns = strategy_returns.copy()
if gross_net == "Net of cost":
    from backtest.costs import liquidity_tiered_cost_bps
    from signals.momentum import tsmom_signal
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=mom.ADV_WINDOW_START)
    from backtest.engine import backtest_signal
    signal = tsmom_signal(close, winning_vol, lookback_months=mom.HEADLINE_LOOKBACK_MONTHS, target_vol=mom.HEADLINE_TARGET_VOL)
    plot_returns = backtest_signal(signal, returns, frequency="monthly", holding_months=1, cost_bps=cost_bps)

equity = (1 + plot_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[0], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(f"Dashed lines mark the train/validation ({TRAIN_END}) and validation/test ({VALIDATION_END}) boundaries.")

st.divider()

st.subheader("Per-Asset Sharpe (headline spec, full sample)")
per_asset = _per_asset(winner)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=per_asset.index, y=per_asset.values, marker_color=CATEGORICAL[0]))
fig2.update_layout(
    xaxis_title=None, yaxis_title="Sharpe",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
st.caption("Moskowitz-Ooi-Pedersen Figure 2 style — before any cross-asset pooling.")

st.divider()

st.subheader("Lookback × Holding Grid (Train Sharpe, descriptive robustness view)")
grid = _grid(winner)
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
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    "The paper's own Table 2 grid, reproduced on TRAIN only — descriptive, never "
    "used to pick the headline spec (k=12, h=1 is fixed a priori)."
)
