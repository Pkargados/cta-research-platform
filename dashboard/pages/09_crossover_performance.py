"""Page 09 — Moving-Average Crossover Performance.

Computes live at render time, reusing `research/crossover.py` directly. All
three pairs reported as parallel Books, no auto-picked winner.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

import crossover as co
from signals.crossover import all_pair_signals, PAIRS
from backtest.engine import backtest_signal, backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe

page_header("Moving-Average Crossover", "Golden/death cross family — SMA-based, three pairs, no headline pick.")


@st.cache_data(ttl=1200)
def _load():
    adj, raw = co.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = co.build_vol(raw)
    from backtest.costs import liquidity_tiered_cost_bps
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=co.ADV_WINDOW_START)
    return close, returns, vol, cost_bps


close, returns, vol, cost_bps = _load()


@st.cache_data(ttl=1200)
def _all_pair_summary():
    signals = all_pair_signals(close, vol, target_vol=co.DEFAULT_TARGET_VOL)
    rows = []
    for name, signal in signals.items():
        turnover = co.annualized_turnover(signal)
        gross = backtest_signal(signal, returns, frequency=co.FREQUENCY)
        net = backtest_signal(signal, returns, frequency=co.FREQUENCY, cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "Pair": name, "Annualized turnover": turnover,
            "Train (gross)": simple_sharpe(g_train), "Validation (gross)": simple_sharpe(g_val), "Test (gross)": simple_sharpe(g_test),
            "Train (net)": simple_sharpe(n_train), "Validation (net)": simple_sharpe(n_val), "Test (net)": simple_sharpe(n_test),
        })
    import pandas as pd
    return pd.DataFrame(rows).set_index("Pair")


summary = _all_pair_summary()
render_key_takeaways([
    "All three pairs reported as parallel Books, **no auto-picked winner** — "
    "same discipline as breakout's two systems.",
    "**50/200 (golden cross)** is the most consistent — positive throughout "
    "train/validation/test. 50/100 fades badly out-of-sample; 100/200 sign-flips "
    "sharply in validation.",
    f"Turnover ~{summary['Annualized turnover'].min():.1f}-{summary['Annualized turnover'].max():.1f}x annualized "
    "— much lower than breakout's ~50-60x, as expected for a slower trend-confirmation signal.",
])

st.divider()

c1, c2 = st.columns(2)
with c1:
    pair = st.selectbox("Pair", list(PAIRS.keys()), format_func=lambda p: f"{p.replace('_', '/')} " + ("(golden cross)" if p == "50_200" else ""))
with c2:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader("All Pairs — Gross/Net Sharpe")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

signals = all_pair_signals(close, vol, target_vol=co.DEFAULT_TARGET_VOL)
signal = signals[pair]
cb = cost_bps if gross_net == "Net of cost" else None
strategy_returns = backtest_signal(signal, returns, frequency=co.FREQUENCY, cost_bps=cb)
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(strategy_returns))}

st.subheader(f"Tearsheet — {pair.replace('_', '/')}, {gross_net}")
c1, c2, c3 = st.columns(3)
for col, key, label in zip([c1, c2, c3], ("train", "validation", "test"), ("Train", "Validation", "Test")):
    s = stats_by_period[key]
    col.metric(f"{label} Sharpe", f"{s['Sharpe']:.3f}" if s['Sharpe'] == s['Sharpe'] else "N/A")
    col.caption(f"Ann Ret {s['Ann Return']:.2%} · Max DD {s['Max DD']:.2%}")

equity = (1 + strategy_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[2], width=1.5)))
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
per_asset_returns = backtest_signal_per_asset(signal, returns, frequency=co.FREQUENCY, cost_bps=cb)
per_asset_sharpe = per_asset_returns.apply(simple_sharpe).sort_values(ascending=False)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=per_asset_sharpe.index, y=per_asset_sharpe.values, marker_color=CATEGORICAL[2]))
fig2.update_layout(
    xaxis_title=None, yaxis_title="Sharpe",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig2 = apply_chart_theme(fig2)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
