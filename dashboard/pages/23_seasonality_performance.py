"""Page 23 — Seasonality Performance.

Computes live at render time, reusing `research/seasonality.py` directly.
Two parallel, UNRELATED specs, no headline pick (see
`signals/seasonality.py`'s own module docstring for why these two and not
the originally-planned RRT/half-month construction):

- half_month (Milonas 1991, this project's own trading-rule interpretation
  of a documented-but-decayed effect) — daily rebalancing, scoped to a
  named 7-asset subset only.
- same_month (Keloharju-Linnainmaa-Nyberg 2014/2016, replicated for
  commodities by Li et al. 2023) — monthly rebalancing, the ADV-filtered
  liquid universe, the paper's own ACTUAL backtested construction.

Per-asset view, not a pooled book (same convention as every other Strategy
Performance page): every number below is one selected asset's own return
stream (`backtest.engine.backtest_signal_per_asset`, sliced to that asset).
half_month reads N/A for any asset outside its 7-name scope, same as
XSMOM's page reads N/A for Copper.
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

import seasonality as season_research
from backtest.engine import backtest_signal_per_asset
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import performance_stats, simple_sharpe
from backtest.costs import liquidity_tiered_cost_bps

page_header(
    "Seasonality",
    "Two unrelated effects, no headline pick — half-month (Milonas 1991) and same-calendar-month "
    "(Keloharju-Linnainmaa-Nyberg 2014/2016, replicated for commodities by Li et al. 2023).",
)


@st.cache_data(ttl=1200)
def _load():
    close, volume, sectors, vol = season_research.load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    signals = season_research.build_signals(close, vol, sectors)
    cost_bps = liquidity_tiered_cost_bps(volume, window_start=season_research.ADV_WINDOW_START)
    return close, returns, signals, cost_bps


close, returns, signals, cost_bps = _load()

asset = st.selectbox("Asset", sorted(close.columns.tolist()))
half_month_scope = set(signals["half_month"].columns)
if asset not in half_month_scope:
    st.info(
        f"{asset} is outside half_month's 7-name scope "
        f"({', '.join(sorted(half_month_scope))}) — its stats will read N/A for that spec."
    )


@st.cache_data(ttl=1200)
def _all_spec_summary(_asset):
    rows = []
    for name, signal in signals.items():
        if _asset not in signal.columns:
            rows.append({
                "Spec": name, "Annualized turnover": float("nan"),
                "Train (gross)": float("nan"), "Validation (gross)": float("nan"), "Test (gross)": float("nan"),
                "Train (net)": float("nan"), "Validation (net)": float("nan"), "Test (net)": float("nan"),
            })
            continue
        freq = season_research.FREQUENCIES[name]
        signal_returns = returns[signal.columns]
        signal_cost_bps = cost_bps[signal.columns]
        turn = season_research.annualized_turnover(signal, freq)
        gross = backtest_signal_per_asset(signal, signal_returns, frequency=freq)[_asset]
        net = backtest_signal_per_asset(signal, signal_returns, frequency=freq, cost_bps=signal_cost_bps)[_asset]
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "Spec": name, "Annualized turnover": turn,
            "Train (gross)": simple_sharpe(g_train), "Validation (gross)": simple_sharpe(g_val), "Test (gross)": simple_sharpe(g_test),
            "Train (net)": simple_sharpe(n_train), "Validation (net)": simple_sharpe(n_val), "Test (net)": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("Spec")


summary = _all_spec_summary(asset)
render_key_takeaways([
    "**half_month** and **same_month** are two independent, unrelated effects from the source "
    "papers — not two views of the same phenomenon. **No headline pick.**",
    "Pooled-book finding, for context: half_month turnover ~166x annualized (deeply negative "
    "net-of-cost); same_month turnover ~7x annualized, weak/mixed (sharply negative in "
    "validation, spanning the 2020 COVID shock — the same pattern every other cross-sectional "
    "family in this project shows there).",
    f"**{asset}**'s own numbers below may differ from the pooled-book pattern.",
])

st.divider()

c1, c2 = st.columns(2)
with c1:
    spec = st.selectbox("Spec", ["half_month", "same_month"], format_func=lambda s: s.replace("_", "-"))
with c2:
    gross_net = st.radio("View", ["Gross", "Net of cost"], horizontal=True)

st.subheader(f"Both Specs — {asset} Gross/Net Sharpe")
st.dataframe(summary.round(3), use_container_width=True)

st.divider()

signal = signals[spec]
freq = season_research.FREQUENCIES[spec]

if asset not in signal.columns:
    st.warning(f"{asset} is not in {spec}'s scope — nothing to show.")
    st.stop()

cb = cost_bps[signal.columns] if gross_net == "Net of cost" else None
signal_returns = returns[signal.columns]
asset_returns = backtest_signal_per_asset(signal, signal_returns, frequency=freq, cost_bps=cb)[asset]
stats_by_period = {k: performance_stats(v) for k, v in zip(("train", "validation", "test"), train_validation_test_split(asset_returns))}

st.subheader(f"Tearsheet — {asset}, {spec.replace('_', '-')}, {gross_net}")
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
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[3], width=1.5)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.caption(
    "half_month is this project's own trading-rule interpretation of a documented-but-decayed "
    "effect (Milonas 1991) — neither source paper backtests it as a strategy, only tests its "
    "statistical significance. same_month is the paper's own actual backtested construction. "
    "See the Seasonality section of WORKFLOW.md Phase 11b for the full build record."
)
