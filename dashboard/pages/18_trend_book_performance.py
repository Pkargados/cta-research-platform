"""Page 18 — Trend Book (Single Strategy Portfolio).

Computes live at render time via `dashboard/_single_strategy_pipeline.py`'s
ONE shared, cached pipeline (see that module's own docstring for why this
must not be a page-local cache — a prior real Streamlit Cloud crash came
from exactly that pattern). Construction (`tsmom_alone`) and vol estimator
(GJR-GARCH) are both already-decided results, not re-derived here — see
WORKFLOW.md decisions #10/#11 and `research/single_strategy_portfolios.py`'s
own bake-off for how they were chosen.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme
from _single_strategy_pipeline import load_and_run, TREND_FLAVOR

from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import simple_sharpe

page_header("Trend Book", "Single Strategy Portfolio — TSMOM alone, GJR-GARCH vol-targeted")

returns, trend_result, carry_result, trend_book, carry_book, combined_pnl = load_and_run()
pnl = trend_result["pnl"]
train, val, test = train_validation_test_split(pnl)
periods_per_year = trend_book.periods_per_year

render_key_takeaways([
    f"**Winning construction: `{TREND_FLAVOR}`** — beat every blend tried against it "
    "(equal-weight, tilted, IC-weighted, risk-parity, a confirmation-filter gate, and a "
    "conviction-deadband construction) in a validation-selected bake-off, not a "
    "numerical hyperparameter grid. TSMOM and the 50/200 crossover are too correlated "
    "(0.65 full-sample, DCC-GARCH) to diversify each other — blending only diluted the "
    "stronger signal.",
    f"Sharpe — train **{simple_sharpe(train, periods_per_year=periods_per_year):.2f}**, "
    f"validation **{simple_sharpe(val, periods_per_year=periods_per_year):.2f}**, "
    f"test **{simple_sharpe(test, periods_per_year=periods_per_year):.2f}**.",
    "Vol-targeting: `GJR-GARCH` (not EWMA) — cut forecast loss (QLIKE) ~60-70% vs. EWMA "
    "on this Book's own realized-PnL volatility, see the Volatility Estimators page's "
    "Book-Level section for the comparison this decision is based on.",
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
c4.metric("Turnover (avg per rebalance)", f"{trend_result.get('turnover', float('nan')):.3f}")
c5.metric("Max Drawdown", f"{trend_result.get('max_dd', float('nan')):.2%}")
c6.metric("Valid rebalance dates", f"{trend_result.get('n_rebalance_dates_valid', 0)}")

st.divider()

st.subheader("Equity Curve")
equity = (1 + pnl.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[0], width=1.8)))
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
