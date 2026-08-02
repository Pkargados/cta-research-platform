"""
dashboard/summary_app.py — the single, public-facing promotional page for
this project. Deliberately separate from dashboard/app.py's 26-page research
tool, which stays as the full local/dev dashboard — THIS is what should be
deployed publicly, per direct instruction (2026-08-02): a 26-page sidebar of
QA/technical-appendix pages is a research tool, not promotional material,
and reads as cluttered on a recruiter's first (and often only) skim.

Shows exactly what a recruiter/interviewer actually wants at a glance: how
the Trend single-strategy portfolio performs standalone, and how the
Trend+Carry multi-strategy combination performs and *behaves* (equity curve,
per-Book attribution, and the dynamic correlation between the two sleeves
that the risk-parity weighting below is built on) — reusing the exact same
cached pipeline (`_single_strategy_pipeline.py`) and already-validated
numbers as the full dashboard's own pages 18/20, not a separate
re-derivation.

Run: `streamlit run dashboard/summary_app.py`
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    # Populates Data/ from the private cta-research-platform-data repo on
    # the public (main-branch) deployment, before any page below reads from
    # it — see dashboard/_bootstrap_data.py's own docstring. Only present on
    # main (Data/ already exists locally on master/for local dev), so this
    # is a no-op ImportError there, same lazy-import discipline
    # src/data/garch_volatility.py already uses for its own main-only
    # dependency.
    import _bootstrap_data  # noqa: E402,F401
except ImportError:
    pass

from lib import page_header, apply_chart_theme, CATEGORICAL  # noqa: E402
from _single_strategy_pipeline import (  # noqa: E402
    load_and_run, compute_risk_parity_weights, TREND_BOOK_NAME, CARRY_BOOK_NAME,
)

from portfolio.allocator import Allocator  # noqa: E402
from portfolio.risk_metrics import historical_var, expected_shortfall  # noqa: E402
from portfolio.correlation import rolling_correlation, ewma_correlation, correlation_summary  # noqa: E402
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split  # noqa: E402
from backtest.performance import simple_sharpe  # noqa: E402

st.set_page_config(
    page_title="Systematic Commodity & Macro Futures Research Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

page_header(
    "Systematic Commodity & Macro Futures Research Platform",
    "42-market systematic futures research — signal research, portfolio construction, honest results.",
)
st.markdown(
    "Nine independently-researched signal families, each matched directly against its "
    "source paper, running through the same train / validation / test discipline — no "
    "signal-spec selection based on its own performance, every signal shifted one day "
    "before it's allowed to trade. "
    "**[Full write-up and 26-page technical dashboard on GitHub →]"
    "(https://github.com/Pkargados/cta-research-platform)**"
)

returns, trend_result, carry_result, trend_book, carry_book, _naive_pnl = load_and_run()
periods_per_year = trend_book.periods_per_year

weights = compute_risk_parity_weights("ewma")
allocator = Allocator(
    [trend_book, carry_book],
    book_weights={TREND_BOOK_NAME: weights["trend"], CARRY_BOOK_NAME: weights["carry"]},
)
combined = allocator.run(returns)
combined_pnl = combined["pnl"]

trend_train, trend_val, trend_test = train_validation_test_split(trend_result["pnl"])
comb_train, comb_val, comb_test = train_validation_test_split(combined_pnl)

st.divider()

cols = st.columns(4)
headline_stats = [
    ("42", "markets"),
    ("9", "signal families"),
    (f"{simple_sharpe(trend_test, periods_per_year=periods_per_year):.2f}", "Trend Sharpe (test)"),
    (f"{simple_sharpe(comb_test, periods_per_year=periods_per_year):.2f}", "Combined Sharpe (test)"),
]
for col, (value, label) in zip(cols, headline_stats):
    with col:
        st.metric(label, value)

st.divider()

# --- Trend single-strategy portfolio -----------------------------------
st.subheader("Trend Single Strategy Portfolio")
st.caption("Time-series momentum (TSMOM), GJR-GARCH vol-targeted — the one consistently positive signal family across every period tested.")

eq_trend = (1 + trend_result["pnl"].dropna()).cumprod()
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=eq_trend.index, y=eq_trend.values, mode="lines", line=dict(color=CATEGORICAL[0], width=2)))
fig_trend.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig_trend.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig_trend.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)", height=340,
    margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig_trend = apply_chart_theme(fig_trend)
st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")

trend_table = pd.DataFrame({"Sharpe": {
    "Train": simple_sharpe(trend_train, periods_per_year=periods_per_year),
    "Validation": simple_sharpe(trend_val, periods_per_year=periods_per_year),
    "Test": simple_sharpe(trend_test, periods_per_year=periods_per_year),
}}).T
st.dataframe(trend_table.round(2), use_container_width=True)

st.divider()

# --- Multi-strategy portfolio -------------------------------------------
st.subheader("Multi-Strategy Portfolio (Trend + Carry)")
st.caption(
    f"EWMA sleeve risk-parity (Trend {weights['trend']/2:.0%} / Carry {weights['carry']/2:.0%} of the risk "
    "budget) — fit once on train sleeve PnL, applied as a fixed weight thereafter, not re-fit walk-forward."
)

eq_combined = (1 + combined_pnl.dropna()).cumprod()
fig_multi = go.Figure()
fig_multi.add_trace(go.Scatter(x=eq_combined.index, y=eq_combined.values, mode="lines", name="Combined", line=dict(color=CATEGORICAL[0], width=2.2)))
for i, (name, result) in enumerate([("Trend", trend_result), ("Carry", carry_result)]):
    eq = (1 + result["pnl"].dropna()).cumprod()
    fig_multi.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name, line=dict(color=CATEGORICAL[i + 1], width=1.2, dash="dot")))
fig_multi.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig_multi.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig_multi.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)", height=380,
    margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.15),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig_multi = apply_chart_theme(fig_multi)
st.plotly_chart(fig_multi, use_container_width=True, theme="streamlit")

var95 = historical_var(combined_pnl, confidence=0.95)
es95 = expected_shortfall(combined_pnl, confidence=0.95)
combined_table = pd.DataFrame({
    "Trend": {
        "Train": simple_sharpe(trend_train, periods_per_year=periods_per_year),
        "Validation": simple_sharpe(trend_val, periods_per_year=periods_per_year),
        "Test": simple_sharpe(trend_test, periods_per_year=periods_per_year),
    },
    "Carry": {
        period: simple_sharpe(series, periods_per_year=periods_per_year)
        for period, series in zip(("Train", "Validation", "Test"), train_validation_test_split(carry_result["pnl"]))
    },
    "Combined": {
        "Train": simple_sharpe(comb_train, periods_per_year=periods_per_year),
        "Validation": simple_sharpe(comb_val, periods_per_year=periods_per_year),
        "Test": simple_sharpe(comb_test, periods_per_year=periods_per_year),
    },
}).T
st.dataframe(combined_table.round(2), use_container_width=True)
st.caption(f"Combined portfolio 95% VaR **{var95:.2%}**, ES **{es95:.2%}** (weekly).")

st.divider()

# --- Dynamic correlation between the two sleeves -------------------------
st.subheader("Dynamic Correlation: Trend vs. Carry")
st.caption(
    "Rolling (52-week) and EWMA (halflife=26 weeks) correlation between the two sleeves' "
    "own weekly PnL — a single full-sample number can hide regime-dependent co-movement, "
    "and this is the diversification effect the risk-parity weighting above is built on."
)

roll_corr = rolling_correlation(trend_result["pnl"], carry_result["pnl"], window=52)
ewma_corr = ewma_correlation(trend_result["pnl"], carry_result["pnl"], halflife=26)

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode="lines", name="Rolling (52w)", line=dict(color=CATEGORICAL[0], width=1.4)))
fig_corr.add_trace(go.Scatter(x=ewma_corr.index, y=ewma_corr.values, mode="lines", name="EWMA (hl=26w)", line=dict(color=CATEGORICAL[3], width=1.4)))
fig_corr.add_hline(y=0, line_dash="dot", line_color="#898781")
fig_corr.update_layout(
    xaxis_title="Date", yaxis_title="Correlation", height=320,
    margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.15),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig_corr = apply_chart_theme(fig_corr)
st.plotly_chart(fig_corr, use_container_width=True, theme="streamlit")

corr_summary = correlation_summary(ewma_corr)
st.caption(
    f"Mean correlation **{corr_summary['mean']:.2f}**, negative **{corr_summary['pct_negative']:.0f}%** "
    "of the time — a real, if imperfect, diversification effect, not assumed from a single static number."
)

st.divider()
st.markdown(
    "**[Full research write-up, all 9 signal families, and the full technical dashboard →]"
    "(https://github.com/Pkargados/cta-research-platform)**"
)
