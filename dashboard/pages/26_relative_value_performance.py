"""Page 26 — Relative Value (cointegration spreads).

WORKFLOW.md §11d. Computes live at render time, reusing `research/
relative_value*.py` directly (same convention as pages 07-12/16/23's live
per-signal-family computation). Three sections:

1. Rolling Engle-Granger cointegration diagnostic (weak-to-no evidence for
   every 2-leg pair — reported honestly, does NOT gate the backtest below).
2. Standalone bake-off — 7 pairs x 2 sizing rules (continuous z-score vs.
   discrete threshold band) x 2 rebalancing frequencies (daily/weekly), no
   headline pick, gross/net toggle.
3. Pooled Relative Value Book — one Book, `single_strategy_portfolios.
   build_book`'s calibration, Ledoit-Wolf (default) vs. Gerber statistic
   covariance comparison (WORKFLOW.md Phase 7's Gerber section, closed out
   for this sleeve).

Single cached load (`@st.cache_data`, one function) — this page is the only
consumer of the RV pipeline, so the multi-page-cache-collision crash pattern
`dashboard/_single_strategy_pipeline.py` documents doesn't apply here, but
everything expensive (hedge-ratio bake-off, 28-row standalone table, 4
covariance-estimator Book runs) is still computed exactly once per cache
window, not once per widget interaction.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme, render_attribution_section

import numpy as np

import relative_value as rv
import relative_value_book as rvb
import relative_value_cointegration_check as rvc
from relative_value_gerber import _estimators, _score
from data.continuous_curve import load_continuous_backadjusted
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import simple_sharpe

page_header("Relative Value", "Cointegration spreads — 7 pairs, three construction groups, one pooled Book.")


@st.cache_data(ttl=1200)
def _load_all():
    adj = load_continuous_backadjusted()
    log_price = np.log(adj["close"])
    coint_report = rvc.rolling_cointegration_report(rvc.EG_PAIRS, log_price)

    result, hedge_winners, spreads = rv.main()
    alpha_df, returns_df, active, cost_bps = rvb.prepare_rv_book_inputs(standalone=(result, hedge_winners, spreads))

    book_rows = []
    book_results = {}
    for name, builder in _estimators().items():
        row = _score(name, alpha_df, returns_df, cost_bps, builder)
        book_rows.append(row)
        book_results[name] = rvb.ssp.build_book("relative_value", alpha_df, returns_df, cov_dict_builder=builder, cost_bps=cost_bps).run(returns_df)
    book_summary = pd.DataFrame(book_rows).set_index("estimator")

    return coint_report, result, hedge_winners, active, book_summary, book_results


coint_report, standalone_result, hedge_winners, active_pairs, book_summary, book_results = _load_all()
excluded_pairs = [p for p in rv.ALL_PAIR_NAMES if p not in active_pairs]

render_key_takeaways([
    "Rolling Engle-Granger cointegration diagnostic found **weak-to-no evidence** for every "
    "2-leg pair (3-11% of rolling windows reject the null at 5%, well below the old, "
    "unverified `Time_Series_Models.ipynb` claim of 45% for Brent/WTI — that notebook "
    "never existed in this repo, see WORKFLOW.md §11d). Diagnostic only — does not gate "
    "which pairs trade.",
    f"**{len(active_pairs)} of 7 pairs** clear the pooled Book's 90% return-density gate: "
    f"{', '.join(active_pairs)}. Excluded (sparse cross-commodity trading calendar, not a "
    f"construction bug): {', '.join(excluded_pairs)}.",
    "Pooled Book (Ledoit-Wolf, net-of-cost): train "
    f"**{book_summary.loc['ledoit_wolf', 'sharpe_train']:.2f}**, validation "
    f"**{book_summary.loc['ledoit_wolf', 'sharpe_validation']:.2f}**, test "
    f"**{book_summary.loc['ledoit_wolf', 'sharpe_test']:.2f}** — genuinely diversified "
    "across 4 low-correlation spreads, positive in validation and test despite most "
    "individual pairs being weak-to-negative standalone.",
    "Gerber statistic covariance: mixed vs. Ledoit-Wolf on this Book — helps test Sharpe "
    "(~0.82-0.85 vs. 0.50) but hurts validation (~0.11-0.16 vs. 0.34) and roughly doubles "
    "turnover/max drawdown. No clean win either way, consistent with this project's other "
    "Gerber-vs-Ledoit-Wolf Book-level comparisons (WORKFLOW.md Phase 7).",
])

st.divider()

st.subheader("1. Rolling Engle-Granger Cointegration Diagnostic")
st.caption("252-day trailing window, 21-day step — strictly point-in-time (CLAUDE.md Hard Rule 2), never full-sample.")
st.dataframe(coint_report.round(3), use_container_width=True)

st.divider()

st.subheader("2. Standalone Bake-off — 7 Pairs x 2 Sizing Rules x 2 Frequencies")
c1, c2 = st.columns(2)
with c1:
    view = st.radio("View", ["Gross", "Net of cost"], horizontal=True, key="rv_gross_net")
with c2:
    freq_filter = st.radio("Frequency", ["daily", "weekly", "both"], horizontal=True, index=1)

display_cols = [c for c in standalone_result.columns if c.endswith("_gross" if view == "Gross" else "_net") or c == "annualized_turnover"]
table = standalone_result[display_cols]
if freq_filter != "both":
    table = table.xs(freq_filter, level="frequency")
st.dataframe(table.round(3), use_container_width=True)
st.caption(f"Hedge-ratio method winners (Kalman vs. rolling-OLS, validation Sharpe): {hedge_winners['corn_wheat'][0]} (corn_wheat), {hedge_winners['rbob_heatingoil'][0]} (rbob_heatingoil).")

st.divider()

st.subheader("3. Pooled Relative Value Book")
estimator = st.selectbox("Covariance estimator", list(book_summary.index), format_func=lambda x: {"ledoit_wolf": "Ledoit-Wolf (default)"}.get(x, x))
result_book = book_results[estimator]

if len(result_book.get("pnl", [])) == 0:
    st.warning("Insufficient data for this estimator (fewer than 20 valid rebalance dates).")
else:
    pnl = result_book["pnl"]
    train, val, test = train_validation_test_split(pnl)
    periods_per_year = rvb.ssp.PERIODS_PER_YEAR

    c1, c2, c3 = st.columns(3)
    for col, label, series in zip([c1, c2, c3], ("Train", "Validation", "Test"), (train, val, test)):
        sh = simple_sharpe(series, periods_per_year=periods_per_year)
        col.metric(f"{label} Sharpe", f"{sh:.3f}" if sh == sh else "N/A")
        col.caption(f"n={len(series.dropna())}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Turnover (avg per rebalance)", f"{result_book.get('turnover', float('nan')):.3f}")
    c5.metric("Max Drawdown", f"{result_book.get('max_dd', float('nan')):.2%}")
    c6.metric("Valid rebalance dates", f"{result_book.get('n_rebalance_dates_valid', 0)}")

    equity = (1 + pnl.dropna()).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[3], width=1.8)))
    fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
    fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative return (x)",
        height=420, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("Weekly rebalancing (Book's own COV_FREQ). Net of the same liquidity-tiered transaction costs every other net-of-cost comparison in this project uses.")

    if "asset_contributions" in result_book:
        st.divider()
        render_attribution_section(result_book["asset_contributions"], {p: "Relative Value" for p in active_pairs}, key_prefix="rv")

st.divider()

st.subheader("Covariance Estimator Comparison — Ledoit-Wolf vs. Gerber Statistic")
st.dataframe(book_summary.round(3), use_container_width=True)
st.caption("Net-of-cost. Ledoit-Wolf is the current default for every live Book in this project (WORKFLOW.md Phase 7) — this comparison is the RV sleeve's own closing of that question, not assumed to reproduce the Trend/Carry finding.")
