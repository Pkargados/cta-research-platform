"""Page 20 — Multi-Strategy Portfolio (Trend + Carry combined).

Computes live at render time via `dashboard/_single_strategy_pipeline.py`'s
ONE shared, cached pipeline (see that module's own docstring for why this
must not be a page-local cache — a prior real Streamlit Cloud crash came
from exactly that pattern). Combines the two Single Strategy Portfolios
(Trend Book, Carry Book — see those pages) two ways, selectable below:
`portfolio.allocator.Allocator`'s naive equal-Book-risk baseline (each Book
already at its own 10% vol target, summed with no further weighting), and
sleeve-level risk parity (`portfolio.risk_parity`, WORKFLOW.md decision #12)
— a general n-sleeve equal-risk-contribution solver, weights fit ONCE on
TRAIN sleeve PnL only and applied as a fixed static weight thereafter (not
re-fit walk-forward — a real, disclosed scope limit, not silently assumed
away).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme, render_attribution_section
from _single_strategy_pipeline import (
    load_and_run, compute_risk_parity_weights, RISK_PARITY_ESTIMATORS,
    TREND_BOOK_NAME, CARRY_BOOK_NAME,
)

from portfolio.allocator import Allocator
from portfolio.risk_metrics import historical_var, expected_shortfall, expanding_var_and_es
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import simple_sharpe
from data.sectors import asset_to_sector

page_header("Multi-Strategy Portfolio", "Trend Book + Carry Book, combined via the Allocator")

returns, trend_result, carry_result, trend_book, carry_book, naive_pnl = load_and_run()
periods_per_year = trend_book.periods_per_year  # both Books share the same weekly cadence

st.subheader("Combination Method")
combo_choice = st.radio(
    "How Trend and Carry are combined",
    ["Naive (equal Book-risk)", "Risk Parity (equal risk contribution)"],
    horizontal=True,
)

if combo_choice.startswith("Risk Parity"):
    estimator_label = st.radio("Sleeve covariance estimator", list(RISK_PARITY_ESTIMATORS.keys()), horizontal=True)
    weights = compute_risk_parity_weights(RISK_PARITY_ESTIMATORS[estimator_label])
    if not weights["converged"]:
        st.warning("This estimator's fit did not converge on the train window — weights may be unstable.")

    allocator = Allocator(
        [trend_book, carry_book],
        book_weights={TREND_BOOK_NAME: weights["trend"], CARRY_BOOK_NAME: weights["carry"]},
    )
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]
    combined_contributions = combined["asset_contributions"]
    combo_caption = (
        f"Risk-parity weights (Trend {weights['trend']/2:.1%} / Carry {weights['carry']/2:.1%} of the total risk "
        f"budget) fit ONCE on TRAIN sleeve PnL only via {estimator_label.split(' (')[0]} covariance, then applied "
        "as a fixed static weight across train/validation/test — not re-fit walk-forward (WORKFLOW.md decision #12)."
    )
else:
    combined_pnl = naive_pnl
    combined_contributions = trend_result["asset_contributions"].add(carry_result["asset_contributions"], fill_value=0.0)
    combo_caption = (
        "Naive equal-Book-risk baseline — each Book targets the same 10% annualized vol independently "
        "(WORKFLOW.md decision #5), then summed with no further risk-weighting (`Allocator`'s own w=[1, 1] default)."
    )

train, val, test = train_validation_test_split(combined_pnl)
var95 = historical_var(combined_pnl, confidence=0.95)
es95 = expected_shortfall(combined_pnl, confidence=0.95)

render_key_takeaways([
    combo_caption,
    f"Combined Sharpe — train **{simple_sharpe(train, periods_per_year=periods_per_year):.2f}**, "
    f"validation **{simple_sharpe(val, periods_per_year=periods_per_year):.2f}**, "
    f"test **{simple_sharpe(test, periods_per_year=periods_per_year):.2f}**.",
    f"Full-sample 95% VaR **{var95:.2%}**, ES **{es95:.2%}** (weekly) — risk MEASUREMENT, "
    "not risk control; position/sector/leverage limits (`next_steps.md` Phase 8) aren't "
    "built yet.",
])

st.divider()

st.subheader("Per-Book Summary")
summary = pd.DataFrame({
    "Trend (tsmom_alone)": {
        "Sharpe (full)": trend_result.get("sharpe"), "Max DD": trend_result.get("max_dd"),
        "Turnover": trend_result.get("turnover"), "Valid rebalance dates": trend_result.get("n_rebalance_dates_valid"),
    },
    "Carry (carry_timing_zero)": {
        "Sharpe (full)": carry_result.get("sharpe"), "Max DD": carry_result.get("max_dd"),
        "Turnover": carry_result.get("turnover"), "Valid rebalance dates": carry_result.get("n_rebalance_dates_valid"),
    },
}).T
st.dataframe(summary.round(4), use_container_width=True)

st.divider()

st.subheader("Combined Equity Curve")
equity = (1 + combined_pnl.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[0], width=2.0)))
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

st.subheader("Per-Book Equity Curves (standalone, not risk-weighted into the combo)")
fig2 = go.Figure()
for i, (name, result) in enumerate([("Trend", trend_result), ("Carry", carry_result)]):
    pnl = result.get("pnl")
    if pnl is None or len(pnl) == 0:
        continue
    eq = (1 + pnl.dropna()).cumprod()
    fig2.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=1.4)))
fig2.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.15),
)
fig2 = apply_chart_theme(fig2)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Attribution — Cumulative Contribution by Book")
_book_weights = {"Trend": weights["trend"], "Carry": weights["carry"]} if combo_choice.startswith("Risk Parity") \
    else {"Trend": 1.0, "Carry": 1.0}
fig_book_attr = go.Figure()
for i, (name, result) in enumerate([("Trend", trend_result), ("Carry", carry_result)]):
    pnl = result.get("pnl")
    if pnl is None or len(pnl) == 0:
        continue
    weighted_pnl = pnl * _book_weights[name]
    fig_book_attr.add_trace(go.Scatter(
        x=weighted_pnl.index, y=weighted_pnl.cumsum(), mode="lines", name=name,
        line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=1.4),
    ))
fig_book_attr.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative contribution (sum of period returns)",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.15),
)
fig_book_attr = apply_chart_theme(fig_book_attr)
st.plotly_chart(fig_book_attr, use_container_width=True, theme="streamlit")
st.caption(
    "Each Book's own weighted cumulative contribution here sums exactly to the combined "
    "portfolio's own cumulative return at any date, no approximation — weighted by the "
    "combination method selected above (equal [1, 1] for naive, the fitted risk-parity "
    "split otherwise)."
)

st.divider()

render_attribution_section(combined_contributions, asset_to_sector(), key_prefix="combined")
st.caption(
    "Combined across both Books (Trend's and Carry's own per-asset gross contributions, "
    "weighted by the combination method selected above and summed date-by-date) — the same "
    "asset can carry a Trend position and a Carry position at once, so this is exposure "
    "attribution for the whole portfolio, not per-Book."
)

st.divider()

st.subheader("Expanding VaR / ES (95%, combined portfolio)")
expanding = expanding_var_and_es(combined_pnl, confidence=0.95, min_periods=24)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=expanding.index, y=expanding["var"], mode="lines", name="VaR", line=dict(color=CATEGORICAL[3], width=1.3)))
fig3.add_trace(go.Scatter(x=expanding.index, y=expanding["es"], mode="lines", name="ES", line=dict(color=CATEGORICAL[5], width=1.3)))
fig3.update_layout(
    xaxis_title="Date", yaxis_title="Return",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.15),
)
fig3 = apply_chart_theme(fig3)
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    "Expanding, not rolling — Book PnL is weekly-periodicity, `min_periods=24` is a "
    "labeled, not validated, default (`portfolio/risk_metrics.py`)."
)
