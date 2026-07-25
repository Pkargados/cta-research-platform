"""Page 13 — Portfolio Construction (Book/Allocator Result).

Computes live at render time, reusing `research/portfolio.py` directly — the
first real Book/Allocator exercise, one representative Book per signal family
(6 total), monthly rebalancing. Combined result + risk metrics (VaR/ES).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL

# research/portfolio.py's own filename collides with the real `portfolio`
# package under src/ (portfolio.covariance, portfolio.book, ...) - loaded via
# an explicit module spec under a non-colliding name instead of sys.path, so
# both "the driver script" and "the src/portfolio package it imports" resolve
# correctly at the same time.
_spec = importlib.util.spec_from_file_location(
    "research_portfolio_driver", Path(__file__).resolve().parent.parent.parent / "research" / "portfolio.py",
)
pf_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf_research)

from portfolio.allocator import Allocator
from portfolio.risk_metrics import historical_var, expected_shortfall, expanding_var_and_es
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from backtest.performance import simple_sharpe

page_header("Portfolio Construction", "One representative Book per signal family, combined via the Optimizer/Allocator.")


@st.cache_data(ttl=1200)
def _load_and_run():
    adj, raw, included, sectors = pf_research.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = pf_research.build_vol(raw)
    alphas = pf_research.build_six_alphas(adj, raw, included, sectors, vol)

    book_results = {}
    books = []
    for name, alpha_df in alphas.items():
        book = pf_research.build_book(name, alpha_df, returns)
        result = book.run(returns)
        book_results[name] = result
        books.append(book)

    allocator = Allocator(books)
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]
    return returns, book_results, combined_pnl


returns, book_results, combined_pnl = _load_and_run()

per_book_summary = pd.DataFrame({
    name: {"Sharpe": res.get("sharpe"), "Max DD": res.get("max_dd"), "Turnover": res.get("turnover"),
           "Valid rebalance dates": res.get("n_rebalance_dates_valid"), "Stale gaps": res.get("n_stale_gaps")}
    for name, res in book_results.items()
}).T

combined_periods = {k: simple_sharpe(v, periods_per_year=pf_research.PERIODS_PER_YEAR)
                     for k, v in zip(("train", "validation", "test"), train_validation_test_split(combined_pnl))}
var95 = historical_var(combined_pnl, confidence=0.95)
es95 = expected_shortfall(combined_pnl, confidence=0.95)

render_key_takeaways([
    "Explicitly NOT the full 19+-spec roster each family's own Book-count "
    "decision calls for — one representative Book per family, a first, "
    "small-scale exercise.",
    f"Combined Allocator Sharpe: train **{combined_periods['train']:.2f}**, "
    f"validation **{combined_periods['validation']:.2f}**, test **{combined_periods['test']:.2f}** "
    "— optimizer beats naive-equal-weight on test, loses on validation (genuinely mixed).",
    f"Full-sample 95% VaR **{var95:.1%}**, ES **{es95:.1%}** (monthly) — risk "
    "MEASUREMENT, not risk control; doesn't throttle anything on its own.",
])

st.divider()

st.subheader("Per-Book Results")
st.dataframe(per_book_summary.round(4), use_container_width=True)
st.caption(
    "`Stale gaps` = rebalance-date gaps longer than `max_gap_days` (default 60), "
    "flattened to zero PnL rather than pricing a stale, unmanaged position "
    "against real subsequent moves — see `portfolio/book.py`'s own documented "
    "2014-2015 finding."
)

st.divider()

st.subheader("Combined Allocator Equity Curve")
equity = (1 + combined_pnl.dropna()).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color=CATEGORICAL[0], width=1.8)))
fig.add_vline(x=TRAIN_END, line_dash="dash", line_color="#898781")
fig.add_vline(x=VALIDATION_END, line_dash="dash", line_color="#898781")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Per-Book Equity Curves")
fig2 = go.Figure()
for i, (name, res) in enumerate(book_results.items()):
    pnl = res.get("pnl")
    if pnl is None or len(pnl) == 0:
        continue
    eq = (1 + pnl).cumprod()
    fig2.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=1.3)))
fig2.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative return (x)",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.15),
)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

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
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.caption(
    "Expanding, not rolling — Book PnL is monthly-periodicity (~120-150 total "
    "observations), too few for a rolling window to leave a meaningful tail "
    "estimate. `min_periods=24` is a labeled, not validated, default — see "
    "`portfolio/risk_metrics.py`'s own module docstring."
)
