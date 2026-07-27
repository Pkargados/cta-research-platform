"""Page 14 — Portfolio Optimizer Health.

Machinery diagnostics — covariance conditioning, vol-target scale pinning,
cap-bind frequency — reusing the SAME 6-Book construction as page 13, no new
backtest logic. `Book.run()`'s per-date diagnostic series (turnover_series,
scale_series, cap_bind_series) drive this page directly.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

_spec = importlib.util.spec_from_file_location(
    "research_portfolio_driver", Path(__file__).resolve().parent.parent.parent / "research" / "portfolio.py",
)
pf_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf_research)

page_header("Optimizer Health", "Covariance conditioning, vol-target scale pinning, and cap-bind frequency — machinery diagnostics, not a signal result.")


@st.cache_data(ttl=1200)
def _load_and_run():
    adj, raw, included, sectors = pf_research.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = pf_research.build_vol(raw)
    alphas = pf_research.build_six_alphas(adj, raw, included, sectors, vol)

    results = {}
    for name, alpha_df in alphas.items():
        book = pf_research.build_book(name, alpha_df, returns)
        run_result = book.run(returns)
        cond_numbers = pd.Series(
            {date: float(np.linalg.cond(cov.values)) for date, cov in book.cov_dict.items()}
        ).sort_index()
        results[name] = {
            "sharpe": run_result.get("sharpe"), "n_cap_bind": run_result.get("n_cap_bind"),
            "avg_scale": run_result.get("avg_scale"), "n_rebalance_dates_valid": run_result.get("n_rebalance_dates_valid"),
            "turnover_series": run_result.get("turnover_series"), "scale_series": run_result.get("scale_series"),
            "cap_bind_series": run_result.get("cap_bind_series"), "cond_numbers": cond_numbers,
        }
    return results


results = _load_and_run()
book_names = list(results.keys())

cap_bind_rate = {
    name: (res["n_cap_bind"] / res["n_rebalance_dates_valid"]) if res["n_rebalance_dates_valid"] else np.nan
    for name, res in results.items()
}
worst_cap_bind = max(cap_bind_rate, key=cap_bind_rate.get) if cap_bind_rate else None

render_key_takeaways([
    f"Cap-bind rate (fraction of rebalance dates where `max_weight` capped the "
    f"vol-target scale before applying) ranges "
    f"{min(cap_bind_rate.values()):.0%}-{max(cap_bind_rate.values()):.0%} across the 6 Books"
    + (f" — highest for **{worst_cap_bind}**." if worst_cap_bind else "."),
    "Covariance condition number (Ledoit-Wolf shrinkage target) — high values "
    "flag a near-singular Sigma_t, where the optimizer's implied leverage "
    "becomes numerically unstable.",
    "This page answers \"is the machinery behaving sanely,\" not \"is the "
    "strategy profitable\" — see page 13 for the actual performance result.",
])

st.divider()

book = st.selectbox("Book", book_names)
res = results[book]

c1, c2, c3 = st.columns(3)
c1.metric("Avg vol-target scale", f"{res['avg_scale']:.2f}" if res['avg_scale'] is not None else "N/A")
c2.metric("Cap-bind rate", f"{cap_bind_rate[book]:.1%}" if cap_bind_rate[book] == cap_bind_rate[book] else "N/A")
c3.metric("Valid rebalance dates", res["n_rebalance_dates_valid"])

st.subheader(f"Covariance Condition Number — {book}")
cond = res["cond_numbers"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=cond.index, y=cond.values, mode="lines", line=dict(color=CATEGORICAL[0], width=1.3)))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Condition number (log scale)", yaxis_type="log",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(
    "Ledoit-Wolf shrinkage (`portfolio/covariance.py`) pulls the estimate toward "
    "a well-conditioned target — a persistently very high condition number here "
    "would flag the shrinkage isn't doing enough for this Book's active universe."
)

st.divider()

st.subheader(f"Vol-Target Scale Applied — {book}")
scale = res["scale_series"]
cap_bind = res["cap_bind_series"]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=scale.index, y=scale.values, mode="lines", name="Scale applied", line=dict(color=CATEGORICAL[1], width=1.3)))
bound_dates = cap_bind[cap_bind].index
if len(bound_dates) > 0:
    fig2.add_trace(go.Scatter(
        x=bound_dates, y=scale.reindex(bound_dates), mode="markers", name="Cap-bound",
        marker=dict(color=CATEGORICAL[5], size=6, symbol="x"),
    ))
fig2.update_layout(
    xaxis_title="Date", yaxis_title="Scale factor",
    height=380, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
fig2 = apply_chart_theme(fig2)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
st.caption(
    "Scale = (target_vol / realized_vol)^2, power-scaled, bounded by "
    "[scale_min, scale_max] and by `max_weight` — 'x' markers show dates where "
    "the `max_weight` cap bound the scale before the final clip."
)

st.divider()

st.subheader("All Books — Machinery Summary")
summary = pd.DataFrame({
    name: {"Sharpe": r["sharpe"], "Avg scale": r["avg_scale"], "Cap-bind rate": cap_bind_rate[name],
           "Median cond. number": r["cond_numbers"].median() if len(r["cond_numbers"]) else np.nan}
    for name, r in results.items()
}).T
st.dataframe(summary.round(4), use_container_width=True)
