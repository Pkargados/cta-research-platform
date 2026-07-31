"""Page 22 — Correlation Estimators.

Technical Appendix page — a methodology comparison, not a new backtest
result, same shape as page 06 (Volatility Estimators): compare estimators,
don't pick a winner, surface where they agree and where they don't. Reads a
precomputed cache (`Data/research/correlation_estimator_{series,summary}.
{parquet,csv}`, `research/correlation_estimator_comparison.py`) — a real
DCC-GARCH MLE fit per pair isn't instant, so this page only ever reads it,
never fits (same never-recomputed-live convention as page 06's GJR-GARCH
vol cache).

Four representative pairs, reused from existing research rather than newly
chosen: the three trend-family signal pairs (`research/trend_correlation.py`
— momentum_12mo, breakout_system1, crossover_50_200) and the Trend/Carry
weekly sleeve pair (`research/sleeve_risk_parity.py`'s own Books,
WORKFLOW.md decision #12).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

from correlation_estimator_comparison import SERIES_CACHE_PATH, SUMMARY_CACHE_PATH, STRESS_WINDOWS

page_header("Correlation Estimators", "Rolling Pearson vs. EWMA Pearson vs. DCC-GARCH — how much do they actually agree?")

ESTIMATOR_LABELS = {"rolling": "Rolling", "ewma": "EWMA", "dcc_garch": "DCC-GARCH"}
ESTIMATOR_COLORS = {"rolling": CATEGORICAL[0], "ewma": CATEGORICAL[1], "dcc_garch": CATEGORICAL[4]}
STRESS_WINDOW_LABELS = {"covid_2020": "2020 COVID crash", "shock_2022": "2022 inflation/rate shock"}

try:
    series = pd.read_parquet(SERIES_CACHE_PATH)
    summary = pd.read_csv(SUMMARY_CACHE_PATH)
except FileNotFoundError:
    st.info(
        "Correlation estimator comparison not yet computed — run `python "
        "research/correlation_estimator_comparison.py` from the repo root to generate it."
    )
    st.stop()

pairs = summary["pair"].tolist()

# Dynamic takeaways, computed from the actual cached numbers rather than
# hardcoded — same discipline as page 06's own avg_winner/rate_winner
# derivation, so this doesn't go stale if the underlying comparison is re-run.
all_converged = summary["dcc_converged"].all()
summary["dcc_vs_rolling_gap"] = (summary["dcc_garch_mean"] - summary["rolling_mean"]).abs()
closest_pair = summary.loc[summary["dcc_vs_rolling_gap"].idxmin(), "pair"]
widest_pair = summary.loc[summary["dcc_vs_rolling_gap"].idxmax(), "pair"]

takeaways = [
    f"DCC-GARCH converged on {'all' if all_converged else 'not all'} {len(pairs)} pairs — "
    "a small-sample fit (as few as ~440 shared observations for the weekly sleeve pair) "
    "isn't automatically trustworthy, so this is checked directly, not assumed.",
    f"**{closest_pair}**: all three estimators land closest together here (full-sample "
    f"rolling/DCC-GARCH mean gap of {summary.loc[summary['pair'] == closest_pair, 'dcc_vs_rolling_gap'].iloc[0]:.3f}) "
    "— the simple and the parametric estimator tell the same story.",
    f"**{widest_pair}**: the widest disagreement (gap of "
    f"{summary.loc[summary['pair'] == widest_pair, 'dcc_vs_rolling_gap'].iloc[0]:.3f}) — worth checking the "
    "per-pair chart below before trusting either estimator's number in isolation here.",
]
render_key_takeaways(takeaways)

st.divider()

st.subheader("Methodology")
st.markdown(
    "**Rolling Pearson** — plain windowed correlation on raw (not demeaned-differently) "
    "returns, the simplest possible estimator, computed per `portfolio.correlation."
    "rolling_correlation`.\n\n"
    "**EWMA Pearson** — RiskMetrics-style exponentially-weighted correlation "
    "(`portfolio.correlation.ewma_correlation`), reacts faster to recent co-movement "
    "than a flat rolling window without an abrupt window-length cutoff.\n\n"
    "**DCC-GARCH** (Engle 2002) — a two-stage fit: univariate GJR-GARCH(1,1,1) per "
    "series, then a dynamic conditional correlation recursion on the standardized "
    "residuals, via the author's own already-validated local `dcc_garch` package "
    "(same fit call `portfolio.sleeve_covariance.dcc_garch_covariance` and `research/"
    "trend_correlation.py` both already use). A genuinely different model class, not "
    "just a different window/halflife choice — its correlation forecast responds to "
    "each series' own conditional volatility, not just their co-movement.\n\n"
    "**Why three, not one:** with as few as ~440 shared weekly observations for the "
    "sleeve pair, DCC's own correlation-recursion parameters are fit on a small "
    "sample — cross-checking against the two simple estimators (which have no fitted "
    "parameters to overfit) is how that risk gets caught, not assumed away."
)

st.divider()

st.subheader("Summary — Full-Sample and Stress-Window Means")
display_cols = {
    "pair": "Pair", "n_obs": "N (shared obs)", "full_sample_corr": "Full-sample corr",
    "dcc_converged": "DCC converged",
    "rolling_mean": "Rolling mean", "ewma_mean": "EWMA mean", "dcc_garch_mean": "DCC-GARCH mean",
}
for window_id, label in STRESS_WINDOW_LABELS.items():
    for est in ("rolling", "ewma", "dcc_garch"):
        display_cols[f"{est}_{window_id}"] = f"{ESTIMATOR_LABELS[est]} — {label}"

st.dataframe(
    summary[list(display_cols.keys())].rename(columns=display_cols).round(3),
    use_container_width=True, hide_index=True,
)
st.caption(
    "Stress-window means (2020 COVID crash: 2020-02-15 to 2020-04-30; 2022 inflation/rate "
    "shock: 2022-01-01 to 2022-12-31) — same two windows `research/trend_correlation.py` "
    "already uses. A widening gap between the rolling/EWMA mean and the full-sample or "
    "stress-window mean signals a genuinely time-varying correlation, not a stable constant."
)

st.divider()

st.subheader("Per-Pair Detail")
pair_choice = st.selectbox("Pair", pairs)
pair_series = series[series["pair"] == pair_choice]
pair_summary = summary[summary["pair"] == pair_choice].iloc[0]

if not pair_summary["dcc_converged"]:
    st.warning("DCC-GARCH did not converge for this pair — treat its correlation path with caution.")

fig = go.Figure()
for start, end in STRESS_WINDOWS.values():
    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(137,135,129,0.12)", line_width=0)
for estimator in ("rolling", "ewma", "dcc_garch"):
    est_series = pair_series[pair_series["estimator"] == estimator].sort_values("date")
    if len(est_series) == 0:
        continue
    fig.add_trace(go.Scatter(
        x=est_series["date"], y=est_series["value"], mode="lines",
        name=ESTIMATOR_LABELS[estimator], line=dict(color=ESTIMATOR_COLORS[estimator], width=1.3),
    ))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Correlation",
    height=440, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(
    "Shaded bands are the two stress windows above. Full-sample static correlation: "
    f"**{pair_summary['full_sample_corr']:.3f}** — compare against how much the rolling/EWMA "
    "lines actually move around that single number over time."
)
