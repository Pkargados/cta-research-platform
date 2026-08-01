"""Page 25 — Covariance Estimators.

Technical Appendix page — a methodology comparison, not a new backtest
result, same shape as page 06 (Volatility Estimators) and page 22
(Correlation Estimators): compare estimators by forecast accuracy, don't
pick a winner by construction, surface where they agree and where they
don't. Reads a precomputed cache (`Data/research/covariance_estimator_
{qlike,summary,pairwise}.{parquet,csv}`, `research/covariance_estimator_
comparison.py`) — same never-recomputed-live convention as pages 06/22.

WORKFLOW.md's "Gerber statistic covariance" plan (Phase 7), step 3 of 4:
Gerber (Gerber, Markowitz, Ernst, Miao, Javid, Sargen 2021) evaluated as a
third covariance-estimator candidate alongside the current Ledoit-Wolf
default, at three thresholds (c=0.5/0.7/0.9), plus plain rolling sample
covariance as the cheap baseline both this comparison and the paper's own
results score against. Diagnostic-only, per the plan's own explicit scope
boundary: this page reports whether Gerber shows a real forecast-accuracy
edge over Ledoit-Wolf — it does not (see the result below) — and no live
Book's actual optimizer covariance is swapped here or anywhere else as a
result of this page.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

from covariance_estimator_comparison import (
    QLIKE_CACHE_PATH, SUMMARY_CACHE_PATH, PAIRWISE_CACHE_PATH,
    STRESS_WINDOWS, MIN_VALID_FRAC,
)

page_header("Covariance Estimators", "Rolling sample vs. Ledoit-Wolf vs. Gerber statistic — does a robust co-movement measure forecast portfolio risk better?")

ESTIMATOR_LABELS = {
    "sample": "Sample", "ledoit_wolf": "Ledoit-Wolf",
    "gerber_c05": "Gerber (c=0.5)", "gerber_c07": "Gerber (c=0.7)", "gerber_c09": "Gerber (c=0.9)",
}
ESTIMATOR_COLORS = {
    "sample": CATEGORICAL[5], "ledoit_wolf": CATEGORICAL[0],
    "gerber_c05": CATEGORICAL[1], "gerber_c07": CATEGORICAL[2], "gerber_c09": CATEGORICAL[4],
}
STRESS_WINDOW_LABELS = {"covid_2020": "2020 COVID crash", "shock_2022": "2022 inflation/rate shock"}

try:
    qlike = pd.read_parquet(QLIKE_CACHE_PATH)
    summary = pd.read_csv(SUMMARY_CACHE_PATH)
    pairwise = pd.read_parquet(PAIRWISE_CACHE_PATH)
except FileNotFoundError:
    st.info(
        "Covariance estimator comparison not yet computed — run `python "
        "research/covariance_estimator_comparison.py` from the repo root to generate it."
    )
    st.stop()

summary = summary.set_index("estimator").loc[list(ESTIMATOR_LABELS.keys())].reset_index()

# Dynamic takeaways, computed from the actual cached numbers rather than
# hardcoded — same discipline as page 06's own avg_winner/rate_winner
# derivation, so this doesn't go stale if the comparison is re-run.
avg_winner = summary.loc[summary["mean_qlike"].idxmin(), "estimator"]
rate_winner = summary.loc[summary["win_rate"].idxmax(), "estimator"]
best_gerber = summary[summary["estimator"].str.startswith("gerber")].sort_values("mean_qlike").iloc[0]
lw_row = summary[summary["estimator"] == "ledoit_wolf"].iloc[0]
gerber_beats_lw = best_gerber["mean_qlike"] < lw_row["mean_qlike"]

takeaways = [
    f"**{ESTIMATOR_LABELS[avg_winner]}** wins pooled-average QLIKE "
    f"({summary.loc[summary['estimator'] == avg_winner, 'mean_qlike'].iloc[0]:.4f}, lower is better) "
    "across all five estimators, evaluated on the global minimum-variance portfolio's "
    "forecast-vs-realized variance (the multivariate analogue of page 06's own QLIKE test).",
    (
        f"**Gerber does not beat Ledoit-Wolf here**: the best Gerber threshold "
        f"({ESTIMATOR_LABELS[best_gerber['estimator']]}, QLIKE {best_gerber['mean_qlike']:.4f}) is "
        f"still worse than Ledoit-Wolf's {lw_row['mean_qlike']:.4f}, and QLIKE gets monotonically "
        "worse as the threshold c increases from 0.5 to 0.9 — reported as found, not tuned "
        "after the fact (CLAUDE.md Rule 1/2)."
        if not gerber_beats_lw else
        f"**Gerber (at {ESTIMATOR_LABELS[best_gerber['estimator']]}) beats Ledoit-Wolf** on pooled "
        f"QLIKE ({best_gerber['mean_qlike']:.4f} vs. {lw_row['mean_qlike']:.4f}) — a real candidate "
        "for further, later evaluation before any live-Book decision (see the plan's own gating rule)."
    ),
    (
        f"Pooled average and per-date win-rate **disagree** on the individual-date leader: "
        f"**{ESTIMATOR_LABELS[rate_winner]}** wins the plurality of individual formation dates "
        f"({summary.loc[summary['estimator'] == rate_winner, 'win_rate'].iloc[0]:.1%}) despite not "
        "having the best pooled average — same tension page 06 already documented between a "
        "mean that a few large-margin dates can dominate and a win-rate that can't."
        if rate_winner != avg_winner else
        f"**{ESTIMATOR_LABELS[avg_winner]}** wins both the pooled average AND the per-date win-rate "
        f"({summary.loc[summary['estimator'] == avg_winner, 'win_rate'].iloc[0]:.1%}) — a consistent result, "
        "not just an artifact of a few outlier dates."
    ),
    "Ledoit-Wolf's shrinkage produces by far the best-conditioned matrices "
    f"(mean condition number {lw_row['mean_condition_number']:,.0f}) — Gerber's PSD-clipping fallback "
    "improves conditioning over plain sample covariance but nowhere near as much as shrinkage does.",
    "**Diagnostic only — no live Book's optimizer covariance is changed by this page.** Per the plan's "
    "own explicit gate, Gerber would need to show a real forecast-accuracy edge here before that's even "
    "considered, and it doesn't.",
]
render_key_takeaways(takeaways)

st.divider()

st.subheader("Methodology")
st.markdown(
    "**Sample** — plain rolling sample covariance, no shrinkage, the cheap baseline.\n\n"
    "**Ledoit-Wolf** (current `Book` default, `portfolio.covariance.build_cov_dict`) — "
    "shrinkage toward a structured target, the standard fix for sample covariance's "
    "own estimation noise in a high-dimension/limited-sample setting.\n\n"
    "**Gerber statistic** (Gerber, Markowitz, Ernst, Miao, Javid, Sargen 2021, "
    "`references/The Gerber Statistic.pdf`, read directly) — a robust, noise-stripping "
    "co-movement measure. For each asset k, a threshold `H_k = c × s_k` (its own sample "
    "std dev); a date/asset is Up if its return >= +H_k, Down if <= -H_k, Neutral "
    "otherwise. A pair's date is concordant if both sides pierce their threshold in the "
    "same direction, discordant if opposite — anything touching Neutral is excluded from "
    "the concordant/discordant count, the whole \"noise-stripping\" point of the "
    "statistic. The paper's own Eq. 11 (the only PSD-safe version used in its empirical "
    "results):\n\n"
    "```\ng_ij = (n_UU + n_DD - n_UD - n_DU) / (T - n_NN)\n```\n\n"
    "Three thresholds (c=0.5/0.7/0.9) built as parallel specs, the paper's own robustness "
    "sweep, fixed a priori — no picking one after seeing results.",
)
st.markdown(
    "**Two disclosed differences from the paper's own setup, not discovered mid-build**: "
    "(1) the paper's own 15-of-15-scenarios win over sample covariance (and 14-of-15 over "
    "Ledoit-Wolf) was demonstrated on 9 clean, monthly, fully-overlapping asset-class "
    "indices, 1990-2020 — a much cleaner setting than this project's own "
    f"~40-asset, ragged-history futures panel doesn't automatically transfer evidence "
    f"from; (2) computed here at the Book's own WEEKLY cadence (`COV_FREQ=\"W-FRI\"`), "
    "not monthly like the paper.\n\n"
    "**Per-pair T_ij, not one global T** — this project's real data has ragged joint "
    "histories, unlike the paper's clean overlapping dataset, so every pair's own "
    "denominator reflects only that pair's own jointly-valid date count "
    "(`portfolio.gerber_covariance.gerber_correlation`).\n\n"
    "**Explicit PSD check + eigenvalue-clipping fallback** on every matrix — the paper "
    "only observed PSD-ness empirically on 9 clean assets, not proven as a theorem, and "
    "not assumed to hold unchecked here.\n\n"
    f"**Universe**: Trend's own already-adopted, compressed universe (WORKFLOW.md decision "
    f"#13), further restricted to assets with >= {MIN_VALID_FRAC:.0%} overall return "
    "coverage — needed because Ledoit-Wolf's row-wise dropna gate produces ZERO usable "
    "weekly dates on the raw, rangier universe (checked directly), while Gerber's own "
    "per-pair tolerance handles that raw panel far better. Restricting the universe here "
    "makes the comparison apples-to-apples across all five estimators, not a thumb on the "
    "scale for either side."
)
st.caption(
    "Forecast-accuracy metric: at each formation date, the GLOBAL MINIMUM-VARIANCE "
    "portfolio implied by that estimator's own Sigma (closed form), scored by QLIKE "
    "(Patton 2011) between its forecast variance and its realized variance held over the "
    "following week — the multivariate analogue of page 06's own single-asset test. "
    "Realized variance from ~5 daily observations per week is genuinely thin (same kind "
    "of small-sample caveat `risk_metrics.py`'s own monthly VaR/ES already carries), but "
    "every estimator is scored on the identical holding period, so the comparison between "
    "them is still fair even if any single number is noisy."
)

st.divider()

st.subheader("Forecast Accuracy — Pooled Summary")
display = summary.copy()
display["estimator"] = display["estimator"].map(ESTIMATOR_LABELS)
st.dataframe(
    display.rename(columns={
        "estimator": "Estimator", "n_dates": "N scored dates",
        "mean_qlike": "Mean QLIKE (lower=better)", "win_rate": "Win rate (per-date)",
        "mean_condition_number": "Mean condition number",
    }).round(4),
    use_container_width=True, hide_index=True,
)
st.caption(
    "Win rate: the fraction of formation dates each estimator's QLIKE was the lowest "
    "among all five — magnitude-robust, but can't distinguish winning by a landslide "
    "from winning by a hair, same tradeoff page 06 already documents for its own "
    "pooled-average-vs-win-rate pair."
)

st.divider()

st.subheader("Condition Number Over Time")
st.caption(
    "A raw numerical-stability diagnostic, not itself a forecast-accuracy score: how "
    "close each estimator's matrix comes to singular at each formation date. Shrinkage "
    "(Ledoit-Wolf) and eigenvalue-clipping (Gerber) both exist specifically to keep this "
    "bounded relative to plain sample covariance."
)
fig_cond = go.Figure()
for est in ESTIMATOR_LABELS:
    series = qlike[qlike["estimator"] == est].sort_values("date")
    fig_cond.add_trace(go.Scatter(
        x=series["date"], y=series["condition_number"], mode="lines",
        name=ESTIMATOR_LABELS[est], line=dict(color=ESTIMATOR_COLORS[est], width=1.1),
    ))
fig_cond.update_layout(
    xaxis_title="Date", yaxis_title="Condition number", yaxis_type="log",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
fig_cond = apply_chart_theme(fig_cond)
st.plotly_chart(fig_cond, use_container_width=True, theme="streamlit")

st.divider()

st.subheader("Per-Pair Detail — Correlation Implied by Each Estimator's Own Sigma")
pairs = sorted(pairwise["pair"].unique())
if not pairs:
    st.info("No representative pair had full coverage in the active universe for this comparison.")
else:
    pair_choice = st.selectbox("Pair", pairs)
    pair_series = pairwise[pairwise["pair"] == pair_choice]

    fig_pair = go.Figure()
    for start, end in STRESS_WINDOWS.values():
        fig_pair.add_vrect(x0=start, x1=end, fillcolor="rgba(137,135,129,0.12)", line_width=0)
    for est in ESTIMATOR_LABELS:
        est_series = pair_series[pair_series["estimator"] == est].sort_values("date")
        if len(est_series) == 0:
            continue
        fig_pair.add_trace(go.Scatter(
            x=est_series["date"], y=est_series["correlation"], mode="lines",
            name=ESTIMATOR_LABELS[est], line=dict(color=ESTIMATOR_COLORS[est], width=1.3),
        ))
    fig_pair.update_layout(
        xaxis_title="Date", yaxis_title="Correlation (implied by Sigma)",
        height=440, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
    )
    fig_pair = apply_chart_theme(fig_pair)
    st.plotly_chart(fig_pair, use_container_width=True, theme="streamlit")
    st.caption(
        "Shaded bands are the two stress windows below. Note this is the correlation "
        "IMPLIED by each estimator's own rolling/shrunk/robust covariance matrix, not a "
        "separately-computed raw rolling correlation — page 22 already covers that "
        "comparison directly on standalone signal/sleeve return series."
    )
    st.caption(
        "Stress windows: 2020 COVID crash (2020-02-15 to 2020-04-30); 2022 inflation/rate "
        "shock (2022-01-01 to 2022-12-31) — same two windows `research/trend_correlation.py`/"
        "`research/correlation_estimator_comparison.py` already use."
    )

st.divider()

st.subheader("Decision")
st.markdown(
    "**Not adopted.** Per WORKFLOW.md's Gerber plan, live-Book integration was gated on "
    "this diagnostic showing a real forecast-accuracy edge over the current Ledoit-Wolf "
    "default — it doesn't, at any of the three thresholds tested. `portfolio.covariance."
    "build_cov_dict` (Ledoit-Wolf) remains every Book's covariance input. "
    "`portfolio.gerber_covariance` is kept as a validated, tested estimator (14 unit "
    "tests) available for future re-evaluation — e.g. if the universe or cadence changes "
    "enough to revisit this — not deleted, but also not wired into any live Book."
    if not gerber_beats_lw else
    "See the takeaways above — the best Gerber spec showed a real forecast-accuracy edge "
    "here, which is a candidate for a separate, later live-Book evaluation, not an "
    "automatic adoption on its own (per the plan's own staged gate)."
)

st.divider()

st.subheader("Book-Level Follow-Up — Does It Help Actual Portfolios Anyway?")
st.markdown(
    "The diagnostic above scores ONE reference portfolio's variance forecast in isolation "
    "— it can't see weight/turnover stability, how the full off-diagonal structure shifts "
    "which alpha a real optimizer crowds down or levers up, or regime-specific behavior a "
    "pooled average washes out. Pursued anyway, per direct instruction, as a disclosed "
    "exception to the plan's own gate — not a re-litigation of the QLIKE result above, a "
    "genuinely different question: does swapping Gerber into a REAL Book change realized, "
    "NET-of-cost Sharpe/turnover/drawdown?"
)

try:
    book_summary = pd.concat([
        pd.read_csv(Path(__file__).resolve().parent.parent.parent / "Data" / "research" / "gerber_book_performance_summary.csv"),
        pd.read_csv(Path(__file__).resolve().parent.parent.parent / "Data" / "research" / "gerber_xsmom_value_seasonality_summary.csv"),
        pd.read_csv(Path(__file__).resolve().parent.parent.parent / "Data" / "research" / "gerber_integrated_value_xsmom_summary.csv"),
    ], ignore_index=True)
    multi_summary = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "Data" / "research" / "gerber_multi_strategy_summary.csv")
    sector_breakdown = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "Data" / "research" / "gerber_sector_breakdown.csv")
except FileNotFoundError:
    st.info(
        "Book-level follow-up not yet computed — run `python research/gerber_book_performance.py`, "
        "`python research/gerber_xsmom_value_seasonality.py`, `python research/gerber_integrated_value_xsmom.py`, "
        "and `python research/gerber_sector_breakdown.py` from the repo root to generate it."
    )
else:
    st.markdown(
        "**Seven Books tested**, Ledoit-Wolf baseline vs. Gerber at all three thresholds, NET of the "
        "same liquidity-tiered transaction costs every other net-of-cost comparison in this project uses:"
    )
    book_display = book_summary.rename(columns={
        "book": "Book", "estimator": "Estimator", "n_rebalance_dates_valid": "N dates",
        "sharpe_train_net": "Train (net)", "sharpe_validation_net": "Validation (net)", "sharpe_test_net": "Test (net)",
        "sharpe_train_gross": "Train (gross)", "sharpe_validation_gross": "Validation (gross)", "sharpe_test_gross": "Test (gross)",
        "turnover": "Turnover (per-period)", "max_dd": "Max DD",
    })
    st.dataframe(
        book_display[["Book", "Estimator", "N dates", "Train (net)", "Validation (net)", "Test (net)",
                       "Turnover (per-period)", "Max DD"]].round(3),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "No consistent rule survives contact with all seven: validation gets worse under Gerber in "
        "6 of 7 Books (Carry is the lone, large exception — its own worst historical stretch cut "
        "dramatically); test is a genuine mixed bag (better for XSMOM/same-month, worse for "
        "Value/Carry/Integrated, roughly tied for both Trend flavors); turnover direction even flips "
        "between strategies (down for Trend/Carry/same-month, UP for XSMOM/Value). The turnover-"
        "smoothing mechanism that looked consistent on the first two Books tested did not generalize."
    )

    st.markdown("**Multi-strategy combination** (naive equal-Book-risk `Allocator`, the current Trend+Carry mandate):")
    multi_display = multi_summary.rename(columns={
        "combo": "Combination", "sharpe_train": "Train", "sharpe_validation": "Validation",
        "sharpe_test": "Test", "n_periods": "N periods",
    })
    st.dataframe(
        multi_display[["Combination", "Train", "Validation", "Test", "N periods"]].round(3),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The per-Book-best mix rescues validation (-0.80 -> +0.04) but makes test WORSE (0.77 -> "
        "0.52) — and since that mix's estimator choice was itself selected by validation Sharpe, its "
        "own validation win is partly mechanical, not free evidence (the same selection-bias caution "
        "this project's own hyperparameter-tuning work already established in Phase 7). Test is the "
        "honest read here, and on test the current all-Ledoit-Wolf mandate still wins outright."
    )

    st.markdown("**Per-sector breakdown** (coarse Commodities/Equities/Rates/FX roll-up, exact decomposition of each Book's own net PnL — not modeled):")
    sector_signal_choice = st.selectbox("Signal", sorted(sector_breakdown["signal"].unique()), key="sector_signal")
    sector_display = sector_breakdown[sector_breakdown["signal"] == sector_signal_choice].rename(columns={
        "estimator": "Estimator", "sector": "Sector", "n_assets": "N assets",
        "sharpe_train": "Train", "sharpe_validation": "Validation", "sharpe_test": "Test",
        "turnover": "Turnover (per-period)", "max_dd": "Max DD",
    })
    st.dataframe(
        sector_display[["Estimator", "Sector", "N assets", "Train", "Validation", "Test", "Turnover (per-period)", "Max DD"]].round(3),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Carry's headline validation improvement under Gerber is NOT spread evenly — it concentrates "
        "in Commodities and especially Rates (a sign flip, -0.495 -> +0.208), while FX and Equities "
        "barely move or get slightly worse. Select \"carry_carry_timing_zero\" above to see this directly."
    )

    st.markdown(
        "**Final: Gerber is not adopted anywhere in this project.** The original diagnostic gate "
        "wasn't cleared, and this extended, disclosed-exception Book-level investigation doesn't clear "
        "a \"clearly better\" bar either — it helps Carry meaningfully, hurts Value and Integrated "
        "Value+XSMOM meaningfully, and is a mixed bag everywhere else, with no predictive rule found "
        "(by strategy speed, cadence, or cross-sectional-vs-time-series construction) that explains the "
        "pattern across all seven Books tested. `portfolio.covariance.build_cov_dict` (Ledoit-Wolf) "
        "remains every live Book's covariance input, with no exception."
    )
