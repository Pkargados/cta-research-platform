"""Page 06 — Volatility Estimators.

Computes live at render time, reusing `research/vol_estimator_comparison.py`
(project-wide, signal-agnostic QLIKE/MSE forecast-accuracy comparison)
directly — no duplicated data-prep logic, no precomputed dashboard_summary/
artifact. GJR-GARCH is read from a precomputed cache
(Data/research/garch_volatility.parquet, `data.garch_volatility`'s
load_or_compute_garch()) — a real MLE fit per asset per refit window, far
too slow to compute live, so this page only ever reads it, never fits.

First page in the "Estimators" nav group — a methodology comparison, not a
status check, evaluated by forecast accuracy rather than backtest
performance (see the methodology section below for why). Shows both the
pooled average QLIKE (can be swayed by a few large-margin assets) and a
per-asset win-rate (magnitude-robust, but blind to how much an estimator
wins by) — the two can and do disagree here, which is itself a finding, not
noise to average away.
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

import vol_estimator_comparison as vec

page_header("Volatility Estimators", "Yang-Zhang vs. EWMA vs. GJR-GARCH — which forecasts realized variance better?")


@st.cache_data(ttl=1200)
def _load():
    adj, raw = vec.load_and_prepare_data()
    vol_estimators, adj_returns = vec.build_vol_estimators_with_garch(adj, raw)
    return vol_estimators, adj_returns


@st.cache_data(ttl=1200)
def _comparison_table():
    vol_estimators, adj_returns = _load()
    rows = [vec.compare_at_horizon(vol_estimators, adj_returns, h) for h in vec.HORIZONS]
    return pd.concat(rows, ignore_index=True)


@st.cache_data(ttl=1200)
def _per_asset_and_derived():
    vol_estimators, adj_returns = _load()
    est_names = list(vol_estimators.keys())
    per_asset = {h: vec.per_asset_comparison(vol_estimators, adj_returns, h) for h in vec.HORIZONS}
    win_rates = {h: vec.win_rate_summary(per_asset[h], est_names) for h in vec.HORIZONS}
    sectors = {h: vec.sector_breakdown(per_asset[h], est_names) for h in vec.HORIZONS}
    return per_asset, win_rates, sectors, est_names


vol_estimators, adj_returns = _load()
comparison = _comparison_table()
per_asset, win_rates, sector_tables, est_names = _per_asset_and_derived()

# Per-horizon winners, both by pooled average and by win-rate -- the two
# genuinely disagree at some horizons here (a finding, see below), so
# neither is computed once and assumed to hold across all of them.
avg_winner = {}
rate_winner = {}
for h in vec.HORIZONS:
    avg_winner[h] = comparison[comparison["horizon_days"] == h].sort_values("mean_qlike").iloc[0]["vol_estimator"]
    rate_winner[h] = win_rates[h]["win_pct"].idxmax()

takeaways = [
    "Three estimators compared by forecast accuracy (QLIKE, Patton 2011), not by "
    "which one produces a higher backtest Sharpe — that would be circular, since "
    "this estimator is shared across three signal families (see Methodology below).",
]
for h in vec.HORIZONS:
    avg_qlike = comparison[(comparison["horizon_days"] == h) & (comparison["vol_estimator"] == avg_winner[h])]["mean_qlike"].iloc[0]
    rate_pct = win_rates[h].loc[rate_winner[h], "win_pct"]
    if avg_winner[h] == rate_winner[h]:
        takeaways.append(
            f"At {h}d, the average and the win-rate **agree**: **{avg_winner[h]}** wins "
            f"pooled-average QLIKE ({avg_qlike:.4f}) and wins **{rate_pct:.0%}** of assets individually."
        )
    else:
        takeaways.append(
            f"At {h}d, the average and the win-rate **disagree**: the pooled average "
            f"favors {avg_winner[h]} ({avg_qlike:.4f}), but **{rate_winner[h]}** actually "
            f"wins **{rate_pct:.0%}** of assets individually — the average isn't the "
            "robust summary it looks like on its own."
        )
takeaways.append(
    "GJR-GARCH is genuinely competitive, not a token third entry: ties or beats "
    "Yang-Zhang on the pooled average and wins the individual-asset win-rate "
    "decisively at every horizon tested."
)
render_key_takeaways(takeaways)

st.divider()

st.subheader("Methodology")
st.markdown(
    "**Yang-Zhang** uses the full OHLC range each day (open/high/low/close) plus the "
    "overnight open-to-previous-close gap — it captures information the closing price "
    "alone misses, at the cost of needing a genuine, clean OHLC panel.\n\n"
    "**EWMA** is an exponentially-weighted moving variance of past close-to-close "
    "returns only — simpler, needs less data, but blind to intraday range and "
    "overnight moves.\n\n"
    "**GJR-GARCH(1,1,1)** is a parametric model fit via maximum likelihood, with "
    "asymmetric response to positive vs. negative return shocks (the \"leverage "
    "effect\") — a well-regarded standard in the vol-forecasting literature. Refit "
    "every 20 trading days on an expanding window, with the conditional-variance "
    "path extended daily between refits using fixed parameters (standard practice — "
    "the recursion itself updates day to day; only the parameters need periodic "
    "re-estimation). Far more expensive to compute than the other two, so it's "
    "precomputed and cached, never fit live on this page.\n\n"
    "**Why compared by forecast accuracy, not backtest Sharpe:** picking a vol "
    "estimator by which one produces a better SIGNAL Sharpe is economically "
    "incoherent — volatility is a property of the asset's price history, not of "
    "whichever signal happens to be consuming it, and this estimator is shared "
    "across three different signal families. QLIKE (Patton 2011) instead measures "
    "each estimator directly against subsequently realized variance — the estimator "
    "either predicts the future accurately or it doesn't, independent of any one "
    "strategy's own quirks."
)
st.caption(
    "A real numerical bug was found and fixed while adding GJR-GARCH: its fixed "
    "internal input scaling left one asset (US_2Y) far outside the fitting "
    "library's own documented stable range, producing a degenerate near-zero "
    "volatility estimate in several windows. Diagnosed from the library's own "
    "convergence warnings and fixed with a per-asset dynamic rescale — see "
    "`data/garch_volatility.py`'s own module docstring for the full account."
)

st.divider()

st.subheader("Forecast Accuracy — Pooled Average, Train Period Only")
st.dataframe(
    comparison.rename(columns={
        "vol_estimator": "Estimator", "horizon_days": "Horizon (days)",
        "mean_qlike": "Mean QLIKE (lower=better)", "mean_mse_vol": "Mean MSE (vol)",
    }),
    use_container_width=True, hide_index=True,
)
st.caption(
    "QLIKE (Patton 2011): 0 at a perfect forecast, penalizes underprediction more "
    "than overprediction of the same relative size — appropriate here since this "
    "vol estimate sizes real positions. Evaluated on TRAIN only (CLAUDE.md Rule "
    "1/2's discipline, applied to forecast-accuracy selection, not backtest performance)."
)

st.divider()

st.subheader("Win Rate — Per-Asset, Not Just the Average")
st.caption(
    "The pooled average above can be swayed by a few assets with unusually large "
    "loss gaps. This counts how many individual assets each estimator actually wins "
    "on — magnitude-robust, though it can't distinguish winning by a landslide from "
    "winning by a hair."
)
wr_cols = st.columns(2)
for col, h in zip(wr_cols, vec.HORIZONS):
    with col:
        st.markdown(f"**{h}-day horizon**")
        st.dataframe(win_rates[h].round(3), use_container_width=True)

st.divider()

st.subheader("Sector Breakdown — Does the Winner Hold Up Across Asset Classes?")
st.caption(
    "DESCRIPTIVE ONLY, not a basis for using a different estimator per sector "
    "without a real significance test — several sectors here have fewer than 5 "
    "members (IndustrialMetals has exactly one), so a per-sector winner is thin "
    "evidence, and naively adopting one without correcting for multiple comparisons "
    "would repeat the exact mistake this project's own hyperparameter-tuning work "
    "already found and corrected for (see WORKFLOW.md Phase 7)."
)
sector_horizon = st.radio("Horizon", list(vec.HORIZONS), horizontal=True, format_func=lambda h: f"{h}d")
sector_table = sector_tables[sector_horizon]
display_cols = {"n_assets": "N assets", **{f"mean_qlike_{n}": f"Mean QLIKE ({n})" for n in est_names}, "winner": "Winner", **{f"wins_{n}": f"Wins ({n})" for n in est_names}}
st.dataframe(sector_table.rename(columns=display_cols).round(4), use_container_width=True)

st.divider()

st.subheader("Per-Asset Detail")
assets = sorted(vol_estimators["yang_zhang"].columns)
asset = st.selectbox("Asset", assets)
detail_horizon = st.radio("Horizon (winner check)", list(vec.HORIZONS), horizontal=True, format_func=lambda h: f"{h}d", key="detail_horizon")

asset_row = per_asset[detail_horizon].loc[asset] if asset in per_asset[detail_horizon].index else None
if asset_row is not None and pd.notna(asset_row.get("winner")):
    st.markdown(
        f"At the {detail_horizon}d horizon, **{asset_row['winner']}** wins for **{asset}** "
        f"specifically (QLIKE margin {asset_row['margin']:.4f})."
    )
else:
    st.info(f"{asset} doesn't have enough valid observations at the {detail_horizon}d horizon for a reliable per-asset comparison.")

yz = vol_estimators["yang_zhang"][asset].dropna()
ewma = vol_estimators["ewma"][asset].dropna()
garch = vol_estimators["gjr_garch"][asset].dropna()

fig = go.Figure()
fig.add_trace(go.Scatter(x=yz.index, y=yz.values, mode="lines", name="Yang-Zhang", line=dict(color=CATEGORICAL[0], width=1.3)))
fig.add_trace(go.Scatter(x=ewma.index, y=ewma.values, mode="lines", name="EWMA", line=dict(color=CATEGORICAL[1], width=1.3)))
fig.add_trace(go.Scatter(x=garch.index, y=garch.values, mode="lines", name="GJR-GARCH", line=dict(color=CATEGORICAL[4], width=1.3)))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Annualized volatility",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.caption(
    "Yang-Zhang is computed off the RAW OHLC curve (roll-masked); EWMA and GJR-GARCH "
    "off the back-adjusted curve's own daily returns — different inputs by design, see "
    "`data/volatility.py`'s module docstring for why the back-adjusted curve "
    "isn't safe for a log-OHLC estimator in this project's older energy segments."
)
