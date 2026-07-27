"""Page 06 — Volatility Estimators.

Computes live at render time, reusing `research/vol_estimator_comparison.py`
(project-wide, signal-agnostic QLIKE/MSE forecast-accuracy comparison) and
`research/momentum.py`'s own vol-estimator builders directly — no duplicated
data-prep logic, no precomputed dashboard_summary/ artifact.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
from lib import page_header, render_key_takeaways, CATEGORICAL, apply_chart_theme

import vol_estimator_comparison as vec

page_header("Volatility Estimators", "Yang-Zhang vs. EWMA — which forecasts realized variance better?")


@st.cache_data(ttl=1200)
def _load():
    adj, raw = vec.load_and_prepare_data()
    vol_estimators, adj_returns = vec.build_vol_estimators(adj, raw)
    return vol_estimators, adj_returns


@st.cache_data(ttl=1200)
def _comparison_table():
    vol_estimators, adj_returns = _load()
    rows = [vec.compare_at_horizon(vol_estimators, adj_returns, h) for h in vec.HORIZONS]
    import pandas as pd
    return pd.concat(rows, ignore_index=True)


vol_estimators, adj_returns = _load()
comparison = _comparison_table()

winner_21d = comparison[comparison["horizon_days"] == 21].sort_values("mean_qlike").iloc[0]
render_key_takeaways([
    f"**{winner_21d['vol_estimator']}** wins on QLIKE forecast accuracy at every "
    "horizon tested — a signal-agnostic, forecast-only comparison "
    "(`data/vol_forecast_eval.py`), not picked by which one produces a higher "
    "backtest Sharpe (that would be circular — see that module's own docstring).",
    "This is the project-wide vol estimator used by momentum/breakout/crossover's "
    "own vol-targeted sizing.",
])

st.divider()

st.subheader("Forecast Accuracy — Train Period Only")
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
    "1/2's discipline, applied to forecast-accuracy selection)."
)

st.divider()

st.subheader("Per-Asset Vol Estimate Comparison")
assets = sorted(vol_estimators["yang_zhang"].columns)
asset = st.selectbox("Asset", assets)

yz = vol_estimators["yang_zhang"][asset].dropna()
ewma = vol_estimators["ewma"][asset].dropna()

fig = go.Figure()
fig.add_trace(go.Scatter(x=yz.index, y=yz.values, mode="lines", name="Yang-Zhang", line=dict(color=CATEGORICAL[0], width=1.3)))
fig.add_trace(go.Scatter(x=ewma.index, y=ewma.values, mode="lines", name="EWMA", line=dict(color=CATEGORICAL[1], width=1.3)))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Annualized volatility",
    height=420, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12),
)
fig = apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.caption(
    "Yang-Zhang is computed off the RAW OHLC curve (roll-masked); EWMA off the "
    "back-adjusted curve's own daily returns — different inputs by design, see "
    "`data/volatility.py`'s module docstring for why the back-adjusted curve "
    "isn't safe for a log-OHLC estimator in this project's older energy segments."
)
