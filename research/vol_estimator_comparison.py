"""
research/vol_estimator_comparison.py — Project-wide Yang-Zhang vs. EWMA
volatility-forecast accuracy comparison, via `data.vol_forecast_eval`'s QLIKE/MSE
losses against forward-realized variance.

Signal-agnostic (no backtest Sharpe anywhere in this file) — see
`data/vol_forecast_eval.py`'s own module docstring for why picking a vol
estimator by which one produces a higher SIGNAL Sharpe (what `research/
momentum.py`/`research/breakout.py` each did independently, per-spec) is
economically incoherent: volatility is a property of the asset's price history,
not of whichever signal happens to be consuming it. This script is the
project-wide, forecast-accuracy-only answer to that question.

Evaluated on the TRAIN period only (CLAUDE.md Rule 1/2's "never pick a spec by
looking at [out-of-sample] results" discipline, applied here to forecast-accuracy
selection, not backtest performance).

Run: `python research/vol_estimator_comparison.py` from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.volatility import yang_zhang_volatility
from data.ewma_volatility import ewma_volatility
from data.vol_forecast_eval import forward_realized_variance, qlike_loss, mse_vol_loss, per_asset_mean_loss
from backtest.splits import TRAIN_END

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
HORIZONS = (21, 63)  # ~1 month, ~1 quarter ahead


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, _ = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    return adj, raw


def build_vol_estimators(adj, raw):
    yang_zhang = yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )
    adj_returns = adj["close"].pct_change(fill_method=None)
    ewma = ewma_volatility(adj_returns)
    return {"yang_zhang": yang_zhang, "ewma": ewma}, adj_returns


def compare_at_horizon(vol_estimators, adj_returns, horizon):
    """Per-estimator mean QLIKE/MSE loss, averaged across assets, TRAIN period only."""
    realized_var = forward_realized_variance(adj_returns, horizon).loc[:TRAIN_END]
    realized_vol = realized_var ** 0.5

    rows = []
    for name, vol in vol_estimators.items():
        forecast_var = (vol.loc[:TRAIN_END]) ** 2
        forecast_vol = vol.loc[:TRAIN_END]

        qlike = qlike_loss(forecast_var, realized_var)
        mse = mse_vol_loss(forecast_vol, realized_vol)

        qlike_mean = per_asset_mean_loss(qlike).mean()
        mse_mean = per_asset_mean_loss(mse).mean()
        rows.append({"vol_estimator": name, "horizon_days": horizon, "mean_qlike": qlike_mean, "mean_mse_vol": mse_mean})
    return pd.DataFrame(rows)


def main():
    adj, raw = load_and_prepare_data()
    vol_estimators, adj_returns = build_vol_estimators(adj, raw)

    print("--- Forecast accuracy, TRAIN period only, mean across assets ---")
    all_rows = []
    for horizon in HORIZONS:
        result = compare_at_horizon(vol_estimators, adj_returns, horizon)
        all_rows.append(result)
        print(f"\nHorizon = {horizon}d:")
        print(result.set_index("vol_estimator").round(4).to_string())

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(Path(__file__).resolve().parent.parent / "Data" / "research" / "vol_estimator_comparison.csv", index=False)

    qlike_winner_by_horizon = full.loc[full.groupby("horizon_days")["mean_qlike"].idxmin()]
    print("\n--- QLIKE winner per horizon (lower is better) ---")
    print(qlike_winner_by_horizon[["horizon_days", "vol_estimator", "mean_qlike"]].to_string(index=False))


if __name__ == "__main__":
    main()
