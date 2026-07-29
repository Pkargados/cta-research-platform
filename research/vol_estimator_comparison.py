"""
research/vol_estimator_comparison.py — Project-wide Yang-Zhang vs. EWMA vs.
GJR-GARCH volatility-forecast accuracy comparison, via `data.vol_forecast_eval`'s
QLIKE/MSE losses against forward-realized variance.

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

GJR-GARCH (`data.garch_volatility`, wrapping the external `dcc_garch` package,
originally built for a separate DCC-GARCH correlation project) added per direct
instruction — a genuine third candidate, not just Yang-Zhang vs. EWMA, since
GJR-GARCH's asymmetric-shock response is a well-regarded standard in the vol-
forecasting literature and validated code already existed. Refit every 20
trading days (~monthly), filtered daily between refits with fixed parameters —
standard GARCH practice, not a backtested/tuned choice. Slow (real MLE fits per
asset per refit window, ~3s/asset for the train period alone) — precomputed
once via `load_or_compute_garch()` and cached to Data/research/garch_volatility.parquet,
never recomputed live.

Run: `python research/vol_estimator_comparison.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.volatility import yang_zhang_volatility
from data.ewma_volatility import ewma_volatility
from data.garch_volatility import gjr_garch_volatility
from data.vol_forecast_eval import forward_realized_variance, qlike_loss, mse_vol_loss, per_asset_mean_loss
from data.sectors import asset_to_sector
from backtest.splits import TRAIN_END

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
HORIZONS = (21, 63)  # ~1 month, ~1 quarter ahead
GARCH_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "garch_volatility.parquet"


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


def load_or_compute_garch(adj_returns: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Cached GJR-GARCH volatility -- computed once (real MLE fits, slow) and
    reused from Data/research/garch_volatility.parquet on every subsequent call,
    never recomputed live."""
    if not force and GARCH_CACHE_PATH.exists():
        return pd.read_parquet(GARCH_CACHE_PATH)
    garch = gjr_garch_volatility(adj_returns)
    GARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    garch.to_parquet(GARCH_CACHE_PATH)
    return garch


def build_vol_estimators_with_garch(adj, raw, force_garch: bool = False):
    """build_vol_estimators() plus a third 'gjr_garch' key, for the full
    3-way comparison. Kept separate from build_vol_estimators() itself so
    every existing caller (the dashboard's live-computed pages) stays fast
    and unaffected -- GARCH is opt-in, not a default dependency."""
    vol_estimators, adj_returns = build_vol_estimators(adj, raw)
    vol_estimators["gjr_garch"] = load_or_compute_garch(adj_returns, force=force_garch)
    return vol_estimators, adj_returns


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


def per_asset_comparison(vol_estimators, adj_returns, horizon):
    """Per-asset mean QLIKE loss for each estimator (TRAIN period only), plus each
    asset's own winner and the margin between them. Complements compare_at_horizon's
    pooled average, which a handful of assets with unusually large loss gaps can
    dominate -- this shows every asset's own comparison instead of collapsing straight
    to one number. `winner`/`margin` are only set where BOTH estimators have a valid
    (min_obs-satisfying) loss for that asset; otherwise NaN/None, not a misleading
    single-estimator "winner"."""
    realized_var = forward_realized_variance(adj_returns, horizon).loc[:TRAIN_END]
    est_names = list(vol_estimators.keys())
    cols = {}
    for name, vol in vol_estimators.items():
        forecast_var = (vol.loc[:TRAIN_END]) ** 2
        qlike = qlike_loss(forecast_var, realized_var)
        cols[name] = per_asset_mean_loss(qlike)
    table = pd.DataFrame(cols)

    mask = table[est_names].notna().all(axis=1)
    table["winner"] = None
    table.loc[mask, "winner"] = table.loc[mask, est_names].idxmin(axis=1)
    table["margin"] = np.nan
    table.loc[mask, "margin"] = table.loc[mask, est_names].max(axis=1) - table.loc[mask, est_names].min(axis=1)
    return table


def win_rate_summary(per_asset_table, estimator_names):
    """Fraction of assets each estimator wins by QLIKE (assets with a valid
    comparison for both estimators only) -- a magnitude-robust complement to the
    pooled average, which can be swayed by a few large-margin assets."""
    valid = per_asset_table.dropna(subset=["winner"])
    total = len(valid)
    counts = valid["winner"].value_counts().reindex(estimator_names, fill_value=0)
    return pd.DataFrame({"wins": counts, "win_pct": counts / total if total else np.nan})


def sector_breakdown(per_asset_table, estimator_names):
    """Mean QLIKE per estimator, grouped by sector, plus each sector's own winner
    and within-sector win-count -- does the pooled/win-rate winner hold up across
    asset classes, or is it uneven? DESCRIPTIVE ONLY: several sectors here have
    fewer than 5 members (IndustrialMetals has exactly one), so a per-sector winner
    is thin evidence, not a basis for using a different estimator per sector without
    a real significance test -- see research/vol_estimator_comparison.py's own
    module docstring on why per-signal-Sharpe-based selection was already rejected
    once for being economically incoherent; a per-sector QLIKE split naively adopted
    without correcting for multiple comparisons would repeat that mistake one level
    down, not fix it."""
    df = per_asset_table.copy()
    df["sector"] = df.index.map(asset_to_sector())
    df = df.dropna(subset=["sector"])
    rows = []
    for sector, group in df.groupby("sector"):
        valid = group.dropna(subset=estimator_names, how="any")
        if valid.empty:
            continue
        means = valid[estimator_names].mean()
        win_counts = valid["winner"].value_counts()
        rows.append({
            "sector": sector, "n_assets": len(valid),
            **{f"mean_qlike_{name}": means[name] for name in estimator_names},
            "winner": means.idxmin(),
            **{f"wins_{name}": int(win_counts.get(name, 0)) for name in estimator_names},
        })
    return pd.DataFrame(rows).set_index("sector")


def main():
    adj, raw = load_and_prepare_data()
    vol_estimators, adj_returns = build_vol_estimators_with_garch(adj, raw)

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

    est_names = list(vol_estimators.keys())
    for horizon in HORIZONS:
        per_asset = per_asset_comparison(vol_estimators, adj_returns, horizon)
        print(f"\n--- Win rate by asset, horizon = {horizon}d (magnitude-robust complement to the pooled average) ---")
        print(win_rate_summary(per_asset, est_names).to_string())
        print(f"\n--- Sector breakdown, horizon = {horizon}d (descriptive only, see function docstring) ---")
        print(sector_breakdown(per_asset, est_names).round(4).to_string())


if __name__ == "__main__":
    main()
