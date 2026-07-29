"""
research/book_vol_targeting_estimator.py — Does `Book._apply_vol_target`'s
EWMA-of-realized-PnL deserve to be the vol-targeting estimator, or does it
inherit that role from the retired stat-arb engine without ever being tested
against alternatives?

Direct follow-up to the asset-level 3-way vol-estimator comparison
(`research/vol_estimator_comparison.py`, YZ/EWMA/GJR-GARCH — EWMA came in
last there). That comparison forecasts each ASSET's own price volatility from
OHLC bars; `Book._apply_vol_target` forecasts something structurally
different — the BOOK's own aggregate REALIZED PNL volatility, used to scale
total Book leverage to hit its target_vol. Different input, so the earlier
result doesn't mechanically transfer. But GJR-GARCH doesn't care whether the
1D return series it's fed is an asset's price return or a Book's own PnL
series, so there's no structural reason not to test the same question here.

Method: each selected Book's (tsmom_alone Trend, carry_timing_zero Carry)
already-solved weekly weight path, marked against DAILY returns via
`portfolio.book.daily_mark_pnl` (built 2026-07-23 for exactly this kind of
finer-grained, non-re-solving analysis — see that function's own docstring).
Two candidate walk-forward vol forecasts on that daily PnL series:
- EWMA (`data.ewma_volatility.ewma_volatility`, unmodified — generic on any
  returns DataFrame, reused as-is, not reimplemented). Cheap, computed live.
- GJR-GARCH (`data.garch_volatility.gjr_garch_volatility`, unmodified, same
  per-series dynamic rescale the US_2Y fix already validated). Slow (real MLE
  fits) — precomputed once and cached to
  Data/research/book_vol_targeting_garch.parquet, same convention
  `research/vol_estimator_comparison.py`'s own `load_or_compute_garch`
  already established, never recomputed live (this is what the dashboard
  page reads).
Same QLIKE/MSE evaluation (`data.vol_forecast_eval`, unmodified) against
forward-realized variance, TRAIN period only (CLAUDE.md Rule 1/2 — this
selects a FORECASTING method, not a backtest spec, but the same "never look
at out-of-sample results to choose" discipline applies).

Result (2026-07-29): GJR-GARCH wins decisively at every horizon, both Books —
QLIKE loss cut ~60-70% vs. EWMA. Adopted: `Book.vol_estimator="garch"` (see
`src/portfolio/book.py`) now backs the Trend/Carry Single Strategy Portfolios.
Only GARCH's own parameter (refit cadence) belongs in the later hyperparameter
grid — EWMA's halflife is moot, the estimator class itself was replaced.

Run: `python research/book_vol_targeting_estimator.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.ewma_volatility import ewma_volatility
from data.garch_volatility import gjr_garch_volatility
from data.vol_forecast_eval import forward_realized_variance, qlike_loss, mse_vol_loss, per_asset_mean_loss
from backtest.splits import TRAIN_END
from portfolio.book import daily_mark_pnl

import single_strategy_portfolios as ssp

HORIZONS = (21, 63)
GARCH_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "book_vol_targeting_garch.parquet"
COMPARISON_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "book_vol_targeting_comparison.csv"
SERIES_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "book_vol_targeting_series.parquet"


def build_selected_books_pnl():
    """Rebuilds exactly the two Books single_strategy_portfolios.py's bake-off
    selected (tsmom_alone, carry_timing_zero) - not re-run through the bake-off
    itself, just the winning construction each. Uses vol_estimator="ewma" here
    deliberately (the DEFAULT, pre-decision Book) - this script is what
    DECIDES whether to switch to "garch" in the first place, so it can't
    already assume the answer."""
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = ssp.build_vol(raw)
    carry_panel, _ = ssp.build_carry_panel(included)

    trend_flavors = ssp.build_trend_flavors(close, vol, returns)
    trend_book = ssp.build_book("trend_tsmom_alone", trend_flavors["tsmom_alone"], returns)
    trend_result = trend_book.run(returns)

    carry_flavors = ssp.build_all_carry_signals(carry_panel, sectors)
    carry_book = ssp.build_book("carry_timing_zero", carry_flavors["carry_timing_zero"], returns)
    carry_result = carry_book.run(returns)

    # daily_mark_pnl's own reindex().ffill() is NaN before a Book's first real
    # weight date - but pandas' .sum(skipna=True) over an all-NaN row returns
    # 0.0, not NaN (documented, deliberate behavior for scattered single-asset
    # gaps, not meant for a multi-year all-NaN prefix). Left untrimmed, ~900
    # days of literal 0.0 "no Book yet" pseudo-returns would sit in every
    # estimator's warmup window and silently corrupt both candidates' vol
    # estimate (GARCH's own zero-variance guard catches this outright and
    # returns all-NaN; EWMA would NOT crash, just quietly understate vol for
    # years - a worse failure mode, not a safer one). Trim each Book's PnL to
    # start at its own first real rebalance date before comparing anything.
    trend_pnl = daily_mark_pnl(trend_result["weights"], returns)
    carry_pnl = daily_mark_pnl(carry_result["weights"], returns)
    return {
        "trend": trend_pnl.loc[trend_result["weights"].index[0]:],
        "carry": carry_pnl.loc[carry_result["weights"].index[0]:],
    }


def load_or_compute_garch(pnl_series: dict, force: bool = False) -> pd.DataFrame:
    """Cached GJR-GARCH volatility for both Books' daily-marked PnL - computed
    once (real MLE fits, slow) and reused from
    Data/research/book_vol_targeting_garch.parquet on every subsequent call,
    never recomputed live. Same convention as `research/
    vol_estimator_comparison.py`'s own `load_or_compute_garch`."""
    if not force and GARCH_CACHE_PATH.exists():
        return pd.read_parquet(GARCH_CACHE_PATH)
    # Combined into one (T x 2) frame so gjr_garch_volatility's own per-column
    # loop handles both Books in one call - each series' differing start date
    # is just a differently-sized leading NaN block, which _asset_walk_forward_
    # vol already drops per-column before fitting (no cross-contamination).
    combined = pd.concat({name: s for name, s in pnl_series.items()}, axis=1)
    garch = gjr_garch_volatility(combined, verbose=True)
    GARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    garch.to_parquet(GARCH_CACHE_PATH)
    return garch


def build_candidate_forecasts(pnl_series: dict, force_garch: bool = False) -> dict:
    ewma = pd.concat({name: ewma_volatility(s.to_frame("x"))["x"] for name, s in pnl_series.items()}, axis=1)
    garch = load_or_compute_garch(pnl_series, force=force_garch)
    return {"ewma": ewma, "gjr_garch": garch}


def compare_book(label: str, pnl: pd.Series, candidates: dict) -> pd.DataFrame:
    print(f"\n=== {label} Book: EWMA vs. GJR-GARCH on its own daily-marked PnL, TRAIN only ===")
    pnl_df = pnl.to_frame(label)
    rows = []
    for horizon in HORIZONS:
        realized_var = forward_realized_variance(pnl_df, horizon).loc[:TRAIN_END]
        for name, vol_df in candidates.items():
            forecast_var = (vol_df[[label]].loc[:TRAIN_END]) ** 2
            forecast_vol = vol_df[[label]].loc[:TRAIN_END]
            realized_vol = realized_var ** 0.5

            qlike = qlike_loss(forecast_var, realized_var)
            mse = mse_vol_loss(forecast_vol, realized_vol)
            qlike_mean = per_asset_mean_loss(qlike).iloc[0]
            mse_mean = per_asset_mean_loss(mse).iloc[0]
            rows.append({"book": label, "horizon": horizon, "estimator": name, "qlike": qlike_mean, "mse_vol": mse_mean})

    result = pd.DataFrame(rows)
    print(result.drop(columns="book").round(6).to_string(index=False))

    for horizon in HORIZONS:
        sub = result[result["horizon"] == horizon].dropna(subset=["qlike"])
        if sub.empty:
            print(f"  Horizon {horizon}d: no valid comparison (insufficient warmup)")
            continue
        winner = sub.loc[sub["qlike"].idxmin(), "estimator"]
        print(f"  Horizon {horizon}d QLIKE winner: {winner}")
    return result


def main():
    pnl_series = build_selected_books_pnl()
    candidates = build_candidate_forecasts(pnl_series)

    trend_result = compare_book("trend", pnl_series["trend"], candidates)
    carry_result = compare_book("carry", pnl_series["carry"], candidates)

    full = pd.concat([trend_result, carry_result], ignore_index=True)
    COMPARISON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(COMPARISON_CACHE_PATH, index=False)
    print(f"\nSaved comparison table to {COMPARISON_CACHE_PATH}")

    # Both estimators' full vol-forecast series, both Books, one flat-columned
    # cache - so the dashboard page never needs to rebuild the two Books (data
    # load + 2x Book.run(), the genuinely slow part) just to plot a curve.
    series = pd.DataFrame({
        "trend_ewma": candidates["ewma"]["trend"], "trend_garch": candidates["gjr_garch"]["trend"],
        "carry_ewma": candidates["ewma"]["carry"], "carry_garch": candidates["gjr_garch"]["carry"],
    })
    series.to_parquet(SERIES_CACHE_PATH)
    print(f"Saved vol series to {SERIES_CACHE_PATH}")


if __name__ == "__main__":
    main()
