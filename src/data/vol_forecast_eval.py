"""
data/vol_forecast_eval.py — Signal-agnostic volatility-forecast accuracy evaluation.

Built 2026-07-21 to fix the flagged problem logged in WORKFLOW.md's Phase 7 section
("vol estimator selection is currently done inconsistently..."): research/momentum.py
and research/breakout.py were each picking Yang-Zhang vs. EWMA independently per spec
by comparing pooled TRAIN Sharpe - a performance-based selection inconsistent with
CLAUDE.md Rule 1/2 (never pick a spec by looking at backtest results) and economically
incoherent (volatility is a property of the asset's price history, not of whichever
signal is consuming it).

This module has no dependency on any trading signal, position, or Sharpe ratio - it
only compares a volatility FORECAST against subsequently REALIZED variance, the
standard approach in the vol-forecasting literature (see references/ for the
project-wide decision this feeds: research/vol_estimator_comparison.py).

QLIKE is Patton (2011)'s standard quasi-likelihood loss for comparing volatility
forecasts against an imperfect realized-variance proxy: normalized so it is exactly 0
at a perfect forecast (forecast == realized) and strictly convex in the ratio
realized/forecast, i.e. it has a genuine, well-behaved minimum rather than being
monotonic in one direction - a loss that's monotonic would reward pushing the forecast
arbitrarily far in one direction, which isn't a meaningful "accuracy" measure. QLIKE
also penalizes underprediction (forecast too low during a high-vol regime) more
heavily than overprediction of the same relative size, appropriate here since this
vol estimate sizes real positions and underprediction is the direction that produces
outsized realized losses.
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252


def forward_realized_variance(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Ex-post, annualized realized variance over the `horizon` trading days AFTER
    each date - the ground-truth a volatility forecast made using information through
    that date is trying to predict.

    RV_t = (252 / horizon) * sum_{i=1}^{horizon} r_{t+i}^2

    This is intentionally forward-looking (t+1 ... t+horizon) - it is a REALIZED-
    VARIANCE PROXY for forecast evaluation only, never fed into a trading signal
    (CLAUDE.md Rule 3's shift(1) discipline governs signals that trade "today," not an
    ex-post accuracy check computed once, offline, in research/vol_estimator_comparison.py).

    Rows within `horizon` trading days of the end of `returns` don't have a full
    forward window and are NaN (min_periods=horizon - a partial trailing window would
    understate variance, not just be a smaller sample).
    """
    squared = returns.pow(2)
    # Reverse, roll forward (=backward in reversed order), reverse back, then shift by
    # -1 to exclude day t itself and start the window at t+1 - see module-level note
    # in WORKFLOW.md for the derivation.
    reversed_sum = squared.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    forward_sum = reversed_sum.shift(-1)
    return forward_sum * (ANNUALIZATION / horizon)


def qlike_loss(forecast_var: pd.DataFrame, realized_var: pd.DataFrame) -> pd.DataFrame:
    """Patton (2011) QLIKE loss, normalized to 0 at a perfect forecast:

    QLIKE = realized/forecast - log(realized/forecast) - 1

    Convex in the ratio realized/forecast with a unique minimum (0) at
    realized == forecast; the -log(.) term makes underprediction (ratio >> 1) far more
    costly than an equally-sized overprediction (ratio << 1), matching the asymmetry
    QLIKE is chosen for in the literature - a bare squared-error term would penalize
    both directions symmetrically. Non-positive forecast or realized values (shouldn't
    occur for a variance, but both estimators can be NaN during warmup) propagate NaN
    rather than raising.
    """
    ratio = realized_var / forecast_var
    return ratio - np.log(ratio) - 1


def mse_vol_loss(forecast_vol: pd.DataFrame, realized_vol: pd.DataFrame) -> pd.DataFrame:
    """Simple squared error on annualized VOLATILITY (not variance) - reported
    alongside QLIKE as a robustness cross-check, since it's symmetric and doesn't
    presume QLIKE's underprediction-averse weighting is the right lens for every
    reader. Vol (not variance) scale chosen because that's the unit both estimators
    are actually consumed in downstream (signals.transforms.vol_targeted_sign_signal
    divides by vol directly), so this loss is in the same units as the thing that
    actually sizes positions.
    """
    return (forecast_vol - realized_vol) ** 2


def per_asset_mean_loss(loss: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """Mean loss per asset (column), NaN-dropped per column first. Assets with fewer
    than `min_obs` valid loss observations return NaN rather than a noisy mean from a
    handful of points - same MIN_OBS discipline as backtest.performance.
    """
    def _mean(col):
        col = col.dropna()
        return col.mean() if len(col) >= min_obs else np.nan

    return loss.apply(_mean)
