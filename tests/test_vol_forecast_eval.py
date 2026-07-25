import numpy as np
import pandas as pd

from data.vol_forecast_eval import (
    forward_realized_variance, qlike_loss, mse_vol_loss, per_asset_mean_loss, ANNUALIZATION,
)


def test_forward_realized_variance_matches_manual_calc():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.DataFrame({"A": [0.01] * 10}, index=dates)
    horizon = 3

    result = forward_realized_variance(returns, horizon)
    # RV_t = (252/horizon) * sum_{i=1..horizon} r_{t+i}^2 - at t=0, uses r[1],r[2],r[3]
    # all = 0.01, so RV = (252/3) * 3 * 0.01^2 = 252 * 0.0001 = 0.0252
    expected = (ANNUALIZATION / horizon) * horizon * (0.01 ** 2)
    assert np.isclose(result["A"].iloc[0], expected)


def test_forward_realized_variance_excludes_todays_own_return():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.DataFrame({"A": [0.0] * 10}, index=dates)
    returns.loc[dates[0], "A"] = 100.0  # a huge same-day return, should NOT appear in RV_0
    horizon = 3

    result = forward_realized_variance(returns, horizon)
    assert np.isclose(result["A"].iloc[0], 0.0)  # r[1],r[2],r[3] are all 0


def test_forward_realized_variance_tail_is_nan_without_full_forward_window():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.DataFrame({"A": [0.01] * 10}, index=dates)
    horizon = 3

    result = forward_realized_variance(returns, horizon)
    # The last `horizon` rows can't see a full forward window.
    assert result["A"].iloc[-horizon:].isna().all()
    assert result["A"].iloc[:-horizon].notna().all()


def test_qlike_loss_is_zero_at_perfect_forecast():
    forecast = pd.DataFrame({"A": [0.01, 0.02, 0.05]})
    realized = forecast.copy()
    loss = qlike_loss(forecast, realized)
    assert np.allclose(loss, 0.0, atol=1e-12)


def test_qlike_loss_penalizes_underprediction_more_than_overprediction():
    realized = pd.DataFrame({"A": [0.04]})
    under = pd.DataFrame({"A": [0.02]})  # forecast half of realized
    over = pd.DataFrame({"A": [0.08]})   # forecast double realized (same ratio magnitude, opposite direction)

    loss_under = qlike_loss(under, realized)
    loss_over = qlike_loss(over, realized)
    assert loss_under["A"].iloc[0] > loss_over["A"].iloc[0]


def test_qlike_loss_is_convex_with_unique_minimum():
    realized = pd.DataFrame({"A": [0.04] * 5})
    forecasts = pd.DataFrame({"A": [0.01, 0.02, 0.04, 0.08, 0.16]})
    losses = qlike_loss(forecasts, realized)["A"]
    min_idx = losses.idxmin()
    assert forecasts["A"].iloc[min_idx] == 0.04  # minimum exactly at the perfect forecast


def test_mse_vol_loss_matches_manual_calc():
    forecast = pd.DataFrame({"A": [0.10, 0.20]})
    realized = pd.DataFrame({"A": [0.12, 0.15]})
    result = mse_vol_loss(forecast, realized)
    expected = (forecast - realized) ** 2
    pd.testing.assert_frame_equal(result, expected)


def test_mse_vol_loss_is_symmetric():
    realized = pd.DataFrame({"A": [0.10]})
    over = pd.DataFrame({"A": [0.15]})
    under = pd.DataFrame({"A": [0.05]})
    assert np.isclose(mse_vol_loss(over, realized)["A"].iloc[0], mse_vol_loss(under, realized)["A"].iloc[0])


def test_per_asset_mean_loss_drops_nan_before_averaging():
    loss = pd.DataFrame({"A": [0.1, np.nan, 0.3] + [0.2] * 20})
    result = per_asset_mean_loss(loss, min_obs=5)
    expected = loss["A"].dropna().mean()
    assert np.isclose(result["A"], expected)


def test_per_asset_mean_loss_below_min_obs_is_nan():
    loss = pd.DataFrame({"A": [0.1, 0.2, 0.3]})  # only 3 valid obs
    result = per_asset_mean_loss(loss, min_obs=20)
    assert np.isnan(result["A"])
