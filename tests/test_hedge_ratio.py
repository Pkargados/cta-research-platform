import numpy as np
import pandas as pd

from signals.hedge_ratio import fixed_ratio, rolling_ols_beta, kalman_hedge_ratio


def test_fixed_ratio_is_constant():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    beta = fixed_ratio(idx, beta=1.0)
    assert (beta == 1.0).all()
    assert len(beta) == 10


def test_fixed_ratio_respects_custom_value():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    beta = fixed_ratio(idx, beta=0.5)
    assert (beta == 0.5).all()


def test_rolling_ols_beta_recovers_a_known_linear_relationship():
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    true_beta = 1.7
    y = true_beta * x + pd.Series(rng.normal(0, 0.01, n), index=dates)

    beta_hat = rolling_ols_beta(y, x, window=100)

    assert np.isnan(beta_hat.iloc[:69]).all()  # min_periods = 0.7*100 = 70, not enough trailing history yet
    assert np.isclose(beta_hat.iloc[-1], true_beta, atol=0.1)


def test_rolling_ols_beta_only_uses_trailing_window_no_lookahead():
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    y = 1.0 * x + pd.Series(rng.normal(0, 0.01, n), index=dates)

    beta_full = rolling_ols_beta(y, x, window=100)
    # Truncating the series after date 200 must not change beta estimated
    # ON OR BEFORE date 200 - a genuine lookahead would let a shorter series
    # change an earlier estimate through some global fit.
    beta_truncated = rolling_ols_beta(y.iloc[:250], x.iloc[:250], window=100)

    pd.testing.assert_series_equal(beta_full.iloc[:250], beta_truncated, check_names=False)


def test_kalman_hedge_ratio_tracks_a_slowly_drifting_beta():
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    # beta drifts linearly from 1.0 to 2.0 over the sample
    true_beta = np.linspace(1.0, 2.0, n)
    y = pd.Series(true_beta * x.to_numpy() + rng.normal(0, 0.01, n), index=dates)

    result = kalman_hedge_ratio(y, x, delta=1e-3, ve=1e-3)

    assert list(result.columns) == ["alpha", "beta"]
    assert len(result) == n
    # Late-sample beta should be closer to 2.0 than early-sample beta was to 1.0-correctness,
    # i.e. the filter tracked the drift rather than staying pinned near its start value.
    # First point sampled just after DEFAULT_KALMAN_WARMUP (index 10 falls inside the
    # masked warmup region and is NaN by construction - not a valid comparison point).
    assert result["beta"].iloc[-1] > result["beta"].iloc[n // 2] > result["beta"].iloc[70]


def test_kalman_hedge_ratio_only_uses_past_observations():
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    y = 1.5 * x + pd.Series(rng.normal(0, 0.01, n), index=dates)

    result_full = kalman_hedge_ratio(y, x)
    result_truncated = kalman_hedge_ratio(y.iloc[:150], x.iloc[:150])

    pd.testing.assert_frame_equal(result_full.iloc[:150], result_truncated)
