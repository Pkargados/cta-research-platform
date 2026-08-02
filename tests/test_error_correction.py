import numpy as np
import pandas as pd

from signals.error_correction import (
    fit_ecm,
    rolling_half_life,
    train_half_life,
    median_rolling_half_life,
    half_life_to_window,
)


def _ou_process(n=1000, half_life=20.0, sigma=1.0, seed=0):
    """Simulate a genuine Ornstein-Uhlenbeck (mean-reverting) series with a
    KNOWN half-life, so fit_ecm's recovered half-life can be checked against
    ground truth, not just "looks plausible"."""
    rng = np.random.default_rng(seed)
    lam = -np.log(2) / half_life  # true adjustment speed implied by half_life
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + lam * x[t - 1] + rng.normal(0, sigma)
    return pd.Series(x, index=dates)


def _random_walk(n=1000, sigma=1.0, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.Series(np.cumsum(rng.normal(0, sigma, n)), index=dates)


def test_fit_ecm_recovers_known_half_life_reasonably():
    spread = _ou_process(n=1500, half_life=20.0)
    fit = fit_ecm(spread, lags=1)

    assert fit["lambda"] < 0
    assert not np.isnan(fit["half_life"])
    # Recovered half-life should be in the right ballpark of the true 20 (noisy estimate)
    assert 10 < fit["half_life"] < 40


def test_fit_ecm_on_a_random_walk_finds_no_mean_reversion():
    spread = _random_walk(n=1000)
    fit = fit_ecm(spread, lags=1)

    # A pure random walk has no error-correction term (lambda ~ 0, could be
    # slightly positive or negative by noise) - half-life should usually be
    # NaN (lambda >= 0) or, if negative by chance, implausibly large.
    if not np.isnan(fit["half_life"]):
        assert fit["half_life"] > 100


def test_fit_ecm_returns_nan_with_too_few_observations():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    spread = pd.Series(np.random.randn(10), index=dates)
    fit = fit_ecm(spread, lags=1)
    assert np.isnan(fit["half_life"])
    assert fit["n_obs"] < 30


def test_rolling_half_life_only_uses_trailing_data_no_lookahead():
    # Mean-reverting first half, random walk second half - a window entirely
    # within the first half must not "see" the second half's behavior.
    reverting = _ou_process(n=500, half_life=15.0, seed=2)
    walk = _random_walk(n=500, seed=3)
    walk.index = pd.date_range(reverting.index[-1] + pd.Timedelta(days=1), periods=500, freq="D")
    spread = pd.concat([reverting, walk])

    result = rolling_half_life(spread, window=252, step=21)
    early_result = result.loc[result.index <= reverting.index[-1]]

    assert len(early_result) > 0
    # Loose bound - this test checks NO LOOKAHEAD (early windows don't see the
    # second-half random walk), not precise recovery accuracy (see
    # test_fit_ecm_recovers_known_half_life_reasonably for that) - a window
    # entirely inside the reverting half should stay in a plausible range, not
    # blow up toward "no mean reversion" the way a window touching the random
    # walk half would.
    assert early_result["half_life"].dropna().between(2, 50).all()


def test_rolling_half_life_truncated_series_matches_full_series_early_estimates():
    spread = _ou_process(n=800, half_life=25.0, seed=4)
    full = rolling_half_life(spread, window=252, step=21)
    truncated = rolling_half_life(spread.iloc[:500], window=252, step=21)

    common_dates = full.index.intersection(truncated.index)
    assert len(common_dates) > 0
    pd.testing.assert_frame_equal(full.loc[common_dates], truncated.loc[common_dates])


def test_train_half_life_only_uses_data_through_train_end():
    spread = _ou_process(n=800, half_life=15.0, seed=5)
    train_end = spread.index[400]

    hl_full_train = train_half_life(spread, train_end)
    hl_truncated = train_half_life(spread.iloc[:401], train_end)

    assert hl_full_train == hl_truncated


def test_median_rolling_half_life_only_uses_windows_through_train_end():
    spread = _ou_process(n=800, half_life=15.0, seed=6)
    train_end = spread.index[500]

    full_median = median_rolling_half_life(spread, train_end, window=252, step=21)
    truncated_median = median_rolling_half_life(spread.iloc[:501], train_end, window=252, step=21)

    assert full_median == truncated_median
    assert 5 < full_median < 40


def test_median_rolling_half_life_empty_before_first_window():
    spread = _ou_process(n=100, half_life=15.0, seed=7)
    result = median_rolling_half_life(spread, spread.index[50], window=252, step=21)
    assert np.isnan(result)


def test_half_life_to_window_applies_multiplier_and_clips():
    assert half_life_to_window(10.0, multiplier=2.0) == 20
    assert half_life_to_window(5.0, multiplier=2.0, min_window=15) == 15  # clipped up
    assert half_life_to_window(200.0, multiplier=2.0, max_window=252) == 252  # 2*200=400, clipped down to 252


def test_half_life_to_window_undefined_half_life_uses_fallback_or_nan():
    assert np.isnan(half_life_to_window(np.nan))
    assert half_life_to_window(np.nan, fallback=63) == 63
    assert half_life_to_window(-5.0, fallback=63) == 63  # non-positive treated as undefined
    assert np.isnan(half_life_to_window(0.0))
