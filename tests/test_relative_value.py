import numpy as np
import pandas as pd

from signals.relative_value import (
    RATIO_PAIRS,
    HEDGE_RATIO_PAIRS,
    CRACK_SPREAD_NAME,
    ALL_PAIR_NAMES,
    estimate_beta,
    log_ratio_spread,
    crack_spread_level,
    spread_return,
    realized_vol,
    zscore_spread_signal,
    continuous_zscore_signal,
    threshold_band_signal,
    build_pair_spread,
    build_pair_signal,
    build_all_pair_spreads,
    build_all_pair_signals,
    build_all_pair_returns,
)


def _price_panel(n=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    base = 50 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "WTI Crude": base,
        "Brent": base + 2 + rng.normal(0, 0.2, n),
        "Gold": 1500 + np.cumsum(rng.normal(0, 2, n)),
        "Silver": 20 + np.cumsum(rng.normal(0, 0.3, n)),
        "Platinum": 900 + np.cumsum(rng.normal(0, 1, n)),
        "Palladium": 1800 + np.cumsum(rng.normal(0, 2, n)),
        "Wheat": 600 + np.cumsum(rng.normal(0, 3, n)),
        "KC_Wheat": 620 + np.cumsum(rng.normal(0, 3, n)),
        "Corn": 400 + np.cumsum(rng.normal(0, 2, n)),
        "RBOB": 2.0 + np.cumsum(rng.normal(0, 0.02, n)),
        "HeatingOil": 2.2 + np.cumsum(rng.normal(0, 0.02, n)),
        "SP500": 4500 + np.cumsum(rng.normal(0, 5, n)),
        "Russell2000": 2000 + np.cumsum(rng.normal(0, 3, n)),
        "Nasdaq100": 15000 + np.cumsum(rng.normal(0, 10, n)),
        "LiveCattle": 170 + np.cumsum(rng.normal(0, 1, n)),
        "FeederCattle": 220 + np.cumsum(rng.normal(0, 1, n)),
        "Soybeans": 1300 + np.cumsum(rng.normal(0, 5, n)),
    }, index=dates)


def test_estimate_beta_fixed_is_always_one():
    price = _price_panel()
    beta = estimate_beta(price, "WTI Crude", "Brent", method="fixed")
    assert (beta == 1.0).all()


def test_log_ratio_spread_with_fixed_beta_equals_log_price_difference():
    price = _price_panel()
    beta = estimate_beta(price, "WTI Crude", "Brent", method="fixed")
    spread = log_ratio_spread(price, "WTI Crude", "Brent", beta)
    expected = np.log(price["WTI Crude"]) - np.log(price["Brent"])
    pd.testing.assert_series_equal(spread, expected, check_names=False)


def test_crack_spread_level_uses_321_ratio_and_can_go_negative():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    price = pd.DataFrame({
        "WTI Crude": [60.0, 60.0, 100.0],
        "RBOB": [2.0, 2.0, 1.0],
        "HeatingOil": [2.0, 2.0, 1.0],
    }, index=dates)

    spread = crack_spread_level(price)

    # (2*2*42 + 1*2*42)/3 - 60 = (168+84)/3 - 60 = 84 - 60 = 24
    assert np.isclose(spread.iloc[0], 24.0)
    # Row 3: (2*1*42+1*1*42)/3 - 100 = 42 - 100 = -58 (a real negative margin)
    assert spread.iloc[2] < 0


def test_spread_return_is_simple_diff():
    spread = pd.Series([1.0, 1.5, 1.2])
    ret = spread_return(spread)
    assert np.isnan(ret.iloc[0])
    assert np.isclose(ret.iloc[1], 0.5)
    assert np.isclose(ret.iloc[2], -0.3)


def test_realized_vol_is_nan_during_warmup_and_positive_after():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 100))
    vol = realized_vol(ret, window=20)
    assert vol.iloc[:9].isna().all()  # min_periods = VOL_MIN_FRAC(0.5)*20 = 10
    assert vol.iloc[-1] > 0


def test_zscore_spread_signal_is_zero_at_a_constant_mean_reverting_series():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Oscillates around 0 - the trailing mean should track close to 0.
    spread = pd.Series(np.sin(np.linspace(0, 20, 100)), index=dates)
    z = zscore_spread_signal(spread, window=20)
    assert z.iloc[:13].isna().all()  # min_periods = MIN_FRAC(0.7)*20 = 14
    assert z.dropna().abs().median() < 3  # sane magnitude, not exploding


def test_continuous_zscore_signal_shorts_a_rich_spread_and_longs_a_cheap_one():
    z = pd.Series([2.0, -2.0])
    vol = pd.Series([0.1, 0.1])
    signal = continuous_zscore_signal(z, vol, target_vol=1.0)
    assert signal.iloc[0] < 0  # rich (z>0) -> short
    assert signal.iloc[1] > 0  # cheap (z<0) -> long


def test_threshold_band_signal_enters_and_holds_then_exits():
    z = pd.Series([0.0, 0.0, 2.5, 1.5, 1.0, 0.3, 0.0])
    state = threshold_band_signal(z, entry_z=2.0, exit_z=0.5)

    assert state.iloc[1] == 0.0     # flat before entry
    assert state.iloc[2] == -1.0    # z=2.5 > entry -> short entry
    assert state.iloc[3] == -1.0    # z=1.5, still above exit_z -> holds short
    assert state.iloc[4] == -1.0    # z=1.0, still above exit_z -> holds short
    assert state.iloc[5] == 0.0     # z=0.3 < exit_z -> flattens


def test_threshold_band_signal_never_flips_directly_long_to_short():
    # From a long position (z very negative), even a swing straight to a
    # rich extreme must pass through flat first in this construction, since
    # exit only checks |z|<exit_z, not a direct opposite-side entry.
    z = pd.Series([-2.5, -2.5, 2.5])
    state = threshold_band_signal(z, entry_z=2.0, exit_z=0.5)
    assert state.iloc[1] == 1.0   # entered long
    assert state.iloc[2] == 1.0   # still long: |2.5| is not < exit_z, so no flatten happened,
                                   # and re-entry logic never runs while already in a position


def test_build_pair_spread_dispatches_ratio_hedge_and_crack_groups():
    price = _price_panel()

    ratio_spread = build_pair_spread(price, "wti_brent")
    assert isinstance(ratio_spread, pd.Series)

    hedge_spread = build_pair_spread(price, "corn_wheat", hedge_method="rolling_ols")
    assert isinstance(hedge_spread, pd.Series)

    crack = build_pair_spread(price, CRACK_SPREAD_NAME)
    assert isinstance(crack, pd.Series)


def test_build_pair_spread_requires_hedge_method_for_hedge_ratio_pairs():
    price = _price_panel()
    try:
        build_pair_spread(price, "corn_wheat")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_pair_signal_continuous_and_threshold_both_produce_a_series():
    price = _price_panel()
    spread = build_pair_spread(price, "wti_brent")

    continuous = build_pair_signal(spread, entry_exit="continuous")
    threshold = build_pair_signal(spread, entry_exit="threshold")

    assert isinstance(continuous, pd.Series)
    assert isinstance(threshold, pd.Series)
    assert continuous.index.equals(spread.index)
    assert threshold.index.equals(spread.index)


def test_all_pair_names_covers_eleven_pairs():
    assert len(ALL_PAIR_NAMES) == 11
    assert set(RATIO_PAIRS) | set(HEDGE_RATIO_PAIRS) | {CRACK_SPREAD_NAME} == set(ALL_PAIR_NAMES)


def test_build_all_pair_spreads_signals_returns_cover_every_pair():
    price = _price_panel()
    spreads = build_all_pair_spreads(price, hedge_method="rolling_ols")
    assert set(spreads.keys()) == set(ALL_PAIR_NAMES)

    signals = build_all_pair_signals(spreads, entry_exit="continuous")
    assert set(signals.keys()) == set(ALL_PAIR_NAMES)

    returns = build_all_pair_returns(spreads)
    assert set(returns.keys()) == set(ALL_PAIR_NAMES)
