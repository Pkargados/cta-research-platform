import numpy as np
import pandas as pd

from signals.combine import combine_alphas, ic_weighted_combine, risk_parity_combine, confirmation_filter_combine


def _two_alphas():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    a = pd.DataFrame({"X": [1.0] * 10, "Y": [2.0] * 10}, index=dates)
    b = pd.DataFrame({"X": [3.0] * 10, "Y": [4.0] * 10}, index=dates)
    return a, b


def test_combine_alphas_single_input_returns_copy():
    a, _ = _two_alphas()
    result = combine_alphas([a])
    pd.testing.assert_frame_equal(result, a)
    assert result is not a  # a copy, not the same object


def test_combine_alphas_empty_raises():
    try:
        combine_alphas([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_combine_alphas_equal_method_averages():
    a, b = _two_alphas()
    result = combine_alphas([a, b], method="equal")
    expected = (a + b) / 2
    pd.testing.assert_frame_equal(result, expected)


def test_combine_alphas_fixed_method_weights_and_sums():
    a, b = _two_alphas()
    result = combine_alphas([a, b], weights=[0.25, 0.75], method="fixed")
    expected = 0.25 * a + 0.75 * b
    pd.testing.assert_frame_equal(result, expected)


def test_combine_alphas_fixed_requires_matching_weights():
    a, b = _two_alphas()
    try:
        combine_alphas([a, b], weights=[1.0], method="fixed")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_combine_alphas_rank_method_averages_cross_sectional_percentile_rank():
    a, b = _two_alphas()
    result = combine_alphas([a, b], method="rank")
    expected = (a.rank(axis=1, pct=True) + b.rank(axis=1, pct=True)) / 2
    pd.testing.assert_frame_equal(result, expected)


def test_combine_alphas_unknown_method_raises():
    a, b = _two_alphas()
    try:
        combine_alphas([a, b], method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ic_weighted_combine_favors_higher_ic_signal():
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    forward_returns = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=dates, columns=["A", "B", "C", "D"])

    # good_signal is exactly the forward return (perfect predictor, IC=1);
    # bad_signal is unrelated noise.
    good_signal = forward_returns.copy()
    bad_signal = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=dates, columns=["A", "B", "C", "D"])

    combined = ic_weighted_combine([good_signal, bad_signal], forward_returns, lookback=20)
    # After the lookback warmup, the combined signal should correlate with
    # good_signal much more than with bad_signal on average (good_signal gets
    # more weight since its trailing IC is higher).
    late = combined.iloc[30:]
    corr_good = late.corrwith(good_signal.iloc[30:], axis=1).mean()
    corr_bad = late.corrwith(bad_signal.iloc[30:], axis=1).mean()
    assert corr_good > corr_bad


def test_ic_weighted_combine_falls_back_to_equal_weight_during_warmup():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    n = len(dates)
    a = pd.DataFrame(1.0, index=dates, columns=["X", "Y"])
    b = pd.DataFrame(3.0, index=dates, columns=["X", "Y"])
    forward_returns = pd.DataFrame(0.0, index=dates, columns=["X", "Y"])

    # lookback=20 with only 10 dates -> min_periods=20 never satisfied -> every
    # date falls back to equal weighting (0.5/0.5).
    combined = ic_weighted_combine([a, b], forward_returns, lookback=20)
    expected = (a + b) / 2
    pd.testing.assert_frame_equal(combined, expected)


def test_risk_parity_combine_favors_lower_vol_signal():
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    a, b = _two_alphas()
    a = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=dates, columns=["X", "Y"])
    b = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=dates, columns=["X", "Y"])

    low_vol_returns = pd.Series(rng.normal(0, 0.001, n), index=dates)
    high_vol_returns = pd.Series(rng.normal(0, 1.0, n), index=dates)

    combined = risk_parity_combine([a, b], [low_vol_returns, high_vol_returns], lookback=20)
    late = combined.iloc[100:]
    # Low-vol signal `a` should dominate the blend after warmup (inverse-vol
    # weighting gives it far more weight than the high-vol signal `b`).
    corr_a = late.corrwith(a.iloc[100:], axis=1).mean()
    corr_b = late.corrwith(b.iloc[100:], axis=1).mean()
    assert corr_a > corr_b


def test_risk_parity_combine_falls_back_to_equal_weight_during_warmup():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    a = pd.DataFrame(1.0, index=dates, columns=["X", "Y"])
    b = pd.DataFrame(3.0, index=dates, columns=["X", "Y"])
    returns_a = pd.Series(0.01, index=dates)
    returns_b = pd.Series(0.02, index=dates)

    # lookback=20 with only 10 dates -> min_periods=20 never satisfied -> every
    # date falls back to equal weighting (0.5/0.5).
    combined = risk_parity_combine([a, b], [returns_a, returns_b], lookback=20)
    expected = (a + b) / 2
    pd.testing.assert_frame_equal(combined, expected)


def test_confirmation_filter_full_size_when_signs_agree():
    dates = pd.date_range("2020-01-01", periods=1)
    primary = pd.DataFrame({"A": [2.0], "B": [-3.0]}, index=dates)
    confirm = pd.DataFrame({"A": [1.0], "B": [-0.5]}, index=dates)  # same signs
    result = confirmation_filter_combine(primary, confirm)
    pd.testing.assert_frame_equal(result, primary)


def test_confirmation_filter_flat_when_signs_disagree():
    dates = pd.date_range("2020-01-01", periods=1)
    primary = pd.DataFrame({"A": [2.0], "B": [-3.0]}, index=dates)
    confirm = pd.DataFrame({"A": [-1.0], "B": [0.5]}, index=dates)  # opposite signs
    result = confirmation_filter_combine(primary, confirm)
    assert (result == 0.0).all().all()


def test_confirmation_filter_custom_disagree_scale_downsizes_not_flattens():
    dates = pd.date_range("2020-01-01", periods=1)
    primary = pd.DataFrame({"A": [2.0]}, index=dates)
    confirm = pd.DataFrame({"A": [-1.0]}, index=dates)
    result = confirmation_filter_combine(primary, confirm, disagree_scale=0.5)
    assert result["A"].iloc[0] == 1.0


def test_confirmation_filter_preserves_nan_on_missing_primary():
    dates = pd.date_range("2020-01-01", periods=1)
    primary = pd.DataFrame({"A": [np.nan]}, index=dates)
    confirm = pd.DataFrame({"A": [1.0]}, index=dates)
    result = confirmation_filter_combine(primary, confirm)
    assert pd.isna(result["A"].iloc[0])
