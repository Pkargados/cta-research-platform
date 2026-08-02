import numpy as np
import pandas as pd

from signals.cointegration import rolling_engle_granger, fraction_cointegrated, rolling_cointegration_report


def _cointegrated_pair(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    common_walk = np.cumsum(rng.normal(0, 1, n))
    y = pd.Series(common_walk + rng.normal(0, 0.1, n), index=dates)
    x = pd.Series(common_walk + rng.normal(0, 0.1, n), index=dates)
    return y, x


def _independent_pair(n=600, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
    return y, x


def test_rolling_engle_granger_finds_cointegration_in_a_genuinely_cointegrated_pair():
    y, x = _cointegrated_pair()
    result = rolling_engle_granger(y, x, window=252, step=21)

    assert len(result) > 0
    assert fraction_cointegrated(result, alpha=0.05) > 0.5


def test_rolling_engle_granger_mostly_fails_to_reject_for_independent_walks():
    y, x = _independent_pair()
    result = rolling_engle_granger(y, x, window=252, step=21)

    assert len(result) > 0
    assert fraction_cointegrated(result, alpha=0.05) < 0.5


def test_rolling_engle_granger_only_uses_trailing_data_no_lookahead():
    # A pair that is cointegrated in the first half and completely decoupled
    # (independent random walks) in the second half - a test window entirely
    # within the first half must not "see" the second half's decoupling.
    y1, x1 = _cointegrated_pair(n=300, seed=2)
    y2, x2 = _independent_pair(n=300, seed=3)
    y2.index = pd.date_range(y1.index[-1] + pd.Timedelta(days=1), periods=300, freq="D")
    x2.index = y2.index

    y = pd.concat([y1, y2])
    x = pd.concat([x1, x2])

    result = rolling_engle_granger(y, x, window=252, step=21)
    early_window = result.loc[result.index <= y1.index[-1]]

    assert len(early_window) > 0
    assert fraction_cointegrated(early_window, alpha=0.05) > 0.5


def test_fraction_cointegrated_empty_result_is_nan():
    empty = pd.DataFrame(columns=["stat", "pvalue"])
    assert np.isnan(fraction_cointegrated(empty))


def test_rolling_cointegration_report_covers_every_pair_and_handles_missing_legs():
    y, x = _cointegrated_pair(n=400)
    price = pd.DataFrame({"A": y, "B": x})
    pairs = {"ab": ("A", "B"), "missing": ("A", "C")}

    report = rolling_cointegration_report(pairs, price, window=252, step=21)

    assert set(report.index) == {"ab", "missing"}
    assert report.loc["ab", "n_windows"] > 0
    assert report.loc["missing", "n_windows"] == 0
    assert np.isnan(report.loc["missing", "fraction_cointegrated"])
