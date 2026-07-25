import pandas as pd

from backtest.splits import train_validation_test_split


def _series():
    dates = pd.date_range("2010-01-01", "2026-01-01", freq="D")
    return pd.Series(range(len(dates)), index=dates)


def test_three_slices_are_contiguous_and_non_overlapping():
    s = _series()
    train, validation, test = train_validation_test_split(s)

    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()


def test_boundary_dates_excluded_from_the_following_slice():
    s = _series()
    train, validation, test = train_validation_test_split(s)

    assert pd.Timestamp("2019-12-31") in train.index
    assert pd.Timestamp("2019-12-31") not in validation.index
    assert pd.Timestamp("2021-12-31") in validation.index
    assert pd.Timestamp("2021-12-31") not in test.index


def test_test_period_starts_2022_01_01():
    s = _series()
    _, _, test = train_validation_test_split(s)

    assert test.index.min() == pd.Timestamp("2022-01-01")


def test_covers_full_index_with_no_gaps():
    s = _series()
    train, validation, test = train_validation_test_split(s)

    recombined = pd.concat([train, validation, test]).sort_index()
    assert recombined.index.equals(s.index)
