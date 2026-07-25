import numpy as np
import pandas as pd

from data.trusted_since import restrict_to_trusted_era


def test_restrict_to_trusted_era_masks_pre_cutoff_per_column():
    dates = pd.date_range("2010-01-01", periods=10, freq="YS")
    df = pd.DataFrame({"A": range(10), "B": range(10, 20)}, index=dates)
    trusted_since = pd.Series({"A": pd.Timestamp("2013-01-01"), "B": pd.Timestamp("2015-01-01")})

    result = restrict_to_trusted_era(df, trusted_since)

    assert result.loc[dates < "2013-01-01", "A"].isna().all()
    assert result.loc[dates >= "2013-01-01", "A"].notna().all()
    assert result.loc[dates < "2015-01-01", "B"].isna().all()
    assert result.loc[dates >= "2015-01-01", "B"].notna().all()


def test_restrict_to_trusted_era_preserves_full_date_index():
    dates = pd.date_range("2010-01-01", periods=5, freq="YS")
    df = pd.DataFrame({"A": range(5)}, index=dates)
    trusted_since = pd.Series({"A": pd.Timestamp("2012-01-01")})

    result = restrict_to_trusted_era(df, trusted_since)
    assert result.index.equals(df.index)  # no rows dropped, only cells masked


def test_restrict_to_trusted_era_leaves_unmentioned_columns_untouched():
    dates = pd.date_range("2010-01-01", periods=5, freq="YS")
    df = pd.DataFrame({"A": range(5), "Untracked": range(5, 10)}, index=dates)
    trusted_since = pd.Series({"A": pd.Timestamp("2012-01-01")})

    result = restrict_to_trusted_era(df, trusted_since)
    assert result["Untracked"].notna().all()
    pd.testing.assert_series_equal(result["Untracked"], df["Untracked"])


def test_restrict_to_trusted_era_does_not_mutate_original():
    dates = pd.date_range("2010-01-01", periods=5, freq="YS")
    df = pd.DataFrame({"A": range(5)}, index=dates)
    original = df.copy()
    trusted_since = pd.Series({"A": pd.Timestamp("2012-01-01")})

    restrict_to_trusted_era(df, trusted_since)
    pd.testing.assert_frame_equal(df, original)
