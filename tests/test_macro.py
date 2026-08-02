import numpy as np
import pandas as pd

import data.macro as macro


def _write_cpi_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "cpi_level_index.csv"
    df.to_csv(path, index=False)
    return path


def _write_yield_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "Yield_Curve_6M_to_30Y.csv"
    df.to_csv(path, index=False)
    return path


def test_load_cpi_aud_bounded_ffill_exactly_two_months(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    # AUD only prints quarterly: real values at months 0 and 3, NaN elsewhere.
    aud = [100.0, np.nan, np.nan, 103.0, np.nan, np.nan]
    _write_cpi_csv(tmp_path, {"Date": dates, "US": [200.0] * 6, "AUD": aud})

    cpi = macro.load_cpi()

    # Months 1-2 (the two in-between months after the real Jan print) should be
    # bounded-ffilled to 100.0 - exactly QUARTERLY_CPI_COUNTRIES["AUD"] = 2 months.
    assert cpi["AUD"].iloc[1] == 100.0
    assert cpi["AUD"].iloc[2] == 100.0
    assert cpi["AUD"].iloc[3] == 103.0  # the next real print


def test_load_cpi_aud_goes_nan_beyond_the_ffill_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    # AUD stops publishing entirely after month 0 - more than 2 months of gap.
    aud = [100.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    _write_cpi_csv(tmp_path, {"Date": dates, "US": [200.0] * 6, "AUD": aud})

    cpi = macro.load_cpi()

    assert cpi["AUD"].iloc[2] == 100.0  # still within the 2-month bound
    assert np.isnan(cpi["AUD"].iloc[3])  # 3rd missing month - beyond the bound


def test_load_cpi_non_quarterly_countries_are_never_forward_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    # EUR (not in QUARTERLY_CPI_COUNTRIES) has a real, unexplained gap - must
    # stay NaN, not silently bridged the way AUD's genuine quarterly cadence is.
    eur = [100.0, np.nan, np.nan, np.nan, 104.0]
    _write_cpi_csv(tmp_path, {"Date": dates, "US": [200.0] * 5, "EUR": eur})

    cpi = macro.load_cpi()
    assert cpi["EUR"].iloc[1:4].isna().all()


def test_load_cpi_sorted_and_indexed_by_date(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    shuffled = {"Date": [dates[2], dates[0], dates[1]], "US": [3.0, 1.0, 2.0]}
    _write_cpi_csv(tmp_path, shuffled)

    cpi = macro.load_cpi()
    assert cpi.index.is_monotonic_increasing
    assert cpi["US"].tolist() == [1.0, 2.0, 3.0]


def test_load_yield_curve_indexed_by_date_and_sorted(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    shuffled = {"Date": [dates[1], dates[0], dates[2]], "2Y": [2.0, 1.0, 3.0], "10Y": [2.5, 1.5, 3.5]}
    _write_yield_csv(tmp_path, shuffled)

    yc = macro.load_yield_curve()
    assert yc.index.is_monotonic_increasing
    assert yc["2Y"].tolist() == [1.0, 2.0, 3.0]
    assert list(yc.index) == list(dates)


def test_quarterly_cpi_countries_is_aud_only_with_limit_two():
    assert macro.QUARTERLY_CPI_COUNTRIES == {"AUD": 2}


def _write_overnight_rates_xlsx(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "overnight_fed_fund_rates_US.xlsx"
    df.to_excel(path, index=False)
    return path


def test_load_overnight_rate_filters_by_rate_type_and_sorts(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    dates = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-01"])
    rows = {
        "Effective Date": list(dates) + [pd.Timestamp("2020-01-01")],
        "Rate Type": ["SOFR", "SOFR", "SOFR", "EFFR"],
        "Rate (%)": [1.60, 1.55, 1.50, 1.58],
    }
    _write_overnight_rates_xlsx(tmp_path, rows)

    sofr = macro.load_overnight_rate("SOFR")
    assert sofr.index.is_monotonic_increasing
    assert sofr.tolist() == [1.50, 1.55, 1.60]
    assert sofr.name == "SOFR"

    effr = macro.load_overnight_rate("EFFR")
    assert effr.tolist() == [1.58]


def test_load_overnight_rate_keeps_last_on_duplicate_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "DATA_DIR", tmp_path)
    rows = {
        "Effective Date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        "Rate Type": ["SOFR", "SOFR"],
        "Rate (%)": [1.50, 1.55],  # a same-day revision - keep the later row
    }
    _write_overnight_rates_xlsx(tmp_path, rows)

    sofr = macro.load_overnight_rate("SOFR")
    assert len(sofr) == 1
    assert sofr.iloc[0] == 1.55


def test_overnight_rate_types_includes_sofr_and_effr():
    assert set(macro.OVERNIGHT_RATE_TYPES) >= {"SOFR", "EFFR"}
