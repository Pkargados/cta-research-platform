import numpy as np
import pandas as pd

from data.panels import load_core_panel, RICE_BAD_PRINT_DATE


def _write_panel(tmp_path):
    # Must actually span RICE_BAD_PRINT_DATE (2024-06-17), otherwise .loc[...]=
    # assignment silently APPENDS a new row (NaN for every other column) instead
    # of patching an existing cell - a real trap for this specific test setup,
    # not a bug in the source module.
    dates = pd.date_range("2024-06-15", periods=5, freq="D")
    cols = ["Rice", "Corn"]

    def _df(value):
        return pd.DataFrame({c: value for c in cols}, index=dates)

    open_ = _df(100.0)
    high = _df(101.0)
    low = _df(99.0)
    close = _df(100.5)
    volume = _df(1000.0)
    volatility = _df(0.2)

    # Give Rice a real (non-NaN) print specifically on the known bad-print date.
    open_.loc[RICE_BAD_PRINT_DATE, "Rice"] = 999.0
    high.loc[RICE_BAD_PRINT_DATE, "Rice"] = 999.0
    low.loc[RICE_BAD_PRINT_DATE, "Rice"] = 999.0
    close.loc[RICE_BAD_PRINT_DATE, "Rice"] = 999.0
    volume.loc[RICE_BAD_PRINT_DATE, "Rice"] = 999.0

    open_.to_parquet(tmp_path / "open.parquet")
    high.to_parquet(tmp_path / "high.parquet")
    low.to_parquet(tmp_path / "low.parquet")
    close.to_parquet(tmp_path / "close.parquet")
    volume.to_parquet(tmp_path / "volume.parquet")
    volatility.to_parquet(tmp_path / "yang_zhang_features.parquet")
    return dates


def test_load_core_panel_returns_expected_keys(tmp_path):
    _write_panel(tmp_path)
    panel = load_core_panel(data_dir=tmp_path)
    assert set(panel.keys()) == {"open", "high", "low", "close", "volume", "volatility"}


def test_load_core_panel_patches_rice_bad_print_in_ohlc_only(tmp_path):
    _write_panel(tmp_path)
    panel = load_core_panel(data_dir=tmp_path)

    for field in ["open", "high", "low", "close"]:
        assert np.isnan(panel[field].loc[RICE_BAD_PRINT_DATE, "Rice"])

    # volume and volatility are NOT part of the patch loop - Rice's bad-print
    # value should still be there, untouched.
    assert panel["volume"].loc[RICE_BAD_PRINT_DATE, "Rice"] == 999.0


def test_load_core_panel_does_not_touch_other_assets_on_bad_print_date(tmp_path):
    _write_panel(tmp_path)
    panel = load_core_panel(data_dir=tmp_path)
    assert panel["close"].loc[RICE_BAD_PRINT_DATE, "Corn"] == 100.5


def test_load_core_panel_leaves_other_dates_untouched(tmp_path):
    dates = _write_panel(tmp_path)
    panel = load_core_panel(data_dir=tmp_path)
    other_date = [d for d in dates if str(d.date()) != RICE_BAD_PRINT_DATE][0]
    assert panel["close"].loc[other_date, "Rice"] == 100.5
