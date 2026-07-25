"""
Scheduled recompute of Yang-Zhang volatility from the current OHLC parquet files.

This is a pure function of Data/open,high,low,close.parquet - not an independent
data pull - so it must run after CTA_DailyDataUpdate each day, not on its own
schedule, or it will compute volatility against yesterday's prices.

Logic mirrors notebooks/volatility.ipynb exactly (Yang & Zhang, 2000). NOTE: this
is a deliberately separate implementation from src/data/volatility.py's
yang_zhang_volatility (which adds min_frac/overnight_min_frac NaN-tolerance and a
roll_mask option for the continuous-curve panel research/dashboard code consumes,
per that module's own docstring) — this job runs against the older, plain core
OHLCV panel (Data/open,high,low,close.parquet), where CLAUDE.md's own investigation
of this panel's NaN gaps attributed them to the panel's real data noise, not a
missing-tolerance bug in this calculation. Not merged into one shared function
without first checking whether that same tolerance is actually appropriate for
this panel too — flagged, not assumed.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
MANIFEST_PATH = DATA_DIR / "volatility_manifest.csv"
HORIZONS = [21, 63, 126, 252]


def yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, window):
    def safe_log(df):
        return np.log(df.where(df > 0))

    log_o = safe_log(open_prices)
    log_h = safe_log(high_prices)
    log_l = safe_log(low_prices)
    log_c = safe_log(close_prices)

    r_o = log_o - log_c.shift(1)
    r_c = log_c - log_o

    rs = (
        (log_h - log_c) * (log_h - log_o)
        + (log_l - log_c) * (log_l - log_o)
    )

    sigma_o2 = r_o.rolling(window).var()
    sigma_c2 = r_c.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs

    return np.sqrt(yz_var)


def main() -> int:
    run_date = datetime.now().isoformat(timespec="seconds")
    try:
        open_prices = pd.read_parquet(DATA_DIR / "open.parquet")
        high_prices = pd.read_parquet(DATA_DIR / "high.parquet")
        low_prices = pd.read_parquet(DATA_DIR / "low.parquet")
        close_prices = pd.read_parquet(DATA_DIR / "close.parquet")

        yz_features = pd.concat(
            {
                f"yz_vol_{w}": yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, w)
                for w in HORIZONS
            },
            axis=1,
        )

        yz_features.to_parquet(DATA_DIR / "yang_zhang_features.parquet")
        status, detail = "OK", f"{yz_features.shape[0]} rows, horizons {HORIZONS}"
        print(f"Yang-Zhang volatility updated: {detail}.")
    except Exception as exc:
        status, detail = "FAILED", str(exc)
        print(f"FAIL volatility update: {exc}")

    manifest_row = pd.DataFrame([{"run_date": run_date, "status": status, "detail": detail}])
    if MANIFEST_PATH.exists():
        manifest_row = pd.concat([pd.read_csv(MANIFEST_PATH), manifest_row], ignore_index=True)
    manifest_row.to_csv(MANIFEST_PATH, index=False)

    return 1 if status == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
