"""
data/panels.py — Load the core OHLCV + volatility panel.

Moved from signal_lib.py (2026-07-21) as part of the notebook -> src/ + research/
migration — see cleanup.md section 2. Same function, same Rice bad-print patch,
just relocated so it sits with the rest of the data-loading layer instead of bundled
into a flat module that also did signal construction and backtest mechanics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

RICE_BAD_PRINT_DATE = "2024-06-17"


def load_core_panel(data_dir=None):
    """Load OHLCV + Yang-Zhang volatility, with the known Rice bad-print patch applied.

    Returns a dict with keys: open, high, low, close, volume, volatility.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    open_prices = pd.read_parquet(data_dir / "open.parquet")
    high_prices = pd.read_parquet(data_dir / "high.parquet")
    low_prices = pd.read_parquet(data_dir / "low.parquet")
    close_prices = pd.read_parquet(data_dir / "close.parquet")
    volume = pd.read_parquet(data_dir / "volume.parquet")
    volatility = pd.read_parquet(data_dir / "yang_zhang_features.parquet")

    for df in (open_prices, high_prices, low_prices, close_prices):
        df.loc[RICE_BAD_PRINT_DATE, "Rice"] = np.nan

    return {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
        "volatility": volatility,
    }
