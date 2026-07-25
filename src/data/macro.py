"""
data/macro.py — Raw loaders for the yield curve and CPI panels, needed as real
signal inputs (not just dashboard display data) for the first time by
signals/value.py.

Same separation-of-concerns convention as data/term_structure.py and every other
data/ module: this module loads/shapes raw panels, signals/ modules take
already-built panels as arguments rather than loading their own data (see
signals/carry.py's `carry_panel` argument for the precedent).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"


def load_yield_curve() -> pd.DataFrame:
    """US Treasury par yields, monthly, `Date, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`."""
    df = pd.read_csv(DATA_DIR / "Yield_Curve_6M_to_30Y.csv", parse_dates=["Date"])
    return df.set_index("Date").sort_index()


QUARTERLY_CPI_COUNTRIES = {"AUD": 2}  # {column: ffill limit in months} - see docstring


def load_cpi() -> pd.DataFrame:
    """CPI level index, 8 countries (`US, EUR, JPY, GBP, AUD, CAD, CHF, MXN`).

    Every country except AUD is genuinely published monthly - a NaN there means
    the source is stale/lagged (real, left unfilled - GBP/CAD/CHF/MXN lag
    1-2 years via their FRED source, see DATA_SCHEMA.md), not a reporting-cadence
    artifact, and is NOT forward-filled at this layer.

    AUD is different: Australia's ABS only PUBLISHES CPI quarterly (confirmed
    2026-07-23 - median gap between real AUD readings is 92 days, vs. ~31 for
    every other country here), so the two in-between calendar months in this
    already-monthly-indexed source are never going to have their own real
    reading - that's not missing data, it's what "the latest known CPI as of
    this date" genuinely is for a quarterly-reporting country. Forward-filled
    here with `limit=2` (exactly one quarter's worth of in-between months) so
    each quarter's real print is used for the months it actually covers, rather
    than every non-print month reading as NaN downstream (this WAS silently
    vetoing ~64% of AUDUSD's value-signal dates, and by extension - via
    portfolio.book.Book.run()'s joint row-wise dropna() over the whole active
    universe - most rebalance dates for any Book that includes AUDUSD, found
    live 2026-07-23 debugging a value+XSMOM mix-vs-integrate comparison).
    `limit=2` still lets AUD go genuinely NaN if it stops publishing entirely
    for more than one quarter - not an unbounded ffill, same discipline as
    signals.value.fx_ppp_value_feature's own bounded reindex_ffill_limit."""
    df = pd.read_csv(DATA_DIR / "cpi_level_index.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    for country, limit in QUARTERLY_CPI_COUNTRIES.items():
        if country in df.columns:
            df[country] = df[country].ffill(limit=limit)
    return df
