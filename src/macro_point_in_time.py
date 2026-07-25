"""
Point-in-time accessors for the 4 macro/auxiliary datasets in Data/.

None of the raw files distinguish "date the value describes" from "date the value
was actually known" - they're stored exactly as each source publishes them, which is
correct for fidelity but unsafe to join directly against a backtest date. These
functions apply each source's confirmed publication lag so `as_of(date)` never returns
a value that wasn't actually knowable on that date.

Lag rules, confirmed 2026-07-14 (see DATA_SCHEMA.md section 3 for sourcing detail):
  - Yield curve:        no lag. Same-day market observation (Treasury par yields via
                         FRED), not survey-based, no revision mechanism.
  - Fed funds rates:     1 business day. NY Fed publishes the rate for day T at ~9am ET
                         on T+1 (confirmed in NY Fed's own "Details on Publication and
                         Revisions" documentation, and empirically: querying on 07-14
                         returned 07-13 as the latest available date). Revisions, when
                         they occur, are same-day-of-publication only, so a value is
                         effectively final once its publish day has passed.
  - GSCPI:               lag to the 6th business day of the following month. NY Fed's
                         own release schedule states the 4th business day; confirmed
                         empirically that the June-2026 value (dated 2026-06-30) was
                         already available by 2026-07-14, consistent with that schedule.
                         Padded to the 6th business day (vs. the stated 4th) as a
                         conservative margin, since GSCPI's revision policy was not
                         confirmed this session (page is JS-rendered, couldn't verify
                         from primary source) - if GSCPI does see minor post-release
                         revisions, this extra margin reduces the chance of catching a
                         first-print value that later moves. Revisit this constant if
                         that gets confirmed either way.
  - Trade Policy         1 business day, conservatively. Empirically near real-time (a
    Uncertainty          pull on 07-14 reached through 07-13), and likely low revision
                         risk given it's a mechanical newspaper-keyword count rather
                         than a survey-based statistic - but no authoritative source
                         confirmed zero revisions this session, so a 1-day lag is
                         applied rather than assuming same-day availability.

Nothing in this project consumes macro data yet (see CLAUDE.md - momentum, breakout,
crossover, and RV spreads don't need it). This module exists so that whenever that
changes, point-in-time correctness is already solved rather than bolted on after the
fact under time pressure.
"""

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

GSCPI_PUBLICATION_LAG_BDAYS = 6
FED_FUNDS_PUBLICATION_LAG_BDAYS = 1
TPU_PUBLICATION_LAG_BDAYS = 1


def _add_bdays(dates, n):
    return dates + n * BDAY


def get_yield_curve_as_of(as_of_date, df=None):
    """No lag - Treasury par yields are a same-day market observation."""
    if df is None:
        df = pd.read_csv(DATA_DIR / "Yield_Curve_6M_to_30Y.csv", parse_dates=["Date"])
    as_of_date = pd.Timestamp(as_of_date)
    available = df[df["Date"] <= as_of_date]
    if available.empty:
        return None
    return available.iloc[-1]


def get_fed_funds_as_of(as_of_date, rate_type="EFFR", df=None):
    """1 business day lag - the rate for day T is published ~9am ET on T+1."""
    if df is None:
        df = pd.read_excel(DATA_DIR / "overnight_fed_fund_rates_US.xlsx", sheet_name="Results")
        df["Effective Date"] = pd.to_datetime(df["Effective Date"])

    df = df[df["Rate Type"] == rate_type].copy()
    df["publish_date"] = _add_bdays(df["Effective Date"], FED_FUNDS_PUBLICATION_LAG_BDAYS)

    as_of_date = pd.Timestamp(as_of_date)
    available = df[df["publish_date"] <= as_of_date]
    if available.empty:
        return None
    return available.sort_values("Effective Date").iloc[-1]


def get_gscpi_as_of(as_of_date, df=None):
    """Lag to the 6th business day of the month following the reference month."""
    if df is None:
        df = pd.read_excel(DATA_DIR / "gscpi_data.xls", sheet_name="GSCPI Monthly Data", skiprows=4)
        df.columns = ["Date", "GSCPI"]
        df = df.dropna()
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.copy()
    next_month_start = df["Date"] + pd.offsets.MonthBegin(1)
    df["publish_date"] = _add_bdays(next_month_start, GSCPI_PUBLICATION_LAG_BDAYS)

    as_of_date = pd.Timestamp(as_of_date)
    available = df[df["publish_date"] <= as_of_date]
    if available.empty:
        return None
    return available.sort_values("Date").iloc[-1]


def get_trade_policy_uncertainty_as_of(as_of_date, df=None):
    """1 business day lag, applied conservatively (see module docstring)."""
    if df is None:
        df = pd.read_csv(DATA_DIR / "trade_policy_uncertainty_US.csv")
        df["Date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

    df = df.copy()
    df["publish_date"] = _add_bdays(df["Date"], TPU_PUBLICATION_LAG_BDAYS)

    as_of_date = pd.Timestamp(as_of_date)
    available = df[df["publish_date"] <= as_of_date]
    if available.empty:
        return None
    return available.sort_values("Date").iloc[-1]
