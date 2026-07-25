"""
Scheduled refresh of the 6 macro/auxiliary datasets in Data/ (DATA_SCHEMA.md section 3).

These are currently collected-but-unused (no notebook references them yet), but kept
current so they're ready whenever a macro-control/regime signal is built. Each source
updates independently - one failing shouldn't block the others - and every run is logged
to Data/macro_data_manifest.csv for auditing.

Sources, identified by inspecting each existing file's schema and confirming the live
endpoint on 2026-07-14 (none of these were previously documented with a source URL):
  - Yield curve:        FRED (fred.stlouisfed.org), series DGS6MO/DGS1/DGS2/DGS3/DGS5/
                         DGS7/DGS10/DGS20/DGS30, no API key required.
  - Fed funds/reference: NY Fed Markets Data API (markets.newyorkfed.org/api/rates),
    rates                no API key required.
  - GSCPI:               NY Fed's own fixed download URL (see GSCPI_URL below).
  - Trade Policy         policyuncertainty.com's fixed download URL (see TPU_URL below).
    Uncertainty
  - VIX:                 yfinance, ticker "^VIX" (CBOE S&P500 implied-vol index) - added
                          2026-07-21 for the short-term reversal signal
                          (references/short_term_reversal_implementation_recipe.md):
                          Nagel (2011) finds reversal-strategy expected returns are
                          strongly predictable with VIX. Same yfinance pipeline already
                          used for the core 42-asset panel (jobs/update_data.py), pulled
                          here rather than there since VIX isn't a tradeable instrument
                          in this project's universe - it's a state variable, same
                          category as the other 4 sources in this file.
  - CPI (8 countries):   FRED (fred.stlouisfed.org), CPI LEVEL INDEX series (not growth
                          rate) - added 2026-07-22 for a prospective value factor
                          (Asness-Moskowitz-Pedersen 2013 "Value and Momentum
                          Everywhere" - FX value there is the 5-year PPP-adjusted real
                          exchange-rate return, which needs relative CPI). Every series
                          ID live-verified to actually resolve AND checked for recency
                          before being hardcoded here (2026-07-22) - a first attempt
                          using the OECD MEI "growth rate previous period" series family
                          (M657N codes) resolved fine but turned out to be stale-to-
                          discontinued for most non-US countries (Japan stuck at
                          2021-06, five years behind), an easy mistake to make from the
                          series ID alone without actually checking the last observation
                          date. Switched to each country's CPI LEVEL index instead - both
                          more current for 6 of 8 countries AND simpler to use for the
                          PPP calc (a direct log-ratio of index levels 5 years apart,
                          vs. compounding a monthly growth-rate series with an easily
                          mixed-up MoM/YoY convention). **Staleness gap, stated
                          plainly**: as of 2026-07-22, US/EUR are fully current
                          (2026-06); GBP/CAD current to 2025-03; CHF to 2025-04; AUD
                          (quarterly - Australia's real-world CPI has no better monthly
                          series) to 2025-01; MXN to 2024-07.
  - CPI, Japan:           e-Stat (api.e-stat.go.jp), Japan's official government
                          statistics API - added 2026-07-23 after FRED's JPNCPIALLMINMEI
                          (and every OECD-MEI-family alternative tried) turned out stuck
                          at 2021-06, ~5 years stale, with no better FRED-hosted monthly
                          alternative found (a World Bank annual series, FPCPITOTLZGJPN,
                          reaches 2025 but at yearly granularity, incompatible with this
                          file's monthly grid). Confirmed live: Japan's PRIMARY source
                          was never actually stale - the international mirror channels
                          (FRED's OECD-MEI copies) were. `api.e-stat.go.jp` needs a free
                          `appId` (registration required, confirmed by testing the API
                          unauthenticated first and getting an explicit "check your
                          application ID" error) - set as the `ESTAT_APP_ID` environment
                          variable, same OS-env-var convention as `DATABENTO_API_KEY`
                          (no `.env` file in this project; adding one now would introduce
                          a second, less safe convention for 3 credentials total).
                          Table `0003427113` (総務省/MIC Statistics Bureau, 2020-base
                          nationwide CPI), filtered to `tab=1` (index level), `cat01=0001`
                          (all items / headline), `area=00000` (nationwide) - codes found
                          via `getMetaInfo`, not guessed. Time-dimension entries mix
                          genuine monthly observations with calendar-year AND fiscal-year
                          averages under the same "time" axis (e.g. "2025年" and
                          "2025年度" alongside "2026年6月") - confirmed live via each
                          entry's own `@level` field (`"4"` = monthly, `"1"` = annual/
                          fiscal) rather than assumed from the numeric time code's
                          positional structure, which encodes the same distinction but
                          is easy to misparse. Result: 1970 -> 2026-05, a completely
                          normal ~2-month publication lag, not stale at all.
"""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
MANIFEST_PATH = DATA_DIR / "macro_data_manifest.csv"

FRED_SERIES = {
    "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5",
    "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}
YIELD_CURVE_PATH = DATA_DIR / "Yield_Curve_6M_to_30Y.csv"

NY_FED_RATES_URL = "https://markets.newyorkfed.org/api/rates/all/search.json"
FED_FUNDS_PATH = DATA_DIR / "overnight_fed_fund_rates_US.xlsx"
FED_FUNDS_LOOKBACK_DAYS = 45  # comfortably covers any plausible gap since the last run

FED_FUNDS_FIELD_MAP = {
    "effectiveDate": "Effective Date",
    "type": "Rate Type",
    "percentRate": "Rate (%)",
    "percentPercentile1": "1st Percentile (%)",
    "percentPercentile25": "25th Percentile (%)",
    "percentPercentile75": "75th Percentile (%)",
    "percentPercentile99": "99th Percentile (%)",
    "volumeInBillions": "Volume ($Billions)",
    "targetRateFrom": "Target Rate From (%)",
    "targetRateTo": "Target Rate To (%)",
    "average30day": "30-Day Average SOFR",
    "average90day": "90-Day Average SOFR",
    "average180day": "180-Day Average SOFR",
    "index": "SOFR Index",
    "revisionIndicator": "Revision Indicator (Y/N)",
}

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
GSCPI_PATH = DATA_DIR / "gscpi_data.xls"

TPU_URL = "https://policyuncertainty.com/media/All_Daily_TPU_Data.csv"
TPU_PATH = DATA_DIR / "trade_policy_uncertainty_US.csv"

VIX_TICKER = "^VIX"
VIX_PATH = DATA_DIR / "vix_data.csv"

# FX universe currencies (data/sectors.py's FX group) -> FRED CPI LEVEL INDEX series,
# plus US as the PPP base. Live-verified 2026-07-22 for both resolution AND recency
# (see module docstring for the staleness gap, worst for JPY at ~5 years).
CPI_FRED_SERIES = {
    "US": "CPIAUCSL",
    "EUR": "CP0000EZ19M086NEST",
    # JPY intentionally absent - sourced from e-Stat directly (ESTAT_JP_CPI_* below),
    # not FRED. See module docstring.
    "GBP": "GBRCPIALLMINMEI",
    "AUD": "AUSCPIALLQINMEI",  # quarterly - see module docstring
    "CAD": "CANCPIALLMINMEI",
    "CHF": "CHECPIALLMINMEI",
    "MXN": "MEXCPIALLMINMEI",
}
CPI_PATH = DATA_DIR / "cpi_level_index.csv"

ESTAT_APP_ID_ENV_VAR = "ESTAT_APP_ID"
ESTAT_STATS_DATA_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
ESTAT_JP_CPI_STATS_DATA_ID = "0003427113"  # 総務省 (MIC Statistics Bureau), 2020-base CPI
ESTAT_JP_CPI_QUERY = {"cdTab": "1", "cdCat01": "0001", "cdArea": "00000"}  # Index / All items / Nationwide

HEADERS = {"User-Agent": "Mozilla/5.0"}


def update_yield_curve():
    series = {}
    for label, fred_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df = df.set_index("observation_date")[fred_id]
        df = pd.to_numeric(df, errors="coerce")
        series[label] = df

    daily = pd.DataFrame(series)
    monthly = daily.resample("MS").mean().dropna(how="any")
    monthly.index.name = "Date"
    monthly = monthly[list(FRED_SERIES.keys())]
    monthly.to_csv(YIELD_CURVE_PATH)
    return len(monthly)


def update_fed_funds_rates():
    end = datetime.now().date()
    start = end - timedelta(days=FED_FUNDS_LOOKBACK_DAYS)
    resp = requests.get(
        NY_FED_RATES_URL,
        params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    records = resp.json().get("refRates", [])
    if not records:
        return 0

    new_data = pd.DataFrame(records).rename(columns=FED_FUNDS_FIELD_MAP)
    new_data["Effective Date"] = pd.to_datetime(new_data["Effective Date"])

    if FED_FUNDS_PATH.exists():
        existing = pd.read_excel(FED_FUNDS_PATH, sheet_name="Results")
        existing["Effective Date"] = pd.to_datetime(existing["Effective Date"])
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (
        combined
        .drop_duplicates(subset=["Effective Date", "Rate Type"], keep="last")
        .sort_values(["Effective Date", "Rate Type"])
        .reset_index(drop=True)
    )
    combined.to_excel(FED_FUNDS_PATH, sheet_name="Results", index=False)
    return len(new_data)


def update_gscpi():
    resp = requests.get(GSCPI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    GSCPI_PATH.write_bytes(resp.content)
    return len(resp.content)


def update_trade_policy_uncertainty():
    resp = requests.get(TPU_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    TPU_PATH.write_bytes(resp.content)
    df = pd.read_csv(TPU_PATH)
    return len(df)


def update_vix():
    df = yf.download(VIX_TICKER, period="max", auto_adjust=False, progress=False)
    if len(df) == 0:
        raise ValueError(f"empty response for {VIX_TICKER}")

    # yf.download returns MultiIndex columns (field, ticker) for a single-ticker
    # call in current yfinance versions - same shape jobs/update_data.py already
    # handles by indexing df["Close"] directly (a Series in that case); do the same
    # here rather than assume a flat-column shape.
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.index.name = "Date"
    out.sort_index(inplace=True)
    out.to_csv(VIX_PATH)
    return len(out)


def update_jp_cpi_from_estat():
    """Japan CPI (nationwide, all-items, index level) from e-Stat directly - see module
    docstring for why (FRED's mirror is ~5 years stale) and how the query codes
    (tab=1/cat01=0001/area=00000) and the monthly-vs-annual time filter (@level=="4")
    were determined (via getMetaInfo/getStatsData live, not guessed).
    """
    app_id = os.environ.get(ESTAT_APP_ID_ENV_VAR)
    if not app_id:
        raise RuntimeError(
            f"{ESTAT_APP_ID_ENV_VAR} not set in the environment. Register a free e-Stat "
            "account, then Mypage -> API -> Issue Application ID, then set it as a "
            "user-level environment variable (same convention as DATABENTO_API_KEY)."
        )

    params = {"appId": app_id, "statsDataId": ESTAT_JP_CPI_STATS_DATA_ID, **ESTAT_JP_CPI_QUERY}
    resp = requests.get(ESTAT_STATS_DATA_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()["GET_STATS_DATA"]
    status = payload["RESULT"]["STATUS"]
    if status != 0:
        raise RuntimeError(f"e-Stat API error (status {status}): {payload['RESULT'].get('ERROR_MSG')}")

    time_classes = next(
        c["CLASS"] for c in payload["STATISTICAL_DATA"]["CLASS_INF"]["CLASS_OBJ"] if c["@id"] == "time"
    )
    # level "4" = genuine monthly observation; level "1" mixes in calendar-year AND
    # fiscal-year averages under the same "time" axis - see module docstring.
    monthly_names = {c["@code"]: c["@name"] for c in time_classes if c["@level"] == "4"}

    rows = {}
    for v in payload["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]:
        name = monthly_names.get(v["@time"])
        if name is None:
            continue
        m = re.match(r"(\d{4})年(\d{1,2})月$", name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        rows[pd.Timestamp(year=year, month=month, day=1)] = float(v["$"])

    return pd.Series(rows, dtype=float).sort_index()


def update_cpi():
    series = {}
    for label, fred_id in CPI_FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df = df.set_index("observation_date")[fred_id]
        series[label] = pd.to_numeric(df, errors="coerce")

    # e-Stat sub-source is independently fault-tolerant - one source failing shouldn't
    # block the other 7 (same principle the module docstring states for the whole file).
    try:
        series["JPY"] = update_jp_cpi_from_estat()
    except Exception as exc:
        print(f"WARN: JPY CPI (e-Stat) failed, continuing with the other 7 countries: {exc}")
        series["JPY"] = pd.Series(dtype=float)

    combined = pd.concat(series, axis=1)
    combined.index.name = "Date"
    # how="all" not how="any" - AUD's quarterly cadence and EUR's later start (1991)
    # would otherwise wipe out real data from the other series - see module docstring.
    combined = combined.dropna(how="all")
    combined.to_csv(CPI_PATH)
    return len(combined)


UPDATERS = {
    "yield_curve": update_yield_curve,
    "fed_funds_rates": update_fed_funds_rates,
    "gscpi": update_gscpi,
    "trade_policy_uncertainty": update_trade_policy_uncertainty,
    "vix": update_vix,
    "cpi": update_cpi,
}


def main() -> int:
    run_date = datetime.now().isoformat(timespec="seconds")
    manifest_rows = []
    any_failed = False

    for name, updater in UPDATERS.items():
        try:
            n = updater()
            manifest_rows.append({"run_date": run_date, "source": name, "status": "OK", "detail": n})
            print(f"OK   {name}: {n}")
        except Exception as exc:
            any_failed = True
            manifest_rows.append({"run_date": run_date, "source": name, "status": "FAILED", "detail": str(exc)})
            print(f"FAIL {name}: {exc}")

    manifest_df = pd.DataFrame(manifest_rows)
    if MANIFEST_PATH.exists():
        manifest_df = pd.concat([pd.read_csv(MANIFEST_PATH), manifest_df], ignore_index=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
