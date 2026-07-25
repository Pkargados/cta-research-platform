"""
Daily forward-capture of individual futures contract-month prices, for term
structure / carry (WORKFLOW.md Phase 4).

Why this exists: yfinance's individual contract-month tickers (e.g. CLQ26.NYM)
are undocumented and stop resolving entirely once a contract expires - there
is no way to retroactively backfill an expired contract's history. So this
script must run daily and append; it cannot be re-run once to "catch up".
Each day it doesn't run is a day of near-dated contract history permanently
lost for whichever contracts expire before the next run.

Output is append-only and deduplicated on (date, contract_symbol), so re-runs
the same day are safe and it can be run more than once without corrupting the
archive.
"""

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_THROTTLE_SECONDS = 0.2


def _fetch_history_once(ticker, period):
    """Single-shot fetch, no retry. Used for discovery, where most probed
    month codes are *expected* to fail (not every month is listed) - retrying
    those would mean retrying dozens of legitimate 404s per asset."""
    time.sleep(REQUEST_THROTTLE_SECONDS)
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


def _fetch_history_with_retry(ticker, period):
    """Retry/backoff fetch for tickers already confirmed to exist - here a
    transient rate-limit hit means permanently losing that day's print for a
    contract, not just a delay, so it's worth a few retries."""
    for attempt in range(RETRY_ATTEMPTS):
        hist = _fetch_history_once(ticker, period)
        if not hist.empty:
            return hist
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return pd.DataFrame()

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ARCHIVE_PATH = DATA_DIR / "term_structure.parquet"
MANIFEST_PATH = DATA_DIR / "term_structure_manifest.csv"

MONTH_CODES = "FGHJKMNQUVXZ"

# (root, exchange suffix) per asset, keyed to match the column names already
# used in Data/close.parquet etc. Exchange suffixes: NYM=NYMEX, CMX=COMEX,
# CBT=CBOT, NYB=ICE/NYBOT, CME=CME.
UNIVERSE = {
    "WTI Crude": ("CL", "NYM"), "Brent": ("BZ", "NYM"), "Natural Gas": ("NG", "NYM"),
    "RBOB": ("RB", "NYM"), "HeatingOil": ("HO", "NYM"),
    "Gold": ("GC", "CMX"), "Silver": ("SI", "CMX"), "Platinum": ("PL", "NYM"), "Palladium": ("PA", "NYM"),
    "Copper": ("HG", "CMX"),
    "Corn": ("ZC", "CBT"), "Soybeans": ("ZS", "CBT"), "Wheat": ("ZW", "CBT"), "KC_Wheat": ("KE", "CBT"),
    "Rice": ("ZR", "CBT"), "Oats": ("ZO", "CBT"),
    "Coffee": ("KC", "NYB"), "Sugar": ("SB", "NYB"), "Cocoa": ("CC", "NYB"), "Cotton": ("CT", "NYB"),
    "OrangeJuice": ("OJ", "NYB"),
    "LiveCattle": ("LE", "CME"), "LeanHogs": ("HE", "CME"), "FeederCattle": ("GF", "CME"),
    "SP500": ("ES", "CME"), "Nasdaq100": ("NQ", "CME"), "Dow": ("YM", "CBT"), "Russell2000": ("RTY", "CME"),
    "US_2Y": ("ZT", "CBT"), "US_5Y": ("ZF", "CBT"), "US_10Y": ("ZN", "CBT"), "US_30Y": ("ZB", "CBT"),
    "UltraBond": ("UB", "CBT"),
    "EURUSD": ("6E", "CME"), "JPYUSD": ("6J", "CME"), "GBPUSD": ("6B", "CME"), "AUDUSD": ("6A", "CME"),
    "CADUSD": ("6C", "CME"),
    # Added 2026-07-17 (WORKFLOW.md Phase 4 asset-expansion note): confirmed via
    # discover_contracts() before adding, not assumed - SOFR resolved 6/6 candidate
    # months, Lumber 3/6 (sparser listing cycle than most), SwissFranc 3/6, MexicanPeso
    # 4/6. All four also have a Databento historical batch pull queued.
    "SOFR": ("SR3", "CME"), "Lumber": ("LBR", "CME"), "SwissFranc": ("6S", "CME"),
    "MexicanPeso": ("6M", "CME"),
}

MAX_CONTRACTS_PER_ASSET = 6
DISCOVERY_MONTHS = 25  # covers quarterly-cycle financial futures out to ~6 listed contracts


def discover_contracts(root, exch, n_months=DISCOVERY_MONTHS, need=MAX_CONTRACTS_PER_ASSET):
    """Probe upcoming month codes and keep the ones that are actually listed.

    There is no "list expiries" API for these tickers, and listing cycles
    aren't uniform across products (e.g. livestock/softs skip some months),
    so discovery is done by trying each candidate rather than assuming a
    fixed cycle.
    """
    today = dt.date.today()
    found = []
    for m in range(n_months):
        month = (today.month - 1 + m) % 12
        year = today.year + (today.month - 1 + m) // 12
        code = MONTH_CODES[month]
        yy = year % 100
        ticker = f"{root}{code}{yy:02d}.{exch}"
        hist = _fetch_history_once(ticker, period="5d")
        if not hist.empty:
            found.append((ticker, code, year))
            if len(found) >= need:
                break
    return found


def pull_contract_history(ticker):
    hist = _fetch_history_with_retry(ticker, period="max")
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # Drop incomplete bars - some field(s) left at exactly 0 while others are
    # populated - rather than write a partially-fabricated print. Found
    # 2026-07-20 on far-dated, zero-volume contracts (e.g. OJH27.NYB
    # 2025-09-30): yfinance serves a flat carried-forward quote for long
    # no-trade stretches (already documented, WORKFLOW.md Phase 4), and
    # occasionally glitches to an exact-zero open/high/low on one isolated day
    # within that stretch instead of repeating the flat value. Reproduced live,
    # on a weekday, ruling out a day-of-week cause. Exact zero, not merely
    # non-positive, mirroring the same fix already validated in
    # databento/transform_databento.py for the same underlying failure mode -
    # a real negative price (e.g. WTI April 2020) never has an exact zero in
    # any field, only a genuinely missing print does.
    return hist[(hist[["Open", "High", "Low", "Close"]] != 0).all(axis=1)]


def run():
    run_date = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    manifest_rows = []

    for asset, (root, exch) in UNIVERSE.items():
        contracts = discover_contracts(root, exch)
        if not contracts:
            manifest_rows.append({
                "run_date": run_date, "asset": asset, "contracts_found": 0,
                "contracts_pulled": 0, "status": "NO_CONTRACTS_FOUND",
            })
            continue

        pulled = 0
        for ticker, code, year in contracts:
            hist = pull_contract_history(ticker)
            if hist.empty:
                continue
            hist["asset"] = asset
            hist["contract_symbol"] = ticker
            hist["root"] = root
            hist["exchange"] = exch
            hist["expiry_code"] = code
            hist["expiry_year"] = year
            rows.append(hist)
            pulled += 1

        manifest_rows.append({
            "run_date": run_date, "asset": asset, "contracts_found": len(contracts),
            "contracts_pulled": pulled,
            "status": "OK" if pulled == len(contracts) else "PARTIAL",
        })

    if not rows:
        raise RuntimeError("No contract data pulled for any asset - aborting without touching the archive.")

    new_data = pd.concat(rows, ignore_index=True)
    new_data = new_data.rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    if ARCHIVE_PATH.exists():
        existing = pd.read_parquet(ARCHIVE_PATH)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (
        combined
        .drop_duplicates(subset=["date", "contract_symbol"], keep="last")
        .sort_values(["asset", "contract_symbol", "date"])
        .reset_index(drop=True)
    )

    combined.to_parquet(ARCHIVE_PATH, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    if MANIFEST_PATH.exists():
        manifest_df = pd.concat([pd.read_csv(MANIFEST_PATH), manifest_df], ignore_index=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    n_assets_ok = sum(1 for r in manifest_rows if r["status"] == "OK")
    print(f"Term structure capture complete: {n_assets_ok}/{len(UNIVERSE)} assets fully captured, "
          f"{len(combined)} total rows in archive.")


if __name__ == "__main__":
    run()
