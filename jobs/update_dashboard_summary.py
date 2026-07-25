"""
Scheduled daily pre-computation of QA dashboard summary artifacts.

This is the data-prep half of the CTA dashboard's QA pages (00-04): it does all the
actual work (pipeline-health roll-up, exchange-calendar gap detection, term-structure
curve snapshotting, volatility/macro snapshots) and writes small, pre-computed
artifacts to Data/dashboard_summary/. Those QA pages only read and render those
artifacts — no computation of their own at render time, same separation of concerns
as every other producer/consumer pair in this project (e.g. capture_term_structure.py
produces, a future notebook consumes). Pages 05-16 (Strategy Performance, Portfolio
Construction, Macro Explorer) do NOT depend on this job — they compute live at render
time instead (see CLAUDE.md's Data QA dashboard row).

Must run after all 4 other scheduled jobs (CTA_DailyDataUpdate 6:00PM,
CTA_VolatilityUpdate 6:05PM, CTA_TermStructureCapture 6:15PM, CTA_MacroDataUpdate
6:20PM) since it summarizes their outputs — scheduled as CTA_DashboardSummary at
6:25PM. Each section updates independently (one failing shouldn't block the others),
logged to Data/dashboard_summary_manifest.csv, same pattern as
jobs/update_macro_data.py.

Gap detection uses real per-asset exchange calendars (pandas_market_calendars), not a
naive business-day check — the 38-asset universe spans CME/COMEX/NYMEX/CBOT/ICE with
different holiday calendars, so a single calendar would over-flag holidays as gaps.
ASSET_CALENDAR below mirrors the sector grouping in DATA_SCHEMA.md and the
(root, exchange) mapping in capture_term_structure.py's UNIVERSE.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from macro_point_in_time import (
    get_yield_curve_as_of,
    get_fed_funds_as_of,
    get_gscpi_as_of,
    get_trade_policy_uncertainty_as_of,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
SUMMARY_DIR = DATA_DIR / "dashboard_summary"
MANIFEST_PATH = DATA_DIR / "dashboard_summary_manifest.csv"

HORIZONS = [21, 63, 126, 252]
MONTH_CODES = "FGHJKMNQUVXZ"  # F=Jan ... Z=Dec, matches capture_term_structure.py
STALE_HOURS = 30  # jobs run once/day ~6-6:25PM; anything older than this is flagged STALE
ACTIVE_CONTRACT_WINDOW_DAYS = 10  # a contract with no print in this long has expired, not gone quiet

# Sector -> exchange calendar, mirrors DATA_SCHEMA.md's sector table and
# capture_term_structure.py's UNIVERSE (root, exchange-suffix) mapping.
ASSET_CALENDAR = {
    "WTI Crude": "CMEGlobex_Energy", "Brent": "CMEGlobex_Energy", "Natural Gas": "CMEGlobex_Energy",
    "RBOB": "CMEGlobex_Energy", "HeatingOil": "CMEGlobex_Energy",
    "Gold": "CMEGlobex_PreciousMetals", "Silver": "CMEGlobex_PreciousMetals",
    "Platinum": "CMEGlobex_PreciousMetals", "Palladium": "CMEGlobex_PreciousMetals",
    "Copper": "CMEGlobex_BaseMetals",
    "Corn": "CMEGlobex_Grains", "Soybeans": "CMEGlobex_Grains", "Wheat": "CMEGlobex_Grains",
    "KC_Wheat": "CMEGlobex_Grains", "Rice": "CMEGlobex_Grains", "Oats": "CMEGlobex_Grains",
    "Coffee": "ICEUS", "Sugar": "ICEUS", "Cocoa": "ICEUS", "Cotton": "ICEUS", "OrangeJuice": "ICEUS",
    "LiveCattle": "CMEGlobex_Livestock", "LeanHogs": "CMEGlobex_Livestock", "FeederCattle": "CMEGlobex_Livestock",
    "SP500": "CME_Equity", "Nasdaq100": "CME_Equity", "Dow": "CME_Equity", "Russell2000": "CME_Equity",
    "US_2Y": "CBOT_InterestRate", "US_5Y": "CBOT_InterestRate", "US_10Y": "CBOT_InterestRate",
    "US_30Y": "CBOT_InterestRate", "UltraBond": "CBOT_InterestRate",
    "EURUSD": "CME_FX", "JPYUSD": "CME_FX", "GBPUSD": "CME_FX", "AUDUSD": "CME_FX", "CADUSD": "CME_FX",
    "SwissFranc": "CME_FX", "MexicanPeso": "CME_FX",
    # Lumber (added 2026-07-17, see WORKFLOW.md Phase 4) has no dedicated calendar in
    # pandas_market_calendars - CME_Agriculture is the closest real fit (CME lists
    # Lumber under its Agricultural product group), not an exact per-product calendar
    # like the other entries above.
    "Lumber": "CME_Agriculture",
}

_CALENDAR_CACHE = {}


def _get_calendar(name):
    if name not in _CALENDAR_CACHE:
        _CALENDAR_CACHE[name] = mcal.get_calendar(name)
    return _CALENDAR_CACHE[name]


def _mtime(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None


def _freshness_status(last_run, now):
    if last_run is None:
        return "UNKNOWN"
    age_hours = (now - last_run).total_seconds() / 3600
    return "OK" if age_hours <= STALE_HOURS else "STALE"


# ------------------------------------------------------------------
# 1. Pipeline health
# ------------------------------------------------------------------
def build_pipeline_health(now):
    rows = []

    # CTA_DailyDataUpdate — no run-timestamp column in metadata.csv; use file mtime
    # of close.parquet plus the data's own max End date as a freshness cross-check.
    close_path = DATA_DIR / "close.parquet"
    last_run = _mtime(close_path)
    meta = pd.read_csv(DATA_DIR / "metadata.csv", parse_dates=["Start", "End"])
    max_end = meta["End"].max()
    rows.append({
        "job": "CTA_DailyDataUpdate",
        "last_run": last_run,
        "status": _freshness_status(last_run, now),
        "detail": f"{len(meta)} assets, latest End={max_end.date()}",
    })

    # CTA_VolatilityUpdate — now has its own manifest (Data/volatility_manifest.csv),
    # same pattern as the other jobs, no more mtime-inference.
    vol_manifest = pd.read_csv(DATA_DIR / "volatility_manifest.csv", parse_dates=["run_date"])
    latest_vol_run = vol_manifest["run_date"].max()
    latest_vol_row = vol_manifest[vol_manifest["run_date"] == latest_vol_run].iloc[-1]
    rows.append({
        "job": "CTA_VolatilityUpdate",
        "last_run": latest_vol_run.to_pydatetime(),
        "status": _freshness_status(latest_vol_run.to_pydatetime(), now) if latest_vol_row["status"] == "OK" else "FAILED",
        "detail": latest_vol_row["detail"],
    })

    # CTA_TermStructureCapture — has a manifest, one row per (run, asset).
    ts_manifest = pd.read_csv(DATA_DIR / "term_structure_manifest.csv", parse_dates=["run_date"])
    latest_run_date = ts_manifest["run_date"].max()
    latest_run_rows = ts_manifest[ts_manifest["run_date"] == latest_run_date]
    n_ok = (latest_run_rows["status"] == "OK").sum()
    rows.append({
        "job": "CTA_TermStructureCapture",
        "last_run": latest_run_date.to_pydatetime(),
        "status": _freshness_status(latest_run_date.to_pydatetime(), now) if n_ok == len(latest_run_rows) else "PARTIAL",
        "detail": f"{n_ok}/{len(latest_run_rows)} assets OK",
    })

    # CTA_MacroDataUpdate — has a manifest, one row per (run, source).
    macro_manifest = pd.read_csv(DATA_DIR / "macro_data_manifest.csv", parse_dates=["run_date"])
    latest_run_date = macro_manifest["run_date"].max()
    latest_run_rows = macro_manifest[macro_manifest["run_date"] == latest_run_date]
    n_ok = (latest_run_rows["status"] == "OK").sum()
    rows.append({
        "job": "CTA_MacroDataUpdate",
        "last_run": latest_run_date.to_pydatetime(),
        "status": _freshness_status(latest_run_date.to_pydatetime(), now) if n_ok == len(latest_run_rows) else "FAILED",
        "detail": f"{n_ok}/{len(latest_run_rows)} sources OK",
    })

    # Databento Transform — not a scheduled daily job (databento/, not jobs/), so no
    # STALE-by-recency check; this reflects last-known per-asset status, not freshness.
    db_manifest_path = DATA_DIR / "databento_transform_manifest.csv"
    if db_manifest_path.exists():
        db_manifest = pd.read_csv(db_manifest_path, parse_dates=["run_date"])
        latest_run_date = db_manifest["run_date"].max()
        latest_run_rows = db_manifest[db_manifest["run_date"] == latest_run_date]
        n_ok = (latest_run_rows["status"] == "OK").sum()
        assets = ", ".join(latest_run_rows["asset"].tolist())
        rows.append({
            "job": "Databento Transform",
            "last_run": latest_run_date.to_pydatetime(),
            "status": "OK" if n_ok == len(latest_run_rows) else "PARTIAL",
            "detail": f"{n_ok}/{len(latest_run_rows)} assets OK ({assets})",
        })
    else:
        rows.append({
            "job": "Databento Transform",
            "last_run": pd.NaT,
            "status": "UNKNOWN",
            "detail": "No manifest yet - databento/transform_databento.py not yet run",
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 2. OHLCV coverage (real exchange-calendar gap detection)
# ------------------------------------------------------------------
def build_ohlcv_coverage():
    meta = pd.read_csv(DATA_DIR / "metadata.csv", parse_dates=["Start", "End"])
    close = pd.read_parquet(DATA_DIR / "close.parquet")
    last_updated = _mtime(DATA_DIR / "close.parquet")

    rows = []
    for _, r in meta.iterrows():
        asset = r["Asset"]
        cal_name = ASSET_CALENDAR.get(asset)
        actual_dates = close[asset].dropna().index.normalize() if asset in close.columns else pd.DatetimeIndex([])

        if cal_name is None or actual_dates.empty:
            rows.append({
                "Asset": asset, "Ticker": r["Ticker"], "Start": r["Start"], "End": r["End"],
                "Rows": r["Rows"], "LastUpdated": last_updated,
                "Calendar": cal_name or "UNMAPPED", "MissingSessions": None, "MissingPct": None,
            })
            continue

        cal = _get_calendar(cal_name)
        expected = cal.schedule(start_date=r["Start"], end_date=r["End"]).index.normalize()
        missing = expected.difference(actual_dates)

        rows.append({
            "Asset": asset, "Ticker": r["Ticker"], "Start": r["Start"], "End": r["End"],
            "Rows": r["Rows"], "LastUpdated": last_updated,
            "Calendar": cal_name,
            "MissingSessions": len(missing),
            "MissingPct": len(missing) / len(expected) if len(expected) else None,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 3. Term structure — latest curve snapshot + coverage
# ------------------------------------------------------------------
def _expiry_sort_key(code, year):
    return year * 12 + MONTH_CODES.index(code)


def _expiry_label(code, year):
    month_num = MONTH_CODES.index(code) + 1
    return pd.Timestamp(year=year, month=month_num, day=1).strftime("%b %Y")


def build_term_structure(now):
    ts = pd.read_parquet(DATA_DIR / "term_structure.parquet")
    # Volume=0 rows are synthetic/theoretical prints, not real trades (DATA_SCHEMA.md 4b).
    ts = ts[ts["volume"] > 0]

    latest = (
        ts.sort_values("date")
        .groupby(["asset", "contract_symbol"], as_index=False)
        .last()[["asset", "contract_symbol", "root", "exchange", "expiry_code", "expiry_year", "close", "date"]]
        .rename(columns={"date": "last_price_date"})
    )

    # A contract whose last print is more than ACTIVE_CONTRACT_WINDOW_DAYS behind `now`
    # has expired/rolled off, not gone quiet - the "last observed price per
    # contract_symbol" query above has no such filter, which was harmless while
    # term_structure.parquet only held ~6 forward-capture contracts per asset. Once the
    # Databento historical backfill added 101 more (long-expired) LiveCattle
    # contract_symbols, this started returning every contract ever seen: a "today's
    # curve" query for LiveCattle came back with 107 points spanning 2010-2028
    # expiries, most years-stale (found 2026-07-16).
    cutoff = now - pd.Timedelta(days=ACTIVE_CONTRACT_WINDOW_DAYS)
    latest = latest[latest["last_price_date"] >= cutoff]

    latest["expiry_sort_key"] = latest.apply(lambda r: _expiry_sort_key(r["expiry_code"], r["expiry_year"]), axis=1)
    latest["expiry_label"] = latest.apply(lambda r: _expiry_label(r["expiry_code"], r["expiry_year"]), axis=1)
    latest = latest.sort_values(["asset", "expiry_sort_key"]).reset_index(drop=True)

    manifest = pd.read_csv(DATA_DIR / "term_structure_manifest.csv", parse_dates=["run_date"])
    latest_run_per_asset = manifest.sort_values("run_date").groupby("asset").last()

    summary_rows = []
    for asset, grp in latest.groupby("asset"):
        grp = grp.sort_values("expiry_sort_key")
        n = len(grp)
        if n >= 2:
            shape = "Contango" if grp["close"].iloc[-1] > grp["close"].iloc[0] else "Backwardation"
        else:
            shape = "N/A (single contract)"
        run_info = latest_run_per_asset.loc[asset] if asset in latest_run_per_asset.index else None
        summary_rows.append({
            "asset": asset,
            "n_contracts": n,
            "curve_shape": shape,
            "front_close": grp["close"].iloc[0],
            "back_close": grp["close"].iloc[-1],
            "last_run": run_info["run_date"] if run_info is not None else None,
            "status": run_info["status"] if run_info is not None else "UNKNOWN",
        })

    return latest.drop(columns=["expiry_sort_key"]), pd.DataFrame(summary_rows)


# Assets with real multi-year depth (Databento historical backfill), not just the ~1
# day of yfinance forward-capture the other 37 have - see WORKFLOW.md Phase 4. Extend
# this list as more assets get their own transform run.
DEEP_HISTORY_ASSETS = ["LiveCattle"]


def build_curve_shape_history(asset_name):
    """Daily contango/backwardation classification across an asset's full captured
    history, not just today's snapshot - only meaningful for DEEP_HISTORY_ASSETS,
    since forward-capture alone (the other 37 assets) only goes back to 2026-07-14."""
    ts = pd.read_parquet(DATA_DIR / "term_structure.parquet")
    ts = ts[(ts["asset"] == asset_name) & (ts["volume"] > 0)].copy()
    ts["expiry_sort_key"] = ts.apply(lambda r: _expiry_sort_key(r["expiry_code"], r["expiry_year"]), axis=1)
    ts = ts.sort_values(["date", "expiry_sort_key"])

    grouped = ts.groupby("date").agg(
        front_close=("close", "first"), back_close=("close", "last"), n_contracts=("close", "size"),
    ).reset_index()
    grouped = grouped[grouped["n_contracts"] >= 2].copy()
    grouped["curve_shape"] = grouped.apply(
        lambda r: "Contango" if r["back_close"] > r["front_close"] else "Backwardation", axis=1
    )
    # Normalized, not raw $ - LiveCattle moved from ~$90 (2010) to ~$250 (2025-26) over
    # this history, so a $5 gap means something different depending on when it occurs.
    # Raw $ isn't comparable across the panel; % is.
    grouped["steepness_pct"] = (grouped["back_close"] - grouped["front_close"]) / grouped["front_close"] * 100
    grouped.insert(0, "asset", asset_name)
    return grouped


def build_curve_rank_history(asset_name, max_rank=10):
    """Long-form (date, rank, close, close_minus_front) for a term-structure heatmap.
    Ranked by position along the curve (0=front/nearest, 1=next, ...), not absolute
    contract month - absolute months roll off every few weeks as contracts expire, so
    indexing by them would make a heatmap mostly-empty/diagonal instead of dense."""
    ts = pd.read_parquet(DATA_DIR / "term_structure.parquet")
    ts = ts[(ts["asset"] == asset_name) & (ts["volume"] > 0)].copy()
    ts["expiry_sort_key"] = ts.apply(lambda r: _expiry_sort_key(r["expiry_code"], r["expiry_year"]), axis=1)
    ts = ts.sort_values(["date", "expiry_sort_key"])
    ts["rank"] = ts.groupby("date").cumcount()
    ts = ts[ts["rank"] < max_rank]

    front = ts[ts["rank"] == 0][["date", "close"]].rename(columns={"close": "front_close"})
    ts = ts.merge(front, on="date", how="left")
    ts["close_minus_front"] = ts["close"] - ts["front_close"]
    return ts[["asset", "date", "rank", "contract_symbol", "close", "close_minus_front"]]


# ------------------------------------------------------------------
# 4. Volatility snapshot
# ------------------------------------------------------------------
def build_volatility_snapshot():
    yz = pd.read_parquet(DATA_DIR / "yang_zhang_features.parquet")
    latest_date = yz.index.max()
    latest = yz.loc[latest_date].unstack(level=0)  # index=asset, columns=yz_vol_*
    latest = latest[[f"yz_vol_{w}" for w in HORIZONS]] * (252 ** 0.5)  # annualize
    latest = latest.rename(columns={f"yz_vol_{w}": f"yz_vol_{w}_ann" for w in HORIZONS})
    latest.insert(0, "AsOfDate", latest_date)
    latest.index.name = "Asset"
    return latest.reset_index()


# ------------------------------------------------------------------
# 5. Macro latest (point-in-time correct, via macro_point_in_time.py)
# ------------------------------------------------------------------
def build_macro_latest(now):
    today = now.date()
    manifest = pd.read_csv(DATA_DIR / "macro_data_manifest.csv", parse_dates=["run_date"])
    latest_run_per_source = manifest.sort_values("run_date").groupby("source").last()

    def _run_info(source):
        if source in latest_run_per_source.index:
            r = latest_run_per_source.loc[source]
            return r["run_date"], r["status"]
        return None, "UNKNOWN"

    rows = []

    r = get_yield_curve_as_of(today)
    last_run, status = _run_info("yield_curve")
    rows.append({
        "source": "yield_curve", "as_of_date": today,
        "value_date": r["Date"].date() if r is not None else None,
        "key_value": r["10Y"] if r is not None else None,
        "key_label": "10Y par yield (%)",
        "last_run": last_run, "job_status": status,
        "data_status": "OK" if r is not None else "NO_DATA_AVAILABLE",
    })

    r = get_fed_funds_as_of(today)
    last_run, status = _run_info("fed_funds_rates")
    rows.append({
        "source": "fed_funds_rates", "as_of_date": today,
        "value_date": r["Effective Date"].date() if r is not None else None,
        "key_value": r["Rate (%)"] if r is not None else None,
        "key_label": "EFFR (%)",
        "last_run": last_run, "job_status": status,
        "data_status": "OK" if r is not None else "NO_DATA_AVAILABLE",
    })

    r = get_gscpi_as_of(today)
    last_run, status = _run_info("gscpi")
    rows.append({
        "source": "gscpi", "as_of_date": today,
        "value_date": r["Date"].date() if r is not None else None,
        "key_value": r["GSCPI"] if r is not None else None,
        "key_label": "GSCPI index",
        "last_run": last_run, "job_status": status,
        "data_status": "OK" if r is not None else "NOT_YET_PUBLISHED",
    })

    r = get_trade_policy_uncertainty_as_of(today)
    last_run, status = _run_info("trade_policy_uncertainty")
    rows.append({
        "source": "trade_policy_uncertainty", "as_of_date": today,
        "value_date": r["Date"].date() if r is not None else None,
        "key_value": r["daily_tpu_index"] if r is not None else None,
        "key_label": "Daily TPU index",
        "last_run": last_run, "job_status": status,
        "data_status": "OK" if r is not None else "NO_DATA_AVAILABLE",
    })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Orchestration — mirrors update_macro_data.py's per-section try/except + manifest.
# ------------------------------------------------------------------
def main() -> int:
    now = datetime.now()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    run_date = now.isoformat(timespec="seconds")
    manifest_rows = []
    any_failed = False

    sections = {
        "pipeline_health": lambda: build_pipeline_health(now).to_csv(SUMMARY_DIR / "pipeline_health.csv", index=False),
        "ohlcv_coverage": lambda: build_ohlcv_coverage().to_csv(SUMMARY_DIR / "ohlcv_coverage.csv", index=False),
        "term_structure": None,  # handled separately, writes two files
        "volatility": lambda: build_volatility_snapshot().to_csv(SUMMARY_DIR / "volatility_snapshot.csv", index=False),
        "macro": lambda: build_macro_latest(now).to_csv(SUMMARY_DIR / "macro_latest.csv", index=False),
    }

    for name, fn in sections.items():
        try:
            if name == "term_structure":
                curve, summary = build_term_structure(now)
                curve.to_parquet(SUMMARY_DIR / "term_structure_curve.parquet", index=False)
                summary.to_csv(SUMMARY_DIR / "term_structure_summary.csv", index=False)
                shape_history = pd.concat(
                    [build_curve_shape_history(a) for a in DEEP_HISTORY_ASSETS], ignore_index=True
                )
                shape_history.to_parquet(SUMMARY_DIR / "term_structure_curve_shape_history.parquet", index=False)
                rank_history = pd.concat(
                    [build_curve_rank_history(a) for a in DEEP_HISTORY_ASSETS], ignore_index=True
                )
                rank_history.to_parquet(SUMMARY_DIR / "term_structure_curve_rank_history.parquet", index=False)
                detail = (f"{summary['n_contracts'].sum()} contracts across {len(summary)} assets, "
                          f"{len(shape_history)} curve-shape-history rows, {len(rank_history)} "
                          f"curve-rank-history rows across {len(DEEP_HISTORY_ASSETS)} deep-history asset(s)")
            else:
                fn()
                detail = "written"
            manifest_rows.append({"run_date": run_date, "section": name, "status": "OK", "detail": detail})
            print(f"OK   {name}")
        except Exception as exc:
            any_failed = True
            manifest_rows.append({"run_date": run_date, "section": name, "status": "FAILED", "detail": str(exc)})
            print(f"FAIL {name}: {exc}")

    manifest_df = pd.DataFrame(manifest_rows)
    if MANIFEST_PATH.exists():
        manifest_df = pd.concat([pd.read_csv(MANIFEST_PATH), manifest_df], ignore_index=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
