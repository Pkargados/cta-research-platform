"""
Thin scheduled re-run of the get_data.ipynb pipeline.

Pulls the full 41-asset universe from yfinance (period="max", same as the notebook) and
overwrites the parquet files in Data/. This is a full re-pull, not an incremental
append — yfinance's "max" period already re-fetches full history cheaply enough that a
true incremental-append design isn't worth the extra complexity (dedup logic, gap
handling) at this stage. See CLAUDE.md / WORKFLOW.md Phase 10b for why this is
deliberately kept thin rather than folded into a shared package: that refactor (Phase 1)
hasn't happened yet, and this script intentionally duplicates get_data.ipynb's logic
rather than anticipate a module structure that doesn't exist yet.

Known limitation carried over from get_data.ipynb: the Rice bad-print patch
(2024-06-17, applied in feature_engineering.ipynb) is NOT applied here — this script
only reproduces the raw pull. See DATA_SCHEMA.md section 1.

Intended to be run manually today, and wired into Windows Task Scheduler later without
modification (run via `python update_data.py`, exit code 0 on success).
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Data"

# =====================================================
# Universe — kept identical to get_data.ipynb.
# If the notebook's universe changes, update both places
# until Phase 1 extracts a shared definition.
# =====================================================

commodities = {
    # ENERGY
    "WTI Crude": "CL=F",
    "Brent": "BZ=F",
    "Natural Gas": "NG=F",
    "RBOB": "RB=F",
    "HeatingOil": "HO=F",
    # PRECIOUS METALS
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    # INDUSTRIAL METALS
    "Copper": "HG=F",
    # GRAINS
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "ZW=F",
    "KC_Wheat": "KE=F",
    "Rice": "ZR=F",
    "Oats": "ZO=F",
    # SOFTS
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cocoa": "CC=F",
    "Cotton": "CT=F",
    "OrangeJuice": "OJ=F",
    # LIVESTOCK
    "LiveCattle": "LE=F",
    "LeanHogs": "HE=F",
    "FeederCattle": "GF=F",
}

futures = {
    # EQUITY INDEX
    "SP500": "ES=F",
    "Nasdaq100": "NQ=F",
    "Dow": "YM=F",
    "Russell2000": "RTY=F",
    # RATES
    "US_2Y": "ZT=F",
    "US_5Y": "ZF=F",
    "US_10Y": "ZN=F",
    "US_30Y": "ZB=F",
    "UltraBond": "UB=F",
    # SOFR (SR3) intentionally excluded - no usable Yahoo continuous ticker
    # (SR3=F/SR1=F resolve but only return 1 row of history) - see get_data.ipynb.
    # FX
    "EURUSD": "6E=F",
    "JPYUSD": "6J=F",
    "GBPUSD": "6B=F",
    "AUDUSD": "6A=F",
    "CADUSD": "6C=F",
    "SwissFranc": "6S=F",
    "MexicanPeso": "6M=F",
    # LUMBER (no natural sector - its own category, per DATA_SCHEMA.md)
    "Lumber": "LBR=F",
}

UNIVERSE = {**commodities, **futures}


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def main() -> int:
    log(f"Starting update — {len(UNIVERSE)} assets, target: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    open_prices = pd.DataFrame()
    high_prices = pd.DataFrame()
    low_prices = pd.DataFrame()
    close_prices = pd.DataFrame()
    adj_close_prices = pd.DataFrame()
    volumes = pd.DataFrame()

    metadata = []
    failures = []

    for asset, ticker in UNIVERSE.items():
        try:
            df = yf.download(
                ticker,
                period="max",
                auto_adjust=False,
                progress=False,
            )

            if len(df) == 0:
                failures.append((asset, ticker, "empty response"))
                log(f"  FAILED (empty) -> {asset} ({ticker})")
                continue

            open_prices[asset] = df["Open"]
            high_prices[asset] = df["High"]
            low_prices[asset] = df["Low"]
            close_prices[asset] = df["Close"]
            adj_close_prices[asset] = df["Adj Close"]
            volumes[asset] = df["Volume"]

            metadata.append(
                {
                    "Asset": asset,
                    "Ticker": ticker,
                    "Rows": len(df),
                    "Start": df.index.min(),
                    "End": df.index.max(),
                }
            )

        except Exception as exc:
            failures.append((asset, ticker, str(exc)))
            log(f"  FAILED -> {asset} ({ticker}): {exc}")

    if failures:
        log(f"{len(failures)}/{len(UNIVERSE)} assets failed to download:")
        for asset, ticker, reason in failures:
            log(f"    {asset} ({ticker}): {reason}")

    if not metadata:
        log("No assets downloaded successfully — aborting without touching Data/.")
        return 1

    datasets = [
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        adj_close_prices,
        volumes,
    ]

    for df in datasets:
        df.sort_index(inplace=True)
        df.index.name = "Date"

    open_prices.to_parquet(OUTPUT_DIR / "open.parquet")
    high_prices.to_parquet(OUTPUT_DIR / "high.parquet")
    low_prices.to_parquet(OUTPUT_DIR / "low.parquet")
    close_prices.to_parquet(OUTPUT_DIR / "close.parquet")
    adj_close_prices.to_parquet(OUTPUT_DIR / "adj_close.parquet")
    volumes.to_parquet(OUTPUT_DIR / "volume.parquet")

    metadata_df = pd.DataFrame(metadata).sort_values("Rows", ascending=False)
    metadata_df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    metadata_df.to_csv(OUTPUT_DIR / "audit.csv", index=False)

    log(
        f"Done — {len(metadata)}/{len(UNIVERSE)} assets updated, "
        f"{len(failures)} failed. Files written to {OUTPUT_DIR}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
