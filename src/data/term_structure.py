"""
data/term_structure.py — Carry construction: real quoted calendar spreads (37 assets)
+ back-differenced ICE-softs proxy (5 assets), joined against data.continuous_curve's
own front-contract assignment.

See references/carry_implementation_recipe.md for the full recipe and reasoning, and
WORKFLOW.md Phase 4 for the same decisions logged in the project's running history.

Formula (LiveCattle prototype, reused verbatim - WORKFLOW.md Phase 4):
    Carry = (F_near - F_far) / F_near * 365 / (days between near and far expiry)

Real-quote path (37 assets): F_near - F_far comes directly from
term_structure_spreads.parquet's own quoted close for the FRONT calendar spread
specifically - near_contract_symbol == that date's front_contract_symbol
(data.continuous_curve's own roll-rule assignment, read from
Data/continuous_futures.parquet, so carry's "front" matches every other signal's, not
a second, competing definition). Where multiple far legs are quoted against the same
front that day (common - CME/ICE list many combinations), the nearest-expiry far leg
is kept - the recipe's own error-analysis is scoped to exactly this front spread, not
far-dated/thin pairs. F_near itself comes from term_structure.parquet's own outright
close for that (date, near_contract_symbol), per the recipe.

Keyed off near_contract_symbol/far_contract_symbol (disambiguated by maturity year via
the definition-schema join already done in src/data/databento_transform.py), never
the raw spread_symbol string - that string is ambiguous across different decades and
will silently merge unrelated instruments if grouped on directly (see
databento/DATA_QUALITY_REPORT.md).

Proxy path (5 ICE softs: Coffee, Sugar, Cocoa, Cotton, OrangeJuice - zero real spread
data exists for any of them, ICE's spread/butterfly schema unsupported by the
transform): F_near - F_far computed by back-differencing two separately-timed OUTRIGHT
closes instead (term_structure.parquet only), same near/far selection (front + the
nearest next-listed contract with a real outright print that day, found by walking
data.continuous_curve's own contract chain). This is a real, measured
non-synchronous-pricing bias, not a data defect - see
compare_real_vs_proxy_spread_error and references/carry_implementation_recipe.md's own
measurement table. Every value this path produces is proxy-derived - see
is_proxy_flags, never blended unlabeled with the 37 real-quote assets (CLAUDE.md
Rule 4).

Expiry is known only to month granularity in this schema (no exact CME/ICE calendar
date) - approximated as 30.44 days/month (the same approximation the LiveCattle
prototype used), consistent across every asset, real or proxy.

Frozen at SPREAD_DATA_END (2026-07-13) - term_structure_spreads.parquet is a one-time
historical Databento pull, not a live feed (the daily forward-capture job only pulls
outright prices, no spread tickers exist via yfinance). A live/forward-going version
of this signal needs the forward-data-gap resolved first (WORKFLOW.md Phase 4,
deliberately deferred) - this module does not silently extend past that date, it
simply has no real-spread rows to join against past it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data.continuous_curve import MONTH_CODES, build_contract_chain

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

# Zero real spread/butterfly data for any of these 5 (ICE's combo-instrument schema
# unsupported by the transform, already known) - confirmed directly against
# term_structure_spreads.parquet (0 rows for any of them), not assumed.
ICE_PROXY_ASSETS = ["Coffee", "Sugar", "Cocoa", "Cotton", "OrangeJuice"]

DAYS_PER_MONTH = 30.44
ANNUALIZATION_DAYS = 365

# term_structure_spreads.parquet's real, quoted history - a one-time pull, not a live
# feed. See module docstring.
SPREAD_DATA_END = "2026-07-13"


def _month_index(year, code):
    """Vectorized (pd.Series) or scalar month index: year*12 + month-of-year (0-11).
    Only ever compared to another value from this same function, so the arbitrary
    zero-point doesn't matter - only the DIFFERENCE (months between near and far) is
    ever used."""
    if isinstance(code, pd.Series):
        month = code.map(lambda c: MONTH_CODES.index(c))
    else:
        month = MONTH_CODES.index(code)
    return year * 12 + month


def load_front_contract_symbols(data_dir=None) -> pd.DataFrame:
    """front_contract_symbol per (date, asset), pivoted wide - data.continuous_curve's
    own roll-rule assignment (Data/continuous_futures.parquet), reused so carry's
    "front" contract definition matches every other signal's, not a second, competing
    definition of "front."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_parquet(
        data_dir / "continuous_futures.parquet", columns=["date", "asset", "front_contract_symbol"]
    )
    return df.pivot(index="date", columns="asset", values="front_contract_symbol")


def load_outrights(data_dir=None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    return pd.read_parquet(data_dir / "term_structure.parquet")


def load_real_spreads(data_dir=None) -> pd.DataFrame:
    """term_structure_spreads.parquet, restricted to same-underlying calendar
    spreads (near_root == far_root) - excludes the rarer inter-commodity spread rows
    (e.g. LiveCattle/LeanHogs cross spreads), which aren't a carry measure for either
    leg."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_parquet(data_dir / "term_structure_spreads.parquet")
    return df[df["near_root"] == df["far_root"]].copy()


def _front_calendar_spread(asset: str, front_symbols: pd.Series, spreads: pd.DataFrame) -> pd.DataFrame:
    """Restrict `spreads` to the FRONT calendar spread specifically, one row per date:
    near_contract_symbol == that date's front (data.continuous_curve's assignment),
    far_contract_symbol = the nearest-expiry leg quoted against it that day (some
    dates have several far legs quoted against the same front - CME/ICE list many
    combinations; only the nearest is the "front spread" the recipe's error analysis
    and this carry measure both use).

    Returns columns: date, near_contract_symbol, far_contract_symbol, close (the
    spread's own quoted close, F_near - F_far), near_month_idx, far_month_idx. Empty
    DataFrame (same columns) if nothing matches.
    """
    cols = ["date", "near_contract_symbol", "far_contract_symbol", "close", "near_month_idx", "far_month_idx"]
    asset_spreads = spreads[spreads["asset"] == asset]
    if asset_spreads.empty:
        return pd.DataFrame(columns=cols)

    asset_spreads = asset_spreads.assign(
        near_month_idx=_month_index(asset_spreads["near_expiry_year"], asset_spreads["near_expiry_code"]),
        far_month_idx=_month_index(asset_spreads["far_expiry_year"], asset_spreads["far_expiry_code"]),
    )

    front_series = front_symbols.dropna().copy()
    front_series.index.name = "date"  # caller's index may be unnamed - pin it explicitly
    front_df = front_series.rename("front_contract_symbol").reset_index()
    if front_df.empty:
        return pd.DataFrame(columns=cols)

    matched = asset_spreads.merge(front_df, on="date", how="inner")
    matched = matched[matched["near_contract_symbol"] == matched["front_contract_symbol"]]
    if matched.empty:
        return pd.DataFrame(columns=cols)

    matched = matched.sort_values(["date", "far_month_idx"])
    nearest = matched.groupby("date", as_index=False).first()
    return nearest[cols]


def real_spread_carry(asset: str, front_symbols: pd.Series, spreads: pd.DataFrame, outrights: pd.DataFrame) -> pd.Series:
    """Annualized carry for one real-quote asset, date-indexed (see module docstring
    for the formula). NaN wherever no front-calendar-spread quote or no matching
    outright close exists that day - left as NaN, not fabricated (CLAUDE.md Rule 4)."""
    front_spread = _front_calendar_spread(asset, front_symbols, spreads)
    if front_spread.empty:
        return pd.Series(np.nan, index=front_symbols.index, name=asset)

    asset_outrights = outrights[outrights["asset"] == asset][["date", "contract_symbol", "close"]]
    near_close = asset_outrights.rename(
        columns={"contract_symbol": "near_contract_symbol", "close": "near_outright_close"}
    )
    merged = front_spread.merge(near_close, on=["date", "near_contract_symbol"], how="left")

    days = (merged["far_month_idx"] - merged["near_month_idx"]) * DAYS_PER_MONTH
    carry = (merged["close"] / merged["near_outright_close"]) * (ANNUALIZATION_DAYS / days)
    carry = carry.where(days > 0)

    result = pd.Series(carry.values, index=merged["date"], name=asset)
    result = result[~result.index.duplicated(keep="first")]
    return result.reindex(front_symbols.index)


def proxy_carry(asset: str, front_symbols: pd.Series, outrights: pd.DataFrame) -> pd.Series:
    """Back-differenced proxy carry for one ICE-softs asset (no real spread data
    exists at all). Near/far selection mirrors the real-quote path (front + nearest
    next-listed contract with a real outright print that day), sourced entirely from
    term_structure.parquet - no spread instrument exists to quote F_near - F_far
    directly for these 5 assets, so it's computed from two separately-timed outright
    closes instead. See module docstring and
    references/carry_implementation_recipe.md for the measured error this
    introduces.

    "Next listed contract" reuses data.continuous_curve.build_contract_chain (the
    same real-expiry-ordered chain every other signal's roll logic is built from),
    walked forward from front for the nearest entry with a real close that date - the
    same forward-scan principle assign_front_contract already uses internally for its
    own "next contract," re-derived here since that scan is coupled to volume/roll
    logic there, not exposed standalone.
    """
    asset_df = outrights[outrights["asset"] == asset]
    if asset_df.empty:
        return pd.Series(np.nan, index=front_symbols.index, name=asset)

    chain = build_contract_chain(asset_df)
    chain_index = {c: i for i, c in enumerate(chain)}
    close_wide = asset_df.pivot(index="date", columns="contract_symbol", values="close")

    expiry = asset_df[["contract_symbol", "expiry_year", "expiry_code"]].drop_duplicates().set_index("contract_symbol")
    month_idx = {c: _month_index(expiry.loc[c, "expiry_year"], expiry.loc[c, "expiry_code"]) for c in chain}

    values = {}
    for date, front in front_symbols.dropna().items():
        if front not in chain_index or date not in close_wide.index:
            continue
        row = close_wide.loc[date]
        near_close = row.get(front)
        if near_close is None or pd.isna(near_close):
            continue

        far = None
        for candidate in chain[chain_index[front] + 1:]:
            candidate_close = row.get(candidate)
            if candidate_close is not None and pd.notna(candidate_close):
                far = candidate
                break
        if far is None:
            continue

        days = (month_idx[far] - month_idx[front]) * DAYS_PER_MONTH
        if days <= 0:
            continue
        values[date] = (near_close - row[far]) / near_close * (ANNUALIZATION_DAYS / days)

    return pd.Series(values, name=asset).reindex(front_symbols.index)


def build_carry_panel(included_assets, data_dir=None) -> tuple:
    """Full (T x N) annualized carry panel for `included_assets` (the project's
    standard ADV-filtered universe), plus an asset-level is_proxy flag (pd.Series,
    bool, indexed by asset - carry's proxy-ness is fixed per asset, not per date, so
    an asset-level flag is the natural, unambiguous label, always visible alongside
    any per-asset view, never silently merged away).

    37 assets get carry from real, exchange-quoted calendar spreads
    (term_structure_spreads.parquet); the 5 ICE softs (ICE_PROXY_ASSETS) get the
    back-differenced proxy (term_structure.parquet outrights only). Assets absent from
    Data/continuous_futures.parquet (no front-contract assignment to key off) are
    silently dropped from the returned columns, not raised - callers should compare
    `carry.columns` against `included_assets` if they need to know what was dropped.
    """
    front = load_front_contract_symbols(data_dir)
    front = front[[a for a in included_assets if a in front.columns]]

    outrights = load_outrights(data_dir)
    spreads = load_real_spreads(data_dir)

    series = {}
    for asset in front.columns:
        if asset in ICE_PROXY_ASSETS:
            series[asset] = proxy_carry(asset, front[asset], outrights)
        else:
            series[asset] = real_spread_carry(asset, front[asset], spreads, outrights)

    carry = pd.DataFrame(series)
    is_proxy = pd.Series({a: (a in ICE_PROXY_ASSETS) for a in carry.columns}, name="is_proxy")
    return carry, is_proxy


def build_databento_only_continuous_curve(asset: str, data_dir=None) -> pd.DataFrame:
    """Continuous back-adjusted curve for an asset with real Databento per-contract
    data but NO core yfinance-based continuous series (currently: SOFR only - see
    DATA_SCHEMA.md §1, "no viable Yahoo continuous ticker exists"). Reuses
    `continuous_curve.build_continuous_curve` directly on term_structure.parquet's
    own outright rows for this asset - same roll-detection (volume crossover) and
    ratio back-adjustment logic every other asset's continuous series is built
    from (CLAUDE.md Rule 6), just sourced from Databento outrights instead of the
    yfinance-backed panel `load_front_contract_symbols` reads. That exclusion from
    the core panel was about the yfinance ticker specifically - the real
    per-contract data needed for a roll-rule/back-adjustment construction was
    captured all along via `jobs/capture_term_structure.py`.

    Returns the same shape as `continuous_curve.build_continuous_curve`: front_
    contract_symbol, is_roll_date, raw_open/high/low/close, volume, adj_open/
    high/low/close - indexed by date.
    """
    from data.continuous_curve import build_continuous_curve

    outrights = load_outrights(data_dir)
    asset_df = outrights[outrights["asset"] == asset]
    if asset_df.empty:
        raise ValueError(f"no term_structure.parquet rows for asset={asset!r}")
    return build_continuous_curve(asset_df)


DATABENTO_ONLY_SPREAD_SCALE = {
    # SOFR's own quoted calendar-spread `close` is on a DIFFERENT scale than its
    # outright `close` - checked directly, not assumed: on 2026-07-13, the
    # SR3Z27-SR3Z28 spread quoted close=-8.5, but the two outrights' own closes
    # were 95.925 and 96.01 (a genuine difference of -0.085, exactly 1/100th of
    # the quoted spread value). SOFR's calendar spreads are quoted directly in
    # basis points of rate (1 tick = 0.01 price point = 1bp), not in the same
    # "100 - rate" price-point units the outrights use - every other real-quote
    # asset in this project's carry panel (Treasuries, commodities) has its
    # spread and outright closes on the SAME scale, confirmed by their carry
    # values already landing in a sane 0.03%-1.4% annualized range; SOFR's did
    # not (mean -55% annualized, min -8.9x) until this scaling was applied,
    # which brings it in line (mean -0.55%, comparable in magnitude to US_2Y/
    # US_10Y). A genuinely different exchange quoting convention, not a bug in
    # the transform pipeline.
    "SOFR": 0.01,
}


def build_databento_only_carry(asset: str, data_dir=None) -> pd.Series:
    """Real-quote carry (see module docstring's formula) for a Databento-only asset
    (see `build_databento_only_continuous_curve`) - always the real-spread path,
    never the ICE proxy, since this asset has genuine exchange-quoted calendar
    spreads (checked directly for SOFR: real spread rows 2018-05-07 to
    SPREAD_DATA_END). Front-contract assignment comes from this function's own
    continuous-curve construction rather than `load_front_contract_symbols`
    (which only covers assets in `continuous_futures.parquet`).

    Rescales this asset's own spread `close` column by `DATABENTO_ONLY_SPREAD_SCALE`
    (default 1.0 - identity - for any asset not listed there) before computing
    carry, since a combo instrument's quoting convention isn't guaranteed to share
    its outright's own price-point scale - see that dict's own docstring for SOFR's
    confirmed 100x mismatch."""
    front_symbols = build_databento_only_continuous_curve(asset, data_dir)["front_contract_symbol"]
    spreads = load_real_spreads(data_dir)
    scale = DATABENTO_ONLY_SPREAD_SCALE.get(asset, 1.0)
    if scale != 1.0:
        spreads = spreads.copy()
        spreads.loc[spreads["asset"] == asset, "close"] *= scale
    outrights = load_outrights(data_dir)
    return real_spread_carry(asset, front_symbols, spreads, outrights)


def compare_real_vs_proxy_spread_error(asset: str, front_symbols: pd.Series, spreads: pd.DataFrame, outrights: pd.DataFrame) -> pd.Series:
    """For a REAL-quote asset only: measures how far the back-differenced proxy
    (near_outright_close - far_outright_close) would have been from the real quoted
    front-calendar-spread close, as a fraction of the near leg's own price - the same
    quantity references/carry_implementation_recipe.md's error table reports (median/
    90th/99th/max), reproduced here from live data rather than just cited from that
    design session. Restricted to the front calendar spread specifically, same
    selection as real_spread_carry - far-dated/thin pairs are not representative of
    what carry actually trades.

    Returns a date-indexed pd.Series of absolute error (fraction, not %), dropna'd.
    """
    front_spread = _front_calendar_spread(asset, front_symbols, spreads)
    if front_spread.empty:
        return pd.Series(dtype=float, name=asset)

    asset_outrights = outrights[outrights["asset"] == asset][["date", "contract_symbol", "close"]]
    near_close = asset_outrights.rename(columns={"contract_symbol": "near_contract_symbol", "close": "near_outright_close"})
    far_close = asset_outrights.rename(columns={"contract_symbol": "far_contract_symbol", "close": "far_outright_close"})

    merged = front_spread.merge(near_close, on=["date", "near_contract_symbol"], how="left")
    merged = merged.merge(far_close, on=["date", "far_contract_symbol"], how="left")

    proxy_spread = merged["near_outright_close"] - merged["far_outright_close"]
    error = (proxy_spread - merged["close"]).abs() / merged["near_outright_close"].abs()

    result = pd.Series(error.values, index=merged["date"], name=asset)
    return result[~result.index.duplicated(keep="first")].dropna()
