"""
Decodes raw Databento `ohlcv-1d` + instrument-`definition` archives
(`Data/databento_raw/*.zip`, from CME Globex `GLBX.MDP3` and ICE Futures US
`IFUS.IMPACT`) into the 6 `Data/term_structure*.parquet` tables (outrights,
spreads, butterflies, condors, averages, packs) plus a per-run manifest.

**The core problem this module solves**: raw exchange symbology encodes a
contract's expiry with a single decade-ambiguous digit (`LEQ5` could be 2015
or 2025), and a multi-leg combo instrument's symbol only names its legs, not
which one anchors the instrument's own listed maturity. The fix — confirmed
directly against real definition-schema rows, not assumed from documentation
alone — is that a combo instrument's own `maturity_year`/`maturity_month`
always matches whichever leg is chronologically nearest, regardless of which
root that leg belongs to. Every other leg's absolute year is then a small
non-negative modular offset from that anchor (`anchor_year + ((leg_digit -
anchor_digit) % 10)` — legs are always listed at or after the anchor
chronologically). When multiple legs share the anchor's month code (a
same-month cross-commodity spread, or a same-month "bundle" butterfly like
SOFR's `:BB`), each candidate is tried as anchor and the one producing the
smallest total leg-to-leg year span wins — a real near-dated combo's legs are
never many years apart, so the tightest-span resolution is the economically
sensible one.

**Tooling**: polars end-to-end — read, join, filter, leg-resolution, and the
final merge-with-existing-data + parquet write. The combo-leg resolution
(classification, parsing, anchor resolution) is batched by unique
`(raw_symbol, maturity_year, maturity_month)` key rather than run once per
row — that triple fully determines the parse outcome, so a market with
hundreds of thousands of raw combo rows reduces to a few thousand unique
instruments to actually resolve, joined back onto the full daily series
afterward. Only the manifest CSV uses pandas, matching every other manifest
file's convention project-wide; a Parquet Datetime(ns) column round-trips
into pandas as `datetime64[ns]` with no behavior change for downstream
pandas-based consumers.

Root/exchange mapping is reused directly from `jobs/capture_term_structure.py`'s
own `UNIVERSE` dict, since that module's `contract_symbol` convention
(`{root}{month_code}{2-digit-year}.{exchange}`) is exactly the identity this
transform must also produce, for "Databento wins on overlap" deduplication to
actually collide correctly between the two sources.

Other data-quality decisions worth knowing about:
- Databento's `UNDEF_PRICE` sentinel (`int64::MAX`) can leak into `ohlcv-1d`
  price fields directly — filtered on the raw pre-scaling integer, before the
  ÷1e9 rescale, for every instrument class.
- Outright rows get an EXACT-zero (not merely non-positive) filter on any of
  open/high/low/close — preserves genuine negative prints (e.g. WTI's real
  April 2020 settlement) while dropping incomplete/fabricated bars. Applied
  to outrights only — spreads/butterflies/etc. legitimately trade negative.
- ICE assets have a second, non-representative `_Z`-suffixed instrument per
  contract month, excluded by `raw_symbol` suffix, and use a structurally
  different `leg_instrument_id`-based spread/butterfly schema this transform
  does not attempt to resolve (logged and skipped, not guessed).
"""

import io
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

# This repo's own path contains non-ASCII characters (Greek "Υπολογιστής") --
# on Windows' default console codepage (cp1252), any print() containing a
# path built from __file__ crashes with UnicodeEncodeError. Every research/
# *.py script already guards against this; this module (and
# continuous_curve's sibling CLI) didn't, and it's a real crash, not
# hypothetical -- found live when this module's own final status print hit
# it after weekly_databento_pipeline.py called run() directly (not through
# transform_databento.py's CLI wrapper).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "jobs"))
from capture_term_structure import UNIVERSE, MONTH_CODES  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "Data" / "databento_raw"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"
MANIFEST_PATH = DATA_DIR / "databento_transform_manifest.csv"

# ICE (IFUS.IMPACT) roots are exactly the assets UNIVERSE tags "NYB" -- the 5
# softs. Confirmed no other UNIVERSE entry uses that exchange code.
ICE_ROOTS = {root for _asset, (root, exch) in UNIVERSE.items() if exch == "NYB"}
ROOT_TO_ASSET = {root: asset for asset, (root, _exch) in UNIVERSE.items()}
ROOT_TO_EXCHANGE = {root: exch for _asset, (root, exch) in UNIVERSE.items()}

UNDEF_PRICE = 9_223_372_036_854_775_807  # signed int64 max -- Databento's "not applicable" sentinel
PRICE_SCALE = 1_000_000_000  # fixed-point, /1e9 = real price

# pandas reads a Parquet DATE32 column back as object-dtype Python
# `datetime.date` values, not `datetime64[ns]` -- every downstream consumer
# in this project expects a real pandas Timestamp column, so dates are
# carried as Datetime(ns) throughout this module, not Date, even though only
# the date (never time-of-day) is ever meaningful here.
DATE_DTYPE = pl.Datetime("ns")

OHLCV_COLUMNS = ["ts_event", "instrument_id", "open", "high", "low", "close", "volume", "symbol"]
DEFINITION_COLUMNS = ["instrument_id", "raw_symbol", "instrument_class", "asset", "maturity_year", "maturity_month"]

_DATE_RE = re.compile(r"(\d{8})")
_LEG_RE = re.compile(r"^([FGHJKMNQUVXZ])(\d{1,2})$")
_ROOTED_LEG_RE = re.compile(r"^([A-Z0-9]{1,4})([FGHJKMNQUVXZ])(\d{1,2})$")

OUTPUT_SCHEMAS = {
    "term_structure": ["date", "open", "high", "low", "close", "volume", "asset", "contract_symbol", "root", "exchange", "expiry_code", "expiry_year"],
    "term_structure_spreads": ["date", "open", "high", "low", "close", "volume", "asset", "spread_symbol",
                                "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
                                "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code"],
    "term_structure_butterflies": ["date", "open", "high", "low", "close", "volume", "asset", "butterfly_symbol",
                                    "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
                                    "mid_root", "mid_contract_symbol", "mid_expiry_year", "mid_expiry_code",
                                    "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code"],
    "term_structure_condors": ["date", "open", "high", "low", "close", "volume", "asset", "condor_symbol",
                                "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
                                "mid1_root", "mid1_contract_symbol", "mid1_expiry_year", "mid1_expiry_code",
                                "mid2_root", "mid2_contract_symbol", "mid2_expiry_year", "mid2_expiry_code",
                                "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code"],
    "term_structure_averages": ["date", "open", "high", "low", "close", "volume", "asset", "average_symbol",
                                 "root", "avg_years", "anchor_contract_symbol", "anchor_expiry_year", "anchor_expiry_code"],
    "term_structure_packs": ["date", "open", "high", "low", "close", "volume", "asset", "pack_symbol", "pack_type",
                              "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
                              "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code"],
}


# ---------------------------------------------------------------------------
# Raw file I/O -- in-memory zstd decompression (no filesystem extraction:
# polars auto-detects zstd from the byte stream, and per-file filesystem
# writes are what actually dominates wall time at this file count, not
# parsing)
# ---------------------------------------------------------------------------

def _zip_paths(root: str, is_ice: bool) -> tuple:
    if is_ice:
        prefix = f"ICE_Futures_US_{root}"
    else:
        prefix = f"CME_Globex_MDP3.0_{root}_FUT"
    return RAW_DIR / f"{prefix}_Definition.zip", RAW_DIR / f"{prefix}_OHLCV.zip"


def _read_zip_csvs(zip_path: Path, columns: list) -> pl.DataFrame:
    """Every `*.csv.zst` member, decompressed in-memory (polars auto-detects
    zstd from the byte stream), concatenated, with a `date` column parsed from
    each member's own filename (not `ts_event` -- avoids any UTC-session-
    boundary ambiguity)."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv.zst")]
        for name in names:
            raw = zf.read(name)
            try:
                df = pl.read_csv(io.BytesIO(raw), columns=columns)
            except Exception:
                continue
            if df.height == 0:
                continue
            m = _DATE_RE.search(name)
            if not m:
                continue
            df = df.with_columns(pl.lit(m.group(1)).str.strptime(DATE_DTYPE, "%Y%m%d").alias("date"))
            frames.append(df)
    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns} | {"date": DATE_DTYPE})
    return pl.concat(frames, how="vertical_relaxed")


def load_raw(root: str, is_ice: bool) -> tuple:
    def_zip, ohlcv_zip = _zip_paths(root, is_ice)
    if not def_zip.exists() or not ohlcv_zip.exists():
        return None, None
    ohlcv = _read_zip_csvs(ohlcv_zip, OHLCV_COLUMNS)
    definition = _read_zip_csvs(def_zip, DEFINITION_COLUMNS)
    return ohlcv, definition


# ---------------------------------------------------------------------------
# Join: exact (date, instrument_id) first, then a per-instrument backward
# asof fallback for rows whose definition stopped being republished before
# their last real print (a CME feed quirk near expiry). raw_symbol/
# instrument_class/maturity_year/maturity_month are immutable for a given
# instrument_id's whole life, so reusing an OLDER definition record is
# exactly as correct as an exact-date one, not a guess.
# ---------------------------------------------------------------------------

def dedup_definition(definition: pl.DataFrame) -> pl.DataFrame:
    """definition is an event-sourced schema (one row per republish, not
    deduplicated per day) -- collapse to one row per (date, instrument_id)."""
    return definition.unique(subset=["date", "instrument_id"], keep="first")


def join_with_fallback(ohlcv: pl.DataFrame, definition: pl.DataFrame) -> tuple:
    definition = dedup_definition(definition)
    exact = ohlcv.join(definition, on=["date", "instrument_id"], how="left")

    matched = exact.filter(pl.col("raw_symbol").is_not_null())
    missing = exact.filter(pl.col("raw_symbol").is_null())

    if missing.height > 0:
        fallback_input = missing.select(ohlcv.columns).sort(["instrument_id", "date"])
        def_sorted = definition.sort(["instrument_id", "date"])
        fallback = fallback_input.join_asof(def_sorted, on="date", by="instrument_id", strategy="backward")
        joined = pl.concat([matched, fallback], how="vertical_relaxed")
    else:
        joined = matched

    n_unmatched = joined.filter(pl.col("raw_symbol").is_null()).height
    return joined, n_unmatched


# ---------------------------------------------------------------------------
# Sentinel / price filters
# ---------------------------------------------------------------------------

def filter_sentinel(joined: pl.DataFrame) -> tuple:
    """UNDEF_PRICE (int64 max) can appear directly in raw OHLC integer
    fields -- filtered BEFORE the /1e9 rescale (dividing the sentinel would
    produce a large but finite, deceptively "valid-looking" positive price).
    Applies to every instrument class -- a sentinel is never real data."""
    before = joined.height
    clean = joined.filter(
        (pl.col("open") != UNDEF_PRICE) & (pl.col("high") != UNDEF_PRICE)
        & (pl.col("low") != UNDEF_PRICE) & (pl.col("close") != UNDEF_PRICE)
    )
    return clean, before - clean.height


def scale_prices(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("open") / PRICE_SCALE).alias("open"),
        (pl.col("high") / PRICE_SCALE).alias("high"),
        (pl.col("low") / PRICE_SCALE).alias("low"),
        (pl.col("close") / PRICE_SCALE).alias("close"),
    ])


def filter_outright_exact_zero(df: pl.DataFrame) -> tuple:
    """Drop outright rows with an EXACT zero (not merely non-positive) in any
    OHLC field -- an incomplete/fabricated bar, not a genuine price (a real
    negative print, e.g. WTI April 2020, never has an exact zero in any
    field). Scoped to outrights ONLY -- spreads/butterflies/condors trade
    negative legitimately."""
    before = df.height
    clean = df.filter((pl.col("open") != 0) & (pl.col("high") != 0) & (pl.col("low") != 0) & (pl.col("close") != 0))
    return clean, before - clean.height


def filter_ice_z_variant(df: pl.DataFrame) -> tuple:
    """ICE only: a second, non-representative instrument per contract month
    (`raw_symbol` suffixed `_Z`, ~97% sentinel/non-positive, not the tradable
    outright) -- excluded by suffix."""
    before = df.height
    clean = df.filter(~pl.col("raw_symbol").str.ends_with("_Z"))
    return clean, before - clean.height


# ---------------------------------------------------------------------------
# Leg resolution -- the anchor-leg-plus-modular-offset algorithm described in
# the module docstring above.
# ---------------------------------------------------------------------------

def _combo_column_dtype(col: str) -> pl.DataType:
    if col == "date":
        return DATE_DTYPE
    if col in ("open", "high", "low", "close"):
        return pl.Float64
    if col == "volume" or col.endswith("_year") or col == "avg_years":
        return pl.Int64
    return pl.Utf8


def _combo_pl_df(rows: list, columns: list) -> pl.DataFrame:
    """Build a polars DataFrame from a plain Python dict list, with an
    explicit column order/schema so an EMPTY list still produces a
    correctly-typed zero-row frame (polars can't infer a schema from
    nothing) -- matters so a later concat against a populated frame never
    hits a schema mismatch."""
    if not rows:
        return pl.DataFrame(schema={c: _combo_column_dtype(c) for c in columns})
    return pl.DataFrame(rows).select(columns)


def _month_index(code: str) -> int:
    return MONTH_CODES.index(code) + 1


def _parse_single_leg(token: str, default_root: str = None):
    """Parse one leg token into (root, month_code, digit_str). `token` is
    either bare `{month_code}{1-2 digit}` (root inherited from
    `default_root`) or `{root}{month_code}{1-2 digit}` (root embedded)."""
    m = _LEG_RE.match(token)
    if m and default_root is not None:
        code, digits = m.group(1), m.group(2)
        return default_root, code, digits
    m = _ROOTED_LEG_RE.match(token)
    if m:
        root, code, digits = m.group(1), m.group(2), m.group(3)
        return root, code, digits
    return None


def _resolve_absolute_year(digit_str: str, anchor_year: int, anchor_digit: int) -> int:
    """A 2-digit trailing code is an unambiguous absolute year (this
    project's pull window, 2010-2036, never needs cross-century
    interpretation). A 1-digit code is resolved as a non-negative modular
    offset from the anchor leg -- legs are always listed at or after the
    anchor chronologically."""
    if len(digit_str) == 2:
        return 2000 + int(digit_str)
    digit = int(digit_str)
    return anchor_year + ((digit - anchor_digit) % 10)


def _root_exchange(root: str):
    """(canonical root, exchange) for a leg's root, or None if it's a real
    but untracked product (logged as spread_unknown_root_exchange and
    skipped, never guessed)."""
    if root in ROOT_TO_EXCHANGE:
        return root, ROOT_TO_EXCHANGE[root]
    return None


def _contract_symbol(root: str, month_code: str, year: int, exchange: str) -> str:
    return f"{root}{month_code}{year % 100:02d}.{exchange}"


# Vectorized lookup for the outright table's `maturity_month` (1-12) ->
# single-char month code, joined rather than computed via a per-row Python
# callback.
MONTH_LOOKUP_DF = pl.DataFrame({"maturity_month": list(range(1, 13)), "expiry_code": list(MONTH_CODES)})


# --- combo-symbol classification -------------------------------------------------

def classify_combo(raw_symbol: str) -> str:
    if raw_symbol.startswith("UD:"):
        return "user_defined"
    if ":" not in raw_symbol:
        return "plain"
    marker = raw_symbol.split(":", 1)[1].split(" ", 1)[0]
    return {
        "BF": "butterfly", "BB": "butterfly3",
        "CF": "condor", "DF": "condor4",
        "AB": "average",
        "SB": "pack",
        "C1": "crack",
    }.get(marker, "labeled_intercommodity")


def _parse_plain_legs(raw_symbol: str, default_root: str):
    parts = raw_symbol.split("-")
    if len(parts) != 2:
        return None
    legs = [_parse_single_leg(p, default_root=default_root) for p in parts]
    if any(leg is None for leg in legs):
        return None
    return legs


def _parse_labeled_intercommodity_legs(raw_symbol: str):
    # "{ROOT1}:{ROOT2} {leg1}-{leg2}" -- both legs bare month+digit.
    head, _, rest = raw_symbol.partition(":")
    marker_root, _, rest = rest.partition(" ")
    parts = rest.split("-")
    if len(parts) != 2:
        return None
    leg1 = _parse_single_leg(parts[0].strip(), default_root=head)
    leg2 = _parse_single_leg(parts[1].strip(), default_root=marker_root)
    if leg1 is None or leg2 is None:
        return None
    return [leg1, leg2]


def _parse_crack_legs(raw_symbol: str):
    # "{ROOT1}:C1 {rootA}[ {legA}]-{rootB} {legB}" -- leg A's month is
    # optional (shares leg B's month when absent -- WTI's own convention;
    # Brent generalizes to an explicit leg-A month).
    _, _, rest = raw_symbol.partition(":C1 ")
    if "-" not in rest:
        return None
    left, _, right = rest.partition("-")
    left = left.strip()
    right = right.strip()
    right_parts = right.split(" ", 1)
    if len(right_parts) != 2:
        return None
    root_b, leg_b_tok = right_parts
    leg_b = _parse_single_leg(leg_b_tok, default_root=root_b)
    if leg_b is None:
        return None

    left_parts = left.split(" ", 1)
    if len(left_parts) == 2:
        root_a, leg_a_tok = left_parts
        leg_a = _parse_single_leg(leg_a_tok, default_root=root_a)
        if leg_a is None:
            return None
    else:
        root_a = left_parts[0]
        # No explicit month for leg A -- shares leg B's month code/digit.
        leg_a = (root_a, leg_b[1], leg_b[2])
    return [leg_a, leg_b]


def _parse_dash_legs(raw_symbol: str, marker: str, n_legs: int, default_root: str):
    # "{ROOT}:{marker} {leg1}-{leg2}[-{leg3}]" -- bare month+digit legs.
    _, _, rest = raw_symbol.partition(f":{marker} ")
    parts = rest.split("-")
    if len(parts) != n_legs:
        return None
    legs = [_parse_single_leg(p.strip(), default_root=default_root) for p in parts]
    if any(leg is None for leg in legs):
        return None
    return legs


def _parse_concat_legs(raw_symbol: str, marker: str, n_legs: int, default_root: str):
    # "{ROOT}:{marker} {leg1}{leg2}{leg3}{leg4}" -- no separator, each leg
    # exactly 2 chars (month_code + single digit; concatenated grammars never
    # carry a 2-digit year in this dataset).
    _, _, rest = raw_symbol.partition(f":{marker} ")
    rest = rest.strip()
    if len(rest) != n_legs * 2:
        return None
    legs = []
    for i in range(n_legs):
        tok = rest[i * 2: i * 2 + 2]
        leg = _parse_single_leg(tok, default_root=default_root)
        if leg is None:
            return None
        legs.append(leg)
    return legs


def _parse_pack_legs(raw_symbol: str, default_root: str):
    # "{ROOT}:SB {PK|NY} {leg1}-{leg2}" -- SOFR-only; a type token (PK, 2Y,
    # 3Y, ...) then a plain 2-leg dash pair.
    _, _, rest = raw_symbol.partition(":SB ")
    parts = rest.split(" ", 1)
    if len(parts) != 2:
        return None
    pack_type, leg_part = parts
    legs = _parse_plain_legs(leg_part.strip(), default_root)
    if legs is None:
        return None
    return pack_type, legs


def _parse_average_leg(raw_symbol: str, default_root: str):
    # "{ROOT}:AB {avg_years}Y {leg}" -- single anchor leg.
    _, _, rest = raw_symbol.partition(":AB ")
    parts = rest.split(" ", 1)
    if len(parts) != 2 or not parts[0].endswith("Y"):
        return None
    avg_years_str, leg_tok = parts
    try:
        avg_years = int(avg_years_str[:-1])
    except ValueError:
        return None
    leg = _parse_single_leg(leg_tok.strip(), default_root=default_root)
    if leg is None:
        return None
    return avg_years, leg


def _resolve_with_anchor(legs, anchor_idx: int, maturity_year: int):
    anchor_digit = int(legs[anchor_idx][2])
    resolved = []
    for root, code, digits in legs:
        if len(digits) == 2:
            year = 2000 + int(digits)
        else:
            year = _resolve_absolute_year(digits, maturity_year, anchor_digit)
        resolved.append((root, code, year))
    return resolved


def resolve_legs(legs, maturity_year: int, maturity_month: int):
    """Anchor = whichever leg's month_code matches the instrument's own
    maturity_month. Returns a list of (root, month_code, absolute_year), same
    order as input, or None if no leg matches (never guessed).

    When MULTIPLE legs share the anchor month code (a same-month cross-
    commodity spread, or a same-month "bundle" butterfly like SOFR's `:BB`),
    each candidate is tried as the anchor and the one producing the SMALLEST
    total year-span across all legs wins -- see the module docstring for why."""
    anchor_code = MONTH_CODES[maturity_month - 1]
    candidates = [i for i, leg in enumerate(legs) if leg[1] == anchor_code and len(leg[2]) == 1]

    if not candidates:
        if all(len(leg[2]) == 2 for leg in legs):
            return [(root, code, 2000 + int(digits)) for root, code, digits in legs]
        return None

    if len(candidates) == 1:
        return _resolve_with_anchor(legs, candidates[0], maturity_year)

    best, best_span = None, None
    for idx in candidates:
        resolved = _resolve_with_anchor(legs, idx, maturity_year)
        years = [y for _, _, y in resolved]
        span = max(years) - min(years)
        if best_span is None or span < best_span:
            best, best_span = resolved, span
    return best


def _legs_to_symbols(resolved_legs):
    """(root, contract_symbol, expiry_year, expiry_code) per leg, sorted
    chronologically, or None if any leg's root is untracked."""
    out = []
    for root, code, year in resolved_legs:
        re_ = _root_exchange(root)
        if re_ is None:
            return None
        canon_root, exch = re_
        out.append({
            "root": canon_root, "contract_symbol": _contract_symbol(canon_root, code, year, exch),
            "expiry_year": year, "expiry_code": code, "_sort_key": year * 12 + _month_index(code),
        })
    out.sort(key=lambda d: d["_sort_key"])
    return out


def _flatten_legs(labeled_legs: list) -> dict:
    """Flatten a list of `(label, leg)` pairs into `{label}_root`/
    `{label}_contract_symbol`/`{label}_expiry_year`/`{label}_expiry_code`
    fields -- shared by every combo bucket's leg-identity construction below."""
    out = {}
    for label, leg in labeled_legs:
        out[f"{label}_root"] = leg["root"]
        out[f"{label}_contract_symbol"] = leg["contract_symbol"]
        out[f"{label}_expiry_year"] = leg["expiry_year"]
        out[f"{label}_expiry_code"] = leg["expiry_code"]
    return out


def _resolve_combo_key(raw_symbol: str, maturity_year: int, maturity_month: int, root: str) -> dict:
    """Classification + leg-resolution outcome for one (raw_symbol,
    maturity_year, maturity_month) key -- these three fields plus the
    asset's own fixed `root` fully determine the parse outcome, so this is
    safe to call once per UNIQUE key rather than once per row (see the
    module docstring). Returns {"reject_reason": None or a reason string}
    plus, on success, {"bucket": ..., <flattened leg/identity fields>}."""
    combo_type = classify_combo(raw_symbol)

    if combo_type == "user_defined":
        return {"combo_type": combo_type, "reject_reason": "user_defined"}

    pack_type = None
    if combo_type == "plain":
        legs = _parse_plain_legs(raw_symbol, default_root=root)
        expected_n = 2
    elif combo_type == "labeled_intercommodity":
        legs = _parse_labeled_intercommodity_legs(raw_symbol)
        expected_n = 2
    elif combo_type == "crack":
        legs = _parse_crack_legs(raw_symbol)
        expected_n = 2
    elif combo_type == "butterfly":
        legs = _parse_dash_legs(raw_symbol, "BF", 3, default_root=root)
        expected_n = 3
    elif combo_type == "butterfly3":
        legs = _parse_dash_legs(raw_symbol, "BB", 3, default_root=root)
        expected_n = 3
    elif combo_type == "condor":
        legs = _parse_concat_legs(raw_symbol, "CF", 4, default_root=root)
        expected_n = 4
    elif combo_type == "condor4":
        legs = _parse_concat_legs(raw_symbol, "DF", 4, default_root=root)
        expected_n = 4
    elif combo_type == "average":
        parsed = _parse_average_leg(raw_symbol, default_root=root)
        legs = [parsed[1]] if parsed else None
        expected_n = 1
    elif combo_type == "pack":
        if root != "SR3":  # NG's own, structurally different ":SB" usage -- deliberately unbuilt
            legs = None
            expected_n = 0
        else:
            parsed = _parse_pack_legs(raw_symbol, default_root=root)
            legs = parsed[1] if parsed else None
            pack_type = parsed[0] if parsed else None
            expected_n = 2
    else:
        legs = None
        expected_n = 0

    if legs is None or len(legs) != expected_n:
        reason = "not_2_legs" if combo_type in ("plain", "labeled_intercommodity", "crack") else "leg_parse_fail"
        return {"combo_type": combo_type, "reject_reason": reason}

    resolved = resolve_legs(legs, maturity_year, maturity_month)
    if resolved is None:
        return {"combo_type": combo_type, "reject_reason": "leg_parse_fail"}

    leg_symbols = _legs_to_symbols(resolved)
    if leg_symbols is None:
        return {"combo_type": combo_type, "reject_reason": "unknown_root"}

    if combo_type in ("plain", "labeled_intercommodity", "crack"):
        near, far = leg_symbols
        return {"reject_reason": None, "bucket": "spread", "spread_symbol": raw_symbol,
                **_flatten_legs([("near", near), ("far", far)])}
    if combo_type in ("butterfly", "butterfly3"):
        near, mid, far = leg_symbols
        return {"reject_reason": None, "bucket": "butterfly", "butterfly_symbol": raw_symbol,
                **_flatten_legs([("near", near), ("mid", mid), ("far", far)])}
    if combo_type in ("condor", "condor4"):
        near, mid1, mid2, far = leg_symbols
        return {"reject_reason": None, "bucket": "condor", "condor_symbol": raw_symbol,
                **_flatten_legs([("near", near), ("mid1", mid1), ("mid2", mid2), ("far", far)])}
    if combo_type == "average":
        anchor = leg_symbols[0]
        return {"reject_reason": None, "bucket": "average",
                "average_symbol": raw_symbol, "root": root, "avg_years": parsed[0],
                "anchor_contract_symbol": anchor["contract_symbol"],
                "anchor_expiry_year": anchor["expiry_year"], "anchor_expiry_code": anchor["expiry_code"]}
    # combo_type == "pack"
    near, far = leg_symbols
    return {"reject_reason": None, "bucket": "pack", "pack_symbol": raw_symbol, "pack_type": pack_type,
            **_flatten_legs([("near", near), ("far", far)])}


_COMBO_JOIN_KEYS = ["raw_symbol", "maturity_year", "maturity_month"]


def _join_bucket(combo_rows_full: pl.DataFrame, keys_list: list, columns: list, asset: str) -> pl.DataFrame:
    """Reconstruct a full per-row (date, OHLCV, identity-fields) combo table
    by joining the full row set against a small per-unique-key resolution
    table, instead of building it row-by-row. `_combo_pl_df`'s empty-schema
    handling is reused so a bucket with zero successful resolutions still
    produces a correctly-typed zero-row frame."""
    if not keys_list:
        return _combo_pl_df([], columns)
    keys_df = pl.DataFrame(keys_list)
    return (
        combo_rows_full.join(keys_df, on=_COMBO_JOIN_KEYS, how="inner")
        .with_columns(pl.lit(asset).alias("asset"))
        .select(columns)
    )


# ---------------------------------------------------------------------------
# Per-asset transform
# ---------------------------------------------------------------------------

def transform_asset(asset: str, root: str, exchange: str) -> dict:
    is_ice = root in ICE_ROOTS
    counters = {
        "rows_read": 0, "join_unmatched_rows": 0, "other_instrument_class_rows": 0,
        "outright_rows": 0, "spread_rows": 0, "butterfly_rows": 0, "condor_rows": 0,
        "average_rows": 0, "pack_rows": 0,
        "non_positive_price_rows": 0, "zero_volume_outright_rows": 0, "ohlc_violation_rows": 0,
    }
    detail_parts = []

    ohlcv, definition = load_raw(root, is_ice)
    if ohlcv is None:
        return {"asset": asset, "root": root, "exchange": exchange, "status": "NO_RAW_DATA", "detail": "raw zip pair not found"}

    counters["rows_read"] = ohlcv.height
    if ohlcv.height == 0:
        return {"asset": asset, "root": root, "exchange": exchange, "status": "NO_RAW_DATA", "detail": "raw OHLCV archive is empty", **counters}

    joined, n_unmatched = join_with_fallback(ohlcv, definition)
    counters["join_unmatched_rows"] = n_unmatched
    joined = joined.filter(pl.col("raw_symbol").is_not_null())

    joined, n_sentinel = filter_sentinel(joined)
    if n_sentinel:
        detail_parts.append(f"sentinel_price_rows_dropped={n_sentinel}")
    joined = scale_prices(joined)

    outrights = joined.filter(pl.col("instrument_class") == "F")
    combos = joined.filter(pl.col("instrument_class") == "S")
    other = joined.filter(~pl.col("instrument_class").is_in(["F", "S"]))
    counters["other_instrument_class_rows"] = other.height

    # --- outrights -----------------------------------------------------
    if is_ice:
        outrights, n_z = filter_ice_z_variant(outrights)
        if n_z:
            detail_parts.append(f"ice_outright_Z_variant_excluded={n_z}")
    outrights, n_zero = filter_outright_exact_zero(outrights)
    if n_zero:
        detail_parts.append(f"outright_incomplete_price_rows_dropped={n_zero}")

    counters["non_positive_price_rows"] = outrights.filter(
        (pl.col("open") <= 0) | (pl.col("high") <= 0) | (pl.col("low") <= 0) | (pl.col("close") <= 0)
    ).height
    counters["zero_volume_outright_rows"] = outrights.filter(pl.col("volume") == 0).height
    counters["ohlc_violation_rows"] = outrights.filter(
        (pl.col("high") < pl.col("low")) | (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
    ).height

    outright_rows = (
        outrights.with_columns([
            pl.lit(asset).alias("asset"),
            pl.lit(root).alias("root"),
            pl.lit(exchange).alias("exchange"),
            pl.col("maturity_year").alias("expiry_year"),
        ])
        # Vectorized: join the 12-row month-code lookup instead of a
        # per-row Python callback -- `expiry_code` is a pure function of
        # `maturity_month` alone.
        .join(MONTH_LOOKUP_DF, on="maturity_month", how="left")
        # Vectorized: plain polars string concatenation instead of a
        # per-row callback into `_contract_symbol` -- `root`/`exchange` are
        # asset-level constants here, so this is exactly
        # `_contract_symbol(root, expiry_code, expiry_year, exchange)`
        # applied to every row at once.
        .with_columns(
            (pl.lit(root) + pl.col("expiry_code") + (pl.col("expiry_year") % 100).cast(pl.Utf8).str.zfill(2)
             + pl.lit(".") + pl.lit(exchange)).alias("contract_symbol")
        )
        .select(OUTPUT_SCHEMAS["term_structure"])
    )
    counters["outright_rows"] = outright_rows.height

    # --- combos (spread / butterfly / condor / average / pack) ---------
    # classify_combo/leg-parsing/resolve_legs/_legs_to_symbols are pure
    # functions of (raw_symbol, maturity_year, maturity_month, root) --
    # called once per UNIQUE key here (not once per row), then joined back
    # onto the full row set in polars. See the module docstring for the
    # row/key compression this achieves at scale.
    spread_keys, bf_keys, condor_keys, avg_keys, pack_keys, reject_keys = [], [], [], [], [], []
    ice_unsupported = 0

    if is_ice:
        ice_unsupported = combos.height
        combo_rows_full = None
    else:
        combo_rows_full = combos.select(["date", "open", "high", "low", "close", "volume"] + _COMBO_JOIN_KEYS)
        unique_keys = combo_rows_full.select(_COMBO_JOIN_KEYS).unique().to_dicts()

        for k in unique_keys:
            raw_symbol, maturity_year, maturity_month = k["raw_symbol"], k["maturity_year"], k["maturity_month"]
            res = _resolve_combo_key(raw_symbol, maturity_year, maturity_month, root)
            key_id = {"raw_symbol": raw_symbol, "maturity_year": maturity_year, "maturity_month": maturity_month}

            if res["reject_reason"] is not None:
                reject_keys.append({**key_id, "reject_reason": res["reject_reason"]})
                continue

            bucket = res.pop("bucket")
            res.pop("reject_reason")
            entry = {**key_id, **res}
            if bucket == "spread":
                spread_keys.append(entry)
            elif bucket == "butterfly":
                bf_keys.append(entry)
            elif bucket == "condor":
                condor_keys.append(entry)
            elif bucket == "average":
                avg_keys.append(entry)
            elif bucket == "pack":
                pack_keys.append(entry)

    if combo_rows_full is not None:
        spread_df = _join_bucket(combo_rows_full, spread_keys, OUTPUT_SCHEMAS["term_structure_spreads"], asset)
        bf_df = _join_bucket(combo_rows_full, bf_keys, OUTPUT_SCHEMAS["term_structure_butterflies"], asset)
        condor_df = _join_bucket(combo_rows_full, condor_keys, OUTPUT_SCHEMAS["term_structure_condors"], asset)
        avg_df = _join_bucket(combo_rows_full, avg_keys, OUTPUT_SCHEMAS["term_structure_averages"], asset)
        pack_df = _join_bucket(combo_rows_full, pack_keys, OUTPUT_SCHEMAS["term_structure_packs"], asset)

        # Reject-reason counters are ROW counts, not unique-key counts -- a
        # join of the full row set against the small per-key reject table,
        # then filtering/counting, expands each key back out to however
        # many actual rows shared it.
        if reject_keys:
            reject_df = combo_rows_full.join(pl.DataFrame(reject_keys), on=_COMBO_JOIN_KEYS, how="inner")
            not_2_legs = reject_df.filter(pl.col("reject_reason") == "not_2_legs").height
            leg_parse_fail = reject_df.filter(pl.col("reject_reason") == "leg_parse_fail").height
            unknown_root = reject_df.filter(pl.col("reject_reason") == "unknown_root").height
            user_defined = reject_df.filter(pl.col("reject_reason") == "user_defined").height
        else:
            not_2_legs = leg_parse_fail = unknown_root = user_defined = 0
    else:
        spread_df = _combo_pl_df([], OUTPUT_SCHEMAS["term_structure_spreads"])
        bf_df = _combo_pl_df([], OUTPUT_SCHEMAS["term_structure_butterflies"])
        condor_df = _combo_pl_df([], OUTPUT_SCHEMAS["term_structure_condors"])
        avg_df = _combo_pl_df([], OUTPUT_SCHEMAS["term_structure_averages"])
        pack_df = _combo_pl_df([], OUTPUT_SCHEMAS["term_structure_packs"])
        not_2_legs = leg_parse_fail = unknown_root = user_defined = 0

    counters["spread_rows"] = spread_df.height
    counters["butterfly_rows"] = bf_df.height
    counters["condor_rows"] = condor_df.height
    counters["average_rows"] = avg_df.height
    counters["pack_rows"] = pack_df.height

    if ice_unsupported:
        detail_parts.append(f"ICE_schema_unsupported_spreads={ice_unsupported}")
    if unknown_root:
        detail_parts.append(f"spread_unknown_root_exchange={unknown_root}")
    if not_2_legs:
        detail_parts.append(f"spread_not_2_legs={not_2_legs}")
    if leg_parse_fail:
        detail_parts.append(f"spread_leg_parse_fail={leg_parse_fail}")
    if user_defined:
        detail_parts.append(f"user_defined_spreads={user_defined}")

    has_anomaly = bool(detail_parts) or counters["non_positive_price_rows"] or counters["ohlc_violation_rows"] or n_unmatched
    status = "PARTIAL" if has_anomaly else "OK"

    return {
        "asset": asset, "root": root, "exchange": exchange, "status": status,
        **counters,
        "coverage_missing_sessions": None, "coverage_missing_pct": None,
        "detail": "; ".join(detail_parts) if detail_parts else None,
        "_outright_df": outright_rows.with_columns(pl.col("date").cast(DATE_DTYPE)),
        "_spread_df": spread_df,
        "_butterfly_df": bf_df,
        "_condor_df": condor_df,
        "_average_df": avg_df,
        "_pack_df": pack_df,
    }


# ---------------------------------------------------------------------------
# Merge + atomic write
# ---------------------------------------------------------------------------

def _write_parquet_atomic(df: pl.DataFrame, path: Path):
    """Write to a temp file then `os.replace()` into the real path -- avoids
    ever asking the OS to overwrite a path a reader may have memory-mapped
    in the same process (a real Windows-only failure mode once these output
    files grow large)."""
    tmp_path = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    os.replace(tmp_path, path)


def _merge_append(new_df: pl.DataFrame, path: Path, dedup_subset: list):
    if new_df is None or new_df.height == 0:
        return
    new_df = new_df.with_columns(pl.col("date").cast(DATE_DTYPE))

    if path.exists():
        # An existing file may have been written by pandas originally
        # (datetime64[ns]) or by an earlier run of this module -- both read
        # back as pl.Datetime, but harmonizing explicitly avoids ever
        # relying on that matching by coincidence.
        existing = pl.read_parquet(path).with_columns(pl.col("date").cast(DATE_DTYPE))
        combined = pl.concat([existing, new_df], how="vertical_relaxed")
    else:
        combined = new_df

    # New (Databento) rows are appended AFTER any existing rows, and
    # `keep="last"` + `maintain_order=True` makes Databento win on overlap --
    # the same convention `jobs/capture_term_structure.py`'s own pandas-based
    # merge already uses (`drop_duplicates(..., keep="last")`).
    combined = (
        combined.unique(subset=dedup_subset, keep="last", maintain_order=True)
        .sort(dedup_subset)
    )
    _write_parquet_atomic(combined, path)


def merge_asset_result(result: dict):
    _merge_append(result.pop("_outright_df", None), DATA_DIR / "term_structure.parquet", ["date", "contract_symbol"])
    _merge_append(result.pop("_spread_df", None), DATA_DIR / "term_structure_spreads.parquet", ["date", "spread_symbol", "asset"])
    _merge_append(result.pop("_butterfly_df", None), DATA_DIR / "term_structure_butterflies.parquet", ["date", "butterfly_symbol", "asset"])
    _merge_append(result.pop("_condor_df", None), DATA_DIR / "term_structure_condors.parquet", ["date", "condor_symbol", "asset"])
    _merge_append(result.pop("_average_df", None), DATA_DIR / "term_structure_averages.parquet", ["date", "average_symbol", "asset"])
    _merge_append(result.pop("_pack_df", None), DATA_DIR / "term_structure_packs.parquet", ["date", "pack_symbol", "asset"])


def append_manifest(rows: list):
    manifest_cols = ["asset", "root", "exchange", "status", "rows_read", "join_unmatched_rows",
                      "other_instrument_class_rows", "outright_rows", "spread_rows", "butterfly_rows",
                      "non_positive_price_rows", "zero_volume_outright_rows", "ohlc_violation_rows",
                      "coverage_missing_sessions", "coverage_missing_pct", "detail", "run_date",
                      "condor_rows", "average_rows", "pack_rows"]
    run_date = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        r["run_date"] = run_date
    new_manifest = pd.DataFrame(rows)[manifest_cols]
    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        combined = pd.concat([existing, new_manifest], ignore_index=True)
    else:
        combined = new_manifest
    combined.to_csv(MANIFEST_PATH, index=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(assets: list = None):
    """Run the transform for `assets` (default: every UNIVERSE asset with a
    raw zip pair present). Each asset's result is merged and written
    immediately (not batched) -- if one asset fails partway through a full
    run, everything before it is already safely on disk."""
    targets = assets if assets is not None else list(UNIVERSE.keys())
    manifest_rows = []
    for asset in targets:
        root, exchange = UNIVERSE[asset]
        print(f"--- {asset} ({root}) ---")
        try:
            result = transform_asset(asset, root, exchange)
        except Exception as exc:
            manifest_rows.append({
                "asset": asset, "root": root, "exchange": exchange, "status": "ERROR",
                "rows_read": None, "join_unmatched_rows": None, "other_instrument_class_rows": None,
                "outright_rows": None, "spread_rows": None, "butterfly_rows": None,
                "non_positive_price_rows": None, "zero_volume_outright_rows": None, "ohlc_violation_rows": None,
                "coverage_missing_sessions": None, "coverage_missing_pct": None,
                "detail": f"{type(exc).__name__}: {exc}", "condor_rows": None, "average_rows": None, "pack_rows": None,
            })
            print(f"  ERROR: {exc}")
            continue

        print(f"  status={result['status']} outright={result.get('outright_rows')} "
              f"spread={result.get('spread_rows')} butterfly={result.get('butterfly_rows')} "
              f"condor={result.get('condor_rows')} average={result.get('average_rows')} "
              f"pack={result.get('pack_rows')} detail={result.get('detail')}")

        if result["status"] != "NO_RAW_DATA":
            merge_asset_result(result)
        for key in ["_outright_df", "_spread_df", "_butterfly_df", "_condor_df", "_average_df", "_pack_df"]:
            result.pop(key, None)
        manifest_rows.append(result)

    append_manifest(manifest_rows)
    n_ok = sum(1 for r in manifest_rows if r["status"] in ("OK", "PARTIAL"))
    print(f"\nDone: {n_ok}/{len(targets)} assets transformed. Manifest: {MANIFEST_PATH}")
