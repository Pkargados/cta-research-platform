"""
data/continuous_curve.py — Continuous futures curve construction (raw + back-adjusted).

Built 2026-07-21, see WORKFLOW.md Phase 0 for the full decision record. Replaces
Yahoo's opaque, unaudited front-month splice (the thing every signal has actually
been backtesting against so far - trusted-era masking never fixed this, it only
removed the noisiest years) with a curve built from term_structure.parquet's real
per-contract Databento data, with an explicit, documented roll rule and adjustment
method instead of an inherited, unverified convention.

General-purpose: every signal that trades a single per-asset price level (momentum,
breakout, crossover) consumes this. RV/cointegration/carry don't - they already
consume term_structure.parquet's real multi-contract data directly.

Two outputs per asset, not one - see WORKFLOW.md Phase 0 for why both are needed:
- raw: real, unadjusted prices (TCA, live/paper-trading comparison, dashboard
  display, and the ground truth to validate this module's own roll logic against).
- back-adjusted (ratio/proportional method): for signal construction and
  backtesting - avoids the fake return spike an unadjusted series creates at every
  roll. Ratio, not additive - corrected 2026-07-21 (see WORKFLOW.md Phase 0's change
  log): additive was originally chosen on the assumption that ratio adjustment would
  be undefined across WTI's real 2020 negative print. Checked directly against this
  dataset and found false - raw front-contract close is never non-positive for any
  of the 42 assets, including WTI (the roll rule below rolls out of each contract
  before expiry, so the front-contract series never actually captures that event).
  Meanwhile additive turned out to have a real, live bug: it pushed HeatingOil,
  RBOB, Brent, WTI Crude, and Oats through zero in old segments, which corrupted
  every downstream percentage-return calculation (momentum, vol) for those assets -
  not just "an old segment can go negative," a percentage-return singularity right
  at the crossing. Ratio adjustment is a product of positive numbers, so it's always
  positive by construction, and it has a second, independent benefit for this
  project: it preserves percentage returns consistently across the whole history,
  which additive doesn't (old, heavily-shifted segments have technically distorted
  pct-returns even where they stay positive) - a better fit for a project whose
  signals are built on percentage returns throughout.

Roll rule: volume crossover (real per-contract volume, already in
term_structure.parquet) is the trigger, confirmed over `confirmation_days` to avoid
a single noisy print flipping the roll; a `backstop_days` fixed-days-before-expiry
rule forces a roll regardless if the front is close to its own last observed date
(used as the real expiry proxy - no external CME/ICE calendar assumed). Open-interest
crossover is the actual industry-standard method and the target once OI is purchased
(DATA_SCHEMA.md section 1, not bought yet) - this module's volume-based rule is the
documented interim, not a permanent design choice.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

# F=Jan ... Z=Dec - matches jobs/capture_term_structure.py's MONTH_CODES. Redefined
# here rather than imported cross-package (same convention already used in
# jobs/update_dashboard_summary.py) so this module has no dependency beyond src/.
MONTH_CODES = "FGHJKMNQUVXZ"


def build_contract_chain(asset_df: pd.DataFrame) -> list:
    """Distinct contracts for one asset, sorted chronologically by real expiry.

    asset_df must have columns: contract_symbol, expiry_year, expiry_code.
    """
    chain = (
        asset_df[["contract_symbol", "expiry_year", "expiry_code"]]
        .drop_duplicates()
        .copy()
    )
    chain["month"] = chain["expiry_code"].map(lambda c: MONTH_CODES.index(c) + 1)
    chain = chain.sort_values(["expiry_year", "month"])
    return chain["contract_symbol"].tolist()


def assign_front_contract(
    asset_df: pd.DataFrame, chain: list, confirmation_days: int = 2, backstop_days: int = 5
) -> pd.Series:
    """Walk forward date by date, deciding which contract is "front" on each date.

    Returns a pd.Series indexed by date (name "date"), values = contract_symbol.

    Roll triggers:
    - Volume crossover, sustained for `confirmation_days` consecutive trading days
      (avoids a single noisy print causing a premature or flip-flopping roll).
    - Backstop: forced roll if the current front is within `backstop_days` trading
      days of its own last observed date in the data (that date is the real expiry
      proxy - no external calendar assumed).

    A contract with no row at all on a given date (never listed / not yet trading
    that day) is never eligible to become front - distinct from a contract that *has*
    a real row with volume == 0 (a synthetic/thin print, still a real observation).

    "Next contract" means the nearest chain entry after front that actually has a
    real row on the current date - not necessarily the immediately-adjacent one.
    Found live on MexicanPeso: some chain entries are near-empty (e.g. a thin
    serial-month FX contract with a two-day burst of prints and nothing else, ever) -
    treating `chain[idx+1]` as always-next got the assignment permanently stuck once
    the front expired next to one of these, since neither leg ever had an overlapping
    real row to compare or roll into again. Scanning forward for the nearest genuinely
    available candidate avoids this generically, for any number of dead entries.
    """
    volume = asset_df.pivot(index="date", columns="contract_symbol", values="volume").sort_index()
    dates = volume.index
    date_pos = {d: i for i, d in enumerate(dates)}
    last_seen = asset_df.groupby("contract_symbol")["date"].max()
    chain_index = {c: i for i, c in enumerate(chain)}

    # Highest-volume contract on day 1, among contracts with a REAL row that day -
    # dropna() before idxmax(), not fillna(0.0). Found live on Coffee: filling absent
    # contracts with 0 makes "not yet trading" tie with "a genuine thin print," and
    # idxmax()'s leftmost tie-break could then pick a contract that doesn't actually
    # start trading until months later, stranding the assignment with nothing to
    # compare against until that contract's real history finally begins.
    front = volume.iloc[0].dropna().idxmax()

    assignments = []
    pending_next, pending_count = None, 0

    for pos, date in enumerate(dates):
        row = volume.loc[date]

        idx = chain_index.get(front)
        next_contract = None
        if idx is not None:
            for candidate in chain[idx + 1:]:
                candidate_raw = row.get(candidate)
                if candidate_raw is not None and pd.notna(candidate_raw):
                    next_contract = candidate
                    break

        front_raw = row.get(front)
        front_available = front_raw is not None and pd.notna(front_raw)
        front_vol = float(front_raw) if front_available else 0.0

        next_raw = row.get(next_contract) if next_contract is not None else None
        next_available = next_raw is not None and pd.notna(next_raw)
        next_vol = float(next_raw) if next_available else 0.0

        volume_signal = next_available and next_vol > 0 and front_vol > 0 and next_vol > front_vol
        if volume_signal:
            pending_count = pending_count + 1 if pending_next == next_contract else 1
            pending_next = next_contract
        else:
            pending_next, pending_count = None, 0

        roll_by_volume = pending_next is not None and pending_count >= confirmation_days

        front_last_seen = last_seen.get(front)
        trading_days_left = (
            date_pos[front_last_seen] - pos if front_last_seen is not None and front_last_seen in date_pos else None
        )
        roll_by_backstop = (
            next_available and trading_days_left is not None and trading_days_left <= backstop_days
        )

        # Only ever roll into a contract that actually has a real row on this date -
        # `next_available` gates both trigger paths above already.
        if (roll_by_volume or roll_by_backstop) and next_available:
            front = next_contract
            pending_next, pending_count = None, 0

        assignments.append(front)

    result = pd.Series(assignments, index=dates, name="contract_symbol")
    result.index.name = "date"
    return result


def build_raw_series(asset_df: pd.DataFrame, front_series: pd.Series) -> pd.DataFrame:
    """Real, unadjusted OHLCV for whichever contract is front on each date.

    Returns a DataFrame indexed by date with columns: front_contract_symbol, open,
    high, low, close, volume.
    """
    front_df = front_series.reset_index()
    merged = front_df.merge(asset_df, on=["date", "contract_symbol"], how="left")
    merged = merged.set_index("date").sort_index()
    return merged[["contract_symbol", "open", "high", "low", "close", "volume"]].rename(
        columns={"contract_symbol": "front_contract_symbol"}
    )


def back_adjust(asset_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ratio (proportional) back-adjustment: scale every date before a roll by the
    roll-date price ratio, compounded backward through every roll, so the most
    recent segment is unadjusted and every earlier segment carries the product of
    every roll ratio between it and today.

    The same scalar is applied to open/high/low/close alike on a given day -
    multiplying by a positive constant preserves each bar's internal OHLC ordering
    exactly, so this step cannot itself introduce a new OHLC-consistency violation
    (same invariant the additive method guaranteed via a level shift, here via a
    positive scale factor instead).

    Adds adj_open/adj_high/adj_low/adj_close/is_roll_date columns to a copy of
    raw_df.
    """
    front = raw_df["front_contract_symbol"]
    prev_front = front.shift(1)
    is_roll = (front != prev_front) & prev_front.notna()

    indexed_close = asset_df.set_index(["date", "contract_symbol"])["close"]

    ratios = pd.Series(1.0, index=raw_df.index)  # 1.0 = no adjustment (identity)
    for date in raw_df.index[is_roll]:
        new_symbol, old_symbol = front.loc[date], prev_front.loc[date]
        new_close = indexed_close.get((date, new_symbol))
        old_close = indexed_close.get((date, old_symbol))
        # Both legs should always have a real, positive row on a roll date -
        # assign_front_contract only ever rolls into a contract with an available
        # row, and the old front necessarily has one too (it was front the day
        # before); raw front-contract close is never non-positive anywhere in this
        # project's universe (checked directly, see module docstring). The <= 0
        # guard is future-proofing, not a currently-reachable path - a ratio is
        # undefined (or sign-flipping) against a non-positive leg, so fall back to
        # the identity (no adjustment at this transition) rather than raise or
        # corrupt the series, consistent with this project's log-and-continue
        # convention for anomalies.
        if new_close is not None and old_close is not None and new_close > 0 and old_close > 0:
            ratios.loc[date] = float(new_close) / float(old_close)

    cumprod_from_here = ratios.iloc[::-1].cumprod().iloc[::-1]
    adjustment = cumprod_from_here / ratios  # product of ratios strictly AFTER this date

    result = raw_df.copy()
    result["is_roll_date"] = is_roll
    for raw_col, adj_col in [("open", "adj_open"), ("high", "adj_high"), ("low", "adj_low"), ("close", "adj_close")]:
        result[adj_col] = result[raw_col] * adjustment
    result = result.rename(columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"})
    return result


def build_continuous_curve(asset_df: pd.DataFrame, confirmation_days: int = 2, backstop_days: int = 5) -> pd.DataFrame:
    """Full construction for one asset: chain -> front assignment -> raw -> back-adjusted.

    Returns a DataFrame indexed by date with columns: front_contract_symbol,
    is_roll_date, raw_open, raw_high, raw_low, raw_close, volume, adj_open, adj_high,
    adj_low, adj_close.
    """
    chain = build_contract_chain(asset_df)
    front_series = assign_front_contract(asset_df, chain, confirmation_days, backstop_days)
    raw_df = build_raw_series(asset_df, front_series)
    return back_adjust(asset_df, raw_df)


def load_continuous_raw(data_dir=None) -> dict:
    """Load the raw (unadjusted) continuous curve, pivoted to the same
    dict-of-wide-(T×N)-DataFrames shape as data.panels.load_core_panel(), so this is
    a drop-in replacement for signal-construction code. Includes an "is_roll_date"
    key (bool, NaN-filled-False for asset/date combos with no row) - callers building
    an OHLC estimator off the raw series need this to identify the one real artifact
    raw prices carry that back-adjustment removes: an "overnight" gap on the roll
    date spans two different contracts, not a real overnight move in one position."""
    result = _load_pivoted(data_dir, {
        "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
        "volume": "volume", "is_roll_date": "is_roll_date",
    })
    # .where(), not .fillna() - fillna's implicit object->bool downcast on a mixed
    # True/False/NaN column raises a FutureWarning in current pandas regardless of
    # a preceding infer_objects() call; .where() sidesteps it cleanly.
    is_roll = result["is_roll_date"]
    result["is_roll_date"] = is_roll.where(is_roll.notna(), False).astype(bool)
    return result


def load_continuous_backadjusted(data_dir=None) -> dict:
    """Load the back-adjusted continuous curve, same shape as load_continuous_raw()
    (minus is_roll_date - not needed once the series is already roll-jump-free)."""
    return _load_pivoted(data_dir, {"open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close", "volume": "volume"})


def _load_pivoted(data_dir, column_map: dict) -> dict:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_parquet(data_dir / "continuous_futures.parquet")
    return {field: df.pivot(index="date", columns="asset", values=col) for field, col in column_map.items()}
