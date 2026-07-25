"""
signals/value.py — Value, Asness-Moskowitz-Pedersen 2013 "Value and Momentum
Everywhere" (`references/Value and Momentum Everywhere.pdf`, same source paper as
`signals/xs_momentum.py`, read directly).

Per direct instruction — a further deliberate scope expansion, following XSMOM's
own precedent (not one of this project's original six signal families).

No book-value measure exists for futures, so the paper substitutes a long-horizon
price-reversal proxy, with two asset-class-specific refinements (Section I.B):

- **Default** (commodities, equity indices, and anything without a special case,
  including FX in this reconstruction's own local-currency term before the PPP
  adjustment is layered on) — negative of the 5-year return, smoothed by averaging
  over a trailing window centered ~5 years back rather than reading a single noisy
  month-end price: `value = log(avg_price[~5y ago] / price_now)`.
- **Bonds** — the 5-year CHANGE in yield (`yield_now - yield_5y_ago`), NOT a
  return — rising yields already mean falling bond prices, so this is directly a
  "cheap now vs. 5 years ago" measure without any sign-flip or log needed.
  Maturity-matched via `BOND_YIELD_MATURITY_MAP` against `data.macro.
  load_yield_curve()`'s own columns (UltraBond has no dedicated point on the curve,
  mapped to 30Y as the closest available maturity).
- **Currencies** — PPP-adjusted real FX return: the same negative-5yr-return
  construction as the default case, minus the 5-year inflation differential
  against the US (`(log(CPI_country_now/CPI_country_5y_ago) -
  log(CPI_US_now/CPI_US_5y_ago))`), via `FX_CPI_COUNTRY_MAP` against `data.macro.
  load_cpi()`. CPI is genuinely stale/missing for some countries (GBP/CAD/CHF/MXN
  lag 1-2yr via their FRED mirror; AUD is bounded-ffilled upstream in
  `data.macro.load_cpi` for its genuinely-quarterly publication cadence) — this
  module does NOT re-forward-fill CPI itself, but DOES apply a bounded
  `reindex_ffill_limit` (default 35 calendar days — just enough to bridge one
  already-monthly-cadence value across the days within its own month, not enough
  to replay a stale reading indefinitely) when reindexing the monthly PPP value
  onto the daily close index. An earlier build of this function used an UNBOUNDED
  `.ffill()` at that step, which would have silently replayed a country's
  last-known PPP value forever once its CPI went missing entirely — this bounded
  limit is that fix, not the original construction.

`build_value_panel` assembles the default (negative-5yr-return) panel first, then
OVERWRITES the bond and FX columns with their special-case values — bonds and FX
never fall through to the generic price-based default.

Rank-weighted within sector (reuses `signals.transforms.cross_sectional_rank`, the
same Eq. 1/Eq. 19-shaped construction XSMOM and carry already share — CLAUDE.md
Rule 6), no vol-scaling, ONE Book — same three departures from the momentum/
breakout/crossover/reversal pattern already established for carry and XSMOM.
"""

import numpy as np
import pandas as pd

from signals.transforms import cross_sectional_rank

DEFAULT_AVG_END_MONTHS = 54  # ~4.5 years back
DEFAULT_AVG_WINDOW_MONTHS = 12  # averaged over the ~12 months before that point
BOND_YIELD_LOOKBACK_MONTHS = 60  # 5 years
FX_CPI_LOOKBACK_MONTHS = 60  # 5 years
FX_REINDEX_FFILL_LIMIT_DAYS = 35

BOND_YIELD_MATURITY_MAP = {
    "US_2Y": "2Y",
    "US_5Y": "5Y",
    "US_10Y": "10Y",
    "US_30Y": "30Y",
    "UltraBond": "30Y",
}

FX_CPI_COUNTRY_MAP = {
    "EURUSD": "EUR",
    "JPYUSD": "JPY",
    "GBPUSD": "GBP",
    "AUDUSD": "AUD",
    "CADUSD": "CAD",
    "SwissFranc": "CHF",
    "MexicanPeso": "MXN",
}

US_CPI_COLUMN = "US"


def negative_5yr_return_value(
    close: pd.DataFrame,
    avg_end_months: int = DEFAULT_AVG_END_MONTHS,
    avg_window_months: int = DEFAULT_AVG_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Default value measure: `log(avg_price / price_now)`, where `avg_price` is
    the trailing `avg_window_months`-month average price ending `avg_end_months`
    months ago (~4.5-5.5 years back) — a smoothed stand-in for "the 5-year-ago
    price," avoiding reading a single noisy month-end observation. Higher value
    (price now far below its level ~5 years ago) means "cheap," matching the
    paper's own "negative of the return" framing without needing an explicit
    sign-flip in code."""
    monthly = close.resample("ME").last()
    avg_price = monthly.shift(avg_end_months).rolling(avg_window_months).mean()
    raw = np.log(avg_price / monthly)
    return raw.reindex(close.index).ffill()


def bond_yield_change_value(
    yield_curve: pd.DataFrame,
    maturity_map: dict = BOND_YIELD_MATURITY_MAP,
    lookback_months: int = BOND_YIELD_LOOKBACK_MONTHS,
    daily_index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """Bonds: `yield_now - yield_5y_ago`, maturity-matched via `maturity_map`
    (asset name -> `yield_curve` column). A rising yield already means a falling
    bond price, so this is directly the value signal — no log, no sign-flip."""
    monthly_yield = yield_curve.resample("ME").last()
    change = monthly_yield - monthly_yield.shift(lookback_months)

    result = pd.DataFrame(np.nan, index=monthly_yield.index, columns=list(maturity_map.keys()))
    for asset, maturity in maturity_map.items():
        if maturity in change.columns:
            result[asset] = change[maturity]

    if daily_index is not None:
        result = result.reindex(daily_index).ffill()
    return result


def fx_ppp_value_feature(
    close: pd.DataFrame,
    cpi: pd.DataFrame,
    country_map: dict = FX_CPI_COUNTRY_MAP,
    avg_end_months: int = DEFAULT_AVG_END_MONTHS,
    avg_window_months: int = DEFAULT_AVG_WINDOW_MONTHS,
    cpi_lookback_months: int = FX_CPI_LOOKBACK_MONTHS,
    reindex_ffill_limit: int = FX_REINDEX_FFILL_LIMIT_DAYS,
) -> pd.DataFrame:
    """Currencies: PPP-adjusted real FX return — the same negative-5yr-return
    construction as `negative_5yr_return_value`, minus the 5-year US-relative
    inflation differential (`country_map` : asset name -> `cpi` column).

    Reindexed onto the daily close index with a BOUNDED `ffill(limit=
    reindex_ffill_limit)`, not an unbounded one — see module docstring for why
    (a stale/missing CPI country must not silently replay its last PPP value
    forever)."""
    monthly_close = close.resample("ME").last()
    monthly_cpi = cpi.resample("ME").last()

    avg_price = monthly_close.shift(avg_end_months).rolling(avg_window_months).mean()
    nominal_fx_component = np.log(avg_price / monthly_close)

    us_cpi = monthly_cpi[US_CPI_COLUMN] if US_CPI_COLUMN in monthly_cpi.columns else None
    us_cpi_change = (
        np.log(us_cpi / us_cpi.shift(cpi_lookback_months)) if us_cpi is not None else None
    )

    result = pd.DataFrame(np.nan, index=monthly_close.index, columns=list(country_map.keys()))
    if us_cpi_change is not None:
        for asset, country in country_map.items():
            if asset not in nominal_fx_component.columns or country not in monthly_cpi.columns:
                continue
            country_cpi = monthly_cpi[country]
            country_cpi_change = np.log(country_cpi / country_cpi.shift(cpi_lookback_months))
            inflation_diff = country_cpi_change - us_cpi_change
            result[asset] = nominal_fx_component[asset] - inflation_diff

    return result.reindex(close.index).ffill(limit=reindex_ffill_limit)


def build_value_panel(
    close: pd.DataFrame,
    yield_curve: pd.DataFrame,
    cpi: pd.DataFrame,
    bond_map: dict = BOND_YIELD_MATURITY_MAP,
    fx_map: dict = FX_CPI_COUNTRY_MAP,
) -> pd.DataFrame:
    """Assembles the default (negative-5yr-return) panel, then OVERWRITES the bond
    and FX columns with their asset-class-specific values — bonds and FX never
    fall through to the generic price-based default."""
    panel = negative_5yr_return_value(close)

    bond_values = bond_yield_change_value(yield_curve, bond_map, daily_index=close.index)
    for asset in bond_map:
        if asset in bond_values.columns:
            panel[asset] = bond_values[asset]

    fx_values = fx_ppp_value_feature(close, cpi, fx_map)
    for asset in fx_map:
        if asset in fx_values.columns:
            panel[asset] = fx_values[asset]

    return panel


def value_signal(
    close: pd.DataFrame,
    yield_curve: pd.DataFrame,
    cpi: pd.DataFrame,
    sectors: dict,
    bond_map: dict = BOND_YIELD_MATURITY_MAP,
    fx_map: dict = FX_CPI_COUNTRY_MAP,
) -> pd.DataFrame:
    """The paper's rank-weighted cross-sectional value signal, sector-scoped, no
    vol-scaling. Monthly rebalancing is a backtest-layer choice, not baked in
    here — same convention as `signals.xs_momentum.xs_momentum_signal`."""
    panel = build_value_panel(close, yield_curve, cpi, bond_map, fx_map)
    return cross_sectional_rank(panel, sectors)
