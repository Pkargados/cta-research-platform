"""
signals/seasonality.py — Phase 11b: two independent, unrelated seasonal effects
from the commodity-seasonality literature, built as parallel specs per this
project's house style (no headline pick). Built per direct instruction after
the original WORKFLOW.md Phase 11b plan (a half-month effect built off Li et
al.'s RRT statistic) turned out, on reading both source papers directly, not
to match how either paper actually constructs a tradeable strategy — see
WORKFLOW.md Phase 11b's build log for the full correction.

Sources, read directly (references/):
- Li, Liu, Miao, Tse (2023), "Return Seasonality in Commodity Futures" — tests
  the half-month effect (Milonas 1991) and backtests the same-calendar-month
  effect (Keloharju, Linnainmaa, Nyberg 2014/2016, replicated here for
  commodities).
- Keloharju, Linnainmaa, Nyberg (2014), "Common Factors in Return
  Seasonalities" — the original same-calendar-month construction.

**half_month_signal** — Milonas (1991)'s half-month effect: the first half of
the month historically outperforms the second half. Li et al.'s own RRT
statistic (their Eq. 2, R_it/sigma_it) is used ONLY to test this effect's
statistical significance (their Table 5) — neither paper backtests a
tradeable half-month strategy from it, so there is no published construction
to reproduce here. This function is this project's own interpretation of the
documented finding as a trading rule (long first half, short second half),
vol-scaled per CLAUDE.md Rule 5 (the same `vol_targeted_sign_signal`
construction already used by momentum/crossover) — NOT a literal
reproduction of RRT, and NOT sourced from Milonas (1991) directly (that paper
could not be located in `references/`, per WORKFLOW.md Phase 11b). Scoped to
a NAMED SUBSET of 7 assets (a per-asset time-series property, not a
cross-sectional one — no peer ranking involved): the 7 of Li et al.'s 9
commodities with a significant 1970-1989 half-month effect that are already
in this project's universe (Corn, KC_Wheat, Wheat, FeederCattle, LeanHogs,
Silver, Soybeans; Soy Oil and Soymeal are not pulled — see WORKFLOW.md open
decision #3).

**same_month_signal** — the paper's ACTUAL backtested strategy (Li et al.
Table 6: 0.66%-1.67% monthly returns, significant only in the 1990-1999
subperiod), Keloharju et al.'s own construction: each month, rank assets by
their trailing 5-20-year historical average return in that SAME calendar
month (excluding the current occurrence). Rank-weighted cross-sectional
(`cross_sectional_rank`, CLAUDE.md Rule 5/6) rather than the paper's own
discrete top-3/bottom-3 long-short portfolio — the same continuous-not-binary
departure already made for carry/XSMOM/value. Scoped to this project's
ADV-filtered liquid universe and sectors (the same within-sector departure
from the paper's own flat full-cross-section stack rank already logged for
carry/XSMOM/value), not restricted to the half-month effect's 7-name subset —
these are two unrelated phenomena in the source papers, not two views of the
same one.

Both null/negative results are the expected, evidence-based outcome per
WORKFLOW.md 11a's synthesis (seasonality "almost completely disappeared"
since 1990 per Li et al.'s own finding) — reported honestly either way
(CLAUDE.md Rule 1/2), not a failure if that's what's found.

**tsmom_seasonal_signal** (Phase 11c, built 2026-07-31) — a completely
different construction from the two above: not a standalone signal, but a
continuous, per-asset seasonal CONVICTION MULTIPLIER applied to
`signals.momentum.tsmom_signal`'s own output (see that function's own
docstring for the full architecture/mechanism rationale). Boosts TSMOM's
existing position for 7 named commodities during their own physically-
motivated seasonal window (winter heating demand, growing-season weather
risk, etc. — `TSMOM_SEASONAL_WINDOWS`, reused verbatim from Phase 11c's own
conviction table), everything else unchanged.
"""

import numpy as np
import pandas as pd

from signals.transforms import cross_sectional_rank, vol_targeted_sign_signal
from signals.momentum import tsmom_signal

SEASONALITY_HALF_MONTH_ASSETS = [
    "Corn", "KC_Wheat", "Wheat", "FeederCattle", "LeanHogs", "Silver", "Soybeans",
]
HALF_MONTH_BOUNDARY_DAY = 15
DEFAULT_TARGET_VOL = 0.40
SAME_MONTH_MIN_YEARS = 5
SAME_MONTH_MAX_YEARS = 20

# Assets with a real, physical/economic seasonal demand driver, per WORKFLOW.md
# Phase 11c's own conviction table (built 2026-07-31 for the TSMOM-modifier
# plan, not yet built as a signal) — Medium confidence or higher only:
#   Natural Gas / HeatingOil - winter heating demand
#   RBOB                     - summer driving season + spring blend-switchover
#   Corn / Soybeans          - growing-season weather risk
#   Wheat / KC_Wheat         - winter-wheat dormancy-through-harvest weather risk
# LiveCattle/FeederCattle/LeanHogs are deliberately excluded here too — 11c's
# own table flags their windows as "Lower confidence — needs a literature
# check before committing to exact dates," never resolved.
#
# Used to test same_month on a hypothesis-driven subset instead of the full
# ADV-filtered universe — a DIFFERENT kind of restriction from CLAUDE.md Rule
# 1's concern (never edit the universe after observing backtest performance):
# this list was fixed from a documented physical-driver theory BEFORE looking
# at same_month's performance on these names specifically, the same discipline
# already applied to 11c's own conviction table and to SEASONALITY_HALF_MONTH_
# ASSETS above (both fixed from a source paper/table, not from a same_month
# result). It intentionally departs from the paper's own KLN/Li et al.
# construction (rank across the full commodity universe, no driver theory
# involved) — a deliberate, economically-motivated exploration beyond what
# either source paper tested, not a reproduction of their methodology.
SEASONALITY_ECONOMIC_DRIVER_ASSETS = [
    "Natural Gas", "HeatingOil", "RBOB", "Corn", "Soybeans", "Wheat", "KC_Wheat",
]


def half_month_direction(index: pd.DatetimeIndex) -> pd.Series:
    """+1 for day-of-month <= HALF_MONTH_BOUNDARY_DAY (first half), -1
    otherwise (second half) — the Lakonishok-Smidt boundary definition, as
    restated by Li et al. Section 2.2 ("the first half of the month is
    considered as the days between the first and fifteenth")."""
    day = index.day
    return pd.Series(np.where(day <= HALF_MONTH_BOUNDARY_DAY, 1.0, -1.0), index=index)


def half_month_signal(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    assets=None,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> pd.DataFrame:
    """Vol-targeted +-1 half-month direction, scoped to `assets` (default
    SEASONALITY_HALF_MONTH_ASSETS). Assets not present in `close` are
    dropped. NaN wherever `close` itself is NaN (no signal for a day the
    asset didn't trade)."""
    assets = SEASONALITY_HALF_MONTH_ASSETS if assets is None else assets
    present = [a for a in assets if a in close.columns]
    direction = half_month_direction(close.index)
    raw = pd.DataFrame({a: direction.to_numpy() for a in present}, index=close.index)
    raw = raw.where(close[present].notna())
    return vol_targeted_sign_signal(raw, vol[present], target_vol=target_vol)


def _same_month_trailing_mean(series: pd.Series, min_years: int, max_years: int) -> pd.Series:
    grouped = series.groupby(series.index.month, group_keys=False)
    result = grouped.apply(lambda s: s.rolling(window=max_years, min_periods=min_years).mean().shift(1))
    return result.reindex(series.index)


def same_month_average_return(
    monthly_returns: pd.DataFrame,
    min_years: int = SAME_MONTH_MIN_YEARS,
    max_years: int = SAME_MONTH_MAX_YEARS,
) -> pd.DataFrame:
    """Trailing average return in the SAME calendar month, min_years-max_years
    years of history, excluding the current occurrence (Li et al. Section
    3.1: "we process these strategies over the past 5-20 years... include
    assets with at least five years of historical return data")."""
    return monthly_returns.apply(lambda col: _same_month_trailing_mean(col, min_years, max_years))


def same_month_signal(
    close: pd.DataFrame,
    sectors: dict,
    min_years: int = SAME_MONTH_MIN_YEARS,
    max_years: int = SAME_MONTH_MAX_YEARS,
) -> pd.DataFrame:
    """Rank-weighted cross-sectional same-calendar-month effect (Keloharju et
    al. 2014/2016's construction, replicated for commodities by Li et al.
    2023 Section 3.1) — continuous rank-demean, not the paper's own discrete
    top-3/bottom-3 portfolio (CLAUDE.md Rule 5/6, same departure as
    carry/XSMOM/value).

    Real bug caught and fixed before shipping (an unexpectedly strong first
    result — positive validation/test Sharpe against this signal family's
    own null/negative prior — is exactly the case CLAUDE.md's own discipline
    says to re-verify, not accept): `same_month_average_return`'s score for
    calendar month m (e.g. January) only needs data through the END of the
    PRIOR month (December) and is labeled at January's own month-end row.
    Every other cross-sectional signal in this project (momentum, carry,
    XSMOM) wants exactly that labeling, because `backtest.engine`'s own
    month-lag convention (form at month-end t, trade month t+1) is designed
    for a signal whose FORMATION and TARGET periods are naturally adjacent,
    different months. A same-calendar-month effect's formation and target
    are the SAME month (m) by construction — so left as originally labeled,
    the standard t -> t+1 lag would trade FEBRUARY on "how good has January
    historically been," a full calendar month off from what the effect
    predicts. `shift(-1)` moves each row's value back one month (the
    January-effect score now sits on December's row) so that the same
    universal t -> t+1 lag lands it on the month it actually describes.
    """
    monthly_close = close.resample("ME").last()
    monthly_returns = monthly_close.pct_change(fill_method=None)
    raw = same_month_average_return(monthly_returns, min_years, max_years)
    raw = raw.shift(-1)
    raw_daily = raw.reindex(close.index).ffill()
    return cross_sectional_rank(raw_daily, sectors)


def build_all_seasonality_signals(close: pd.DataFrame, vol: pd.DataFrame, sectors: dict) -> dict:
    """Both parallel specs, no headline pick."""
    return {
        "half_month": half_month_signal(close, vol),
        "same_month": same_month_signal(close, sectors),
    }


# --- Phase 11c: seasonality as a TSMOM modifier (built 2026-07-31) ---
#
# Architecture (already decided, WORKFLOW.md Phase 11c, not re-litigated here):
# NOT a regime_lookup case (Book-level granularity) - a per-asset, alpha-
# construction-time modifier applied to signals.momentum.tsmom_signal's own
# output, upstream of Book entirely, same category as vol-scaling.
#
# Mechanism (already decided): a CONTINUOUS seasonal weight, not a binary
# gate - direct precedent against a gate already exists in this project
# (tsmom_deadband: best train Sharpe of 7 flavors, WORST validation Sharpe,
# higher turnover than continuously-resized alternatives).
#
# Windows reused VERBATIM from Phase 11c's own conviction table - fixed
# before any backtest (CLAUDE.md Rule 1), Medium confidence or higher only
# (matches SEASONALITY_ECONOMIC_DRIVER_ASSETS - LiveCattle/FeederCattle/
# LeanHogs excluded, "Lower confidence, needs a literature check," never
# resolved). Explicitly NOT applied to FX, Rates, Equity indices, or
# precious/industrial metals - no physical seasonal demand driver there.
TSMOM_SEASONAL_WINDOWS = {
    "Natural Gas": (11, 3),  # Nov-Mar, winter heating demand
    "HeatingOil": (10, 2),   # Oct-Feb, winter heating demand
    "RBOB": (4, 9),          # Apr-Sep, summer driving season + spring blend-switchover
    "Corn": (6, 8),          # Jun-Aug, growing-season weather risk (July pollination)
    "Soybeans": (6, 8),      # Jun-Aug, growing-season weather risk
    "Wheat": (3, 6),         # Mar-Jun, winter-wheat dormancy-through-harvest weather risk
    "KC_Wheat": (3, 6),      # Mar-Jun, winter-wheat dormancy-through-harvest weather risk
}
# Fixed a priori, not tuned from any backtest result (CLAUDE.md Rule 1/2) - a
# moderate, round-number conviction boost: up to +50% at a window's center,
# smoothly tapering to +0% (no modification at all) at its edges and beyond.
TSMOM_SEASONAL_AMPLITUDE = 0.5
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # non-leap reference year - a 1-day/4-year wobble is immaterial for a smooth seasonal weight


def _month_start_doy(month: int) -> int:
    return sum(_DAYS_IN_MONTH[: month - 1]) + 1


def _window_center_and_half_width(start_month: int, end_month: int) -> tuple:
    """Center and half-width of a (start_month, end_month) window, in
    day-of-year units (1-365), handling year-end wraparound (e.g. Nov-Mar)."""
    start_doy = _month_start_doy(start_month)
    end_doy = _month_start_doy(end_month) + _DAYS_IN_MONTH[end_month - 1] - 1
    if end_month >= start_month:
        length = end_doy - start_doy + 1
        center = start_doy + length / 2.0
    else:
        length = (365 - start_doy + 1) + end_doy
        center = (start_doy + length / 2.0 - 1) % 365 + 1
    return center, length / 2.0


def _circular_doy_distance(doy: pd.Series, center: float, period: float = 365.0) -> pd.Series:
    d = (doy - center).abs()
    return np.minimum(d, period - d)


def seasonal_weight_multiplier(
    index: pd.DatetimeIndex,
    assets=None,
    windows: dict = TSMOM_SEASONAL_WINDOWS,
    amplitude: float = TSMOM_SEASONAL_AMPLITUDE,
) -> pd.DataFrame:
    """(T x N) continuous multiplier, 1.0 everywhere except `assets` (default:
    every key in `windows`) during their own seasonal window, where it rises
    smoothly (a raised-cosine / Hann taper - continuously differentiable, no
    jump discontinuities) to `1 + amplitude` at the window's center and back
    down to exactly 1.0 at the window's edges and beyond. Sign-preserving by
    construction (always >= 1.0 - amplitude*0 = a pure magnitude scale, never
    flips or zeroes a position)."""
    assets = list(windows.keys()) if assets is None else assets
    doy = pd.Series(index.dayofyear.astype(float), index=index).clip(upper=365.0)

    result = pd.DataFrame(1.0, index=index, columns=assets)
    for asset in assets:
        if asset not in windows:
            continue
        center, half_width = _window_center_and_half_width(*windows[asset])
        dist = _circular_doy_distance(doy, center)
        hann = 0.5 * (1.0 + np.cos(np.pi * dist / half_width))
        hann = hann.where(dist <= half_width, 0.0)
        result[asset] = 1.0 + amplitude * hann
    return result


def tsmom_seasonal_signal(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    lookback_months: int = None,
    target_vol: float = None,
    windows: dict = TSMOM_SEASONAL_WINDOWS,
    amplitude: float = TSMOM_SEASONAL_AMPLITUDE,
) -> pd.DataFrame:
    """TSMOM (`signals.momentum.tsmom_signal`), scaled by the continuous
    seasonal weight above for the named assets, 1.0 (unchanged) for every
    other asset. `lookback_months`/`target_vol` default to `tsmom_signal`'s
    own headline-spec defaults if not given (not re-derived here)."""
    kwargs = {}
    if lookback_months is not None:
        kwargs["lookback_months"] = lookback_months
    if target_vol is not None:
        kwargs["target_vol"] = target_vol
    base = tsmom_signal(close, vol, **kwargs)

    assets = [a for a in windows if a in close.columns]
    weight = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    weight[assets] = seasonal_weight_multiplier(close.index, assets=assets, windows=windows, amplitude=amplitude)
    return base * weight
