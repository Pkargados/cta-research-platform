"""
signals/relative_value.py — Relative Value cointegration-spread sleeve (WORKFLOW.md
§11d), originally the seven pairs decided 2026-07-31, rebuilt from a genuine
greenfield state (§11d's "Real gap found 2026-08-01" addendum —
`Time_Series_Models.ipynb` never existed in this repo, so nothing here is ported
from prior work). Extended to 11 pairs 2026-08-01 (same day, follow-on session) —
see the four new entries below, added after examining which sectors in this
project's universe had ZERO RV exposure yet (Livestock, Grains-ex-wheat, Equities).

Three construction groups, one shared spread/signal mechanic (CLAUDE.md Rule 6):

- **RATIO_PAIRS** (fixed beta=1, log-price space): WTI-Brent, Gold-Silver,
  Platinum-Palladium, Wheat-KC_Wheat — a structural reason to track 1:1 in log
  terms (same commodity/units, or a real market-quoted ratio convention).
  **Russell2000-SP500, Nasdaq100-SP500** (added 2026-08-01) follow the same
  logic: equity-index relative-strength ratios (small-cap/large-cap,
  tech/broad-market) are themselves a real, commonly tracked market
  convention, not a regression-estimated relationship — same justification
  Gold-Silver's own docstring already uses ("the gold-silver ratio IS the
  market convention"). Log-ratio is scale-agnostic (Russell2000 ~2000s,
  SP500 ~5000s, Nasdaq100 ~18000s — only relative CHANGES matter, same
  reasoning `signals.crossover`/`signals.breakout` already rely on for
  absolute price level not mattering).
- **HEDGE_RATIO_PAIRS** (estimated, time-varying beta, baked off rolling-OLS vs.
  Kalman filter — `signals.hedge_ratio`): Corn-Wheat, RBOB-HeatingOil — no
  natural 1:1 equivalence. **LiveCattle-FeederCattle, Corn-Soybeans** (added
  2026-08-01) are the same shape: no natural 1:1 price ratio (feeder cattle
  are a DIFFERENT product stage from live cattle, not a fixed conversion;
  corn and soybeans trade at structurally different price levels per
  bushel), but a real economic linkage justifies testing a spread at all
  (feeder-to-live is a genuine production relationship — young cattle are
  fattened into live cattle, closer to crack spread's engineering logic than
  a market-convention ratio; corn-soybeans is the classic crop-substitution
  spread — farmers choose which to plant based on relative price).
- **Crack spread** (3-leg, fixed 3:2:1 economic ratio — refinery conversion
  chemistry, not statistically estimated at all): WTI/RBOB/HeatingOil. Crude leg
  is WTI, confirmed by direct instruction 2026-08-01 (NYMEX's own market
  convention; Brent not built as a parallel spec).

**Sectors considered and rejected for a new pair, 2026-08-01** (see the research
session's own writeup, not repeated here): Natural Gas (correlation with WTI/
HeatingOil both ~0.13-0.17 — confirmed no natural in-universe partner, not just
assumed); FX (conceptually overlaps with `signals.value`'s existing PPP-based
mean-reversion mechanism); yield curve (2s10s/5s30s-style spreads are the most
standard REAL rates-desk RV trade, but bond futures price sensitivity scales
with duration — a log-price-ratio spread would be dominated by duration
mismatch, not a clean curve signal; needs a duration-adjusted construction,
flagged as a separate later extension, not bundled in here).

Every pair (including crack) reduces to one LEVEL series (`spread`), z-scored on
a trailing window (`zscore_spread_signal`), then sized one of two ways — both
built as parallel specs per direct instruction 2026-08-01, no headline pick:

- `continuous_zscore_signal` — continuous, vol-scaled mean-reversion sizing
  (`-target_vol * z / vol`), matching CLAUDE.md Rule 5's own "binary
  underperforms continuous" finding and every other signal family's house
  style.
- `threshold_band_signal` — the classic literal pairs-trading construction
  (enter at |z| > entry_z, exit at |z| < exit_z, hold in between) — genuinely
  path-dependent, same per-asset walk-forward state-machine pattern as
  `signals.breakout.donchian_state_machine`.

Units caveat, disclosed not hidden: the ratio/Kalman-beta pairs' spread is in
LOG-price units (`diff()` approximates a $1-long/$1-short synthetic return, the
Gatev-Goetzmann-Rouwenhorst 1999 convention), while the crack spread's level is
a raw $/bbl refining margin (`diff()` is a raw dollar change, not a normalized
return) — these are NOT directly comparable magnitudes before pooling. Resolved
at the Book stage exactly the way `research/single_strategy_portfolios.py`'s
`build_book` already resolves the same problem for XSMOM/Value (rescale each
spread's alpha by its own train-period std before pooling) — not a new
resolution invented here.
"""

import numpy as np
import pandas as pd

from signals.hedge_ratio import fixed_ratio, rolling_ols_beta, kalman_hedge_ratio

RATIO_PAIRS = {
    "wti_brent": ("WTI Crude", "Brent"),
    "gold_silver": ("Gold", "Silver"),
    "platinum_palladium": ("Platinum", "Palladium"),
    "wheat_kcwheat": ("Wheat", "KC_Wheat"),
    "russell_sp500": ("Russell2000", "SP500"),
    "nasdaq_sp500": ("Nasdaq100", "SP500"),
}

# y, x order for the estimated-beta pairs: spread = log(y) - beta*log(x).
HEDGE_RATIO_PAIRS = {
    "corn_wheat": ("Corn", "Wheat"),
    "rbob_heatingoil": ("RBOB", "HeatingOil"),
    "livecattle_feedercattle": ("LiveCattle", "FeederCattle"),
    "corn_soybeans": ("Corn", "Soybeans"),
}

CRACK_SPREAD_NAME = "crack_spread"
CRACK_CRUDE_LEG = "WTI Crude"  # confirmed 2026-08-01: WTI, not Brent (NYMEX convention)
GALLONS_PER_BARREL = 42.0

DEFAULT_OLS_WINDOW = 252
DEFAULT_Z_WINDOW = 63
DEFAULT_VOL_WINDOW = 63
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5
DEFAULT_TARGET_VOL = 1.0

# Same tolerance/rationale as signals.hedge_ratio.MIN_FRAC (and crossover/
# volatility/covariance's own versions of this fix) — a strict min_periods ==
# window rolling mean/std found ZERO valid dates for Wheat/KC_Wheat (KC_Wheat
# is only 63% dense), checked live while building this module. Used for
# LEVEL-based rolling estimates (zscore_spread_signal's mean/std over the
# spread level itself), which run at roughly the same 80-95% density as the
# established use cases this 0.7 tolerance was already calibrated for.
MIN_FRAC = 0.7

# A SEPARATE, lower tolerance for realized_vol, which operates on
# spread_return (a DIFFERENCED series), not a level. Checked live: diff()
# structurally lowers density below the underlying spread's own — corn_wheat's
# spread is ~96% dense but its diff is only ~64% dense (an isolated one-day
# spread gap corrupts TWO diff values, not one), and Wheat/KC_Wheat's diff
# density is only ~50%. At those densities, MIN_FRAC=0.7 (calibrated for
# level series) leaves realized_vol almost entirely NaN (corn_wheat: 626 of
# 4170 dates; wheat_kcwheat: 8 of 5019) — not a marginal reduction in
# coverage, effectively broken. 0.5 was checked directly to recover the vast
# majority of dates for every one of the 7 pairs (corn_wheat: 4782;
# wheat_kcwheat: 3872; the more liquid pairs: >4900 each) without requiring a
# near-empty window.
VOL_MIN_FRAC = 0.5


# ---------------------------------------------------------------------------
# Spread construction
# ---------------------------------------------------------------------------

def estimate_beta(price: pd.DataFrame, leg_y: str, leg_x: str, method: str, window: int = DEFAULT_OLS_WINDOW) -> pd.Series:
    """`method`: "fixed" (beta=1, RATIO_PAIRS), "rolling_ols", or "kalman"
    (HEDGE_RATIO_PAIRS bake-off) — estimated in LOG-price space regardless of
    method, so `log_ratio_spread` always receives a beta on the same scale."""
    if method == "fixed":
        return fixed_ratio(price.index, 1.0)

    log_y = np.log(price[leg_y])
    log_x = np.log(price[leg_x])
    if method == "rolling_ols":
        return rolling_ols_beta(log_y, log_x, window=window).reindex(price.index)
    elif method == "kalman":
        return kalman_hedge_ratio(log_y, log_x)["beta"].reindex(price.index)
    else:
        raise ValueError("method must be 'fixed', 'rolling_ols', or 'kalman'")


def log_ratio_spread(price: pd.DataFrame, leg_y: str, leg_x: str, beta: pd.Series) -> pd.Series:
    """log(Y) - beta * log(X) — the mean-reverting object for every ratio/
    estimated-beta pair. `beta` reindexed to `price.index`; a NaN beta (e.g.
    before an OLS/Kalman warmup window fills) produces a NaN spread for that
    date, not a fabricated one."""
    log_y = np.log(price[leg_y])
    log_x = np.log(price[leg_x])
    return log_y - beta.reindex(price.index) * log_x


def crack_spread_level(price: pd.DataFrame, crude_leg: str = CRACK_CRUDE_LEG) -> pd.Series:
    """3:2:1 crack spread refining margin, $/bbl: `(2*RBOB + 1*HeatingOil) *
    42 gal/bbl / 3 - crude`. Fixed economic ratio (real refinery conversion
    chemistry — 2 barrels of gasoline + 1 barrel of heating oil per 3 barrels
    of crude), not a statistically estimated relationship — no beta, no
    rolling/Kalman estimation. RBOB/HeatingOil are quoted $/gallon on NYMEX;
    the 42 gal/bbl conversion puts them on the same $/bbl footing as crude
    before differencing. Can legitimately go negative (a real refining-margin
    compression event) — this is why the spread stays in raw dollar units
    rather than log-ratio space (undefined/discontinuous through zero)."""
    gasoline = price["RBOB"] * GALLONS_PER_BARREL
    heating_oil = price["HeatingOil"] * GALLONS_PER_BARREL
    crude = price[crude_leg]
    return (2.0 * gasoline + 1.0 * heating_oil) / 3.0 - crude


def spread_return(spread: pd.Series) -> pd.Series:
    """Day-over-day change in the spread level — the synthetic-instrument
    "return" `backtest.engine.backtest_signal` trades this pair against. For a
    log-ratio spread this approximates the continuously-compounded return of a
    $1-long-Y/$beta-short-X position (Gatev-Goetzmann-Rouwenhorst 1999's own
    $1-long/$1-short convention). For the crack spread's raw $/bbl level, this
    is a raw dollar change, not a normalized return — see module docstring for
    how that units mismatch is resolved (train-std rescaling at the Book
    stage, not here)."""
    return spread.diff()


def realized_vol(returns: pd.Series, window: int = DEFAULT_VOL_WINDOW, annualize: bool = True) -> pd.Series:
    """Trailing realized vol of a spread's own `spread_return` series — the
    substitute for Yang-Zhang OHLC vol every other signal family uses, since a
    synthetic spread has no real OHLC bars of its own. `shift(1)`'d is NOT
    applied here (consistent with `data.volatility`'s own convention — the
    final SIGNAL gets shift(1)'d once, inside `backtest.engine`, not every
    intermediate rolling estimate separately). `min_periods` uses VOL_MIN_FRAC,
    a lower tolerance than MIN_FRAC — see VOL_MIN_FRAC's own docstring for why
    a differenced return series needs a different threshold than a level."""
    vol = returns.rolling(window, min_periods=max(1, int(window * VOL_MIN_FRAC))).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


# ---------------------------------------------------------------------------
# Signal: z-score + two parallel sizing rules
# ---------------------------------------------------------------------------

def zscore_spread_signal(spread: pd.Series, window: int = DEFAULT_Z_WINDOW, min_periods: int = None) -> pd.Series:
    """Rolling z-score of the spread level: `(spread - rolling_mean) /
    rolling_std`. Positive z = spread rich relative to its own recent history
    (short the spread); negative z = spread cheap (long the spread) — sign
    convention consumed by both sizing rules below. `min_periods` defaults to
    `MIN_FRAC * window`, not a full-density `window` requirement (see
    MIN_FRAC's docstring)."""
    min_periods = min_periods or max(1, int(window * MIN_FRAC))
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std()
    return (spread - mean) / std


def continuous_zscore_signal(z: pd.Series, vol: pd.Series, target_vol: float = DEFAULT_TARGET_VOL) -> pd.Series:
    """Continuous, vol-scaled mean-reversion sizing: `-target_vol * z / vol` —
    magnitude-preserving (bigger deviation = more conviction, unlike a sign-
    only rule), the RV-specific mean-reversion analog of
    `signals.transforms.vol_targeted_sign_signal`'s momentum-direction sizing
    (negated here since RV bets AGAINST the current deviation, not with it)."""
    return -target_vol * z / vol


def threshold_band_signal(z: pd.Series, entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z) -> pd.Series:
    """Classic discrete pairs-trading entry/exit band: from flat, enter short
    when z > entry_z (spread too rich) or long when z < -entry_z (spread too
    cheap); once in a position, hold until |z| < exit_z (flatten — the exit
    band is deliberately looser than the entry band, same asymmetric-channel
    idea as `signals.breakout.donchian_state_machine`, so a position isn't
    stopped out by the same noise that would block a fresh entry). Genuinely
    path-dependent (which threshold applies depends on the state carried in
    from the prior day) — implemented as an explicit per-date walk-forward,
    same pattern as `donchian_state_machine`, not vectorized.

    Returns {-1, 0, 1, NaN} — a caller sizes it with
    `signals.transforms.vol_targeted_sign_signal` (sign() is a no-op on an
    already-{-1,0,1} series, same reuse `breakout_signal` already does)."""
    z_vals = z.to_numpy()
    n = len(z_vals)
    state = np.full(n, np.nan)
    current = 0.0
    started = False

    for i in range(n):
        zi = z_vals[i]
        if np.isnan(zi):
            state[i] = np.nan
            continue
        started = True
        if current == 0.0:
            if zi > entry_z:
                current = -1.0
            elif zi < -entry_z:
                current = 1.0
        else:
            if abs(zi) < exit_z:
                current = 0.0
        state[i] = current if started else np.nan

    return pd.Series(state, index=z.index)


# ---------------------------------------------------------------------------
# Per-pair spread + signal builders
# ---------------------------------------------------------------------------

def build_pair_spread(price: pd.DataFrame, pair_name: str, hedge_method: str = None, ols_window: int = DEFAULT_OLS_WINDOW) -> pd.Series:
    """Dispatch to the right construction group by `pair_name`. `hedge_method`
    ("rolling_ols" or "kalman") is required for a HEDGE_RATIO_PAIRS name, and
    ignored otherwise."""
    if pair_name in RATIO_PAIRS:
        leg_y, leg_x = RATIO_PAIRS[pair_name]
        beta = estimate_beta(price, leg_y, leg_x, method="fixed")
        return log_ratio_spread(price, leg_y, leg_x, beta)
    elif pair_name in HEDGE_RATIO_PAIRS:
        if hedge_method not in ("rolling_ols", "kalman"):
            raise ValueError("hedge_method must be 'rolling_ols' or 'kalman' for a HEDGE_RATIO_PAIRS name")
        leg_y, leg_x = HEDGE_RATIO_PAIRS[pair_name]
        beta = estimate_beta(price, leg_y, leg_x, method=hedge_method, window=ols_window)
        return log_ratio_spread(price, leg_y, leg_x, beta)
    elif pair_name == CRACK_SPREAD_NAME:
        return crack_spread_level(price)
    else:
        raise ValueError(f"unknown pair_name: {pair_name}")


def build_pair_signal(
    spread: pd.Series, entry_exit: str = "continuous",
    z_window: int = DEFAULT_Z_WINDOW, vol_window: int = DEFAULT_VOL_WINDOW,
    target_vol: float = DEFAULT_TARGET_VOL, entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z,
) -> pd.Series:
    """`entry_exit`: "continuous" (`continuous_zscore_signal`) or "threshold"
    (`threshold_band_signal`, vol-targeted via `sign()` same as breakout)."""
    z = zscore_spread_signal(spread, window=z_window)
    ret = spread_return(spread)
    vol = realized_vol(ret, window=vol_window)

    if entry_exit == "continuous":
        return continuous_zscore_signal(z, vol, target_vol=target_vol)
    elif entry_exit == "threshold":
        state = threshold_band_signal(z, entry_z=entry_z, exit_z=exit_z)
        return target_vol * np.sign(state) / vol
    else:
        raise ValueError("entry_exit must be 'continuous' or 'threshold'")


ALL_PAIR_NAMES = list(RATIO_PAIRS.keys()) + list(HEDGE_RATIO_PAIRS.keys()) + [CRACK_SPREAD_NAME]


def build_all_pair_spreads(price: pd.DataFrame, hedge_method: str = "rolling_ols", ols_window: int = DEFAULT_OLS_WINDOW) -> dict:
    """All 7 pairs' spread LEVEL series, keyed by pair name. `hedge_method`
    picks which estimator HEDGE_RATIO_PAIRS uses (bake off both by calling this
    twice — once per method — in the research driver, not by building both
    inline here)."""
    return {name: build_pair_spread(price, name, hedge_method=hedge_method, ols_window=ols_window) for name in ALL_PAIR_NAMES}


def build_all_pair_signals(spreads: dict, entry_exit: str = "continuous", **signal_kwargs) -> dict:
    """All 7 pairs' signals from an already-built `spreads` dict (see
    `build_all_pair_spreads`), one `entry_exit` spec at a time — the research
    driver calls this twice (continuous, threshold) to report both as parallel
    specs, no headline pick."""
    return {name: build_pair_signal(spread, entry_exit=entry_exit, **signal_kwargs) for name, spread in spreads.items()}


def build_all_pair_returns(spreads: dict) -> dict:
    return {name: spread_return(spread) for name, spread in spreads.items()}
