"""
signals/error_correction.py — Error-Correction Model (ECM) half-life estimation for
the Relative Value sleeve, replacing the uniform 63-day z-score window
(`signals.relative_value.DEFAULT_Z_WINDOW`) with a per-pair, half-life-informed one.

Per direct instruction 2026-08-02, following up on `references/Mean Reversion Using
Machine Learning.pdf` (a low-credibility source — see WORKFLOW.md's own writeup — but
its "calibrate the lookback from mean-reversion speed" idea is worth testing on its own
merits, independent of that paper's rigor).

Single-equation ECM on the spread (residual) series — the Engle-Granger two-step
procedure's own natural second step (`signals.cointegration.rolling_engle_granger` only
covers step 1, the cointegrating regression, plus a stationarity test on the residual;
this module is the step-2 DYNAMICS estimate that was never built):

    Δspread_t = c + λ·spread_{t-1} + Σ_{k=1}^{p} φ_k·Δspread_{t-k} + ε_t

`λ` is the error-correction (adjustment) speed: how much of a deviation from the
spread's own trailing level gets corrected each period. `half_life = -ln(2)/λ` is the
number of periods for half of a deviation to decay — undefined (NaN, not fabricated)
when `λ >= 0` (no correction, or explosive/non-mean-reverting over that window).

**Deliberately NOT a full bivariate VECM** (separate equations for each leg's own
Δy_t/Δx_t, which would additionally reveal ASYMMETRIC adjustment speed between the two
legs — a genuinely richer question). The practical goal here is one half-life number
per pair to calibrate a signal-construction window, not a standalone research question
about leg-level asymmetry — a deliberate, disclosed scope decision (CLAUDE.md: "don't
add complexity for its own sake"), not an oversight. Flagged as a real extension if the
asymmetry question ever becomes its own question worth answering.

`lags=1` default (a single lagged difference term as a short-run control) — this is
what distinguishes an ECM from the simpler, un-augmented AR(1)-on-the-residual half-life
estimate (Ernest Chan's own popular version, `Δspread_t = λ·spread_{t-1} + ε_t`, no
lagged-difference terms at all): the lagged term controls for short-run serial
correlation in the spread's own returns, which would otherwise bias λ. Not a full
AIC-selected lag order — a small, fixed, documented choice, since this module's goal is
a stable calibration input, not a formal hypothesis test where lag-order
misspecification would bias a p-value.

Point-in-time safe by construction (CLAUDE.md Hard Rule 2): `rolling_half_life` walks
forward using ONLY a trailing window ending at each test date, mirroring
`signals.cointegration.rolling_engle_granger`'s own convention exactly. `train_half_life`
fits ONCE on the train-period spread only (CLAUDE.md Rule 1/2 — a window-length
CALIBRATION decided from train data alone, applied as a fixed constant thereafter, not
re-estimated by looking at validation/test).
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 252
DEFAULT_STEP = 21
DEFAULT_LAGS = 1
MIN_OBS_FOR_FIT = 30  # need real degrees of freedom beyond [const, level_lag1, lag diffs]

DEFAULT_WINDOW_MULTIPLIER = 2.0  # z-score window = multiplier * half-life
MIN_IMPLIED_WINDOW = 10
MAX_IMPLIED_WINDOW = 252


def _ecm_design(spread: pd.Series, lags: int) -> tuple:
    """Build (y, X) for the ECM regression over the WHOLE input series (no
    windowing here — callers slice the input before calling). y = Δspread_t,
    X = [const, spread_{t-1}, Δspread_{t-1}, ..., Δspread_{t-lags}]. Rows with
    any NaN (warmup, or a gap in `spread` itself) are dropped."""
    level_lag1 = spread.shift(1)
    dspread = spread.diff()
    cols = {"const": pd.Series(1.0, index=spread.index), "level_lag1": level_lag1}
    for k in range(1, lags + 1):
        cols[f"dlag{k}"] = dspread.shift(k)
    X = pd.DataFrame(cols, index=spread.index)
    y = dspread
    valid = X.notna().all(axis=1) & y.notna()
    return y[valid], X[valid]


def fit_ecm(spread: pd.Series, lags: int = DEFAULT_LAGS) -> dict:
    """Fit the single-equation ECM on one spread series (already the window the
    caller wants fit on — no internal windowing). Returns
    {"lambda": float|nan, "half_life": float|nan, "n_obs": int}.

    `half_life` is NaN whenever `lambda >= 0` (not mean-reverting over this
    window) or there isn't enough data to fit at all — a genuine "undefined",
    not a fabricated fallback value."""
    y, X = _ecm_design(spread.dropna(), lags)
    n_obs = len(y)
    if n_obs < MIN_OBS_FOR_FIT:
        return {"lambda": np.nan, "half_life": np.nan, "n_obs": n_obs}

    beta, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
    lam = float(beta[1])  # coefficient on level_lag1
    half_life = -np.log(2.0) / lam if lam < 0 else np.nan
    return {"lambda": lam, "half_life": half_life, "n_obs": n_obs}


def rolling_half_life(spread: pd.Series, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP, lags: int = DEFAULT_LAGS) -> pd.DataFrame:
    """Walk forward in `step`-sized increments; at each test point, fit the ECM
    on the trailing `window` observations ending at that point (inclusive) —
    strictly historical, no look-ahead, same pattern as
    `signals.cointegration.rolling_engle_granger`.

    Returns a DataFrame indexed by the LAST date of each trailing window,
    columns ["lambda", "half_life", "n_obs"]. Empty if `spread` is too short
    for even one window.
    """
    s = spread.dropna().sort_index()
    n = len(s)

    rows = {}
    for i in range(window, n + 1, step):
        window_slice = s.iloc[i - window:i]
        date = s.index[i - 1]
        rows[date] = fit_ecm(window_slice, lags=lags)

    if not rows:
        return pd.DataFrame(columns=["lambda", "half_life", "n_obs"])
    return pd.DataFrame(rows).T.sort_index()


def train_half_life(spread: pd.Series, train_end, lags: int = DEFAULT_LAGS) -> float:
    """Single ECM fit on the FULL train-period spread (`spread.loc[:train_end]`)
    — used to pick ONE fixed z-score window per pair, decided from train data
    only (CLAUDE.md Rule 1/2), not re-estimated walk-forward for the live
    signal. Returns half-life (float) or NaN if undefined."""
    train_spread = spread.loc[:train_end]
    return fit_ecm(train_spread, lags=lags)["half_life"]


def median_rolling_half_life(
    spread: pd.Series, train_end, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP, lags: int = DEFAULT_LAGS,
) -> float:
    """Median of `rolling_half_life`'s own per-window estimates, restricted to
    windows ending at or before `train_end` — a more robust train-period
    calibration than `train_half_life`'s single long regression.

    Checked directly on RBOB-HeatingOil while building this: the single
    full-train regression gives one number (~97 days), but the underlying
    process is genuinely time-varying (rolling 252-day estimates over the
    same train span range from ~9 to ~350 days, median ~21) — a single long
    OLS fit averages over regimes the pair actually moved through, which is
    a real loss of information, not just noise. The median of many shorter
    rolling fits is a more robust summary of "typical" reversion speed than
    one regression spanning the whole, regime-mixed train period.
    """
    result = rolling_half_life(spread, window=window, step=step, lags=lags)
    if result.empty:
        return np.nan
    train_rows = result.loc[result.index <= pd.Timestamp(train_end)]
    if train_rows.empty:
        return np.nan
    return float(train_rows["half_life"].median())


def half_life_to_window(
    half_life: float, multiplier: float = DEFAULT_WINDOW_MULTIPLIER,
    min_window: int = MIN_IMPLIED_WINDOW, max_window: int = MAX_IMPLIED_WINDOW,
    fallback: int = None,
) -> int:
    """`round(multiplier * half_life)`, clipped to `[min_window, max_window]` —
    a "look back roughly `multiplier` mean-reversion cycles" heuristic (2x is a
    common practitioner convention, not backtested/tuned here — CLAUDE.md Rule
    1/2: this is a construction DECISION, not something selected by comparing
    validation Sharpe across multiplier choices).

    `fallback` (default None, meaning "raise" is NOT the behavior — see below):
    when `half_life` is NaN or non-positive (undefined — no genuine mean
    reversion detected over the fitting window), returns `fallback` if given,
    else NaN (propagates the "undefined" state rather than silently guessing a
    window, matching this project's own "don't fabricate a value for a
    genuinely undefined estimate" discipline). A caller wiring this into
    `signals.relative_value.zscore_spread_signal` should pass its own default
    (e.g. `DEFAULT_Z_WINDOW=63`) as `fallback` explicitly, not rely on a
    silent internal guess.
    """
    if half_life is None or np.isnan(half_life) or half_life <= 0:
        return fallback if fallback is not None else np.nan
    window = int(round(multiplier * half_life))
    return int(np.clip(window, min_window, max_window))
