"""
signals/hedge_ratio.py — Hedge-ratio construction for the Relative Value sleeve
(WORKFLOW.md §11d), three methods, one per construction group:

1. `fixed_ratio` — beta=1 in log-price space (`log(A) - log(B)`), no estimation
   at all. For pairs with a structural reason to track 1:1 in log terms: WTI-Brent,
   Gold-Silver, Platinum-Palladium, Wheat-KC_Wheat.
2. `rolling_ols_beta` — rolling-window OLS hedge ratio (`Cov(y,x)/Var(x)` over a
   trailing window — algebraically the rolling-regression slope, no intercept
   needed since the spread is z-scored downstream anyway). NEVER a static
   full-sample OLS (CLAUDE.md Hard Rule 2 — the exact look-ahead bug this
   project's own dossier already found once) — always trailing-window.
3. `kalman_hedge_ratio` — time-varying [alpha, beta] via a 2-state Kalman filter
   (the classic Ernest-Chan-style pairs-trading construction: `y_t = alpha_t +
   beta_t * x_t + eps_t`, state follows a random walk). Implemented directly in
   numpy (~40 lines) rather than adding `pykalman`/`filterpy` as a dependency —
   both confirmed absent from this environment, and this project already prefers
   a from-scratch implementation of a well-specified estimator over a niche
   package (e.g. the Gerber statistic, built from the paper's own formula rather
   than pulled from pip). Baked off against `rolling_ols_beta` per pair
   (`research/relative_value.py`), never adopted by assumption.

Corn-Wheat and RBOB-HeatingOil (no natural 1:1 equivalence) use methods 2/3,
baked off against each other. The crack spread's 3:2:1 economic ratio lives in
`signals.relative_value` instead — it's a fixed real-world conversion ratio, not
a statistically estimated one, so it doesn't belong in an "estimation methods"
module.

All functions are point-in-time safe by construction: at any date t, a rolling
or Kalman estimate uses only data through t (inclusive) — the FINAL spread
signal built from these ratios is still `shift(1)`'d before trading inside
`backtest.engine` (CLAUDE.md Rule 3), same convention as every other signal.
"""

import numpy as np
import pandas as pd

DEFAULT_OLS_WINDOW = 252
DEFAULT_KALMAN_DELTA = 1e-4
DEFAULT_KALMAN_VE = 1e-3

# Same tolerance and rationale as signals.crossover's MIN_FRAC / data.volatility's
# min_frac / portfolio.covariance's DEFAULT_MIN_FRAC: this project's multi-decade,
# multi-asset panel has scattered per-asset NaN gaps dense enough that a strict
# min_periods == window (require every row in the window non-null) finds ZERO
# valid windows for a sparser-calendar pair — checked live for Corn/Wheat (max
# consecutive jointly-valid streak: 70 days, well under a 252-day window) and
# Wheat/KC_Wheat (KC_Wheat only 63% dense) while building this module; a strict
# min_periods produced 0 non-NaN beta estimates for either pair. Reusing the
# already-validated 0.7 tolerance rather than a fresh guess.
MIN_FRAC = 0.7


def fixed_ratio(index: pd.Index, beta: float = 1.0) -> pd.Series:
    """Constant hedge ratio (beta=1 by default) — no estimation, for pairs with
    a structural 1:1 log-price relationship."""
    return pd.Series(beta, index=index)


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int = DEFAULT_OLS_WINDOW, min_periods: int = None) -> pd.Series:
    """Rolling-window OLS slope of y on x: `Cov(y,x)_t / Var(x)_t` over the
    trailing `window` observations ending at t — the vectorized equivalent of
    re-running a windowed linear regression at every date, without a per-date
    Python loop. No intercept term: the spread built from this beta is z-scored
    downstream (`signals.relative_value.zscore_spread_signal`), which already
    demeans it, so a separately-estimated intercept would be redundant.

    `min_periods` defaults to `MIN_FRAC * window` (not a full-density `window`
    requirement — see MIN_FRAC's own docstring for why a strict requirement
    finds zero valid windows for this project's sparser-calendar pairs).
    """
    min_periods = min_periods or max(1, int(window * MIN_FRAC))
    common = y.index.union(x.index)
    y, x = y.reindex(common), x.reindex(common)
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var()
    return cov / var


DEFAULT_KALMAN_WARMUP = 60


def kalman_hedge_ratio(
    y: pd.Series, x: pd.Series, delta: float = DEFAULT_KALMAN_DELTA, ve: float = DEFAULT_KALMAN_VE,
    warmup: int = DEFAULT_KALMAN_WARMUP,
) -> pd.DataFrame:
    """Time-varying [alpha, beta] via a 2-state Kalman filter: observation
    `y_t = alpha_t + beta_t * x_t + eps_t`, state `[alpha_t, beta_t]` follows a
    random walk with transition covariance `Vw = delta/(1-delta) * I` (the
    standard Ernest-Chan pairs-trading parameterization — `delta` controls how
    fast the estimated beta is allowed to drift; smaller delta -> smoother,
    slower-moving beta). `ve` is the assumed observation-noise variance.

    Point-in-time safe by construction: the state estimate at t is the
    predict/update of the state at t-1 against the OBSERVATION at t only — no
    future observation ever enters the recursion for an earlier date, and the
    filter never re-runs backward (no smoothing pass).

    **`warmup` (added 2026-08-01, a real bug found live, not a pre-emptive
    guess)**: the first `warmup` alpha/beta values are masked to NaN. The
    module docstring's own original claim ("beta is genuinely uninformative
    for the first several observations, same real-warmup discipline as every
    other rolling/recursive estimator") was true in PROSE but never actually
    enforced in code — every prior version returned a real (fabricated-looking)
    value from observation 1 onward. Checked directly on LiveCattle-
    FeederCattle: theta starts at a flat [0, 0] prior, so beta jumps from 0
    toward its steady state (~0.86) within the first 2-3 observations — this
    swing alone produced a single-day spread `diff()` of -0.93 (economically
    absurd) that dominated the ENTIRE multi-decade series' reported std
    (excluding just the first 5 observations cut `spread_return`'s std by
    ~10x, from 0.017 to 0.0018) — silently corrupting `realized_vol` and every
    downstream Sharpe/turnover number for every Kalman-based pair, not a
    localized artifact. `warmup=60` (~1 quarter) checked directly against all
    4 Kalman-consuming pairs at the time this was added (Corn-Wheat,
    RBOB-HeatingOil, Corn-Soybeans, LiveCattle-FeederCattle): the first three
    converge within ~20 observations (a fast cold-start jump, exactly the
    failure mode this masks); RBOB-HeatingOil converges much more slowly and
    smoothly (still drifting at observation 300) but produces no single-day
    outlier the way the other three do, so a 60-day mask costs it negligible
    history without fixing a problem it didn't have.

    Returns a DataFrame (columns "alpha", "beta") indexed by the intersection
    of y's and x's non-NaN dates, first `warmup` rows NaN.
    """
    common = y.dropna().index.intersection(x.dropna().index).sort_values()
    y_vals = y.loc[common].to_numpy(dtype=float)
    x_vals = x.loc[common].to_numpy(dtype=float)
    n = len(common)

    theta = np.zeros(2)
    P = np.zeros((2, 2))
    Vw = (delta / (1.0 - delta)) * np.eye(2)

    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)

    for t in range(n):
        H = np.array([1.0, x_vals[t]])

        P = P + Vw
        yhat = H @ theta
        e = y_vals[t] - yhat
        Q = H @ P @ H.T + ve
        K = (P @ H) / Q
        theta = theta + K * e
        P = P - np.outer(K, H) @ P

        alphas[t] = theta[0]
        betas[t] = theta[1]

    if warmup > 0:
        alphas[:warmup] = np.nan
        betas[:warmup] = np.nan

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=common)
