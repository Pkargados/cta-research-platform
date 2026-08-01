"""
portfolio/gerber_covariance.py — Gerber statistic covariance estimator
(Gerber, Markowitz, Ernst, Miao, Javid, Sargen 2021, "The Gerber Statistic:
A Robust Co-Movement Measure for Portfolio Construction", `references/The
Gerber Statistic.pdf`, read directly). Scoped as a third covariance-
estimator candidate alongside the current Ledoit-Wolf default
(`portfolio.covariance.build_cov_dict`) — WORKFLOW.md's "Gerber statistic
covariance" plan, Phase 7. Diagnostic-only at this stage (see
`research/covariance_estimator_comparison.py`); not wired into any live
Book — that is a separate, later decision per the plan's own scope
boundary.

Formula is the paper's own Eq. 11, the only version the paper's empirical
results actually use — the naive Eq. 4 version is NOT positive-semidefinite
-safe and is not implemented here:

    g_ij = (n_UU + n_DD - n_UD - n_DU) / (T - n_NN)

For each asset k, a threshold H_k = c * s_k (s_k = that asset's own sample
std dev over the window passed in). A date/asset is Up if its return is >=
+H_k, Down if <= -H_k, Neutral otherwise (and excluded entirely if the
return is missing — NaN is not Neutral). A pair's date is concordant (UU or
DD) if both sides pierce their threshold in the same direction, discordant
(UD or DU) if opposite directions; any date where at least one side is
Neutral contributes to neither the concordant nor discordant count, but
DOES still count in the denominator unless BOTH sides are Neutral (n_NN) —
that asymmetry is the paper's own designed noise penalty, not an
approximation of it.

Real, disclosed difference from the paper: **per-pair T_ij, not one global
T**. The paper's own 9-index dataset is fully overlapping; this project's
~40-asset futures panel has ragged per-asset histories (different listing
dates, different exchange calendars), so every pair's own denominator here
reflects only that pair's own jointly-valid (non-NaN) date count — unlike
`portfolio.covariance.build_cov_dict`'s Ledoit-Wolf fit, which drops an
entire window row if ANY asset in the whole panel is missing that date.
Gerber therefore retains more data per pair than Ledoit-Wolf sees in the
same nominal window — a real methodological difference to account for when
comparing the two, not a bug in either.

PSD handling: the paper only observed empirical PSD-ness on 9 clean,
fully-overlapping indices — not proven as a theorem, and not assumed here
to hold unchecked on this project's noisier, raggeder panel. Every
covariance this module builds goes through `_nearest_psd_correlation`
(eigenvalue-clip, then rescale back to a unit diagonal) before being
returned, and `gerber_covariance` also drops whichever assets can't form a
complete (NaN-free) pairwise correlation sub-matrix, rather than silently
feeding NaN downstream (CLAUDE.md Rule 4's "label missing, don't fake it"
principle, applied here to a covariance entry instead of a carry proxy).

Caveat for any future live-Book use (not exercised by this diagnostic
pass): `build_gerber_cov_dict`'s per-date matrices can have NaN entries for
assets that lacked enough coverage in that specific window, unlike
`build_cov_dict`'s output which is always fully dense once a date is
produced at all. A real Book's `Sigma_t = cov_dict[date].loc[assets,
assets]` (see `book.py`) would need those dates skipped or that asset
excluded before this could be a true drop-in replacement — deferred until
the diagnostic in `research/covariance_estimator_comparison.py` shows a
real forecast-accuracy edge, per the plan.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from portfolio.covariance import real_period_end_dates, DEFAULT_MIN_FRAC

DEFAULT_C = 0.5
DEFAULT_WINDOW = 252  # matches portfolio.covariance.DEFAULT_WINDOW - same lookback, for an apples-to-apples comparison
DEFAULT_FREQ = "ME"  # caller overrides to "W-FRI" to match Book's own weekly cadence, same convention as build_cov_dict
_PSD_EIGENVALUE_FLOOR = 1e-8


def _up_down_neutral(returns: pd.DataFrame, c: float):
    """Up/Down/Neutral boolean indicator DataFrames, one column per asset.
    NaN returns are excluded from all three (see module docstring)."""
    valid = returns.notna()
    threshold = c * returns.std(ddof=1)
    up = valid & returns.ge(threshold, axis=1)
    down = valid & returns.le(-threshold, axis=1)
    neutral = valid & ~up & ~down
    return valid, up, down, neutral


def gerber_correlation(returns: pd.DataFrame, c: float = DEFAULT_C) -> pd.DataFrame:
    """Gerber statistic correlation matrix (Eq. 11) over the full `returns`
    panel passed in — caller controls the estimation window by slicing
    `returns` before calling, same convention as `sleeve_covariance`'s
    estimators. Uses per-pair T_ij, not one global T (see module
    docstring).

    `g_ii == 1` exactly by construction: an asset can't be simultaneously
    Up and Down against itself on the same date, so n_UD = n_DU = 0 and the
    numerator reduces to n_UU + n_DD, exactly equal to the denominator
    (T_ii - n_NN_ii = n_up + n_down). The diagonal is still pinned
    explicitly below to avoid any floating-point division noise.

    NaN in the output means a pair had no usable joint signal for this
    window (every jointly-valid date was Neutral for at least one side, or
    there were zero jointly-valid dates at all) — left as NaN, not silently
    zeroed.
    """
    valid, up, down, neutral = _up_down_neutral(returns, c)
    U = up.astype(float).to_numpy()
    D = down.astype(float).to_numpy()
    N = neutral.astype(float).to_numpy()
    V = valid.astype(float).to_numpy()

    n_UU = U.T @ U
    n_DD = D.T @ D
    n_UD = U.T @ D
    n_DU = D.T @ U
    n_NN = N.T @ N
    T_ij = V.T @ V

    numer = n_UU + n_DD - n_UD - n_DU
    denom = T_ij - n_NN

    g = np.full(numer.shape, np.nan)
    ok = denom > 0
    g[ok] = numer[ok] / denom[ok]
    g = np.clip(g, -1.0, 1.0)
    np.fill_diagonal(g, 1.0)
    return pd.DataFrame(g, index=returns.columns, columns=returns.columns)


def _nearest_psd_correlation(corr: np.ndarray, floor: float = _PSD_EIGENVALUE_FLOOR) -> np.ndarray:
    """Eigenvalue-clip `corr` to PSD if it isn't already, then rescale back
    to a unit diagonal so it remains a valid correlation matrix after
    clipping. No-op (returns `corr` unchanged) if already PSD — the paper
    only observed PSD empirically on 9 clean assets, not proven as a
    theorem, so this is checked rather than assumed (see module
    docstring)."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    if eigvals.min() >= -1e-10:
        return corr
    clipped = np.clip(eigvals, floor, None)
    reconstructed = eigvecs @ np.diag(clipped) @ eigvecs.T
    d = np.sqrt(np.diag(reconstructed))
    rescaled = reconstructed / np.outer(d, d)
    np.fill_diagonal(rescaled, 1.0)
    return rescaled


def drop_until_complete(g: pd.DataFrame) -> pd.DataFrame:
    """Repeatedly drop whichever asset has the most NaN pairwise entries
    until the remaining sub-matrix has no NaN off-diagonal entries left —
    guarantees a complete matrix to eigen-decompose. Some assets can
    individually be well-covered yet still fail to share enough valid
    dates with specific peers (see `gerber_correlation`'s own NaN note);
    this resolves down to whichever subset is mutually complete, rather
    than failing the whole matrix. Terminates in at most `len(g.columns)`
    passes.

    Generic to any symmetric NaN-containing matrix, not Gerber-specific —
    `research/covariance_estimator_comparison.py` reuses this directly to
    clean up Ledoit-Wolf/sample-covariance matrices too, rather than
    reimplementing the same NaN-elimination logic (CLAUDE.md Rule 6)."""
    remaining = list(g.columns)
    while remaining:
        sub = g.loc[remaining, remaining]
        nan_counts = sub.isna().sum()
        if nan_counts.max() == 0:
            return sub
        remaining = [a for a in remaining if a != nan_counts.idxmax()]
    return g.loc[[], []]


def gerber_covariance(returns: pd.DataFrame, c: float = DEFAULT_C) -> pd.DataFrame:
    """Gerber covariance: PSD-corrected `gerber_correlation` rescaled by
    each surviving asset's own sample std dev over the same `returns`
    window (`diag(sigma) @ G @ diag(sigma)`, the paper's own construction,
    the same correlation-to-covariance conversion `portfolio.covariance`'s
    module docstring already describes for Ledoit-Wolf).

    May return a matrix over a SUBSET of `returns.columns` — any asset that
    can't form a complete, NaN-free correlation sub-matrix with the rest
    (see `drop_until_complete`) is dropped rather than propagating NaN.
    Returns an empty DataFrame if no such subset of size >= 1 survives.
    """
    g_raw = gerber_correlation(returns, c=c)
    g_complete = drop_until_complete(g_raw)
    if g_complete.empty:
        return g_complete
    g_psd = _nearest_psd_correlation(g_complete.to_numpy())
    sigma = returns[g_complete.columns].std(ddof=1).to_numpy()
    cov = np.diag(sigma) @ g_psd @ np.diag(sigma)
    return pd.DataFrame(cov, index=g_complete.columns, columns=g_complete.columns)


def build_gerber_cov_dict(returns_df: pd.DataFrame, window: int = DEFAULT_WINDOW, freq: str = DEFAULT_FREQ,
                           c: float = DEFAULT_C, min_frac: float = DEFAULT_MIN_FRAC) -> OrderedDict:
    """Rolling Gerber covariance, one matrix per real rebalance date (see
    `portfolio.covariance.real_period_end_dates`) — same date-indexed dict
    shape as `build_cov_dict`, for direct side-by-side comparison in
    `research/covariance_estimator_comparison.py`. Same `window`/`freq`/
    full-warmup convention as `build_cov_dict` (a date is skipped unless
    `window` full trading days of history precede it) so the two estimators
    are compared at an identical lookback and cadence, not confounded by
    different window lengths.

    Per date, an asset needs >= `window * min_frac` valid (non-NaN)
    observations within that window to be considered at all (same
    DEFAULT_MIN_FRAC tolerance `build_cov_dict` already uses, reused not
    re-guessed). Assets failing that — or failing `drop_until_complete`'s
    own completeness requirement inside `gerber_covariance` — are NaN in
    that date's matrix rather than
    skipping the whole date; this is the actual point of Gerber's per-pair
    T_ij design versus Ledoit-Wolf's all-or-nothing row-wise dropna (see
    module docstring's live-Book caveat before ever treating this as a
    literal drop-in replacement).
    """
    min_obs = max(2, int(window * min_frac))
    reb_dates = real_period_end_dates(returns_df.index, freq)
    cov_dict = OrderedDict()
    for date in reb_dates:
        window_data = returns_df.loc[:date].iloc[-window:]
        if len(window_data) < window:
            continue
        eligible_cols = window_data.columns[window_data.notna().sum() >= min_obs]
        if len(eligible_cols) < 2:
            continue
        cov = gerber_covariance(window_data[eligible_cols], c=c)
        if cov.empty:
            continue
        full = pd.DataFrame(np.nan, index=returns_df.columns, columns=returns_df.columns)
        full.loc[cov.index, cov.columns] = cov.to_numpy()
        cov_dict[date] = full
    return cov_dict
