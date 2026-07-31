"""
portfolio/sleeve_covariance.py — Sleeve-level (Book-level) covariance
estimators for `risk_parity.risk_parity_weights`.

Unlike `portfolio.covariance.build_cov_dict` (asset-level, produces a whole
`{date: matrix}` history keyed to real rebalance dates, for `Book`'s own
optimizer), these take a (T x N) sleeve-returns DataFrame — one column per
Book/strategy, e.g. Trend/Carry — and return a SINGLE current covariance
matrix: the risk-parity solver only ever needs today's Sigma, not a full
path. Built 2026-07-29 per the Trend/Carry combination discussion —
WORKFLOW.md decision #12.

Two estimators, both returned in decimal-return covariance units (same
units as `sleeve_returns.cov()`), so they're directly comparable and either
one can be dropped straight into `risk_parity.risk_parity_weights`:

- `rolling_covariance` / `ewma_covariance` — plain pandas sample covariance,
  the cheap baseline.
- `dcc_garch_covariance` — Engle (2002) DCC-GARCH conditional covariance,
  via the author's own already-validated local `dcc_garch` package (see
  `research/trend_correlation.py`, the same fitting pattern reused here
  verbatim). With only ~140-250 weekly sleeve observations, DCC's own
  correlation-recursion parameters (a, b) are fit on a small sample — cross-
  check against the simple estimators above rather than trusting this
  blindly, per direct instruction.

The `dcc_garch` import is deferred to inside `dcc_garch_covariance` (not at
module load time), same discipline as `data.garch_volatility`'s own lazy
import: it's a private local sibling repo, not a pip package, and the rest
of this module (the two pandas estimators) must keep working in any
environment — including the public dashboard's — where that sibling repo
doesn't exist.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROLLING_WINDOW = 63  # in observation units, not calendar time - caller picks window to match its own data frequency
DEFAULT_EWMA_HALFLIFE = 26  # in observation units, same convention as research/tune_all_books.py's EWMA_HALFLIFE

# dcc_garch itself scales returns x100 internally (see gjr_garch.py) so that
# GARCH sigmas come out in "% daily" units — H = sigma_i * R_ij * sigma_j is
# therefore in (%)^2 units. Divide by this to get back to decimal-return
# covariance units, matching rolling_covariance/ewma_covariance's output.
_DCC_PCT_SQUARED_TO_DECIMAL = 100.0 ** 2


def rolling_covariance(sleeve_returns: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW) -> pd.DataFrame:
    """Plain sample covariance over the trailing `window` observations ending
    at the last available date. Rows with any NaN are dropped first (a
    sleeve's PnL can start later than another's), then the most recent
    `window` clean rows are used — not point-in-time-indexed like
    `portfolio.covariance.build_cov_dict`, since the caller only ever wants
    "the current matrix," not a rebalance-dated history."""
    clean = sleeve_returns.dropna(how="any")
    windowed = clean.iloc[-window:]
    if len(windowed) < 2:
        raise ValueError(f"Need at least 2 clean observations, got {len(windowed)}")
    return windowed.cov()


def ewma_covariance(sleeve_returns: pd.DataFrame, halflife: float = DEFAULT_EWMA_HALFLIFE) -> pd.DataFrame:
    """EWMA sample covariance (pandas `.ewm(halflife=...).cov()`), evaluated
    at the last available date — heavier recent weighting than
    `rolling_covariance`'s flat window, same halflife-in-observation-units
    convention already used project-wide (e.g. `research/tune_all_books.py`'s
    `EWMA_HALFLIFE`)."""
    clean = sleeve_returns.dropna(how="any")
    if len(clean) < 2:
        raise ValueError(f"Need at least 2 clean observations, got {len(clean)}")
    cov_panel = clean.ewm(halflife=halflife).cov()
    last_date = clean.index[-1]
    return cov_panel.loc[last_date]


def dcc_garch_covariance(sleeve_returns: pd.DataFrame) -> dict:
    """DCC-GARCH (Engle 2002) conditional covariance, current date only.

    Fits a GJR-GARCH(1,1,1) independently per sleeve, then DCC on the
    standardized residuals — the exact same two-stage call sequence as
    `research/trend_correlation.py`'s `dcc_report`. Sleeve PnL is already
    strategy-level, vol-targeted, normal-magnitude daily/weekly return data
    (like trend_correlation.py's own inputs), so no per-asset rescale is
    applied here (contrast `data.garch_volatility`'s per-asset dynamic
    rescale, needed there for raw price-return series with wildly different
    native scales).

    Returns {"cov": pd.DataFrame (N x N, decimal-return units),
             "converged": bool} — the convergence flag is surfaced, not
    swallowed, since a small-sample DCC fit (~140-250 obs) can silently fail
    to converge to a stable (a, b).
    """
    clean = sleeve_returns.dropna(how="any")
    if len(clean) < 2:
        raise ValueError(f"Need at least 2 clean observations, got {len(clean)}")
    names = list(clean.columns)

    dcc_garch_src = Path(__file__).resolve().parent.parent.parent.parent / "DCC Garch Rompolis" / "src"
    if str(dcc_garch_src) not in sys.path:
        sys.path.insert(0, str(dcc_garch_src))
    from dcc_garch.garch.gjr_garch import fit_multivariate_gjr
    from dcc_garch.dcc.optimizer import fit as fit_dcc

    fit = fit_multivariate_gjr(clean.values)
    dcc = fit_dcc(fit["Z"], fit["sigmas"], model="DCC")

    H_last_pct2 = dcc["H"][-1]
    cov = pd.DataFrame(
        np.asarray(H_last_pct2) / _DCC_PCT_SQUARED_TO_DECIMAL,
        index=names, columns=names,
    )
    return {"cov": cov, "converged": bool(dcc["converged"])}
