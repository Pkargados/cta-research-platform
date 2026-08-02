"""
signals/ipca.py — Instrumented Principal Components Analysis (Kelly, Pruitt &
Su 2018, "Characteristics Are Covariances," read directly -
references/Characteristics are Covariances IPCA.pdf), implemented directly in
numpy - the paper's own ALS algorithm, "solvable via alternating least
squares in a matter of seconds even for high dimension systems" - rather
than adding a new dependency (no maintained Python IPCA package is installed
in this environment, checked directly: `pip show ipca` / `import ipca` both
fail). Same convention this project already uses for the Kalman filter and
Gerber statistic: a well-specified, compact estimator implemented from
scratch over a niche package.

**Scale caveat, logged before any use, not after** (WORKFLOW.md §11e): this
is a SMALL-N adaptation (a handful of Rates futures), not the paper's own
large-cross-section setting (12,000+ stocks). The estimator below uses the
paper's own explicitly-sanctioned APPROXIMATION (Section 2.1.1: "if we were
to approximate the Rayleigh quotient denominators with a constant... the
solution... would be to set Gamma_beta equal to the first K eigenvectors of
the sample second moment matrix of managed portfolio returns... This is a
close approximation to the exact solution as long as Z_t'Z_t is not too
volatile") rather than the full per-date-reweighted ALS - a deliberate,
disclosed simplification given this project's tiny, slowly-varying
characteristic panel, not a claim of exact equivalence to the paper's own
estimator.

Gamma_alpha (the anomaly/mispricing loading) is estimated by extending the
identical approximation: treating Z_t'Z_t as roughly time-invariant lets
Gamma_alpha be recovered via one more closed-form step (the time-average of
the restricted model's own managed-portfolio residual, GMM-style) rather
than the paper's own joint numerical estimation of the unrestricted model -
significance is still assessed via a residual bootstrap in the same spirit
as the paper's own (Section 3.1.1: resample the residual managed-portfolio
series, wild-bootstrap with a random Student-t multiplier), just applied to
this simplified estimator.
"""

import numpy as np
import pandas as pd


def _valid_mask(z_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
    return ~np.isnan(r_t) & ~np.isnan(z_t).any(axis=1)


def build_managed_portfolios(z_panel: np.ndarray, r_next: np.ndarray):
    """z_panel: (T, N, L) characteristics, r_next: (T, N) forward returns
    (already aligned - r_next[t] is the return realized from t to t+1, the
    caller's job to align via shift, matching this project's own shift(1)
    discipline elsewhere - CLAUDE.md Rule 3).

    Returns X (T, L) managed-portfolio returns (`x_t = Z_t' r_{t+1}`, the
    paper's own equation 9) and ZZ (T, L, L) per-date Z_t'Z_t, both computed
    only over assets with a non-missing characteristic row AND a
    non-missing forward return that date (the paper's own footnote 13
    NaN-tolerant convention)."""
    T, N, L = z_panel.shape
    X = np.full((T, L), np.nan)
    ZZ = np.full((T, L, L), np.nan)
    for t in range(T):
        valid = _valid_mask(z_panel[t], r_next[t])
        if valid.sum() == 0:
            continue
        z_v, r_v = z_panel[t][valid], r_next[t][valid]
        X[t] = z_v.T @ r_v
        ZZ[t] = z_v.T @ z_v
    return X, ZZ


def fit_ipca_restricted(z_panel: np.ndarray, r_next: np.ndarray, k: int) -> dict:
    """Restricted (Gamma_alpha=0) IPCA fit via the paper's own sanctioned
    eigenvector approximation (see module docstring): Gamma_beta = the first
    `k` eigenvectors of the managed portfolios' own sample second-moment
    matrix (valid dates only), and each date's factor realization is the
    projection of that date's managed-portfolio vector onto Gamma_beta -
    exactly the paper's own "first K principal components of the managed
    portfolio panel" description of this approximation.

    Returns dict: Gamma_beta (L,K), F (T,K), X (T,L), ZZ (T,L,L), valid_t
    (bool mask over T), eigvals (L,, descending)."""
    X, ZZ = build_managed_portfolios(z_panel, r_next)
    valid_t = ~np.isnan(X).any(axis=1)

    X_valid = X[valid_t]
    second_moment = X_valid.T @ X_valid
    eigvals, eigvecs = np.linalg.eigh(second_moment)
    order = np.argsort(eigvals)[::-1]
    Gamma_beta = eigvecs[:, order[:k]]

    F = np.full((X.shape[0], k), np.nan)
    F[valid_t] = X[valid_t] @ Gamma_beta

    return {
        "Gamma_beta": Gamma_beta, "F": F, "X": X, "ZZ": ZZ, "valid_t": valid_t,
        "eigvals": eigvals[order],
    }


def total_r2(fit: dict) -> float:
    """Fraction of managed-portfolio variance explained by the K estimated
    factors - the paper's own total-R2 diagnostic, computed on managed
    portfolios X (not raw asset returns), consistent with the
    managed-portfolio representation this small-N estimator uses
    throughout."""
    valid_t = fit["valid_t"]
    X = fit["X"][valid_t]
    Gamma_beta, F = fit["Gamma_beta"], fit["F"][valid_t]
    fitted = F @ Gamma_beta.T
    resid = X - fitted
    return 1.0 - float(np.nansum(resid ** 2) / np.nansum(X ** 2))


def managed_portfolio_residuals(fit: dict) -> np.ndarray:
    """d_t = X_t - Gamma_beta @ F_t, the restricted model's own residual in
    managed-portfolio space - the paper's own object for its alpha test
    (Section 3.1.1's `d_{t+1} = Z_t' epsilon*_{t+1}`), consistent with how F
    was derived in `fit_ipca_restricted` (`F_t = X_t @ Gamma_beta`, i.e. NOT
    itself weighted by `ZZ_t` - the whole point of this module's own
    "approximate the Rayleigh quotient denominator with a constant"
    simplification is that `ZZ_t` drops out of the factor-fitting step
    entirely). Sanity property, checked directly: when K=L (no dimension
    reduction), Gamma_beta spans the full characteristic space and this
    residual is exactly zero (to floating-point precision) - an earlier
    version of this function multiplied by `ZZ_t` here, which is
    inconsistent with the `F_t = X_t @ Gamma_beta` derivation and produced a
    nonzero, spuriously large "residual" even in the perfect-fit K=L case -
    caught by that exact sanity check before this was ever used for a real
    alpha estimate."""
    X, valid_t = fit["X"], fit["valid_t"]
    Gamma_beta, F = fit["Gamma_beta"], fit["F"]
    T, L = X.shape
    d = np.full((T, L), np.nan)
    for t in np.where(valid_t)[0]:
        d[t] = X[t] - Gamma_beta @ F[t]
    return d


def _solve_or_lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def estimate_gamma_alpha(fit: dict) -> np.ndarray:
    """Gamma_alpha via the same time-invariant-Z'Z approximation used for
    Gamma_beta: the time-average restricted-model residual, GMM-style -
    `Gamma_alpha_hat = mean(ZZ_t)^-1 @ mean(d_t)`, over valid dates only."""
    d = managed_portfolio_residuals(fit)
    valid_t = fit["valid_t"]
    ZZ_bar = np.nanmean(fit["ZZ"][valid_t], axis=0)
    d_bar = np.nanmean(d[valid_t], axis=0)
    return _solve_or_lstsq(ZZ_bar, d_bar)


def bootstrap_alpha_test(fit: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Wald-type test for Gamma_alpha != 0, via a residual wild-bootstrap in
    the same spirit as the paper's own (Section 3.1.1): resample d_t with
    replacement, each draw multiplied by an independent Student-t(5) random
    variable (the "wild" part, robust to heteroskedasticity), recompute the
    same closed-form Gamma_alpha estimator on the resampled residual series
    (mean residual = 0 under this resampling, i.e. the null), and compare
    the observed `W_alpha = Gamma_alpha' Gamma_alpha` against the empirical
    bootstrap distribution.

    Returns dict: gamma_alpha (L,), w_alpha (float), p_value (float),
    boot_stats (n_boot,)."""
    d = managed_portfolio_residuals(fit)
    valid_idx = np.where(fit["valid_t"])[0]
    d_valid = d[valid_idx]
    ZZ_valid = fit["ZZ"][valid_idx]
    ZZ_bar = ZZ_valid.mean(axis=0)

    gamma_alpha = estimate_gamma_alpha(fit)
    w_alpha = float(gamma_alpha @ gamma_alpha)

    rng = np.random.default_rng(seed)
    n_t = len(valid_idx)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_t, size=n_t)
        t_mult = rng.standard_t(df=5, size=n_t)
        d_boot_bar = (d_valid[idx] * t_mult[:, None]).mean(axis=0)
        gamma_boot = _solve_or_lstsq(ZZ_bar, d_boot_bar)
        boot_stats[b] = gamma_boot @ gamma_boot

    p_value = float((boot_stats >= w_alpha).mean())
    return {"gamma_alpha": gamma_alpha, "w_alpha": w_alpha, "p_value": p_value, "boot_stats": boot_stats}


def alpha_signal(z_panel: np.ndarray, gamma_alpha: np.ndarray) -> np.ndarray:
    """alpha_{i,t} = z_{i,t}' Gamma_alpha - the actual per-asset, time-varying
    trading signal (not the latent factors themselves): positive means this
    asset's characteristics currently imply a return ABOVE what its risk
    loadings (Gamma_beta) alone would compensate for. Returns (T, N)."""
    return np.einsum("tnl,l->tn", z_panel, gamma_alpha)
