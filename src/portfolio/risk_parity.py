"""
portfolio/risk_parity.py — General n-sleeve equal-risk-contribution (risk
parity) solver.

Built 2026-07-29, per direct instruction, following the Trend/Carry
combination discussion: for exactly two sleeves, equal-risk-contribution
weighting has a closed form (w_1/w_2 = sigma_2/sigma_1 — correlation
cancels out of the RATIO), so this general solver isn't strictly needed for
the current two-Book mandate. Built anyway since more strategy sleeves
(Relative Value, etc.) are planned, and the numerical machinery needed for
n>=3 sleeves doesn't get simpler by special-casing n=2 first — see
WORKFLOW.md decision #12.

Pure functions, no optimizer dependency beyond scipy — covariance-matrix-
agnostic (doesn't care whether Sigma came from rolling/EWMA sample
covariance or DCC-GARCH, see sleeve_covariance.py for both).
"""

import numpy as np
from scipy.optimize import minimize


def risk_contributions(weights, cov_matrix) -> np.ndarray:
    """
    Each sleeve's exact contribution to total portfolio risk (volatility),
    given weights and a covariance matrix.

    RC_i = w_i * (Sigma @ w)_i / sqrt(w' Sigma w)

    Sums exactly to portfolio vol (Euler's theorem for the positively-
    homogeneous-degree-1 function sigma_p(w) = sqrt(w'Sigma*w)) — not an
    approximation. Used both to verify `risk_parity_weights`'s own solution
    and as a standalone diagnostic for any weight vector.
    """
    w = np.asarray(weights, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)
    portfolio_var = float(w @ Sigma @ w)
    if portfolio_var <= 0:
        return np.zeros_like(w)
    portfolio_vol = np.sqrt(portfolio_var)
    marginal = Sigma @ w
    return w * marginal / portfolio_vol


def risk_parity_weights(cov_matrix, risk_budgets=None, w_init=None, max_iter=1000) -> np.ndarray:
    """
    Solve for weights w (positive, sum to 1) such that each sleeve's risk
    contribution matches its target risk budget — default equal budgets
    (1/n each), the standard "risk parity" / equal-risk-contribution (ERC)
    solution. Pass `risk_budgets` explicitly for a conviction-weighted split
    instead (same solver, different targets — e.g. a research committee's
    judgment that one sleeve's evidence warrants more risk than another,
    not something this function decides).

    Convex reformulation (Spinu 2013; Bruder & Roncalli 2012), not the naive
    "minimize sum of squared risk-contribution differences" version:

        minimize f(w) = 0.5 * w'Sigma*w - sum_i budget_i * log(w_i)
        subject to w_i > 0

    This is convex whenever Sigma is positive semi-definite (true for any
    real covariance matrix), so it has a UNIQUE minimizer. The naive
    squared-difference objective is not convex and can converge to
    different local optima depending on the starting point — an
    unnecessary risk for a problem that has a provably well-behaved
    formulation available.

    At the (sign-constrained, scale-unconstrained) optimum, the first-order
    condition is `w_i * (Sigma @ w)_i = budget_i` for every i — risk
    contributions are exactly proportional to the budgets, before
    normalization. Rescaling w to sum to 1 afterward preserves this ratio
    exactly: risk_contributions(w) is homogeneous of degree 1 in w (RC_i
    itself is w_i * (Sigma@w)_i / ||w||_Sigma, and both numerator and
    denominator scale together under w -> c*w), so RC_i/RC_j — the thing
    that actually matters for "equal risk contribution" — is scale-
    invariant. Solve unconstrained-in-scale, normalize once at the end.

    Two sleeves, equal budgets: this reduces to the closed-form inverse-
    volatility solution w_1/w_2 = sigma_2/sigma_1 (see this module's own
    docstring, and tests/test_risk_parity.py's direct check against that
    closed form).
    """
    Sigma = np.asarray(cov_matrix, dtype=float)
    n = Sigma.shape[0]

    if risk_budgets is None:
        budgets = np.full(n, 1.0 / n)
    else:
        budgets = np.asarray(risk_budgets, dtype=float)
        budgets = budgets / budgets.sum()

    if w_init is None:
        diag_vol = np.sqrt(np.diag(Sigma))
        w0 = (1.0 / diag_vol) if np.all(diag_vol > 0) else np.full(n, 1.0 / n)
        w0 = w0 / w0.sum()
    else:
        w0 = np.asarray(w_init, dtype=float)

    def objective(w):
        return 0.5 * w @ Sigma @ w - np.sum(budgets * np.log(w))

    def gradient(w):
        return Sigma @ w - budgets / w

    bounds = [(1e-8, None)] * n
    result = minimize(
        objective, w0, jac=gradient, method="L-BFGS-B", bounds=bounds,
        # L-BFGS-B's default ftol (~2.2e-9) / gtol (1e-5) leave the risk
        # contributions off by ~1e-5 relative for a problem this small and
        # cheap to solve tighter — tightened both so RC_i matches its
        # budget to ~1e-10 instead, confirmed in tests/test_risk_parity.py.
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Risk parity solve failed to converge: {result.message}")

    return result.x / result.x.sum()
