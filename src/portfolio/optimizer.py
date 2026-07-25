"""
portfolio/optimizer.py — Quadratic mean-variance optimizer with turnover regularization.

Adapted from the retired Cross Asset Stat Arb Engine's `chernov_weights`
(engine/portfolio/optimizer.py) — see cleanup.md section 3 for the full evaluation of
what transfers and what doesn't. Two changes from the original:

1. Dollar-neutrality is now optional (`dollar_neutral=False` by default). The original
   was a market-neutral stat-arb book; a directional CTA book has no reason to be flat
   net exposure.
2. No assumption that alpha is in Fama-MacBeth-aligned E[r] units. This is a pure
   function — it optimizes whatever alpha units it's given. Getting alpha into
   comparable-across-assets units, if that matters for a given signal family, is a
   signal-construction-time decision (see signals/combine.py), not this function's job.

Pure function — no global state, no I/O, no mutation of inputs.
"""

import numpy as np


def solve_weights(alpha_t, Sigma_t, x_prev, gamma, kappa, lambd, max_weight, dollar_neutral=False):
    """
    Quadratic mean-variance optimizer with position-inertia regularization.

    Objective: maximize alpha'x - (gamma/2)*x'Sigma*x - (kappa/2)*||x - x_prev||^2
               (with an L1 transaction-cost adjustment)
    Closed-form solution: (gamma*Sigma + kappa*I)*x = alpha + kappa*x_prev - lambda*sign(x_prev)

    Parameters
    ----------
    alpha_t        : array-like (n,)   — alpha vector at time t, in whatever units the
                                          caller's signal construction produces
    Sigma_t        : array-like (n, n) — covariance matrix at time t
    x_prev         : array-like (n,)   — positions held at t-1
    gamma          : float             — risk-aversion coefficient
    kappa          : float             — position-inertia coefficient (penalizes turnover)
    lambd          : float             — L1 penalty coefficient (transaction-cost proxy)
    max_weight     : float             — per-asset position limit (absolute value)
    dollar_neutral : bool              — if True, recenter so weights sum to zero
                                          (market-neutral books only — leave False for CTA)

    Returns
    -------
    np.ndarray shape (n,) — new target weights, within +/-max_weight.
    """
    alpha_t = np.asarray(alpha_t, dtype=float).reshape(-1, 1)
    x_old = np.asarray(x_prev, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma_t, dtype=float)
    n = len(alpha_t)

    A = gamma * Sigma + kappa * np.eye(n)
    b = alpha_t + kappa * x_old - lambd * np.sign(x_old)
    x = np.linalg.solve(A, b).flatten()

    if dollar_neutral:
        x = x - x.mean()
    x = np.clip(x, -max_weight, max_weight)
    return x
