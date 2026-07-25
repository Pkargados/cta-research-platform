"""
data/ewma_volatility.py — Exponentially weighted volatility estimator, exactly
Moskowitz-Ooi-Pedersen (2012) Eq. 1.

sigma_t^2 = 261 * sum_{i=0}^inf (1-delta)*delta^i * (r_{t-1-i} - rbar_t)^2

delta is chosen so the weights' center of mass is 60 trading days:
delta/(1-delta) = 60. pandas' `com` parameter *is* this center-of-mass
parametrization directly (com = delta/(1-delta)), so no manual delta derivation is
needed - `returns.ewm(com=60)` matches the paper's weighting exactly.

Two deliberate departures from the paper, both immaterial to what this project uses
the estimate for:
- Annualized by 252, not the paper's 261 (261 was their own multi-decade average
  trading-days/year; 252 matches every other annualization in this codebase -
  Yang-Zhang, performance_stats - so estimates stay comparable within this project).
- No internal shift/lag: this returns sigma_t built from data through date t itself
  (same convention as every other feature in this codebase - e.g. the momentum
  feature itself uses close_t). The no-look-ahead safety margin is applied once,
  generically, by backtest.engine.backtest_signal's own positions.shift(1) - not
  re-implemented per feature (CLAUDE.md Rule 3).
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252
DEFAULT_COM = 60


def ewma_volatility(returns: pd.DataFrame, com: int = DEFAULT_COM, annualize: bool = True) -> pd.DataFrame:
    """Exponentially weighted volatility of daily returns, center-of-mass `com`.

    Returns annualized volatility by default (`annualize=True`); pass False for the
    raw daily-scale estimate.
    """
    ewma_var = returns.pow(2).ewm(com=com, adjust=False).mean()
    vol = np.sqrt(ewma_var)
    if annualize:
        vol = vol * np.sqrt(ANNUALIZATION)
    return vol
