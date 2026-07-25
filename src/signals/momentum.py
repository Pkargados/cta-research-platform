"""
signals/momentum.py — Time-series momentum (TSMOM), Moskowitz-Ooi-Pedersen (2012).

Rebuilt to match the paper exactly (`references/Time Series Momentum.pdf`, read
directly), not the earlier binary/continuous/rank transform comparison
(`build_momentum_features` below — kept only for `signal_lib.py`'s backward-compat
import surface, see `signals.transforms`' module docstring).

Paper mechanics (Section 3.2, Eq. 5):
    r^TSMOM,s_{t,t+1} = sign(r^s_{t-12,t}) * (40% / sigma^s_t) * r^s_{t,t+1}

i.e. go long/short based on the SIGN of the trailing k-month return, sized to a
constant 40% annualized ex-ante vol target, headline spec k=12 months, h=1 month
holding (no skip month — the skip convention is a cross-sectional-momentum
import, not part of TSMOM, see `build_momentum_features` below). Position sizing
itself is `signals.transforms.vol_targeted_sign_signal` — generic across signal
families, not re-derived here.

Momentum lookback is computed in trading-day terms (`lookback_months * 21`, the
standard trading-days-per-month approximation) directly off DAILY closes, not a
month-end resample at the signal-construction step — `backtest.engine`'s own
`holding_period_positions`/`monthly_positions` already handle the month-end
formation + holding-period blending (Jegadeesh-Titman-style overlapping-vintage
averaging, matching the paper's own methodology, Section 3.2) at the BACKTEST
layer, not here. Keeping lookback computation daily and resampling downstream
(rather than resampling first) means the same raw signal can be backtested at any
holding_months without recomputing it.

Vol estimator is deliberately NOT baked into this module — the paper's own vol
estimate (`data.ewma_volatility`) and this project's Yang-Zhang
(`data.volatility`) are compared on train evidence by the caller
(`research/momentum.py`), decoupled from lookback (one fixed-horizon vol estimate
regardless of which k is tested, matching the paper and correcting an earlier
codebase's implicit 1:1 coupling between vol horizon and momentum lookback).
"""

import numpy as np
import pandas as pd

from signals.transforms import binary_signal, continuous_signal, rank_signal, vol_targeted_sign_signal

TRADING_DAYS_PER_MONTH = 21
HEADLINE_LOOKBACK_MONTHS = 12
HEADLINE_HOLDING_MONTHS = 1
HEADLINE_TARGET_VOL = 0.40

GRID_MONTHS = (1, 3, 6, 9, 12, 24, 36, 48)


def raw_momentum(close: pd.DataFrame, lookback_months: int, skip_months: int = 0) -> pd.DataFrame:
    """Trailing return over `lookback_months`, ending `skip_months` before today
    (skip_months=0 for TSMOM's own headline spec — no skip month).

    `close` should be the BACK-ADJUSTED continuous curve
    (`data.continuous_curve.load_continuous_backadjusted`) — momentum needs clean
    percentage returns across rolls, the same reasoning `data.continuous_curve`'s
    own module docstring gives for building this series in the first place.
    """
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = skip_months * TRADING_DAYS_PER_MONTH
    base = close.shift(skip_days) if skip_days > 0 else close
    return base / base.shift(lookback_days) - 1.0


def tsmom_signal(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    lookback_months: int = HEADLINE_LOOKBACK_MONTHS,
    target_vol: float = HEADLINE_TARGET_VOL,
) -> pd.DataFrame:
    """The paper's Eq. 5 position, for one (lookback, target_vol) spec. `vol` must
    already be annualized (both `data.volatility.yang_zhang_volatility` and
    `data.ewma_volatility.ewma_volatility` default to `annualize=True`) - the 40%
    target itself is an ANNUALIZED vol target (paper Section 4.1).

    Holding period (h) is NOT a parameter here - apply it at the backtest layer via
    `backtest.engine.backtest_signal(..., frequency="monthly", holding_months=h)`,
    which reproduces the paper's own overlapping-vintage-averaging construction.
    """
    mom = raw_momentum(close, lookback_months, skip_months=0)
    return vol_targeted_sign_signal(mom, vol, target_vol=target_vol)


def momentum_grid_signals(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    lookback_months_grid=GRID_MONTHS,
    target_vol: float = HEADLINE_TARGET_VOL,
) -> dict:
    """One `tsmom_signal` per lookback (k) in `lookback_months_grid` - the paper's
    Table 2 grid varies both k (lookback) and h (holding), but h is a
    BACKTEST-layer parameter (see `tsmom_signal`'s docstring), so this only needs
    to vary k; the caller backtests each returned signal at every h in the same
    grid via `backtest.engine.backtest_signal`'s own `holding_months`.

    Returns {k: signal_df}. Descriptive/robustness view only, per CLAUDE.md Rule 1/
    2 - the headline spec (k=12, h=1) is fixed a priori, never chosen by looking at
    which grid cell scores highest.
    """
    return {k: tsmom_signal(close, vol, lookback_months=k, target_vol=target_vol) for k in lookback_months_grid}


# ---------------------------------------------------------------------------
# Legacy pre-rebuild feature builder — kept only for signal_lib.py's backward-
# compat import surface (see signals/transforms.py's module docstring). This is
# the original binary/continuous/rank comparison that CLAUDE.md Rule 5's finding
# ("binary momentum signals are a disaster") came from, before the TSMOM rebuild
# above replaced it as the project's actual momentum signal.
# ---------------------------------------------------------------------------

DEFAULT_SKIP_MONTHS = 1  # cross-sectional-momentum convention (Jegadeesh-Titman) - not used by tsmom_signal above


def build_momentum_features(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    lookback_months: int = HEADLINE_LOOKBACK_MONTHS,
    skip_months: int = DEFAULT_SKIP_MONTHS,
) -> dict:
    """Legacy transform comparison: {"binary", "continuous", "rank"} feature dict,
    each a (T x N) DataFrame, built off the same skip-adjusted trailing return.
    Superseded by `tsmom_signal` above for anything actually traded in this
    project - kept only so old callers (`signal_lib.py`'s re-export surface) still
    resolve."""
    mom = raw_momentum(close, lookback_months, skip_months)
    return {
        "binary": binary_signal(mom),
        "continuous": continuous_signal(mom, vol),
        "rank": rank_signal(mom),
    }
