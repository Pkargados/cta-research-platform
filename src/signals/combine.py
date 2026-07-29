"""
signals/combine.py — Blend multiple alpha DataFrames into one.

Gives two ways to combine signals without forcing the decision now (per direct
request — research will explore both): blend here into one alpha, feed to a single
Book ("integrate" — AQR's own term, see references/AQR - Portfolio Construction
Matters.pdf: blend each asset's per-signal scores into one composite BEFORE building
any portfolio); or keep alphas separate, build one Book per alpha, and let
portfolio/allocator.py combine them at the PnL level ("mix" — build each portfolio
separately, then combine the already-built portfolios). This module's earlier
docstring had these two labels backwards relative to AQR's actual terminology —
corrected 2026-07-23, see research/value_momentum_combine.py for the first real use
of both routes. Both routes reuse the same Book/Allocator primitives — this module
only adds the blending step itself.

Pure functions, no optimizer dependency, same convention as every other signal module
in this project (CLAUDE.md Architecture section).
"""

import numpy as np
import pandas as pd


def combine_alphas(alpha_dfs, weights=None, method="equal"):
    """
    Blend multiple (T×N) alpha DataFrames into one.

    Parameters
    ----------
    alpha_dfs : list[pd.DataFrame] — same columns (assets), same units, one per
                                      signal to blend
    weights   : list[float] | None — per-signal weight, required for method="fixed"
                                      (ignored otherwise)
    method    : str
        "equal" — simple average across signals (default)
        "fixed" — weights[i] * alpha_dfs[i], summed (caller's responsibility to
                  normalize weights if that matters for their purposes)
        "rank"  — average of each signal's cross-sectional rank per date rather than
                  its raw value — use when signals are in different units/scales

    Returns
    -------
    pd.DataFrame (T×N) — combined alpha, same shape as the inputs.
    """
    if len(alpha_dfs) == 0:
        raise ValueError("combine_alphas requires at least one alpha DataFrame")
    if len(alpha_dfs) == 1:
        return alpha_dfs[0].copy()

    if method == "equal":
        return sum(alpha_dfs) / len(alpha_dfs)

    if method == "fixed":
        if weights is None or len(weights) != len(alpha_dfs):
            raise ValueError("method='fixed' requires one weight per alpha_df")
        return sum(w * df for w, df in zip(weights, alpha_dfs))

    if method == "rank":
        ranked = [df.rank(axis=1, pct=True) for df in alpha_dfs]
        return sum(ranked) / len(ranked)

    raise ValueError(f"Unknown method: {method!r}. Use 'equal', 'fixed', or 'rank'.")


def ic_weighted_combine(alpha_dfs, forward_returns, lookback=252):
    """
    Weight each signal by its trailing rolling information coefficient (cross-
    sectional rank correlation with forward returns) rather than a fixed weight —
    signals that have recently been more predictive get more weight.

    Vectorized (no per-date Python loop): builds a (T, n_signals) matrix of trailing
    ICs, normalizes it into per-date weights, and combines via broadcast multiply.

    Parameters
    ----------
    alpha_dfs       : list[pd.DataFrame] (T×N) — one per signal to blend
    forward_returns : pd.DataFrame (T×N)       — same index/columns, the return each
                                                   alpha is trying to predict
    lookback        : int                       — trailing window (days) for the
                                                   rolling IC estimate

    Returns
    -------
    pd.DataFrame (T×N) — combined alpha. Dates where all trailing ICs are ~0 or NaN
    (e.g. the lookback warmup period) fall back to an equal-weighted blend rather
    than producing NaN.
    """
    ics = []
    for alpha_df in alpha_dfs:
        # shift(1): today's combination weight uses only ICs knowable before today.
        daily_ic = alpha_df.corrwith(forward_returns, axis=1, method="spearman")
        rolling_ic = daily_ic.rolling(lookback, min_periods=20).mean().shift(1)
        ics.append(rolling_ic.clip(lower=0))  # negative trailing IC -> zero weight, not negative

    ic_df = pd.concat(ics, axis=1)
    ic_df.columns = range(len(alpha_dfs))

    weight_sum = ic_df.sum(axis=1)
    needs_fallback = weight_sum < 1e-8

    normalized = ic_df.div(weight_sum.where(~needs_fallback), axis=0)
    normalized = normalized.where(~needs_fallback, other=1.0 / len(alpha_dfs))

    return sum(alpha_dfs[i].mul(normalized[i], axis=0) for i in range(len(alpha_dfs)))


def risk_parity_combine(alpha_dfs, strategy_returns, lookback=252, min_periods=20):
    """
    Weight each signal inversely by its own trailing realized volatility (of its
    OWN standalone strategy return stream, not the underlying assets') — each
    signal contributes roughly equal RISK to the blend, rather than equal raw
    alpha magnitude (method="equal") or equal predictive power
    (`ic_weighted_combine`).

    Same trailing-window / shift(1) / warmup-fallback shape as
    `ic_weighted_combine`, for consistency — weights are computed once per date
    (not per-asset) from each signal's own aggregate return series (`backtest.
    engine.backtest_signal`'s daily-marked output, matching `research/
    signal_correlation.py`'s own convention), then broadcast across every asset.

    Parameters
    ----------
    alpha_dfs        : list[pd.DataFrame] (T×N) — one per signal to blend
    strategy_returns : list[pd.Series] — each signal's OWN standalone strategy
                        return series, one per alpha_df, same daily index
    lookback         : int — trailing window (periods) for the rolling vol estimate
    min_periods      : int — minimum trailing observations before a real (non-
                        fallback) weight is used

    Returns
    -------
    pd.DataFrame (T×N) — combined alpha, same shape as the inputs. Dates where
    all trailing vols are ~0 or NaN (warmup) fall back to an equal-weighted blend.
    """
    inv_vols = []
    for returns in strategy_returns:
        # shift(1): today's weight uses only volatility knowable before today.
        rolling_vol = returns.rolling(lookback, min_periods=min_periods).std().shift(1)
        inv_vols.append(1.0 / rolling_vol.replace(0.0, float("nan")))

    inv_vol_df = pd.concat(inv_vols, axis=1)
    inv_vol_df.columns = range(len(alpha_dfs))

    weight_sum = inv_vol_df.sum(axis=1)
    needs_fallback = weight_sum.isna() | (weight_sum < 1e-8)

    normalized = inv_vol_df.div(weight_sum.where(~needs_fallback), axis=0)
    normalized = normalized.where(~needs_fallback, other=1.0 / len(alpha_dfs))

    return sum(alpha_dfs[i].mul(normalized[i], axis=0) for i in range(len(alpha_dfs)))


def confirmation_filter_combine(primary, confirm, agree_scale=1.0, disagree_scale=0.0):
    """
    Trade `primary` at `agree_scale`x its own size when `confirm` agrees on
    direction (same sign), and at `disagree_scale`x when it doesn't — a gate,
    not a blend. Qualitatively different from `combine_alphas`/
    `ic_weighted_combine`/`risk_parity_combine` (all weighted averages of two
    scores): here `confirm` only ever votes on DIRECTION, never contributes its
    own magnitude to the combined position size.

    `disagree_scale=0.0` (the default) means genuinely FLAT when the two signals
    disagree, not just downsized — the "not constantly trading" behavior this was
    built for: `primary` and `confirm` must actually agree before any position is
    taken at all. NaN-safe: `np.sign(NaN) == np.sign(x)` is always False, so a
    missing `primary` or `confirm` value is treated as "disagree" (scale applied),
    and a NaN `primary` stays NaN after scaling (NaN * float = NaN) rather than
    being silently zeroed — missing data is never mistaken for "no conviction."

    Parameters
    ----------
    primary        : pd.DataFrame (T×N) — the signal whose magnitude sets position size
    confirm        : pd.DataFrame (T×N) — only its SIGN is used, as a gate
    agree_scale    : float — multiplier on `primary` when signs agree
    disagree_scale : float — multiplier on `primary` when signs disagree (or either
                     side is NaN)

    Returns
    -------
    pd.DataFrame (T×N) — gated alpha, same shape as `primary`.
    """
    agree = np.sign(primary) == np.sign(confirm)
    scale = pd.DataFrame(disagree_scale, index=primary.index, columns=primary.columns)
    scale = scale.where(~agree, other=agree_scale)
    return primary * scale
