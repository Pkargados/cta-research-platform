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
