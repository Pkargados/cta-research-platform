"""
signals/cointegration.py — Rolling-window Engle-Granger cointegration test.

Built 2026-08-01 for the Relative Value sleeve (WORKFLOW.md §11d), replacing the
"Foundation done" claim that turned out to be unverified — `Time_Series_Models.ipynb`
does not exist anywhere in this repo (checked directly, see WORKFLOW.md §11d's "Real
gap found 2026-08-01" addendum) — this is a genuine rebuild, not a resumption.

CLAUDE.md Hard Rule 2 is the whole reason this module exists in this shape: "Never
run a stationarity or cointegration test on the full sample... always use a rolling
or expanding window with a strict train/test boundary." `rolling_engle_granger`
below walks forward and, at each test date, only ever sees the trailing `window`
observations ending at (and including) that date — no future data ever enters a
test window, so a "45% of windows cointegrated" style statistic is a genuine
point-in-time-safe diagnostic, not a full-sample statistic dressed up as one.

Engle-Granger (not Johansen): `statsmodels.tsa.stattools.coint` runs the textbook
two-step test (OLS regression of y on x, then an ADF test on the residual, using
the Engle-Granger-specific critical values, not the plain ADF ones) directly — no
need to hand-roll the two steps.

This module is a DIAGNOSTIC only — it reports whether/how often a pair looks
cointegrated over time. It does not decide which pairs to trade (the 7 pairs in
WORKFLOW.md §11d were chosen on economic/structural grounds, not by screening
cointegration results — the same "don't pick a universe by having seen a result"
discipline as CLAUDE.md Rule 1, one level removed) and it does not gate the spread
signal itself (see `signals.relative_value` — the signal trades every date once
inputs are available, same "continuous, always-on" convention as every other
pair/spread construction in this codebase; a cointegration-gated ON/OFF signal
would itself be a binary construction, which Rule 5 already found underperforms).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

DEFAULT_WINDOW = 252
DEFAULT_STEP = 21
DEFAULT_ALPHA = 0.05


def rolling_engle_granger(y: pd.Series, x: pd.Series, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP) -> pd.DataFrame:
    """Walk forward in `step`-sized increments; at each test point, run
    Engle-Granger on the trailing `window` observations ending at that point
    (inclusive) — strictly historical, no look-ahead.

    Returns a DataFrame indexed by the LAST date of each trailing window (the
    date as of which the test result was knowable), columns ["stat", "pvalue"].
    A pair with too little overlapping history to fill even one window returns
    an empty DataFrame.
    """
    common = y.dropna().index.intersection(x.dropna().index).sort_values()
    y, x = y.loc[common], x.loc[common]
    n = len(common)

    rows = {}
    for i in range(window, n + 1, step):
        y_win = y.iloc[i - window:i]
        x_win = x.iloc[i - window:i]
        date = common[i - 1]
        try:
            stat, pvalue, _ = coint(y_win, x_win)
        except Exception:
            continue
        rows[date] = {"stat": stat, "pvalue": pvalue}

    if not rows:
        return pd.DataFrame(columns=["stat", "pvalue"])
    return pd.DataFrame(rows).T.sort_index()


def fraction_cointegrated(result: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> float:
    """Share of tested windows where the null of no cointegration is rejected at
    `alpha` — the "45% of windows cointegrated" style summary statistic. NaN if
    `result` has no rows (nothing was tested)."""
    if result.empty:
        return float("nan")
    return float((result["pvalue"] < alpha).mean())


def rolling_cointegration_report(pairs: dict, log_price: pd.DataFrame, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Run `rolling_engle_granger` for a dict of 2-leg pairs {name: (leg_a, leg_b)}
    against a (T x N) log-price panel, one row per pair in the summary.

    3-leg pairs (crack spread) aren't tested here — Engle-Granger is a two-series
    test; the crack spread's fixed economic ratio isn't a statistically-estimated
    relationship in the first place (see `signals.hedge_ratio`), so there's no
    "is this pair cointegrated" question to ask for it the same way.
    """
    rows = []
    for name, (leg_a, leg_b) in pairs.items():
        if leg_a not in log_price.columns or leg_b not in log_price.columns:
            rows.append({"pair": name, "leg_a": leg_a, "leg_b": leg_b, "n_windows": 0, "fraction_cointegrated": np.nan})
            continue
        result = rolling_engle_granger(log_price[leg_a], log_price[leg_b], window=window, step=step)
        rows.append({
            "pair": name, "leg_a": leg_a, "leg_b": leg_b,
            "n_windows": len(result),
            "fraction_cointegrated": fraction_cointegrated(result, alpha=alpha),
        })
    return pd.DataFrame(rows).set_index("pair")
