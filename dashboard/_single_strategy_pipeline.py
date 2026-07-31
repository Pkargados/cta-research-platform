"""
dashboard/_single_strategy_pipeline.py — Shared, cached entry point for the
two decided Single Strategy Portfolios (Trend = tsmom_alone, Carry =
carry_timing_zero, both GJR-GARCH vol-targeted — WORKFLOW.md decisions #10/
#11) plus their combined Allocator.

One shared `st.cache_resource` entry, not one per page — `dashboard/
_portfolio_pipeline.py`'s own docstring documents a real Streamlit Cloud
crash (confirmed from platform logs) caused by two pages each independently
caching the same expensive pipeline: visiting both close together ran it
TWICE simultaneously and OOM'd the container. Same fix here, applied before
it has a chance to repeat: Trend Book, Carry Book, and Multi-Strategy pages
all import and call THIS one function.

Does NOT re-run the 7-flavor Trend / 4-flavor Carry bake-off — that's already
decided (`research/single_strategy_portfolios.py`, run once, logged in
WORKFLOW.md). This only ever constructs and runs the two WINNING alpha
constructions, so it's a small fraction of that script's own full cost.

**Universe, per direct instruction, 2026-07-31 (WORKFLOW.md decision #13):**
Trend runs on `compress_for_family(included, "trend")` (next_steps.md Phase 2
layers B/C) — a clean, adopted improvement (test Sharpe 1.356 -> 1.600,
turnover 0.78x -> 0.62x, both better). Carry deliberately stays on the
FULL uncompressed universe, unchanged from the original published
decision #10/#11 — the same compression flipped Carry's own bake-off winner
to a knife-edge, unstable construction (carry1m, test Sharpe -2.035 on 26
observations) via a marginal data-sufficiency threshold effect, not genuine
outperformance, so it was reverted rather than adopted.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "pages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import single_strategy_portfolios as ssp
import seasonality_single_strategy as sss
from single_strategy_portfolios import TREND_FLAVOR, CARRY_FLAVOR
from portfolio.allocator import Allocator
from portfolio.risk_parity import risk_parity_weights
from portfolio.sleeve_covariance import rolling_covariance, ewma_covariance, dcc_garch_covariance
from backtest.splits import TRAIN_END

TREND_BOOK_NAME = "trend_" + TREND_FLAVOR
CARRY_BOOK_NAME = "carry_" + CARRY_FLAVOR
SEASONALITY_BOOK_NAME = "same_month_economic"

# Sleeve-covariance EWMA halflife, in weekly-observation units - matches
# single_strategy_portfolios.py's own EWMA_HALFLIFE rescale for weekly cadence
# (WORKFLOW.md decision #12).
RISK_PARITY_EWMA_HALFLIFE = 87

RISK_PARITY_ESTIMATORS = {
    "Rolling (full-train sample)": "rolling",
    "EWMA (halflife=87, train)": "ewma",
    "DCC-GARCH (train, cross-check)": "dcc_garch",
}


@st.cache_resource(ttl=1200)
def load_and_run():
    """Returns (returns, trend_result, carry_result, trend_book, carry_book, combined_pnl).

    `combined_pnl` is the NAIVE equal-Book-risk Allocator combination (w=[1, 1])
    — see `compute_risk_parity_weights` for the risk-parity alternative
    (WORKFLOW.md decision #12), computed separately so the dashboard's
    estimator toggle doesn't need to re-run this expensive pipeline."""
    returns, trend_book, carry_book = ssp.build_adopted_books()
    trend_result = trend_book.run(returns)
    carry_result = carry_book.run(returns)

    allocator = Allocator([trend_book, carry_book])
    combined = allocator.run(returns)

    return returns, trend_result, carry_result, trend_book, carry_book, combined["pnl"]


@st.cache_data(ttl=1200)
def compute_risk_parity_weights(estimator: str = "rolling") -> dict:
    """Fixed sleeve risk-parity weights ({"trend": w, "carry": w, "converged": bool}),
    fit ONCE on TRAIN sleeve PnL only (CLAUDE.md Rule 1/2 — validation/test must not
    influence the weight choice) via one of `portfolio.sleeve_covariance`'s three
    estimators. Weights are rescaled by n_sleeves=2 (risk_parity_weights' own output
    sums to 1) so they sit on the same total gross-weight budget as the naive
    Allocator's implicit [1, 1] — Sharpe is invariant to this rescale (it only depends
    on the trend/carry weight RATIO), but reported vol levels stay directly
    comparable across combination methods this way. Same construction as
    `research/sleeve_risk_parity.py`, already validated on this same real data
    (WORKFLOW.md decision #12) — reused here, not reimplemented.

    `st.cache_data`, not `st.cache_resource` — this returns plain floats/bools, not
    the heavy Book objects `load_and_run` caches, so a separate lightweight cache
    keyed by `estimator` lets the dashboard's estimator toggle stay fast without
    re-running Book construction."""
    _returns, trend_result, carry_result, _trend_book, _carry_book, _naive_pnl = load_and_run()
    sleeve_returns = pd.concat({"trend": trend_result["pnl"], "carry": carry_result["pnl"]}, axis=1)
    train_sleeve_returns = sleeve_returns.loc[:TRAIN_END]

    if estimator == "rolling":
        n_train = len(train_sleeve_returns.dropna(how="any"))
        cov = rolling_covariance(train_sleeve_returns, window=n_train)
        converged = True
    elif estimator == "ewma":
        cov = ewma_covariance(train_sleeve_returns, halflife=RISK_PARITY_EWMA_HALFLIFE)
        converged = True
    elif estimator == "dcc_garch":
        dcc_result = dcc_garch_covariance(train_sleeve_returns.dropna(how="any"))
        cov, converged = dcc_result["cov"], dcc_result["converged"]
    else:
        raise ValueError(f"Unknown sleeve covariance estimator: {estimator!r}")

    w = risk_parity_weights(cov.values)
    return {"trend": float(2 * w[0]), "carry": float(2 * w[1]), "converged": bool(converged)}


@st.cache_resource(ttl=1200)
def load_and_run_seasonality():
    """Returns (returns, season_result, season_book) — the economic-driver
    same_month Single Strategy Portfolio (WORKFLOW.md decision #13's
    follow-up, `seasonality_single_strategy.build_economic_seasonality_book`).
    A SEPARATE cache from `load_and_run()` (its own independent pipeline,
    different universe/data path) — kept in this same shared module (not
    page-local) for the same reason `load_and_run` itself is shared: both the
    Seasonality Book page and the Multi-Strategy page's own "does Seasonality
    help" section need it, and caching it twice independently is exactly the
    pattern that caused the documented Streamlit Cloud OOM crash."""
    returns, season_book = sss.build_economic_seasonality_book()
    season_result = season_book.run(returns)
    return returns, season_result, season_book


@st.cache_data(ttl=1200)
def compute_risk_parity_weights_n(sleeve_names: tuple, estimator: str = "rolling") -> dict:
    """General n-sleeve risk-parity weights (WORKFLOW.md decision #12,
    generalized 2026-07-31 for the Seasonality-inclusive combinations —
    `research/multi_strategy_seasonality_risk_parity.py`'s own pattern).
    `sleeve_names`: a tuple (hashable, for st.cache_data's key) of "trend",
    "carry", "seasonality" in any combination including "trend" alone being
    meaningless (needs >= 2 to combine). Returns {name: weight, ...,
    "converged": bool} — weights rescaled by n_sleeves so the total
    gross-weight budget matches the naive Allocator's own implicit
    all-ones weighting.

    Covariance is rescaled by 1e4 before solving `risk_parity_weights` — its
    log-barrier solver can fail to converge (ABNORMAL termination) when
    Sigma's own values are this small (~1e-4 to 1e-5, weekly Book PnL
    variance); the solution is provably scale-invariant after normalization
    (Sigma -> c*Sigma solves for w/sqrt(c), which normalizes identically) —
    a safe, local rescale, not a change to the shared, already-tested
    `risk_parity_weights` function itself. Found and fixed the same day
    building the 3-sleeve comparison this generalizes."""
    _returns, trend_result, carry_result, _tb, _cb, _naive = load_and_run()
    pnls = {"trend": trend_result["pnl"], "carry": carry_result["pnl"]}
    if "seasonality" in sleeve_names:
        _r, season_result, _sb = load_and_run_seasonality()
        pnls["seasonality"] = season_result["pnl"]

    sleeve_returns = pd.concat({name: pnls[name] for name in sleeve_names}, axis=1)
    train_sleeve_returns = sleeve_returns.loc[:TRAIN_END]
    n = len(sleeve_names)

    if estimator == "rolling":
        cov = rolling_covariance(train_sleeve_returns, window=len(train_sleeve_returns.dropna(how="any")))
        converged = True
    elif estimator == "ewma":
        cov = ewma_covariance(train_sleeve_returns, halflife=RISK_PARITY_EWMA_HALFLIFE)
        converged = True
    elif estimator == "dcc_garch":
        dcc_result = dcc_garch_covariance(train_sleeve_returns.dropna(how="any"))
        cov, converged = dcc_result["cov"], dcc_result["converged"]
    else:
        raise ValueError(f"Unknown sleeve covariance estimator: {estimator!r}")

    w = risk_parity_weights(cov.values * 1e4)
    result = {name: float(n * w[i]) for i, name in enumerate(sleeve_names)}
    result["converged"] = bool(converged)
    return result
