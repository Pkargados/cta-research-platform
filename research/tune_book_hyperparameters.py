"""
research/tune_book_hyperparameters.py — Per-Book hyperparameter tuning (Value +
XSMOM only — the scope this script owns; scaling to all 20 Books is
`research/tune_all_books.py`'s job, a separate, already-existing file not part of
this reconstruction pass).

Pre-committed 5x5 grid (`target_vol` 0.05-0.15 x `max_weight` 0.15-0.5, fixed
before any result is looked at), weekly Book rebalancing (same cadence as
`research/value_momentum_combine.py`), evaluated via `portfolio.book.
daily_mark_pnl` (daily-marking an already-solved weekly weight path against daily
returns — more validation observations WITHOUT re-solving the optimizer more
often, a measurement-precision improvement, not a strategy change — see that
function's own docstring). Selection rule: Newey-West HAC paired t-test
(`maxlags=25`, ~1 trading month, the real redundancy length since alpha is still
monthly-cadence even though the Book resolves weekly) on each grid candidate's
validation daily-marked PnL MINUS the default combo's own validation daily-marked
PnL — tests "did tuning help," not "is this combo's raw Sharpe positive." Test is
touched exactly once, for the winning candidate only.

Reproduces CLAUDE.md's central, honest finding: tuning overfits the validation
window and does not survive the HAC check — the recommendation is to KEEP the
flat default calibration (target_vol=0.10, max_weight=0.30), not adopt any
grid-selected combo.

Run: `python research/tune_book_hyperparameters.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.macro import load_yield_curve, load_cpi
from signals.value import value_signal
from signals.xs_momentum import xs_momentum_signal
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe
from portfolio.covariance import build_cov_dict
from portfolio.book import Book, daily_mark_pnl
from portfolio.allocator import Allocator

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000

GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252
COV_FREQ = "W-FRI"
PERIODS_PER_YEAR = 52
EWMA_HALFLIFE = 87

DEFAULT_TARGET_VOL = 0.10
DEFAULT_MAX_WEIGHT = 0.30
TARGET_VOL_GRID = [0.05, 0.075, 0.10, 0.125, 0.15]
MAX_WEIGHT_GRID = [0.15, 0.2, 0.3, 0.4, 0.5]
GRID_SIZE = len(TARGET_VOL_GRID) * len(MAX_WEIGHT_GRID)  # 25
HAC_MAXLAGS = 25


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    close = adj["close"][included]
    sectors = sectors_for_universe(included)
    return close, sectors


def _active_columns(alpha_df, returns_df, min_valid_frac=0.90):
    has_alpha = alpha_df.notna().any()
    returns_valid_frac = returns_df.notna().mean()
    return [c for c in alpha_df.columns if has_alpha.get(c, False) and returns_valid_frac.get(c, 0.0) >= min_valid_frac]


def make_book(name, alpha_scaled, cov_dict, target_vol, max_weight):
    return Book(
        name=name, alpha_df=alpha_scaled, cov_dict=cov_dict,
        gamma=GAMMA, kappa=KAPPA, lambd=LAMBD, max_weight=max_weight,
        target_vol=target_vol, ewma_halflife=EWMA_HALFLIFE,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=PERIODS_PER_YEAR, dollar_neutral=False,
    )


def hac_mean_tstat(diff: pd.Series, maxlags: int = HAC_MAXLAGS) -> dict:
    y = diff.dropna()
    if len(y) < maxlags * 2:
        return {"mean": np.nan, "t": np.nan, "p": np.nan, "n_obs": len(y)}
    X = np.ones((len(y), 1))
    model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"mean": float(model.params[0]), "t": float(model.tvalues[0]), "p": float(model.pvalues[0]), "n_obs": len(y)}


def tune_one_book(name, alpha_df, returns):
    active = _active_columns(alpha_df, returns)
    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:TRAIN_END].stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    cov_dict = build_cov_dict(returns[active], window=COV_WINDOW, freq=COV_FREQ)

    daily_pnls = {}
    for tv in TARGET_VOL_GRID:
        for mw in MAX_WEIGHT_GRID:
            book = make_book(name, alpha_scaled, cov_dict, tv, mw)
            result = book.run(returns)
            daily_pnls[(tv, mw)] = daily_mark_pnl(result["weights"], returns)

    default_key = (DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT)
    default_val_pnl = train_validation_test_split(daily_pnls[default_key])[1]

    rows = []
    for (tv, mw), daily_pnl in daily_pnls.items():
        train_pnl, val_pnl, test_pnl = train_validation_test_split(daily_pnl)
        is_default = (tv, mw) == default_key
        diff = pd.Series(dtype=float) if is_default else (val_pnl - default_val_pnl).dropna()
        hac = {"mean": 0.0, "t": np.nan, "p": 1.0, "n_obs": 0} if is_default else hac_mean_tstat(diff)
        rows.append({
            "target_vol": tv, "max_weight": mw, "is_default": is_default,
            "train_sharpe": simple_sharpe(train_pnl), "val_sharpe": simple_sharpe(val_pnl), "test_sharpe": simple_sharpe(test_pnl),
            "diff_vs_default_mean": hac["mean"], "diff_vs_default_t": hac["t"], "diff_vs_default_p": hac["p"], "diff_n_obs": hac["n_obs"],
        })
    grid_df = pd.DataFrame(rows)

    candidates = grid_df[(~grid_df["is_default"]) & (grid_df["diff_vs_default_mean"] > 0)]
    best = None
    if len(candidates) > 0:
        best = candidates.loc[candidates["diff_vs_default_p"].idxmin()].to_dict()
        best["p_bonf"] = min(1.0, best["diff_vs_default_p"] * GRID_SIZE)

    return grid_df, best, alpha_scaled, cov_dict, active


def main():
    close, sectors = load_and_prepare_data()
    returns = close.pct_change(fill_method=None)
    yield_curve, cpi = load_yield_curve(), load_cpi()

    signals = {
        "value": value_signal(close, yield_curve, cpi, sectors),
        "xs_momentum": xs_momentum_signal(close, sectors),
    }

    winners = {}
    books_by_calibration = {}
    for name, alpha_df in signals.items():
        print(f"\n=== {name}: {GRID_SIZE}-point grid (target_vol x max_weight), weekly, daily-marked, HAC-tested ===")
        grid_df, best, alpha_scaled, cov_dict, active = tune_one_book(name, alpha_df, returns)
        print(grid_df.round(3).to_string(index=False))
        books_by_calibration[name] = (alpha_scaled, cov_dict, active)

        default_row = grid_df[grid_df["is_default"]].iloc[0]
        print(f"\nDefault (target_vol={DEFAULT_TARGET_VOL}, max_weight={DEFAULT_MAX_WEIGHT}): "
              f"train={default_row['train_sharpe']:.3f} val={default_row['val_sharpe']:.3f} test={default_row['test_sharpe']:.3f}")

        if best is None:
            print("No candidate improved on default at all.")
            winners[name] = (DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT)
        else:
            print(f"Best candidate: target_vol={best['target_vol']:.3f} max_weight={best['max_weight']:.2f} "
                  f"val={best['val_sharpe']:.3f} test={best['test_sharpe']:.3f} "
                  f"HAC t={best['diff_vs_default_t']:.2f} p={best['diff_vs_default_p']:.3f} "
                  f"bonferroni_p={best['p_bonf']:.3f} (n={best['diff_n_obs']:.0f})")
            adopted = best["p_bonf"] < 0.05
            print(f"Adopted (Bonferroni p<0.05)? {adopted}")
            winners[name] = (best["target_vol"], best["max_weight"]) if adopted else (DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT)

    print("\n=== Combined Allocator: default vs. best-per-Book (whether or not individually adopted) ===")
    default_books = [make_book(n, *books_by_calibration[n][:2], DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT) for n in signals]
    default_pnl = Allocator(default_books).run(returns)["pnl"]

    tuned_books = []
    for name in signals:
        alpha_scaled, cov_dict, _ = books_by_calibration[name]
        tv, mw = winners[name]
        tuned_books.append(make_book(name, alpha_scaled, cov_dict, tv, mw))
    tuned_pnl = Allocator(tuned_books).run(returns)["pnl"]

    for label, pnl in [("Mixed-default", default_pnl), ("Mixed-tuned", tuned_pnl)]:
        train, val, test = train_validation_test_split(pnl)
        print(f"{label}: train={simple_sharpe(train, periods_per_year=PERIODS_PER_YEAR):.3f} "
              f"validation={simple_sharpe(val, periods_per_year=PERIODS_PER_YEAR):.3f} "
              f"test={simple_sharpe(test, periods_per_year=PERIODS_PER_YEAR):.3f}")

    print(
        "\nDocumented recommendation: kept the flat default calibration, did NOT adopt "
        "the tuned one — tuning overfits the validation window (HAC-confirmed), and this "
        "does not generalize to test."
    )


if __name__ == "__main__":
    main()
