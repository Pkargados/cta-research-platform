"""
research/single_strategy_portfolios.py — "Single Strategy Portfolios" phase
(the user's own name for next_steps.md Phase 4: build canonical single-
strategy Books, one per accepted family under the two-Book mandate — decision
#7 in WORKFLOW.md's open-decisions log).

Two thematic bake-offs, each selected on the VALIDATION period only, test
touched exactly once for the winner (CLAUDE.md Rule 1/2) — a small number of
economically-motivated constructions per Book, not a numerical grid search, so
this is NOT the same multiple-comparisons situation `tune_all_books.py`/
`tune_all_books_cpcv.py` found overfits (19 Books x 25-50 grid points). No
Bonferroni/FDR correction applied here — the earlier finding doesn't transfer
to comparing 5-7 conceptually distinct constructions of ONE Book.

Trend Book flavors (decision #8: TSMOM + crossover 50/200, breakout parked):
- tsmom_alone      — baseline reference, not really an "ensemble" flavor
- equal_weight     — combine_alphas(method="equal")
- fixed_70_30      — combine_alphas(method="fixed", weights=[0.7, 0.3]), TSMOM-tilted
- ic_weighted      — signals.combine.ic_weighted_combine, adaptive trailing-IC weight
- risk_parity      — signals.combine.risk_parity_combine, inverse-trailing-vol weight
- confirmation     — signals.combine.confirmation_filter_combine, TSMOM sized,
                      crossover as a direction-only gate (flat when they disagree)
- tsmom_deadband   — signals.transforms.vol_targeted_sign_signal_with_deadband on
                      TSMOM alone: genuinely flat (not just downsized) when trend
                      strength is below its own trailing median — the "not
                      constantly trading" construction, per direct request

Carry Book flavors: the 4 already-existing parallel specs from
`signals.carry.build_all_carry_signals` (carry1m, carry1_12, carry_timing_zero,
carry_timing_mean) — no new construction needed, just run head-to-head on the
same validation-selection discipline as Trend.

Weekly Book rebalancing (COV_FREQ="W-FRI", PERIODS_PER_YEAR=52,
EWMA_HALFLIFE=87) — not monthly (research/portfolio.py's original pilot) —
reusing the exact precedent already established in `research/
value_momentum_combine.py`/`research/tune_book_hyperparameters.py` for the same
reason: monthly rebalancing left too few validation observations (~16-21) to
compare candidates meaningfully; weekly gives ~4.3x that (~70-90), without
moving the deliberately-placed TRAIN_END/VALIDATION_END boundaries themselves
(COVID stays in validation, the 2022 shock stays the untouched test period's
own stress event — see backtest/splits.py's own docstring for why those
boundaries aren't the right lever to pull).

Run: `python research/single_strategy_portfolios.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from data.term_structure import build_carry_panel
from signals.momentum import tsmom_signal, raw_momentum, HEADLINE_LOOKBACK_MONTHS, HEADLINE_TARGET_VOL
from signals.crossover import crossover_pair_signal
from signals.carry import build_all_carry_signals
from signals.combine import combine_alphas, ic_weighted_combine, risk_parity_combine, confirmation_filter_combine
from signals.transforms import vol_targeted_sign_signal_with_deadband
from portfolio.covariance import build_cov_dict
from portfolio.book import Book
from portfolio.allocator import Allocator
from portfolio.risk_metrics import historical_var, expected_shortfall
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe
from backtest.engine import backtest_signal

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
TARGET_VOL_SIGNAL = 0.40

# Book calibration - weekly cadence, per the module docstring above (NOT
# research/portfolio.py's original monthly pilot).
GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252
COV_FREQ = "W-FRI"
PERIODS_PER_YEAR = 52
EWMA_HALFLIFE = 87
TARGET_VOL_BOOK = 0.10  # per-Book vol target, decision #5 - standard CTA-sleeve convention
MAX_WEIGHT = 0.30


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded: {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    sectors = sectors_for_universe(included)
    return adj, raw, included, sectors


def build_vol(raw):
    return yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )


def build_trend_flavors(close, vol, returns):
    momentum = tsmom_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=TARGET_VOL_SIGNAL)
    crossover = crossover_pair_signal(close, vol, "50_200", target_vol=TARGET_VOL_SIGNAL)

    momentum_returns = backtest_signal(momentum, returns, frequency="monthly", holding_months=1)
    crossover_returns = backtest_signal(crossover, returns, frequency="daily")

    # shift(-1): IC should measure PREDICTIVE correlation (alpha at t vs. the
    # return realized AFTER t), not contemporaneous correlation. The weight
    # itself is still shift(1)'d inside ic_weighted_combine, so today's combine
    # weight only ever uses information knowable before today either way - this
    # shift is about what "IC" means, not about look-ahead safety.
    forward_returns = returns.shift(-1)

    mom_raw = raw_momentum(close, HEADLINE_LOOKBACK_MONTHS, skip_months=0)
    deadband = vol_targeted_sign_signal_with_deadband(
        mom_raw, vol, target_vol=TARGET_VOL_SIGNAL, deadband_quantile=0.5, deadband_window=252, min_periods=60,
    )

    return {
        "tsmom_alone": momentum,
        "equal_weight": combine_alphas([momentum, crossover], method="equal"),
        "fixed_70_30": combine_alphas([momentum, crossover], weights=[0.7, 0.3], method="fixed"),
        "ic_weighted": ic_weighted_combine([momentum, crossover], forward_returns, lookback=252),
        "risk_parity": risk_parity_combine([momentum, crossover], [momentum_returns, crossover_returns], lookback=252),
        "confirmation": confirmation_filter_combine(momentum, crossover),
        "tsmom_deadband": deadband,
    }


def _active_columns(alpha_df, returns_df, min_valid_frac=0.90):
    has_alpha = alpha_df.notna().any()
    returns_valid_frac = returns_df.notna().mean()
    return [c for c in alpha_df.columns if has_alpha.get(c, False) and returns_valid_frac.get(c, 0.0) >= min_valid_frac]


def build_book(name, alpha_df, returns, vol_estimator="ewma"):
    active = _active_columns(alpha_df, returns)
    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:TRAIN_END].stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    cov_dict = build_cov_dict(returns[active], window=COV_WINDOW, freq=COV_FREQ)
    return Book(
        name=name, alpha_df=alpha_scaled, cov_dict=cov_dict,
        gamma=GAMMA, kappa=KAPPA, lambd=LAMBD, max_weight=MAX_WEIGHT,
        target_vol=TARGET_VOL_BOOK, ewma_halflife=EWMA_HALFLIFE,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=PERIODS_PER_YEAR, dollar_neutral=False,
        vol_estimator=vol_estimator,
    )


def evaluate_flavor(name, alpha_df, returns):
    book = build_book(name, alpha_df, returns)
    result = book.run(returns)
    # Book.run() returns a minimal dict (no "turnover"/"max_dd") when fewer than
    # 20 valid rebalance dates exist at all (its own early-return guard) - .get()
    # rather than direct indexing so a genuinely too-sparse flavor is reported as
    # such, not a crash.
    train, val, test = train_validation_test_split(result["pnl"])
    return {
        "flavor": name,
        "train_sharpe": simple_sharpe(train, periods_per_year=PERIODS_PER_YEAR),
        "val_sharpe": simple_sharpe(val, periods_per_year=PERIODS_PER_YEAR),
        "test_sharpe": simple_sharpe(test, periods_per_year=PERIODS_PER_YEAR),
        "n_train": len(train.dropna()), "n_val": len(val.dropna()), "n_test": len(test.dropna()),
        "n_rebalance_dates_valid": result.get("n_rebalance_dates_valid"),
        "turnover": result.get("turnover", np.nan), "max_dd": result.get("max_dd", np.nan),
        "book": book, "result": result,
    }


def bake_off(flavors, returns, label):
    print(f"\n=== {label} bake-off (validation-selected, test touched once for the winner) ===")
    rows = []
    for name, alpha_df in flavors.items():
        evaluation = evaluate_flavor(name, alpha_df, returns)
        rows.append(evaluation)
        if evaluation["n_rebalance_dates_valid"] is not None and evaluation["n_rebalance_dates_valid"] < 20:
            print(f"  {name:16s} INSUFFICIENT DATA (only {evaluation['n_rebalance_dates_valid']} valid rebalance dates, need >= 20) - excluded from bake-off")
            continue
        print(f"  {name:16s} train={evaluation['train_sharpe']:.3f}  "
              f"val={evaluation['val_sharpe']:.3f} (n={evaluation['n_val']})  "
              f"turnover={evaluation['turnover']:.2f}  max_dd={evaluation['max_dd']:.3f}")

    valid = [r for r in rows if not np.isnan(r["val_sharpe"])]
    winner = max(valid, key=lambda r: r["val_sharpe"])
    print(f"\n  Winner by validation Sharpe: {winner['flavor']} (val={winner['val_sharpe']:.3f})")
    print(f"  Winner's test Sharpe (touched once): {winner['test_sharpe']:.3f} (n={winner['n_test']})")
    return winner, rows


def main():
    adj, raw, included, sectors = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    carry_panel, _ = build_carry_panel(included)

    # Flavor bake-off runs under vol_estimator="ewma" (unchanged, matches the
    # already-logged WORKFLOW.md decision #10 numbers) - all flavors within
    # each bake-off use the SAME estimator, so the ranking comparison among
    # them stays fair even though it wasn't re-run under GARCH. The winning
    # CONSTRUCTION (which alpha to trade) and the winning VOL ESTIMATOR
    # (decision #11: GARCH beats EWMA at forecasting a Book's own realized-PnL
    # vol) are two separate decisions, made at two different times - not
    # re-litigating the first to adopt the second.
    trend_flavors = build_trend_flavors(close, vol, returns)
    trend_winner, trend_rows = bake_off(trend_flavors, returns, "Trend Book")

    carry_flavors = build_all_carry_signals(carry_panel, sectors)
    carry_winner, carry_rows = bake_off(carry_flavors, returns, "Carry Book")

    print(f"\nSelected Trend Book construction: {trend_winner['flavor']}")
    print(f"Selected Carry Book construction: {carry_winner['flavor']}")

    print("\n=== Final Single Strategy Portfolios: winning construction, GARCH vol-targeting ===")
    trend_book_garch = build_book("trend_" + trend_winner["flavor"], trend_flavors[trend_winner["flavor"]], returns, vol_estimator="garch")
    trend_result_garch = trend_book_garch.run(returns)
    carry_book_garch = build_book("carry_" + carry_winner["flavor"], carry_flavors[carry_winner["flavor"]], returns, vol_estimator="garch")
    carry_result_garch = carry_book_garch.run(returns)

    for label, result in (("Trend", trend_result_garch), ("Carry", carry_result_garch)):
        print(f"\n  {label} ({result.get('n_rebalance_dates_valid')} valid rebalance dates):")
        for period, series in zip(("train", "validation", "test"), train_validation_test_split(result["pnl"])):
            print(f"    {period}: Sharpe={simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR):.3f} (n={len(series.dropna())})")
        print(f"    turnover={result.get('turnover', float('nan')):.3f}  max_dd={result.get('max_dd', float('nan')):.3f}")

    print("\n=== Combined two-Book Allocator (GARCH vol-targeting, equal Book-risk baseline) ===")
    allocator = Allocator([trend_book_garch, carry_book_garch])
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(combined_pnl)):
        print(f"  {period}: Sharpe={simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR):.3f} (n={len(series.dropna())})")

    var95 = historical_var(combined_pnl, confidence=0.95)
    es95 = expected_shortfall(combined_pnl, confidence=0.95)
    print(f"\n  95% VaR: {var95:.3f}  ES: {es95:.3f} (weekly)")


if __name__ == "__main__":
    main()
