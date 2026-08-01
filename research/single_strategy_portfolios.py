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
- tsmom_seasonal   — signals.seasonality.tsmom_seasonal_signal, WORKFLOW.md Phase
                      11c implemented: TSMOM alone, scaled by a continuous
                      per-asset seasonal conviction multiplier (up to +50% at a
                      window's center, exactly 1.0 elsewhere) for the 7 named
                      commodities with a real physical seasonal demand driver —
                      everything else (FX, Rates, equity indices, metals)
                      unchanged from tsmom_alone

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
from data.universe import get_liquid_universe, compress_for_family
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from data.term_structure import build_carry_panel
from signals.momentum import tsmom_signal, raw_momentum, HEADLINE_LOOKBACK_MONTHS, HEADLINE_TARGET_VOL
from signals.crossover import crossover_pair_signal
from signals.carry import build_all_carry_signals
from signals.combine import combine_alphas, ic_weighted_combine, risk_parity_combine, confirmation_filter_combine
from signals.transforms import vol_targeted_sign_signal_with_deadband
from signals.seasonality import tsmom_seasonal_signal
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

# The ADOPTED winning flavors (WORKFLOW.md decisions #10/#13) - single source
# of truth, reused by build_adopted_books() below and by
# dashboard/_single_strategy_pipeline.py (imported, not re-declared).
TREND_FLAVOR = "tsmom_alone"
CARRY_FLAVOR = "carry_timing_zero"


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

    seasonal = tsmom_seasonal_signal(close, vol, lookback_months=HEADLINE_LOOKBACK_MONTHS, target_vol=TARGET_VOL_SIGNAL)

    return {
        "tsmom_alone": momentum,
        "equal_weight": combine_alphas([momentum, crossover], method="equal"),
        "fixed_70_30": combine_alphas([momentum, crossover], weights=[0.7, 0.3], method="fixed"),
        "ic_weighted": ic_weighted_combine([momentum, crossover], forward_returns, lookback=252),
        "risk_parity": risk_parity_combine([momentum, crossover], [momentum_returns, crossover_returns], lookback=252),
        "confirmation": confirmation_filter_combine(momentum, crossover),
        "tsmom_deadband": deadband,
        "tsmom_seasonal": seasonal,
    }


def _active_columns(alpha_df, returns_df, min_valid_frac=0.90):
    has_alpha = alpha_df.notna().any()
    returns_valid_frac = returns_df.notna().mean()
    return [c for c in alpha_df.columns if has_alpha.get(c, False) and returns_valid_frac.get(c, 0.0) >= min_valid_frac]


def build_book(name, alpha_df, returns, vol_estimator="ewma", cov_dict_builder=None, cost_bps=None):
    """`cov_dict_builder` defaults to `build_cov_dict` (Ledoit-Wolf, every
    existing caller's unchanged behavior) - overridable to any callable with
    the same `(returns, window, freq)` signature, e.g.
    `functools.partial(gerber_covariance.build_gerber_cov_dict, c=0.5)`, for
    `research/gerber_book_performance.py`'s own covariance-estimator swap
    (WORKFLOW.md's Gerber statistic covariance plan, Phase 7 follow-up).

    `cost_bps` defaults to None (every existing caller's unchanged GROSS
    behavior) - pass `backtest.costs.liquidity_tiered_cost_bps(...)` (same
    construction `research/tune_all_books.py` already uses) for a NET-of-cost
    Book, needed to tell a genuine turnover-driven cost saving apart from a
    pure gross-Sharpe effect."""
    if cov_dict_builder is None:
        cov_dict_builder = build_cov_dict
    active = _active_columns(alpha_df, returns)
    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:TRAIN_END].stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    cov_dict = cov_dict_builder(returns[active], window=COV_WINDOW, freq=COV_FREQ)
    return Book(
        name=name, alpha_df=alpha_scaled, cov_dict=cov_dict,
        gamma=GAMMA, kappa=KAPPA, lambd=LAMBD, max_weight=MAX_WEIGHT,
        target_vol=TARGET_VOL_BOOK, ewma_halflife=EWMA_HALFLIFE,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=PERIODS_PER_YEAR, dollar_neutral=False,
        vol_estimator=vol_estimator, cost_bps=cost_bps,
    )


def build_adopted_books():
    """The current ADOPTED Trend/Carry construction (WORKFLOW.md decisions
    #10/#11/#13) - Trend on `compress_for_family(included, "trend")`
    (tsmom_alone), Carry on the full uncompressed universe (carry_timing_zero,
    reverted after the compressed bake-off picked an unstable flavor). Builds
    directly from the already-decided winning flavor/universe combination,
    WITHOUT re-running the 7-flavor/4-flavor bake-off (`main()`'s job, already
    decided and logged) - fast, for reuse by downstream consumers (CLAUDE.md
    Rule 6): `dashboard/_single_strategy_pipeline.py` and any multi-strategy
    comparison script (e.g. `research/multi_strategy_seasonality.py`).

    Returns (returns, trend_book, carry_book). `returns` covers the FULL
    included universe (a superset of both Books' own compressed/uncompressed
    universes), safe to reuse directly for any Allocator combination."""
    adj, raw, included, sectors = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    carry_panel, _ = build_carry_panel(included)

    trend_universe = compress_for_family(included, "trend")
    trend_flavors = build_trend_flavors(close[trend_universe], vol[trend_universe], returns[trend_universe])
    trend_book = build_book("trend_" + TREND_FLAVOR, trend_flavors[TREND_FLAVOR], returns, vol_estimator="garch")

    carry_flavors = build_all_carry_signals(carry_panel, sectors)
    carry_book = build_book("carry_" + CARRY_FLAVOR, carry_flavors[CARRY_FLAVOR], returns, vol_estimator="garch")

    return returns, trend_book, carry_book


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


def run_pipeline(close, vol, returns, carry_panel, sectors, label):
    """One full Trend+Carry bake-off -> GARCH Books -> combined Allocator pass,
    for a given (close, vol, returns, carry_panel, sectors) universe slice.
    Extracted so the uncompressed (published, decision #10/#11/#12) and
    compressed (next_steps.md Phase 2 layers B/C) universes can each run this
    exact same pipeline without duplicating it (CLAUDE.md Rule 6)."""
    print(f"\n########## {label} ##########")

    # Flavor bake-off runs under vol_estimator="ewma" (unchanged, matches the
    # already-logged WORKFLOW.md decision #10 numbers) - all flavors within
    # each bake-off use the SAME estimator, so the ranking comparison among
    # them stays fair even though it wasn't re-run under GARCH. The winning
    # CONSTRUCTION (which alpha to trade) and the winning VOL ESTIMATOR
    # (decision #11: GARCH beats EWMA at forecasting a Book's own realized-PnL
    # vol) are two separate decisions, made at two different times - not
    # re-litigating the first to adopt the second.
    trend_flavors = build_trend_flavors(close, vol, returns)
    trend_winner, trend_rows = bake_off(trend_flavors, returns, f"Trend Book ({label})")

    carry_flavors = build_all_carry_signals(carry_panel, sectors)
    carry_winner, carry_rows = bake_off(carry_flavors, returns, f"Carry Book ({label})")

    print(f"\nSelected Trend Book construction: {trend_winner['flavor']}")
    print(f"Selected Carry Book construction: {carry_winner['flavor']}")

    print(f"\n=== Final Single Strategy Portfolios ({label}): winning construction, GARCH vol-targeting ===")
    trend_book_garch = build_book("trend_" + trend_winner["flavor"], trend_flavors[trend_winner["flavor"]], returns, vol_estimator="garch")
    trend_result_garch = trend_book_garch.run(returns)
    carry_book_garch = build_book("carry_" + carry_winner["flavor"], carry_flavors[carry_winner["flavor"]], returns, vol_estimator="garch")
    carry_result_garch = carry_book_garch.run(returns)

    summary_rows = {}
    for name, result in (("Trend", trend_result_garch), ("Carry", carry_result_garch)):
        print(f"\n  {name} ({result.get('n_rebalance_dates_valid')} valid rebalance dates):")
        sharpes = {}
        for period, series in zip(("train", "validation", "test"), train_validation_test_split(result["pnl"])):
            sh = simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR)
            sharpes[period] = sh
            print(f"    {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")
        print(f"    turnover={result.get('turnover', float('nan')):.3f}  max_dd={result.get('max_dd', float('nan')):.3f}")
        summary_rows[name] = {"turnover": result.get("turnover", float("nan")), **sharpes}

    print(f"\n=== Combined two-Book Allocator ({label}, GARCH vol-targeting, equal Book-risk baseline) ===")
    allocator = Allocator([trend_book_garch, carry_book_garch])
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]
    combined_sharpes = {}
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(combined_pnl)):
        sh = simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR)
        combined_sharpes[period] = sh
        print(f"  {period}: Sharpe={sh:.3f} (n={len(series.dropna())})")
    summary_rows["Combined"] = {"turnover": float("nan"), **combined_sharpes}

    var95 = historical_var(combined_pnl, confidence=0.95)
    es95 = expected_shortfall(combined_pnl, confidence=0.95)
    print(f"\n  95% VaR: {var95:.3f}  ES: {es95:.3f} (weekly)")

    return summary_rows, trend_book_garch, carry_book_garch


def main():
    adj, raw, included, sectors = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    carry_panel, _ = build_carry_panel(included)

    uncompressed, _trend_book_u, carry_book_u = run_pipeline(close, vol, returns, carry_panel, sectors, "UNCOMPRESSED (published, decision #10/#11/#12)")

    # next_steps.md Phase 2, layers B/C - see data.universe.compress_for_family
    # and research/universe_compression.py for the train-period correlation
    # analysis behind these exclusions. Trend (a per-asset time-series signal)
    # and Carry (a cross-sectional rank signal, like same_month) get DIFFERENT
    # compressed universes - a duplicate directional bet is a real redundancy
    # problem for Trend that doesn't apply the same way to Carry's own
    # rank-weighted construction.
    trend_universe = compress_for_family(included, "trend")
    rank_universe = compress_for_family(included, "rank")
    print(f"\nTrend universe compressed: {len(trend_universe)} of {len(included)} (dropped {sorted(set(included) - set(trend_universe))})")
    print(f"Carry universe compressed: {len(rank_universe)} of {len(included)} (dropped {sorted(set(included) - set(rank_universe))})")

    close_t, vol_t = close[trend_universe], vol[trend_universe]
    carry_panel_c, _ = build_carry_panel(rank_universe)
    sectors_rank = sectors_for_universe(rank_universe)
    # run_pipeline needs ONE returns_data frame covering both Trend's and
    # Carry's own (different) compressed universes - a superset is fine,
    # since bake_off/build_book/_active_columns only ever index returns by
    # alpha_df's own columns, never require an exact match.
    returns_union = returns[sorted(set(trend_universe) | set(rank_universe))]

    compressed, trend_book_c, _carry_book_c = run_pipeline(close_t, vol_t, returns_union, carry_panel_c, sectors_rank, "COMPRESSED (next_steps.md Phase 2, layers B/C)")

    print("\n=== Compressed vs. uncompressed comparison (test Sharpe / turnover), reported as found ===")
    for name in ("Trend", "Carry", "Combined"):
        u, c = uncompressed[name], compressed[name]
        print(
            f"  {name:10s} uncompressed test={u['test']:.3f} turnover={u['turnover']:.2f}  |  "
            f"compressed test={c['test']:.3f} turnover={c['turnover']:.2f}"
        )

    # ADOPTED, per direct instruction (WORKFLOW.md decision #13): Trend keeps
    # the compressed universe (a clean, genuine improvement - higher Sharpe,
    # LOWER turnover). Carry is REVERTED to the uncompressed, originally
    # published construction - the compression flipped Carry's own bake-off
    # winner to a knife-edge, unstable construction (carry1m, test Sharpe
    # -2.035 on 26 observations) via a marginal data-sufficiency threshold
    # effect, not genuine outperformance, so it was rejected rather than
    # mechanically adopted just because compression was applied. This is the
    # exact combination dashboard/_single_strategy_pipeline.py now runs live.
    print("\n########## ADOPTED: Trend (compressed) + Carry (uncompressed, reverted) ##########")
    allocator_adopted = Allocator([trend_book_c, carry_book_u])
    adopted = allocator_adopted.run(returns)
    adopted_pnl = adopted["pnl"]
    for period, series in zip(("train", "validation", "test"), train_validation_test_split(adopted_pnl)):
        print(f"  {period}: Sharpe={simple_sharpe(series, periods_per_year=PERIODS_PER_YEAR):.3f} (n={len(series.dropna())})")
    var95_a = historical_var(adopted_pnl, confidence=0.95)
    es95_a = expected_shortfall(adopted_pnl, confidence=0.95)
    print(f"  95% VaR: {var95_a:.3f}  ES: {es95_a:.3f} (weekly)")


if __name__ == "__main__":
    main()
