"""
research/relative_value.py — Relative Value sleeve: standalone bake-off for all
7 pairs (WORKFLOW.md §11d), before any pooling into a Book.

Two things get baked off, on VALIDATION Sharpe only (CLAUDE.md Rule 1/2 — test
touched exactly once, for the winner):

1. **Hedge-ratio method**, Corn-Wheat and RBOB-HeatingOil only (the two
   HEDGE_RATIO_PAIRS with no natural 1:1 relationship): rolling-OLS vs. Kalman
   filter (`signals.hedge_ratio`). Selected using each pair's CONTINUOUS-sizing
   spec (the house-style default) — the winning method's spread is then reused
   for that pair's own continuous/threshold comparison below (a fair "same
   spread, two sizing rules" comparison rather than re-litigating both
   dimensions jointly).
2. NOT baked off, reported as PARALLEL SPECS (per direct instruction
   2026-08-01, "test both"): continuous vol-scaled z-score sizing vs. the
   discrete threshold entry/exit band. 7 pairs x 2 sizing rules = 14 rows, no
   headline pick — same discipline as every other signal family's own
   multi-spec reporting (momentum's lookback grid, breakout's two systems,
   crossover's three MA pairs, carry's four specs).

**Frequency: daily AND weekly, both reported as parallel specs** (per direct
instruction 2026-08-01, after the first daily-only pass came back with very
high turnover — 9x-106x annualized, worse than every family except short-term
reversal — and net-of-cost consistently well below gross). Daily was the
original a-priori choice (same reasoning as breakout's own daily-direction
requirement: a signal that only re-evaluates at month-end badly lags the
deviation it's trading). Weekly is added as a genuine second construction, NOT
selected after seeing which one wins — both are reported for all 7 pairs x 2
sizing rules (28 rows total), same "report every spec, no headline pick"
discipline as every other multi-spec comparison in this project. The
UNDERLYING signal (z-score, hedge ratio, spread) is still computed daily in
both cases — only the REBALANCING cadence differs (`backtest.engine`'s own
`frequency` parameter resamples an already-daily signal, doesn't rebuild it at
a coarser step), matching breakout's own "direction always daily, sizing
cadence is the caller's choice" convention.

**Cost model**: each pair is a SYNTHETIC instrument (no real bid/ask of its
own) — its one-way cost is approximated as the SUM of its real legs'
liquidity-tiered cost (`backtest.costs.liquidity_tiered_cost_bps`, same
per-asset ADV tiering every other signal family already uses): trading the
spread means trading every real leg, so the costs are additive, not averaged.
The crack spread (3 legs) sums three legs' costs; every 2-leg pair sums two.

Run: `python research/relative_value.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from data.universe import get_liquid_universe
from signals.relative_value import (
    ALL_PAIR_NAMES, RATIO_PAIRS, HEDGE_RATIO_PAIRS, CRACK_SPREAD_NAME,
    build_pair_spread, build_pair_signal, spread_return,
)
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
FREQUENCIES = ("daily", "weekly")
HEDGE_BAKEOFF_FREQUENCY = "daily"  # house-style default, used only to pick rolling_ols vs kalman
TARGET_VOL_SIGNAL = 0.40

PAIR_LEGS = {
    **RATIO_PAIRS,
    **HEDGE_RATIO_PAIRS,
    CRACK_SPREAD_NAME: ("WTI Crude", "RBOB", "HeatingOil"),
}


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    close = adj["close"]
    volume = adj["volume"]
    included, excluded = get_liquid_universe(volume, ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    return close, volume, included


def pair_cost_bps(cost_by_leg: pd.Series, pair_name: str) -> float:
    legs = PAIR_LEGS[pair_name]
    return float(sum(cost_by_leg.get(leg, 0.0) for leg in legs))


def _one_col(series: pd.Series, col: str = "pair") -> pd.DataFrame:
    return series.to_frame(col)


def backtest_pair(signal: pd.Series, ret: pd.Series, cost_bps_value: float, frequency: str):
    signal_df, ret_df = _one_col(signal), _one_col(ret)
    cost_bps_series = pd.Series({"pair": cost_bps_value})
    gross = backtest_signal(signal_df, ret_df, frequency=frequency)
    net = backtest_signal(signal_df, ret_df, frequency=frequency, cost_bps=cost_bps_series)
    return gross, net


def annualized_turnover(signal: pd.Series, frequency: str) -> float:
    positions = normalized_positions(_one_col(signal), frequency)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate_spec(pair_name, spec_label, frequency, signal, ret, cost_bps_value):
    gross, net = backtest_pair(signal, ret, cost_bps_value, frequency)
    g_train, g_val, g_test = train_validation_test_split(gross)
    n_train, n_val, n_test = train_validation_test_split(net)
    return {
        "pair": pair_name, "spec": spec_label, "frequency": frequency,
        "annualized_turnover": annualized_turnover(signal, frequency),
        "train_gross": simple_sharpe(g_train), "train_net": simple_sharpe(n_train),
        "validation_gross": simple_sharpe(g_val), "validation_net": simple_sharpe(n_val),
        "test_gross": simple_sharpe(g_test), "test_net": simple_sharpe(n_test),
    }


def hedge_method_bakeoff(price: pd.DataFrame, cost_by_leg: pd.Series):
    """rolling_ols vs. kalman for each HEDGE_RATIO_PAIRS pair, selected on
    VALIDATION gross Sharpe of the continuous-sizing spec at
    HEDGE_BAKEOFF_FREQUENCY (daily) — a separate decision from the
    daily-vs-weekly REBALANCING comparison below, kept at one fixed frequency
    so this bake-off isn't itself a 2x2 comparison. Returns
    {pair_name: (winning_method, spread)}."""
    winners = {}
    print("\n=== Hedge-ratio method bake-off (Corn-Wheat, RBOB-HeatingOil), validation Sharpe, daily ===")
    for pair_name in HEDGE_RATIO_PAIRS:
        candidates = {}
        for method in ("rolling_ols", "kalman"):
            spread = build_pair_spread(price, pair_name, hedge_method=method)
            signal = build_pair_signal(spread, entry_exit="continuous", target_vol=TARGET_VOL_SIGNAL)
            ret = spread_return(spread)
            gross, _ = backtest_pair(signal, ret, pair_cost_bps(cost_by_leg, pair_name), HEDGE_BAKEOFF_FREQUENCY)
            _, val, _ = train_validation_test_split(gross)
            val_sharpe = simple_sharpe(val)
            candidates[method] = (val_sharpe, spread)
            print(f"  {pair_name:20s} {method:12s} val_sharpe={val_sharpe:.3f}")

        valid = {m: v for m, v in candidates.items() if not np.isnan(v[0])}
        winning_method = max(valid, key=lambda m: valid[m][0]) if valid else "rolling_ols"
        print(f"  -> winner: {winning_method}")
        winners[pair_name] = (winning_method, candidates[winning_method][1])
    return winners


def build_all_pair_spreads_final(price: pd.DataFrame, hedge_winners: dict) -> dict:
    spreads = {}
    for name in RATIO_PAIRS:
        spreads[name] = build_pair_spread(price, name)
    spreads[CRACK_SPREAD_NAME] = build_pair_spread(price, CRACK_SPREAD_NAME)
    for name, (method, spread) in hedge_winners.items():
        spreads[name] = spread
    return spreads


def main():
    close, volume, included = load_and_prepare_data()
    cost_by_leg = liquidity_tiered_cost_bps(volume[included], window_start=ADV_WINDOW_START)

    hedge_winners = hedge_method_bakeoff(close, cost_by_leg)
    spreads = build_all_pair_spreads_final(close, hedge_winners)

    print(f"\n=== Standalone bake-off, all 7 pairs x 2 sizing rules x {len(FREQUENCIES)} frequencies ===")
    rows = []
    for pair_name in ALL_PAIR_NAMES:
        spread = spreads[pair_name]
        ret = spread_return(spread)
        cost_value = pair_cost_bps(cost_by_leg, pair_name)
        for spec_label in ("continuous", "threshold"):
            signal = build_pair_signal(spread, entry_exit=spec_label, target_vol=TARGET_VOL_SIGNAL)
            for frequency in FREQUENCIES:
                rows.append(evaluate_spec(pair_name, spec_label, frequency, signal, ret, cost_value))

    result = pd.DataFrame(rows).set_index(["pair", "spec", "frequency"])
    print(result.round(3).to_string())

    print("\nHedge-ratio method winners: " + ", ".join(f"{p}={m}" for p, (m, _) in hedge_winners.items()))
    print("\nReminder: rolling Engle-Granger diagnostic (research/relative_value_cointegration_check.py) "
          "found weak-to-no cointegration evidence for every 2-leg pair (3-11% of windows at alpha=0.05) — "
          "reported as found, not a gate on this backtest.")

    return result, hedge_winners, spreads


if __name__ == "__main__":
    main()
