"""
research/relative_value_halflife.py — does an ECM-half-life-informed z-score window
beat the current uniform 63-day window (`signals.relative_value.DEFAULT_Z_WINDOW`)?

Per direct instruction 2026-08-02, following up on `references/Mean Reversion Using
Machine Learning.pdf` (low-credibility source, see WORKFLOW.md — but its "calibrate the
lookback from mean-reversion speed" idea is worth testing on its own merits).

**Calibration, decided from TRAIN data only (CLAUDE.md Rule 1/2)**: for each pair,
`signals.error_correction.median_rolling_half_life` (the median of many rolling
252-day-window ECM fits within train — checked directly to be far more robust than a
single long regression over the whole train span, which is statistically fragile for a
weakly-cointegrated spread; see that function's own docstring) gives one half-life
number, converted to a window via `half_life_to_window` (2x half-life, clipped to
[10, 252], falling back to the current 63-day default when undefined). This one number
per pair is then used as a FIXED z_window for the whole backtest — not re-estimated
walk-forward, same discipline as every other per-pair construction constant in this
sleeve (e.g. the hedge-ratio-method choice).

**Controlled comparison**: for each pair, both the current fixed-63 window and the
half-life-implied window are backtested using the SAME entry/exit spec and frequency
already selected in `research/relative_value.py`'s own standalone bake-off — isolating
the z_window's own marginal effect, not re-litigating spec/frequency at the same time.

Run: `python research/relative_value_halflife.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import relative_value as rv
from signals.relative_value import ALL_PAIR_NAMES, build_pair_signal, spread_return, DEFAULT_Z_WINDOW
from signals.error_correction import median_rolling_half_life, half_life_to_window
from backtest.splits import TRAIN_END, train_validation_test_split
from backtest.performance import simple_sharpe

WINDOW_MULTIPLIER = 2.0
MIN_WINDOW = 10
MAX_WINDOW = 252


def pick_best_spec_frequency(result: pd.DataFrame, pair_name: str) -> tuple:
    """The (entry_exit, frequency) combo with the highest validation NET Sharpe
    for this pair, from relative_value.py's already-computed standalone table
    — reused so this comparison isolates z_window, not re-picking spec/frequency
    too."""
    sub = result.xs(pair_name, level="pair")["validation_net"].dropna()
    if sub.empty:
        return "continuous", "daily"
    return sub.idxmax()  # (spec, frequency)


def main():
    result, hedge_winners, spreads = rv.main()

    print("\n=== Half-life calibration per pair (median of rolling train-period ECM fits) ===")
    rows = []
    for pair in ALL_PAIR_NAMES:
        spread = spreads[pair]
        hl = median_rolling_half_life(spread, TRAIN_END)
        window = half_life_to_window(hl, multiplier=WINDOW_MULTIPLIER, min_window=MIN_WINDOW, max_window=MAX_WINDOW, fallback=DEFAULT_Z_WINDOW)
        rows.append({"pair": pair, "median_half_life": hl, "implied_window": window})
        print(f"  {pair:26s} half_life={hl:.1f}" + (f"  implied_window={window}" if hl == hl else f"  implied_window={window} (fallback, half-life undefined)"))

    calib = pd.DataFrame(rows).set_index("pair")

    print("\n=== Controlled comparison: fixed 63-day window vs. half-life-implied window ===")
    print("(same entry_exit spec and frequency per pair, held fixed - only z_window changes)\n")

    close, volume, included = rv.load_and_prepare_data()
    cost_by_leg = rv.liquidity_tiered_cost_bps(volume[included], window_start=rv.ADV_WINDOW_START)

    final_rows = []
    for pair in ALL_PAIR_NAMES:
        spec, freq = pick_best_spec_frequency(result, pair)
        spread = spreads[pair]
        ret = spread_return(spread)
        cost_value = rv.pair_cost_bps(cost_by_leg, pair)
        implied_window = calib.loc[pair, "implied_window"]

        signal_fixed = build_pair_signal(spread, entry_exit=spec, z_window=DEFAULT_Z_WINDOW, target_vol=rv.TARGET_VOL_SIGNAL)
        signal_hl = build_pair_signal(spread, entry_exit=spec, z_window=int(implied_window), target_vol=rv.TARGET_VOL_SIGNAL)

        gross_f, net_f = rv.backtest_pair(signal_fixed, ret, cost_value, freq)
        gross_h, net_h = rv.backtest_pair(signal_hl, ret, cost_value, freq)

        _, val_f, test_f = train_validation_test_split(net_f)
        train_f, _, _ = train_validation_test_split(net_f)
        _, val_h, test_h = train_validation_test_split(net_h)
        train_h, _, _ = train_validation_test_split(net_h)

        final_rows.append({
            "pair": pair, "spec": spec, "frequency": freq, "implied_window": implied_window,
            "fixed63_train": simple_sharpe(train_f), "fixed63_val": simple_sharpe(val_f), "fixed63_test": simple_sharpe(test_f),
            "halflife_train": simple_sharpe(train_h), "halflife_val": simple_sharpe(val_h), "halflife_test": simple_sharpe(test_h),
        })

    comparison = pd.DataFrame(final_rows).set_index("pair")
    print(comparison.round(3).to_string())

    print("\n=== Validation Sharpe delta (half-life - fixed63), positive = half-life window helps ===")
    delta = comparison["halflife_val"] - comparison["fixed63_val"]
    print(delta.round(3).sort_values(ascending=False).to_string())

    return comparison


if __name__ == "__main__":
    main()
