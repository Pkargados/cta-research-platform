"""
research/tune_all_books_cpcv.py — CSCV/CPCV Probability-of-Backtest-Overfitting
diagnostic, scaled to ALL 20 Books.

Built 2026-07-23, per direct instruction, immediately following `research/
tune_books_cpcv.py`'s 3-Book pilot (momentum_12mo, breakout_system1, value —
PBO 0.09/0.24/0.37, all comfortably below the noise-like 0.5 threshold). That
spread was reassuring enough to justify scaling, matching this project's own
"validate small, then scale" precedent (`tune_book_hyperparameters.py`'s
2-Book pass -> `tune_all_books.py`'s 20-Book scale-up followed the identical
pattern) — this script is the CPCV analogue of that same step, a NEW script
rather than an overwrite of the 3-Book pilot, so both stay on record
side-by-side (see WORKFLOW.md Phase 7 for both, and for the "is the
'default never wins IS' pattern real or a measurement artifact" open
question this run's own output should be read against, not yet resolved by
this script alone).

Reuses `tune_books_cpcv.py`'s own `build_pnl_matrix` (imported directly, not
copy-pasted — the same "one research script imports another" exception
already logged there) unchanged: weekly-frequency-only, 25-point
(target_vol x max_weight) grid, `backtest.cpcv.cscv_pbo` with the same
n_groups=8 / purge_periods=5 / embargo_periods=5 as the pilot, so this run's
numbers are directly comparable to the pilot's three Books, not a
re-parameterized re-run.

Run: `python research/tune_all_books_cpcv.py` from the repo root. Results
written to `Data/research/tune_all_books_cpcv/` (a separate directory from
the 3-Book pilot's `Data/research/tune_books_cpcv/`, which is left
untouched).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tune_all_books as tab
from tune_books_cpcv import build_pnl_matrix, N_GROUPS, N_TRAIN_GROUPS, PURGE_PERIODS, EMBARGO_PERIODS
from backtest.costs import liquidity_tiered_cost_bps
from backtest.cpcv import cscv_pbo, pbo
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Data" / "research" / "tune_all_books_cpcv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = tab.load_and_prepare_data()
    close = prepared["close"]
    returns = close.pct_change(fill_method=None)
    vol = tab.build_vol(prepared)
    cost_bps = liquidity_tiered_cost_bps(prepared["volume"], window_start=tab.ADV_WINDOW_START)

    all_signals = tab.build_all_signals(prepared, vol)
    default_col = f"tv={tab.DEFAULT_TARGET_VOL:.3f}_mw={tab.DEFAULT_MAX_WEIGHT:.2f}"

    summary_rows = []
    for name, alpha_df in all_signals.items():
        print(f"\n=== {name} ===")
        pnl_matrix = build_pnl_matrix(name, alpha_df, returns, cost_bps)
        if pnl_matrix is None:
            summary_rows.append({"book": name, "skipped": True})
            continue
        pnl_matrix.to_csv(OUTPUT_DIR / f"{name}_pnl_matrix.csv")

        result = cscv_pbo(
            pnl_matrix, n_groups=N_GROUPS, n_train_groups=N_TRAIN_GROUPS,
            purge_periods=PURGE_PERIODS, embargo_periods=EMBARGO_PERIODS,
        )
        result.to_csv(OUTPUT_DIR / f"{name}_cscv_combinations.csv", index=False)

        if len(result) == 0:
            print(f"  {name}: 0 usable combinations (too few valid IS/OOS observations after purge/embargo)")
            summary_rows.append({"book": name, "skipped": True})
            continue

        p = pbo(result)
        winner_counts = result["is_winner"].value_counts()
        default_win_rate = float(winner_counts.get(default_col, 0)) / len(result)
        median_logit = float(result["logit"].median())

        val_sharpes = {c: simple_sharpe(train_validation_test_split(pnl_matrix[c])[1]) for c in pnl_matrix.columns}
        single_window_best = max(val_sharpes, key=lambda k: (val_sharpes[k] if pd.notna(val_sharpes[k]) else -np.inf))

        print(f"  n_combinations_evaluated: {len(result)}")
        print(f"  PBO: {p:.3f}   median logit: {median_logit:.3f}   default IS win-rate: {default_win_rate:.1%}")
        print(f"  most frequent IS-winner: {winner_counts.index[0]} ({winner_counts.iloc[0] / len(result):.1%}) "
              f"vs. single-window pick: {single_window_best}")

        summary_rows.append({
            "book": name, "skipped": False, "n_days": len(pnl_matrix), "n_combinations": len(result),
            "pbo": p, "median_logit": median_logit, "default_win_rate": default_win_rate,
            "most_frequent_is_winner": winner_counts.index[0],
            "most_frequent_is_winner_rate": float(winner_counts.iloc[0]) / len(result),
            "single_window_pick": single_window_best,
        })

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(result["logit"], bins=20, color="#4C72B0", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} — CSCV logit distribution (PBO={p:.2f})")
        ax.set_xlabel("logit(omega_c)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{name}_logit_histogram.png", dpi=100)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    evaluable = summary_df[summary_df["skipped"] == False]
    print("\n=== Summary (sorted by PBO, ascending = most reassuring) ===")
    print(evaluable.sort_values("pbo")[
        ["book", "pbo", "median_logit", "default_win_rate", "most_frequent_is_winner", "most_frequent_is_winner_rate"]
    ].to_string(index=False))

    n_skipped = int((summary_df["skipped"] == True).sum())
    n_default_never_wins = int((evaluable["default_win_rate"] == 0.0).sum())
    print(f"\n{len(evaluable)} of {len(summary_df)} Books evaluable ({n_skipped} skipped — too few active "
          f"assets/rebalance dates/purged observations).")
    print(f"Mean PBO across evaluable Books: {evaluable['pbo'].mean():.3f}   "
          f"Median PBO: {evaluable['pbo'].median():.3f}")
    print(f"Books where the flat default NEVER won in-sample: {n_default_never_wins} of {len(evaluable)}")
    print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
