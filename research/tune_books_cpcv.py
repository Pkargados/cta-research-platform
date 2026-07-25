"""
research/tune_books_cpcv.py — Per-Book hyperparameter selection evidence,
re-measured with CSCV/CPCV instead of one fixed 2020-2021 validation block.

Built 2026-07-23, per direct instruction, as the first concrete step of
WORKFLOW.md Phase 7's "beyond naive Bonferroni/FDR" plan. `research/
tune_all_books.py`'s two runs (2D grid, then cost-inclusive 3D grid) both
found 0 of 19 evaluable Books survive a Bonferroni+FDR-corrected test against
ONE validation window (2020-2021) — a real result, but one that can't
distinguish "tuning never helps here" from "one historical regime is too
thin a sample to tell." This script re-poses the same underlying question
(does picking the best-looking grid point actually generalize, or is the
selection process itself overfitting-prone?) across MANY combinatorial
train/test recombinations of the FULL ~15yr history, using
`backtest.cpcv.cscv_pbo` — see that module's own docstring for the exact
algorithm (Bailey/Borwein/López de Prado/Zhu 2017's CSCV) and the
purge/embargo refinement, both read directly from source before being
implemented, not from memory.

**Deliberately a SMALL-SCALE first pass, not all 20 Books** — matching this
project's own established "validate small, then scale" precedent (the first
Book/Allocator pass used 6 of the eventual ~20 Books; the tuning work itself
started as a 2D grid before the cost-inclusive 3D grid). Three Books,
pre-committed for reasons independent of any CPCV result already seen (not
cherry-picked after running this):

- `momentum_12mo` — the project's own flagship, most-validated spec
  (Moskowitz-Ooi-Pedersen's headline horizon, CLAUDE.md's own current-state
  table), a natural "does the mechanism even make sense on the strategy we
  trust most" check.
- `breakout_system1` — used in the ORIGINAL 6-Book portfolio-construction
  pilot (WORKFLOW.md Phase 7), for continuity with that earlier small-scale
  precedent.
- `value` — the Book that came CLOSEST to surviving `tune_all_books.py`'s
  own correction (within-Book Bonferroni p=0.032 in the 2D/no-cost run, 0.056
  in the cost-inclusive 3D run — right at the edge either way) — the single
  most informative case to re-examine with more independent evidence, since
  it's the one where "was this actually close to real, or was the single
  validation window flattering it" is a live, unresolved question.

**Scope limits, stated explicitly**: uses the DEFAULT rebalance frequency
only (weekly — `tune_all_books.py`'s own FREQUENCY_GRID dimension is
dropped here, not because it doesn't matter, but to keep this first CPCV
pass to a 25-point target_vol x max_weight grid per Book, matching the
original 2D grid's own size, rather than compounding two new dimensions of
scope increase in one pass). Reuses `tune_all_books.py`'s own data loading,
signal construction, and Book calibration verbatim (imported directly, not
copy-pasted) — this is the first time one `research/*.py` driver script
imports another; a deliberate, logged exception to avoid duplicating ~150
lines of already-established 20-Book construction logic that would
otherwise silently drift out of sync with the canonical version.

Run: `python research/tune_books_cpcv.py` from the repo root. Results
written to `Data/research/tune_books_cpcv/`.
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
from portfolio.book import daily_mark_pnl
from backtest.costs import liquidity_tiered_cost_bps
from backtest.cpcv import cscv_pbo, pbo
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Data" / "research" / "tune_books_cpcv"

BOOKS_TO_TEST = ["momentum_12mo", "breakout_system1", "value"]

N_GROUPS = 8          # C(8,4) = 70 combinations per Book — see backtest/cpcv.py docstring
N_TRAIN_GROUPS = 4     # symmetric split, matching the CSCV paper's own design
PURGE_PERIODS = 5      # 1 trading week, matching this project's current weekly Book cadence
EMBARGO_PERIODS = 5


def build_pnl_matrix(name, alpha_df, returns, cost_bps):
    """(T x 25) daily-marked, cost-inclusive PnL matrix — one column per
    (target_vol, max_weight) grid point at the DEFAULT (weekly) frequency
    only. Reuses tune_all_books.py's own grid/constants/Book construction
    verbatim, just collects the full daily PnL series per grid point instead
    of immediately slicing to one validation window."""
    active = tab._active_columns(alpha_df, returns)
    if len(active) < 3:
        print(f"  {name}: skipped — only {len(active)} active assets (<3)")
        return None

    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:tab.TRAIN_END].stack().std() if hasattr(tab, "TRAIN_END") else alpha_active.stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    freq_name, freq_str, ppy, halflife = next(f for f in tab.FREQUENCY_GRID if f[0] == tab.DEFAULT_FREQUENCY)
    cov_dict = tab.build_cov_dict(returns[active], window=tab.COV_WINDOW, freq=freq_str)
    if len(cov_dict) < 20:
        print(f"  {name}: skipped — <20 usable rebalance dates at {freq_name} frequency")
        return None

    columns = {}
    for tv in tab.TARGET_VOL_GRID:
        for mw in tab.MAX_WEIGHT_GRID:
            book = tab.make_book(name, alpha_scaled, cov_dict, tv, mw, ppy, halflife, cost_bps)
            result = book.run(returns)
            if "weights" not in result or len(result.get("pnl", [])) == 0:
                continue
            columns[f"tv={tv:.3f}_mw={mw:.2f}"] = daily_mark_pnl(result["weights"], returns, cost_bps=cost_bps)

    if len(columns) < 3:
        print(f"  {name}: skipped — only {len(columns)} viable grid points (<3)")
        return None

    return pd.DataFrame(columns)


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

    for name in BOOKS_TO_TEST:
        print(f"\n=== {name} ===")
        pnl_matrix = build_pnl_matrix(name, all_signals[name], returns, cost_bps)
        if pnl_matrix is None:
            continue
        pnl_matrix.to_csv(OUTPUT_DIR / f"{name}_pnl_matrix.csv")

        n_days = len(pnl_matrix)
        n_days_per_group = n_days // N_GROUPS
        print(f"  {n_days} daily PnL obs, {N_GROUPS} groups (~{n_days_per_group} days/group), "
              f"C({N_GROUPS},{N_TRAIN_GROUPS})={70} combinations")

        result = cscv_pbo(
            pnl_matrix, n_groups=N_GROUPS, n_train_groups=N_TRAIN_GROUPS,
            purge_periods=PURGE_PERIODS, embargo_periods=EMBARGO_PERIODS,
        )
        result.to_csv(OUTPUT_DIR / f"{name}_cscv_combinations.csv", index=False)

        p = pbo(result)
        winner_counts = result["is_winner"].value_counts()
        default_win_rate = float(winner_counts.get(default_col, 0)) / len(result) if len(result) else np.nan
        median_logit = float(result["logit"].median()) if len(result) else np.nan

        # For comparison: what the SINGLE-validation-window approach already
        # found for this Book's default vs. its own grid-best (reusing
        # tune_all_books.py's own train/validation/test slicer directly).
        _, val_default, _ = train_validation_test_split(pnl_matrix[default_col])
        val_sharpes = {c: simple_sharpe(train_validation_test_split(pnl_matrix[c])[1]) for c in pnl_matrix.columns}
        single_window_best = max(val_sharpes, key=lambda k: (val_sharpes[k] if pd.notna(val_sharpes[k]) else -np.inf))

        print(f"  n_combinations_evaluated: {len(result)}")
        print(f"  PBO (fraction of combos where the IS-winner underperformed the OOS median): {p:.3f}")
        print(f"  median logit: {median_logit:.3f}  (>0 = IS-optimal choice typically beats OOS median; <=0 = typically doesn't)")
        print(f"  IS-winner was the DEFAULT config in {default_win_rate:.1%} of combinations")
        print(f"  single-fixed-validation-window's own pick: {single_window_best} "
              f"(vs. CPCV's most frequent IS-winner: {winner_counts.index[0] if len(winner_counts) else 'n/a'}, "
              f"picked in {winner_counts.iloc[0] / len(result):.1%} of combos)" if len(winner_counts) else "")

        summary_rows.append({
            "book": name, "n_days": n_days, "n_combinations": len(result), "pbo": p,
            "median_logit": median_logit, "default_win_rate": default_win_rate,
            "most_frequent_is_winner": winner_counts.index[0] if len(winner_counts) else None,
            "most_frequent_is_winner_rate": float(winner_counts.iloc[0]) / len(result) if len(result) else np.nan,
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
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
