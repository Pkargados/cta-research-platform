"""
research/relative_value_cointegration_check.py — Rebuild of the rolling-window
Engle-Granger cointegration diagnostic WORKFLOW.md §11d's plan originally cited
as already done (`Time_Series_Models.ipynb`'s claimed Corn/Wheat, Gold/Silver,
Brent/WTI results, including the "45% of windows cointegrated" Brent/WTI figure).
That notebook does not exist anywhere in this repo (checked directly, see §11d's
"Real gap found 2026-08-01" addendum) — this driver re-verifies the diagnostic
from scratch as real `src/`/`research/` code, not a notebook, and reports
whatever the real numbers come out to be, not the old unverified claim.

Tests all 6 two-leg pairs (the 3 originally cited plus the 4 new ones from the
7-pair list — the crack spread is excluded, see `signals.cointegration`'s own
docstring: it's a fixed 3:2:1 economic ratio, not a statistically estimated
relationship, so there's no "is this pair cointegrated" question to ask it the
same way). Uses the BACK-ADJUSTED curve (same choice as breakout/crossover, for
the same reason: a raw roll-date price jump in either leg could spuriously
register as a structural break in the spread that isn't a real economic event).

This is a DIAGNOSTIC only, run once up front — it does not gate which of the 7
pairs get built into a signal (already decided on economic/structural grounds,
WORKFLOW.md §11d) and does not gate the signal's trading logic date-by-date
(see `signals.relative_value`'s own docstring for why).

Run: `python research/relative_value_cointegration_check.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted
from signals.cointegration import rolling_cointegration_report, DEFAULT_WINDOW, DEFAULT_STEP, DEFAULT_ALPHA
from signals.relative_value import RATIO_PAIRS, HEDGE_RATIO_PAIRS

EG_PAIRS = {**RATIO_PAIRS, **HEDGE_RATIO_PAIRS}

ORIGINALLY_CITED = {"wti_brent": "Brent/WTI", "gold_silver": "Gold/Silver", "corn_wheat": "Corn/Wheat"}


def main():
    adj = load_continuous_backadjusted()
    close = adj["close"]
    log_price = np.log(close)

    print(f"Rolling Engle-Granger, window={DEFAULT_WINDOW}d, step={DEFAULT_STEP}d, alpha={DEFAULT_ALPHA}")
    print(f"Pairs tested: {list(EG_PAIRS.keys())}\n")

    report = rolling_cointegration_report(EG_PAIRS, log_price, window=DEFAULT_WINDOW, step=DEFAULT_STEP, alpha=DEFAULT_ALPHA)
    print(report.round(3).to_string())

    print("\n--- Originally-cited pairs (Time_Series_Models.ipynb's claimed result, unverified/never existed) ---")
    for name, label in ORIGINALLY_CITED.items():
        if name in report.index:
            row = report.loc[name]
            print(f"  {label:14s} fraction_cointegrated={row['fraction_cointegrated']:.3f} (n_windows={int(row['n_windows'])})")
    print("  Old claimed figure: Brent/WTI 45% of windows cointegrated. Real number reported above — not assumed to match.")

    return report


if __name__ == "__main__":
    main()
