"""Page 21 — Other Signal Families (summary).

Static content, not live-computed (same convention as the Overview page,
17_overview.py) — every number here is already established, reproduced
directly from each family's own research driver
(research/{breakout,crossover,short_term_reversal,carry,xs_momentum,value}.py),
not re-derived or rounded differently on this page.

Why this page exists: momentum is the one signal family with a consistently
positive train/validation/test result (see the Momentum page) and is the
only one of the seven that's part of the actual Trend Book in the Single
Strategy Portfolios section. The other six each got a full dedicated page
with the same chart-heavy treatment regardless of whether the result
justified it - diluting the one strong result under six weak-to-mixed ones
rather than clarifying anything. Nothing here is hidden or omitted: every
number and finding below is the same one already reported on that family's
own page (still reachable in the source tree, just not routed in the public
nav) and logged in WORKFLOW.md/CLAUDE.md - this is a presentation change,
not a results change.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import page_header, render_key_takeaways

page_header("Other Signal Families", "Six families tested, reported honestly, and not part of the current Trend/Carry mandate")

render_key_takeaways([
    "Momentum (TSMOM) is the one consistently positive family across train/validation/"
    "test, and the only one of these seven actually trading in the Single Strategy "
    "Portfolios section (Trend Book) — see the Momentum page for its full detail.",
    "The six below are reported here exactly as found — weak, negative, or genuinely "
    "mixed — not filtered out. Full per-asset detail, robustness grids, and net-of-cost "
    "breakdowns for each still exist in `research/*.py` and are reproducible directly.",
    "Carry's own cross-sectional/timing variants below are a different question from "
    "the Carry Book on the Multi-Strategy Portfolios page: that Book uses "
    "`carry_timing_zero`, selected by a validation-based bake-off among these same "
    "specs, and is part of the actual current mandate — its own page has the full detail.",
])

st.divider()

st.subheader("Headline spec, gross Sharpe (train / validation / test)")

rows = [
    {
        "Family": "Breakout", "Headline spec": "Turtle System 1 (20d entry / 10d exit), daily resizing",
        "Train": 0.088, "Validation": 0.069, "Test": 0.101,
        "Net (train/val/test)": "-0.35 / -0.29 / -0.31",
        "Why it didn't clear the bar": "Turnover ~50-60x annualized swamps any gross edge net-of-cost; the driver is pooling many independently-triggered regimes under one daily book, not the signal logic itself.",
    },
    {
        "Family": "Crossover", "Headline spec": "50/200 SMA (golden cross) — the most consistent of 3 pairs tested",
        "Train": 0.14, "Validation": 0.37, "Test": 0.27,
        "Net (train/val/test)": "0.07 / 0.30 / 0.20",
        "Why it didn't clear the bar": "Not weak on its own, but 0.65 full-sample correlated with TSMOM (DCC-GARCH) — a bake-off found it adds no diversification to the Trend Book, only dilutes the stronger signal.",
    },
    {
        "Family": "Short-Term Reversal", "Headline spec": "Individual-asset, 5-day lag (of 6 specs tested)",
        "Train": 0.067, "Validation": 0.418, "Test": -0.686,
        "Net (train/val/test)": "-1.69 / -0.81 / -2.09",
        "Why it didn't clear the bar": "The only outright-unprofitable family net-of-cost across every one of its 6 specs — turnover 160-360x annualized. VIX-conditioning (Nagel 2011) is statistically real at the sector level (HAC p=0.09) but economically too small to rescue it.",
    },
    {
        "Family": "Carry (standalone)", "Headline spec": "Cross-sectional, 1-12mo (of 4 specs tested)",
        "Train": 0.33, "Validation": -0.36, "Test": -0.06,
        "Net (train/val/test)": "≈ gross (turnover collapsed to ~1-3x after matching the paper's monthly rebalance exactly)",
        "Why it didn't clear the bar": "Genuinely mixed, not a clean win or loss — carry-timing variants underperform in validation (spans COVID, consistent with the paper's own documented crisis underperformance) but turn positive in test.",
    },
    {
        "Family": "Cross-Sectional Momentum", "Headline spec": "MOM2-12 (12-month, skip-1-month)",
        "Train": -0.34, "Validation": -1.39, "Test": 0.04,
        "Net (train/val/test)": "-0.39 / -1.42 / -0.01",
        "Why it didn't clear the bar": "Validation spans the 2020 COVID crash — sharply negative, consistent with the well-documented \"momentum crash\" phenomenon around violent V-shaped reversals.",
    },
    {
        "Family": "Value", "Headline spec": "Negative-5yr-return, asset-class-adjusted",
        "Train": -0.01, "Validation": 0.13, "Test": -0.69,
        "Net (train/val/test)": "-0.03 / 0.09 / -0.71",
        "Why it didn't clear the bar": "Weak-to-negative and inconsistent in sign across periods; per-asset results are genuinely mixed rather than one construction flaw explaining the whole result.",
    },
]

df = pd.DataFrame(rows).set_index("Family")
st.dataframe(df, use_container_width=True)

st.caption(
    "Every number above is reproduced from that family's own `research/*.py` driver "
    "(unchanged), not recomputed on this page. Full detail — per-asset breakdowns, "
    "robustness grids, VIX-conditioning regressions — remains in the source tree even "
    "though these six no longer have their own routed dashboard page."
)
