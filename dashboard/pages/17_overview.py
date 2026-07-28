"""Page 17 — Overview (public-build landing page).

Static content only, no data reads and no live computation — this is the
first page every visitor hits, so it has to load instantly and can never be
the thing that crashes. The narrative/numbers here are pulled directly from
README.md (the project's own "sell the work honestly" document) rather than
re-derived, so the two stay in sync by construction.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import page_header

page_header(
    "Systematic Commodity & Macro Futures Research Platform",
    "A 42-market systematic futures research platform — signal research, portfolio construction, and honest results.",
)

st.markdown(
    "A systematic futures research platform spanning **42 markets** across commodities, "
    "FX, rates, and equity indices — built on a self-collected, two-source daily data "
    "pipeline, **seven independently-researched signal families**, and a covariance-aware "
    "portfolio construction layer with rigorous overfitting controls.\n\n"
    "Results on this dashboard are reported honestly, including where signals are weak "
    "or mixed — the platform is built to make that distinction impossible to fudge, not "
    "to hide it. Every result below is out-of-sample: strict train / validation / test "
    "splits, no signal-spec selection based on its own performance, every signal "
    "shifted one day before it's allowed to trade."
)

st.divider()

cols = st.columns(4)
stats = [
    ("42", "markets covered"),
    ("7", "signal families"),
    ("233+", "tests"),
    ("18yr", "backtest history"),
]
for col, (value, label) in zip(cols, stats):
    with col:
        st.metric(label, value)

st.divider()

st.subheader("Signal Research")
st.markdown(
    "Seven signal families, each implemented as pure functions with no optimizer "
    "dependency, matched directly against their source papers rather than built from "
    "memory. Results are gross Sharpe, train / validation / test:"
)
st.markdown(
    "| Family | Source | Train | Validation | Test |\n"
    "|---|---|---:|---:|---:|\n"
    "| Time-series momentum | Moskowitz-Ooi-Pedersen (2012) | 0.24 | 0.48 | 0.40 |\n"
    "| Crossover (50/200 golden cross) | Classic trend-following | 0.14 | 0.37 | 0.27 |\n"
    "| Carry (cross-sectional, 1-12mo) | Koijen-Moskowitz-Pedersen-Vrugt (2018) | 0.33 | -0.36 | -0.06 |\n"
    "| Cross-sectional momentum | Asness-Moskowitz-Pedersen (2013) | -0.34 | -1.39 | 0.04 |\n"
    "| Value | Asness-Moskowitz-Pedersen (2013) | -0.01 | 0.13 | -0.69 |\n"
    "| Combined portfolio (6-Book pilot) | Ledoit-Wolf + turnover-penalized MVO | 0.46 | -1.21 | 0.09 |\n"
)
st.caption(
    "Momentum is the one consistently positive family; the rest are genuinely mixed. "
    "That's reported as found — the platform's value is in the discipline that produced "
    "these numbers, not in engineering a cleaner-looking table. Full detail for every "
    "family, including breakout and short-term reversal, is under **Strategy Performance** "
    "in the sidebar."
)

st.divider()

st.subheader("Portfolio Construction")
st.markdown(
    "- **`Book` / `Allocator` architecture** — each signal family owns its own alpha, "
    "covariance estimate, and vol target end-to-end; the Allocator combines Books and "
    "applies any regime conditioning *before* the optimizer runs.\n"
    "- **Ledoit-Wolf shrinkage covariance**, rolling-estimated with a minimum-coverage "
    "gate.\n"
    "- **Turnover-penalized mean-variance optimization** with position-size caps and a "
    "dollar-neutral constraint.\n"
    "- **Historical VaR / Expected Shortfall** at the combined-portfolio level.\n"
    "- Per-Book hyperparameters were tuned across all 20 Books, then independently "
    "cross-checked with Combinatorially Symmetric / Purged Cross-Validation (Bailey, "
    "Borwein, López de Prado & Zhu, 2017) — both methods agree the flat default "
    "calibration holds up better than anything tuned, a result that held under two "
    "independent statistical tests rather than being taken on faith from the first one."
)
st.caption("Full detail under **Portfolio Construction** in the sidebar.")

st.divider()

st.subheader("Data Infrastructure")
st.markdown(
    "- Daily OHLCV across 41 of 42 markets via a scheduled `yfinance` pipeline.\n"
    "- Daily per-contract term-structure data via Databento (CME Globex and ICE Futures "
    "US raw daily archives) for 38 core assets, decoded from raw exchange feeds rather "
    "than a vendor's pre-cleaned panel.\n"
    "- A fully vectorized `polars` transform pipeline rebuilding all 6 term-structure "
    "tables for the full universe end-to-end in under 15 minutes.\n"
    "- A continuous futures curve built with volume-crossover roll detection and ratio "
    "(not additive) back-adjustment."
)
st.caption("Full detail under **Coverage** in the sidebar, and in `DATA_SCHEMA.md` on GitHub.")

st.divider()

st.markdown(
    "**[Full write-up on GitHub →](https://github.com/Pkargados/cta-research-platform)** "
    "— architecture, references, and testing methodology."
)
