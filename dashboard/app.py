"""
CTA Research Dashboard — entry point (public build).

Launch with: streamlit run dashboard/app.py

Navigation-router pattern: this file only defines the sidebar structure via
st.navigation()/st.Page() and renders nothing itself. Every page computes
live at render time — see each page's own module docstring.

Fourteen pages are intentionally not registered in this public build's
navigation (files still exist, just unrouted here — see them on master):
Term Structure and Continuous Curve display actual Databento-sourced price/
curve levels directly, a licensing distinction (everything else here is a
backtest/statistical result derived from that data, not the raw series
itself); Pipeline Health, OHLCV Coverage, Volatility, and Macro are all
"is the pipeline healthy right now" status/monitoring pages — useful for
the maintainer, not evidence of research quality for a visitor, and a
stale/missing-data status would read as "something's broken" with no
context to a stranger. Volatility Estimators and Optimizer Health are kept,
but live in the "Technical Appendix" group at the end of the nav rather
than up front — both validate machinery/methodology (does the vol estimator
forecast well, is the optimizer behaving sanely) rather than reporting a
result, so they're placed after the actual research (Strategy Performance,
Portfolio Construction) for whoever wants to verify the rigor behind it,
not as the first thing a visitor sees. The entire "Strategy Performance"
group (Momentum plus the six weaker families that were briefly folded into
an "Other Signal Families" summary page) is removed from this build's nav —
Single Strategy Portfolios (Trend Book, Carry Book) supersedes it as the
public-facing story: those two pages show the actual decided construction
and vol-targeting choice for each, not the individual per-signal research
that fed into deciding them. That underlying research (07_momentum_
performance.py, 21_other_signal_families.py, and the rest) still exists and
stays routed on master.

Overview (page 17) is the default landing page — a static, no-computation
project narrative, since a visitor's first click shouldn't land on a live
signal/portfolio computation.
"""
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CTA Data QA",
    layout="wide",
    initial_sidebar_state="expanded",
)

import _bootstrap_data  # noqa: E402,F401 -- populates Data/ before any page reads it

PAGES_DIR = Path(__file__).parent / "pages"

overview = st.Page(
    PAGES_DIR / "17_overview.py",
    title="Overview",
    icon=":material/rocket_launch:",
    default=True,
)
volatility_estimators = st.Page(
    PAGES_DIR / "06_volatility_estimators.py",
    title="Volatility Estimators",
    icon=":material/query_stats:",
)
portfolio_performance = st.Page(
    PAGES_DIR / "13_portfolio_performance.py",
    title="Portfolio Construction",
    icon=":material/account_tree:",
)
portfolio_optimizer_health = st.Page(
    PAGES_DIR / "14_portfolio_optimizer_health.py",
    title="Optimizer Health",
    icon=":material/insights:",
)
macro_explorer = st.Page(
    PAGES_DIR / "15_macro_explorer.py",
    title="Macro Explorer",
    icon=":material/query_stats:",
)
trend_book_performance = st.Page(
    PAGES_DIR / "18_trend_book_performance.py",
    title="Trend Book",
    icon=":material/show_chart:",
)
carry_book_performance = st.Page(
    PAGES_DIR / "19_carry_book_performance.py",
    title="Carry Book",
    icon=":material/percent:",
)
multi_strategy_portfolio = st.Page(
    PAGES_DIR / "20_multi_strategy_portfolio.py",
    title="Trend + Carry",
    icon=":material/account_tree:",
)

nav = st.navigation({
    "Overview": [overview],
    "Portfolio Construction": [portfolio_performance],
    "Single Strategy Portfolios": [trend_book_performance, carry_book_performance],
    "Multi-Strategy Portfolios": [multi_strategy_portfolio],
    "Macro Data": [macro_explorer],
    "Technical Appendix": [volatility_estimators, portfolio_optimizer_health],
})

nav.run()
