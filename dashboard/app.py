"""
CTA Research Dashboard — entry point (public build).

Launch with: streamlit run dashboard/app.py

Navigation-router pattern: this file only defines the sidebar structure via
st.navigation()/st.Page() and renders nothing itself. Every page computes
live at render time — see each page's own module docstring.

Six pages are intentionally not registered in this public build's
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
not as the first thing a visitor sees.

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
momentum_performance = st.Page(
    PAGES_DIR / "07_momentum_performance.py",
    title="Momentum",
    icon=":material/trending_up:",
)
breakout_performance = st.Page(
    PAGES_DIR / "08_breakout_performance.py",
    title="Breakout",
    icon=":material/bolt:",
)
crossover_performance = st.Page(
    PAGES_DIR / "09_crossover_performance.py",
    title="Crossover",
    icon=":material/candlestick_chart:",
)
short_term_reversal_performance = st.Page(
    PAGES_DIR / "10_short_term_reversal_performance.py",
    title="Short-Term Reversal",
    icon=":material/swap_horiz:",
)
carry_performance = st.Page(
    PAGES_DIR / "11_carry_performance.py",
    title="Carry",
    icon=":material/percent:",
)
xs_momentum_performance = st.Page(
    PAGES_DIR / "12_xs_momentum_performance.py",
    title="Cross-Sectional Momentum",
    icon=":material/leaderboard:",
)
value_performance = st.Page(
    PAGES_DIR / "16_value_performance.py",
    title="Value",
    icon=":material/sell:",
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

nav = st.navigation({
    "Overview": [overview],
    "Strategy Performance": [
        momentum_performance, breakout_performance, crossover_performance,
        short_term_reversal_performance, carry_performance, xs_momentum_performance,
        value_performance,
    ],
    "Portfolio Construction": [portfolio_performance],
    "Macro Data": [macro_explorer],
    "Technical Appendix": [volatility_estimators, portfolio_optimizer_health],
})

nav.run()
