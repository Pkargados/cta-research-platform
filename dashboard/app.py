"""
CTA Research Dashboard — entry point (public build).

Launch with: streamlit run dashboard/app.py

Navigation-router pattern: this file only defines the sidebar structure via
st.navigation()/st.Page() and renders nothing itself. Pipeline Health reads
pre-computed artifacts from Data/dashboard_summary/ — produced by
jobs/update_dashboard_summary.py. Every other page computes live at render
time instead — see each page's own module docstring.

Two pages that display actual Databento-sourced price/curve levels directly
(Term Structure, Continuous Curve) are intentionally not registered in this
public build's navigation — everything shown here is a backtest/statistical
result derived from that data, not the underlying price series itself.

Overview (page 17) is the default landing page — a static, no-computation
project narrative, since a visitor's first click shouldn't land on an
internal QA page (Pipeline Health) or wait on a live signal/portfolio
computation.
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
pipeline_health = st.Page(
    PAGES_DIR / "00_pipeline_health.py",
    title="Pipeline Health",
    icon=":material/monitor_heart:",
)
ohlcv_coverage = st.Page(
    PAGES_DIR / "02_ohlcv_coverage.py",
    title="OHLCV Coverage",
    icon=":material/candlestick_chart:",
)
volatility = st.Page(
    PAGES_DIR / "03_volatility.py",
    title="Volatility",
    icon=":material/ssid_chart:",
)
macro = st.Page(
    PAGES_DIR / "04_macro.py",
    title="Macro",
    icon=":material/public:",
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
    "Monitoring": [pipeline_health],
    "Coverage": [ohlcv_coverage, volatility, macro, volatility_estimators],
    "Strategy Performance": [
        momentum_performance, breakout_performance, crossover_performance,
        short_term_reversal_performance, carry_performance, xs_momentum_performance,
        value_performance,
    ],
    "Portfolio Construction": [portfolio_performance, portfolio_optimizer_health],
    "Macro Data": [macro_explorer],
})

nav.run()
