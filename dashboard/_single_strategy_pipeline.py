"""
dashboard/_single_strategy_pipeline.py — Shared, cached entry point for the
two decided Single Strategy Portfolios (Trend = tsmom_alone, Carry =
carry_timing_zero, both GJR-GARCH vol-targeted — WORKFLOW.md decisions #10/
#11) plus their combined Allocator.

One shared `st.cache_resource` entry, not one per page — `dashboard/
_portfolio_pipeline.py`'s own docstring documents a real Streamlit Cloud
crash (confirmed from platform logs) caused by two pages each independently
caching the same expensive pipeline: visiting both close together ran it
TWICE simultaneously and OOM'd the container. Same fix here, applied before
it has a chance to repeat: Trend Book, Carry Book, and Multi-Strategy pages
all import and call THIS one function.

Does NOT re-run the 7-flavor Trend / 4-flavor Carry bake-off — that's already
decided (`research/single_strategy_portfolios.py`, run once, logged in
WORKFLOW.md). This only ever constructs and runs the two WINNING alpha
constructions, so it's a small fraction of that script's own full cost.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "pages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
from portfolio.allocator import Allocator

TREND_FLAVOR = "tsmom_alone"
CARRY_FLAVOR = "carry_timing_zero"


@st.cache_resource(ttl=1200)
def load_and_run():
    """Returns (returns, trend_result, carry_result, trend_book, carry_book, combined_pnl)."""
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = ssp.build_vol(raw)
    carry_panel, _ = ssp.build_carry_panel(included)

    trend_flavors = ssp.build_trend_flavors(close, vol, returns)
    trend_book = ssp.build_book("trend_" + TREND_FLAVOR, trend_flavors[TREND_FLAVOR], returns, vol_estimator="garch")
    trend_result = trend_book.run(returns)

    carry_flavors = ssp.build_all_carry_signals(carry_panel, sectors)
    carry_book = ssp.build_book("carry_" + CARRY_FLAVOR, carry_flavors[CARRY_FLAVOR], returns, vol_estimator="garch")
    carry_result = carry_book.run(returns)

    allocator = Allocator([trend_book, carry_book])
    combined = allocator.run(returns)

    return returns, trend_result, carry_result, trend_book, carry_book, combined["pnl"]
