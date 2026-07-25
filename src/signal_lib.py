"""
signal_lib.py — Backward-compatibility shim, superseded 2026-07-21.

This module used to hold all shared signal/backtest logic in one flat file. It has
been split into data/, signals/, and backtest/ (see cleanup.md section 2 for the
target shape and WORKFLOW.md's 2026-07-21 change log for why now) - new work should
import from those directly, not from here.

Kept only because Time_Series_Models.ipynb (cointegration, untouched by this split)
still imports `load_price_data` and `get_liquid_universe` from this name. Retire this
file once that notebook migrates to the new structure too (Phase 2d).
"""

from data.panels import load_core_panel as load_price_data
from data.universe import get_liquid_universe
from signals.momentum import build_momentum_features
from signals.transforms import binary_signal, continuous_signal, rank_signal
from backtest.engine import weekly_positions, monthly_positions, backtest_signal
from backtest.performance import performance_stats

__all__ = [
    "load_price_data",
    "get_liquid_universe",
    "build_momentum_features",
    "binary_signal",
    "continuous_signal",
    "rank_signal",
    "weekly_positions",
    "monthly_positions",
    "backtest_signal",
    "performance_stats",
]
