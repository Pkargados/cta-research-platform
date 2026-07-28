"""
dashboard/_portfolio_pipeline.py — Shared, cached entry point for the 6-Book
portfolio pipeline (data load, alpha construction, per-Book rolling
Ledoit-Wolf covariance + optimizer solve, Allocator combine).

Previously Portfolio Construction (13) and Optimizer Health (14) each had
their own independently `@st.cache_data`-wrapped copy of this same
computation — two separate cache keys meant visiting both pages close
together ran and held the full pipeline in memory TWICE simultaneously.
Confirmed as the actual crash cause from Streamlit Cloud's own logs: the
pipeline's "Universe: 36 of 42 assets" diagnostic print fired twice within
~20 seconds, immediately before a connection-reset/OOM-style kill. One
shared cache entry instead.

`st.cache_resource`, not `st.cache_data`: nothing downstream mutates the
returned Book objects or DataFrames (both pages only read from them), so
reference-sharing is correct here and avoids the deep-copy/pickle overhead
`st.cache_data` would otherwise pay on every cache write — relevant given
the crash was a memory problem, not a correctness one.
"""
import importlib.util
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_spec = importlib.util.spec_from_file_location(
    "research_portfolio_driver", Path(__file__).resolve().parent.parent / "research" / "portfolio.py",
)
pf_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf_research)

from portfolio.allocator import Allocator


@st.cache_resource(ttl=1200)
def load_and_run():
    """Returns (returns, book_results, books_by_name, combined_pnl)."""
    adj, raw, included, sectors = pf_research.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = pf_research.build_vol(raw)
    alphas = pf_research.build_six_alphas(adj, raw, included, sectors, vol)

    book_results = {}
    books_by_name = {}
    books = []
    for name, alpha_df in alphas.items():
        book = pf_research.build_book(name, alpha_df, returns)
        result = book.run(returns)
        book_results[name] = result
        books_by_name[name] = book
        books.append(book)

    allocator = Allocator(books)
    combined = allocator.run(returns)
    combined_pnl = combined["pnl"]
    return returns, book_results, books_by_name, combined_pnl
