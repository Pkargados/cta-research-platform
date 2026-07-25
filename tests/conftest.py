"""
tests/conftest.py — makes `src/` importable as top-level packages (data,
signals, backtest, portfolio, regime) for every test module, the same way
every research/*.py driver script adds src/ to sys.path manually.

RECONSTRUCTED 2026-07-23 — the original conftest.py was lost (never read in
full this session, only inferred to exist from context); this is a standard,
minimal recreation of what a conftest.py doing this job looks like, not a
recovery of the original file's exact content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
