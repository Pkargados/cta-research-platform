"""
data/universe.py — Asset universe filtering.

Moved from signal_lib.py (2026-07-21) — see cleanup.md section 2.
"""


def get_liquid_universe(volume, window_start="2024-07-14", threshold=1000):
    """ADV liquidity floor (CLAUDE.md Rule 1) - assets with < `threshold` contracts/day
    average volume over the trailing window are excluded. Threshold decided from
    liquidity alone, independent of any signal's performance."""
    adv = volume.loc[window_start:].mean()
    included = adv[adv >= threshold].index.tolist()
    excluded = adv[adv < threshold].index.tolist()
    return included, excluded
