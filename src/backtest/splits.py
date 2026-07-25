"""
backtest/splits.py — Train / validation / test split.

Decided 2026-07-21 (see WORKFLOW.md Phase 1 change log), replacing the original
two-way train/held-out split (CLAUDE.md Rule 1, TRAIN_END = 2019-12-31) that
feature_engineering.ipynb used for momentum. Rationale for the three-way version:

- Momentum was the only signal family so far, with no real hyperparameters to tune -
  a two-way split was enough. Breakout (window length), crossover (fast/slow MA
  pairs), and cointegration (z-score entry/exit thresholds) all have genuine
  parameters, and picking them by looking at the same data used for the final
  reported performance is exactly the data-snooping failure mode CLAUDE.md Rule 2
  already exists to prevent for statistical tests - the same discipline applies here.
- 2020-2021 (COVID) is deliberately placed in validation, not train and not test.
  A fast, violent, V-shaped crash like COVID doesn't discriminate much between "the
  strategy saw it during fitting" and "it didn't" - the whipsaw speed is what hurts
  trend-following either way - so its value as a pristine, untouched test period is
  low. Putting it in validation instead lets spec selection be informed by a real
  stress regime, while the untouched test period (2022+) still contains its own
  genuine stress event (the 2022 inflation/rate-hike/commodity shock), so "test isn't
  just quiet markets" is preserved with a different anchor.
- Train stays anchored at the original TRAIN_END (2019-12-31) rather than being
  redefined, so this is a genuine three-way split of the existing methodology, not a
  new one - validation and test are carved out of what used to be the single
  held-out block.

TEST is discipline-critical: touched exactly once, at the very end of a research
pass, never used to pick a specification. VALIDATION may be looked at repeatedly
during development.
"""

import pandas as pd

TRAIN_END = "2019-12-31"
VALIDATION_END = "2021-12-31"


def train_validation_test_split(df: pd.DataFrame, train_end=TRAIN_END, validation_end=VALIDATION_END):
    """Split a (T x N) DataFrame (or Series) into three contiguous, non-overlapping
    slices by calendar time. Boundary dates are excluded from the following slice.

    Returns (train, validation, test).
    """
    train = df.loc[:train_end]
    validation = df.loc[train_end:validation_end].iloc[1:]
    test = df.loc[validation_end:].iloc[1:]
    return train, validation, test
