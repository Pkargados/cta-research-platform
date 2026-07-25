"""
backtest/cpcv.py — Combinatorially Symmetric Cross-Validation (CSCV) and the
purged/embargoed refinement (CPCV), for measuring whether a hyperparameter
selection procedure is picking up real signal or overfitting to one fixed
historical window.

Built 2026-07-23, per direct instruction, as the first step of WORKFLOW.md
Phase 7's "beyond naive Bonferroni/FDR" plan — the diagnosed root problem
with `research/tune_all_books.py`'s per-Book hyperparameter search is that
its selection evidence comes from ONE fixed 2020-2021 validation block; no
multiple-testing correction turns one historical regime into many
independent looks. This module generates many train/test recombinations
across the FULL history instead.

**Sources read directly before implementing (CLAUDE.md discipline — not
implemented from memory or from a blog summary):**

- **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017),
  "The Probability of Backtest Overfitting," Journal of Computational
  Finance 20(4), 39-69** (`https://www.davidhbailey.com/dhbpapers/
  backtest-prob.pdf`, fetched and read directly) — Algorithm 2.3 (CSCV) is
  reproduced here essentially verbatim: partition a (T x N) matrix M of N
  trials' performance series into an even number S of contiguous row-blocks,
  form all C(S, S/2) ways to select S/2 blocks as the training set (the
  complement is the test set), and for each combination determine the
  IS-optimal trial and its relative OOS rank. The paper's own logit
  transform (`lambda_c = ln(omega_c / (1 - omega_c))`, `omega_c` = OOS
  relative rank of the IS-optimal trial) and PBO definition (`PBO =
  P[lambda_c <= 0]`, i.e. the fraction of combinations where the IS-optimal
  choice performs at or below the OOS median) are implemented exactly as
  Definition 2.2 / Section 3.1 state them, not approximated.
- **skfolio's `CombinatorialPurgedCV` documentation** (fetched directly,
  itself a precise, code-level restatement of López de Prado's *Advances in
  Financial Machine Learning* purging/embargo definitions, which are not
  freely available to fetch directly — the book itself was not accessible
  this session, flagged honestly rather than implemented from a second-hand
  paraphrase without checking) — `purged_size`: drop observations from the
  END of a training block that immediately PRECEDES a test block, AND from
  the START of a training block that immediately FOLLOWS one (bidirectional,
  symmetric each side). `embargo_size`: an ADDITIONAL, one-directional drop
  from the START of a training block that immediately follows a test block
  only (layered on top of purging on that side specifically, not the
  pre-test side).

**Where this deliberately reinterprets, not blindly copies, the ML-CV
motivation for purging — logged so this doesn't get "corrected" back to a
misapplied literal reading later.** Classical purged CV exists because a
FITTED estimator's training labels can overlap a test observation's own
outcome window, leaking future information into the fit. Nothing in this
project's `Book`/optimizer pipeline is "fit" per fold in that sense — a
given (target_vol, max_weight, frequency) configuration produces the SAME
full-history weight path and PnL series regardless of which CSCV combination
is being scored (`research/tune_all_books.py`'s `daily_mark_pnl` output is
computed ONCE per grid point, not re-solved per fold). What purging/embargo
actually guards against here is different but real: `Book` PnL is
mechanically autocorrelated across adjacent days (`kappa` position inertia
carries `x_prev` forward, the EWMA vol tracker carries state, a rebalance
weight is held for up to a week) — a training-set day immediately adjacent
to a test-set day is not independent evidence of that configuration's true
skill, it's largely the same regime bleeding across the boundary. Purging
and embargo remove exactly that adjacency, for that reason, not because any
model here is refit on IS data.

**Group count / purge / embargo defaults**: `n_groups=8` (each block spans
roughly 2 of this project's ~18 years of history — few enough that group
count doesn't get lost in noise, many enough that C(8,4)=70 combinations
recombine across genuinely different multi-year stretches, not just
month-to-month noise) with `n_train_groups=4` (the original paper's own
symmetric S/2 design). `purge_periods=5, embargo_periods=5` (5 trading days
= one calendar week either side of a boundary) — sized to this project's
current weekly Book rebalancing cadence (WORKFLOW.md Phase 7), a labeled,
not-tuned default matching the same "size a constant to the cadence, don't
backtest it" discipline already used for `EWMA_HALFLIFE` elsewhere in this
project. Not run against daily rebalancing (excluded from the tuning grid
itself for compute reasons — see `tune_all_books.py`'s own docstring).
"""

import itertools

import numpy as np
import pandas as pd

from backtest.performance import simple_sharpe


def split_into_groups(dates, n_groups: int) -> list:
    """Partition a sorted, deduplicated date index into `n_groups` contiguous,
    near-equal-size, non-overlapping blocks — Algorithm 2.3 step 2 ("we
    partition M across rows, into an even number S of disjoint submatrices of
    equal dimensions")."""
    unique_sorted = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    return [pd.DatetimeIndex(block) for block in np.array_split(unique_sorted, n_groups)]


def generate_combinations(n_groups: int, n_train_groups: int) -> list:
    """All C(n_groups, n_train_groups) ways to choose which blocks form the
    TRAINING (IS) set for one combination — the complement is that
    combination's TEST (OOS) set. With `n_train_groups = n_groups // 2` this
    reproduces the paper's own symmetric CSCV design exactly (Algorithm 2.3
    step 3); a general `n_train_groups` also supports skfolio's more flexible
    `n_folds`/`n_test_folds` parameterization if a non-symmetric split is ever
    wanted."""
    return list(itertools.combinations(range(n_groups), n_train_groups))


def purge_and_embargo(groups: list, train_group_idxs: tuple, purge_periods: int, embargo_periods: int):
    """Returns (train_dates, test_dates) for one CSCV combination.

    `test_dates` is the untouched union of every group NOT in
    `train_group_idxs` — OOS is never trimmed, only IS (training) is.
    `train_dates` drops `purge_periods` observations from the tail of a
    training block immediately BEFORE a test block, and
    `purge_periods + embargo_periods` observations from the head of a
    training block immediately AFTER a test block — see this module's own
    docstring for the exact skfolio-sourced definition being reproduced.
    """
    n = len(groups)
    train_set = set(train_group_idxs)
    test_dates = pd.DatetimeIndex(sorted(set().union(
        *[set(groups[i]) for i in range(n) if i not in train_set]
    ))) if len(train_set) < n else pd.DatetimeIndex([])

    train_blocks = []
    for gi in sorted(train_set):
        block = pd.DatetimeIndex(sorted(groups[gi]))
        preceded_by_test = (gi - 1) >= 0 and (gi - 1) not in train_set
        followed_by_test = (gi + 1) < n and (gi + 1) not in train_set

        head_drop = (purge_periods + embargo_periods) if preceded_by_test else 0
        tail_drop = purge_periods if followed_by_test else 0
        keep = block[head_drop: len(block) - tail_drop] if len(block) > head_drop + tail_drop else pd.DatetimeIndex([])
        train_blocks.append(keep)

    train_dates = pd.DatetimeIndex(sorted(set().union(*[set(b) for b in train_blocks]))) if train_blocks else pd.DatetimeIndex([])
    return train_dates, test_dates


def cscv_pbo(
    pnl_matrix: pd.DataFrame,
    n_groups: int = 8,
    n_train_groups: int = None,
    purge_periods: int = 5,
    embargo_periods: int = 5,
    stat_fn=simple_sharpe,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Run the CSCV procedure (Algorithm 2.3, Bailey/Borwein/López de
    Prado/Zhu 2017) on an already-computed (T x N) matrix of N
    configurations' full-history daily PnL series, with purging/embargo
    applied to the training side of each combination.

    `pnl_matrix` : one column per grid-point/configuration, same convention
    as the paper's own M ("each column n represents a vector of profits and
    losses... a particular model configuration"). NOT re-solved per fold —
    see this module's own docstring for why that's a legitimate
    simplification for this project's Book pipeline specifically.

    `n_train_groups` defaults to `n_groups // 2` (the paper's own symmetric
    design).

    Returns one row per combination with columns: `train_group_idxs`,
    `is_winner` (the column name with the best `stat_fn` on the PURGED
    training dates), `n_train_obs`, `n_test_obs`, `oos_relative_rank`
    (`omega_c` in the paper, in (0, 1)), `logit` (`lambda_c = ln(omega_c /
    (1 - omega_c))`). A combination is skipped (omitted from the returned
    DataFrame) if purging leaves too few training observations, or too few
    test observations, to compute `stat_fn` for every column.

    PBO (`phi` in the paper) is `(result["logit"] <= 0).mean()` over the
    returned rows — the fraction of combinations where the IS-optimal
    configuration underperformed the OOS median. Compute it directly on the
    returned DataFrame; not returned as a single scalar here so the caller
    can also inspect the full distribution (Figure 2 in the paper plots this
    distribution, not just its integral).
    """
    if n_train_groups is None:
        n_train_groups = n_groups // 2

    groups = split_into_groups(pnl_matrix.index, n_groups)
    combos = generate_combinations(n_groups, n_train_groups)
    n_cols = len(pnl_matrix.columns)

    rows = []
    for train_group_idxs in combos:
        train_dates, test_dates = purge_and_embargo(groups, train_group_idxs, purge_periods, embargo_periods)

        train_pnl = pnl_matrix.reindex(train_dates)
        test_pnl = pnl_matrix.reindex(test_dates)

        is_stats = train_pnl.apply(stat_fn)
        if is_stats.isna().all():
            continue
        is_winner = is_stats.idxmax()

        oos_stats = test_pnl.apply(stat_fn)
        if oos_stats.isna().sum() > 0.5 * n_cols or pd.isna(oos_stats.get(is_winner, np.nan)):
            continue

        oos_ranks = oos_stats.rank(method="average")
        omega_c = float(oos_ranks[is_winner]) / (n_cols + 1)
        omega_c = min(max(omega_c, 1e-6), 1 - 1e-6)  # avoid +/-inf logit at the boundary
        logit = float(np.log(omega_c / (1 - omega_c)))

        rows.append({
            "train_group_idxs": train_group_idxs,
            "is_winner": is_winner,
            "n_train_obs": int(train_pnl.dropna(how="all").shape[0]),
            "n_test_obs": int(test_pnl.dropna(how="all").shape[0]),
            "oos_relative_rank": omega_c,
            "logit": logit,
        })

    return pd.DataFrame(rows)


def pbo(cscv_result: pd.DataFrame) -> float:
    """Probability of Backtest Overfitting — the fraction of CSCV
    combinations whose IS-optimal configuration ranked at or below the OOS
    median (Definition 2.2 / Section 3.1 of the paper: `phi =
    integral_{-inf}^{0} f(lambda) dlambda`, estimated here as the empirical
    fraction with `logit <= 0`, equivalent for a finite sample)."""
    if len(cscv_result) == 0:
        return np.nan
    return float((cscv_result["logit"] <= 0).mean())
