import numpy as np
import pandas as pd

from backtest.cpcv import (
    split_into_groups, generate_combinations, purge_and_embargo, cscv_pbo, pbo,
)


def _dates(n=800):
    return pd.date_range("2010-01-01", periods=n, freq="D")


def test_split_into_groups_partitions_all_dates_with_no_overlap():
    idx = _dates(800)
    groups = split_into_groups(idx, n_groups=8)

    assert len(groups) == 8
    recombined = sorted(set().union(*[set(g) for g in groups]))
    assert recombined == sorted(idx)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not set(groups[i]) & set(groups[j])


def test_generate_combinations_count_matches_binomial_coefficient():
    # C(8, 4) = 70, per Bailey/Borwein/Lopez de Prado/Zhu (2017) Eq. 2.3
    combos = generate_combinations(8, 4)
    assert len(combos) == 70
    assert all(len(c) == 4 for c in combos)
    assert len(set(combos)) == 70  # no duplicates


def test_purge_and_embargo_drops_boundary_dates_only():
    idx = _dates(800)
    groups = split_into_groups(idx, n_groups=8)  # 100 dates/group
    # Train = groups 0,1 ; Test = groups 2..7 (group 2 immediately follows train)
    train_dates, test_dates = purge_and_embargo(groups, (0, 1), purge_periods=5, embargo_periods=5)

    # Group 1 is followed by a test group -> loses purge_periods from its tail.
    assert len(train_dates) == len(groups[0]) + len(groups[1]) - 5
    # Test dates are never trimmed.
    assert len(test_dates) == sum(len(groups[i]) for i in range(2, 8))
    assert set(train_dates) & set(test_dates) == set()


def test_purge_and_embargo_drops_more_from_train_block_after_test():
    idx = _dates(800)
    groups = split_into_groups(idx, n_groups=8)  # 100 dates/group
    # Train = groups 2,3 ; Test = groups 0,1,4,5,6,7 -> group 2 is preceded by
    # test (group 1) and followed by train (group 3): head loses purge+embargo,
    # tail is untouched by group 3 being train too... but group 3 is followed by
    # test (group 4), so group 3's tail loses purge_periods only.
    train_dates, test_dates = purge_and_embargo(groups, (2, 3), purge_periods=5, embargo_periods=5)

    expected = len(groups[2]) - 10 + len(groups[3]) - 5
    assert len(train_dates) == expected


def test_cscv_pbo_flags_pure_noise_as_overfit_prone():
    # 20 configurations, each pure iid noise with no real edge -> whichever
    # looks "best" in-sample should be close to a coin flip out-of-sample,
    # so PBO should land near 0.5, not near 0 (which would mean genuine skill).
    rng = np.random.default_rng(0)
    idx = _dates(2000)
    pnl = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), 20)), index=idx,
                        columns=[f"cfg_{i}" for i in range(20)])

    result = cscv_pbo(pnl, n_groups=8, purge_periods=5, embargo_periods=5)

    assert len(result) > 0
    p = pbo(result)
    assert 0.3 < p < 0.7


def test_cscv_pbo_recognizes_a_genuinely_dominant_configuration():
    # One configuration has a real, large, consistent edge; the rest are pure
    # noise. That one should almost always be the IS winner AND rank well OOS,
    # so PBO should be low, not near 0.5.
    rng = np.random.default_rng(1)
    idx = _dates(2000)
    data = {f"cfg_{i}": rng.normal(0, 0.01, size=len(idx)) for i in range(19)}
    data["cfg_best"] = rng.normal(0.01, 0.01, size=len(idx))
    pnl = pd.DataFrame(data, index=idx)

    result = cscv_pbo(pnl, n_groups=8, purge_periods=5, embargo_periods=5)

    assert len(result) > 0
    assert pbo(result) < 0.2
    assert (result["is_winner"] == "cfg_best").mean() > 0.8


def test_pbo_empty_result_is_nan():
    assert np.isnan(pbo(pd.DataFrame()))
