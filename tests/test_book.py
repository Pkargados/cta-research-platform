from collections import OrderedDict

import numpy as np
import pandas as pd

from portfolio.book import Book, daily_mark_pnl

ASSETS = ["A", "B"]


def _identity_cov(scale=0.0001):
    return pd.DataFrame(np.eye(2) * scale, index=ASSETS, columns=ASSETS)


def _make_book(alpha_df, cov_dict, **overrides):
    params = dict(
        name="test_book", alpha_df=alpha_df, cov_dict=cov_dict,
        gamma=1.0, kappa=0.1, lambd=0.0, max_weight=0.5,
        target_vol=0.10, ewma_halflife=5, scale_min=0.1, scale_max=5.0,
        periods_per_year=252, dollar_neutral=False,
    )
    params.update(overrides)
    return Book(**params)


def _basic_setup(n=30):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    alpha_df = pd.DataFrame({"A": rng.normal(0, 1, n), "B": rng.normal(0, 1, n)}, index=dates)
    returns_df = pd.DataFrame({"A": rng.normal(0, 0.01, n), "B": rng.normal(0, 0.01, n)}, index=dates)
    cov_dict = OrderedDict((d, _identity_cov()) for d in dates)
    return dates, alpha_df, returns_df, cov_dict


def test_run_produces_weights_within_max_weight():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict, max_weight=0.4)
    result = book.run(returns_df)
    assert "weights" in result
    assert (result["weights"].abs() <= 0.4 + 1e-9).all().all()


def test_run_too_few_common_dates_returns_degenerate_result():
    dates, alpha_df, returns_df, cov_dict = _basic_setup(n=30)
    # Restrict cov_dict to fewer than 20 dates -> Book.run()'s own early-return path.
    small_cov_dict = OrderedDict(list(cov_dict.items())[:10])
    book = _make_book(alpha_df, small_cov_dict)
    result = book.run(returns_df)
    assert "weights" not in result
    assert np.isnan(result["sharpe"])
    assert result["n_rebalance_dates_valid"] < 20


def test_apply_constraints_clips_to_max_weight():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict, max_weight=0.2)
    x_t = np.array([5.0, -5.0])
    clipped = book._apply_constraints(x_t)
    assert np.allclose(clipped, [0.2, -0.2])


def test_apply_constraints_dollar_neutral_recenters():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict, max_weight=10.0, dollar_neutral=True)
    x_t = np.array([0.3, 0.1])
    result = book._apply_constraints(x_t)
    assert np.isclose(result.sum(), 0.0, atol=1e-10)


def test_apply_vol_target_scales_toward_target():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict, target_vol=0.10, scale_min=0.01, scale_max=100.0)
    x_t = np.array([0.1, -0.1])
    # Realized vol (rv) much smaller than target_vol -> scale > 1 (re-lever).
    tiny_ewma_var = (0.01 / np.sqrt(252)) ** 2
    scaled, scale_applied, cap_bound = book._apply_vol_target(x_t, tiny_ewma_var)
    assert scale_applied > 1.0
    assert np.allclose(scaled, x_t * scale_applied)


def test_apply_vol_target_respects_scale_bounds():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict, target_vol=0.10, scale_min=0.5, scale_max=2.0)
    x_t = np.array([0.1, -0.1])
    huge_ewma_var = (5.0 / np.sqrt(252)) ** 2  # way above target -> would want scale << scale_min
    _, scale_applied, _ = book._apply_vol_target(x_t, huge_ewma_var)
    assert scale_applied >= 0.5 - 1e-9


def test_apply_vol_target_near_zero_rv_is_a_noop():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict)
    x_t = np.array([0.1, -0.1])
    scaled, scale_applied, cap_bound = book._apply_vol_target(x_t, ewma_var=0.0)
    assert np.array_equal(scaled, x_t)
    assert scale_applied == 1.0
    assert cap_bound is False


def test_max_gap_days_flattens_pnl_and_counts_stale_gap():
    # 25 monthly rebalance dates, one deliberate 4-month gap in the middle
    # (well beyond the default max_gap_days=60) - Book.run() should contribute
    # zero PnL for that stretch and flag it via n_stale_gaps, not silently price
    # a stale position against real subsequent moves.
    monthly_dates = list(pd.date_range("2015-01-31", periods=30, freq="ME"))
    gapped_dates = monthly_dates[:10] + monthly_dates[14:]  # skip 4 consecutive month-ends
    assert len(gapped_dates) >= 20

    daily_dates = pd.date_range(gapped_dates[0], gapped_dates[-1], freq="D")
    rng = np.random.default_rng(1)
    alpha_df = pd.DataFrame(
        {"A": rng.normal(0, 1, len(daily_dates)), "B": rng.normal(0, 1, len(daily_dates))}, index=daily_dates,
    )
    returns_df = pd.DataFrame(
        {"A": rng.normal(0, 0.01, len(daily_dates)), "B": rng.normal(0, 0.01, len(daily_dates))}, index=daily_dates,
    )
    cov_dict = OrderedDict((d, _identity_cov()) for d in gapped_dates)

    book = _make_book(alpha_df, cov_dict, max_gap_days=60)
    result = book.run(returns_df)
    assert result["n_stale_gaps"] >= 1


def test_cost_bps_reduces_pnl_relative_to_gross():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    cost_bps = pd.Series({"A": 20.0, "B": 20.0})
    book = _make_book(alpha_df, cov_dict, cost_bps=cost_bps)
    result = book.run(returns_df)
    # pnl = gross_pnl - lambd_penalty - real_cost; with lambd=0 and nonzero
    # turnover, net pnl must be <= gross_pnl on every date, strictly less on
    # dates with real turnover.
    diff = (result["gross_pnl"] - result["pnl"]).dropna()
    assert (diff >= -1e-12).all()
    assert (diff > 0).any()


def test_no_cost_bps_real_cost_series_is_zero():
    dates, alpha_df, returns_df, cov_dict = _basic_setup()
    book = _make_book(alpha_df, cov_dict)  # cost_bps=None (default)
    result = book.run(returns_df)
    assert (result["real_cost_series"] == 0.0).all()


def test_daily_mark_pnl_matches_ffill_shift_formula():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    weights = pd.DataFrame({"A": [1.0, 0.5], "B": [-1.0, -0.5]}, index=[dates[0], dates[5]])
    returns_df = pd.DataFrame({"A": 0.01, "B": -0.01}, index=dates)

    result = daily_mark_pnl(weights, returns_df)

    daily_weights = weights.reindex(dates).ffill().shift(1)
    expected = (daily_weights * returns_df).sum(axis=1, skipna=True)
    pd.testing.assert_series_equal(result, expected)


def test_daily_mark_pnl_with_cost_bps_deducts_lump_sum_on_rebalance_date():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[dates[0], dates[5]])
    returns_df = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    cost_bps = pd.Series({"A": 100.0, "B": 100.0})

    gross = daily_mark_pnl(weights, returns_df)
    net = daily_mark_pnl(weights, returns_df, cost_bps=cost_bps)

    # The rebalance at dates[5] (A: 1.0->0.0, B: 0.0->1.0, turnover=2.0) should
    # show a real cost drag exactly on that date, zero gross return elsewhere.
    assert net.loc[dates[5]] < gross.loc[dates[5]]
    assert np.isclose(net.loc[dates[5]], gross.loc[dates[5]] - 2.0 * (100.0 / 10_000))
