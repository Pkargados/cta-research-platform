from collections import OrderedDict

import numpy as np
import pandas as pd

from portfolio.allocator import Allocator
from portfolio.book import Book

ASSETS = ["A", "B"]


class FakeBook:
    """Minimal stand-in for portfolio.book.Book, used to isolate Allocator's own
    combination/regime logic from the real optimizer's mechanics."""

    def __init__(self, name, alpha_df, pnl_index, is_active=True):
        self.name = name
        self.alpha_df = alpha_df
        self.is_active = is_active
        self._pnl_index = pnl_index

    def run(self, returns_df):
        # PnL is exactly the alpha_df's own column-A values on its own index -
        # a run() result that directly reflects whatever alpha_df this call
        # actually received, so regime-scaling can be verified end-to-end.
        pnl = self.alpha_df["A"].reindex(self._pnl_index)
        return {"pnl": pnl, "sharpe": np.nan}


def _identity_cov(scale=0.0001):
    return pd.DataFrame(np.eye(2) * scale, index=ASSETS, columns=ASSETS)


def _real_book(n=30, **overrides):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    alpha_df = pd.DataFrame({"A": rng.normal(0, 1, n), "B": rng.normal(0, 1, n)}, index=dates)
    returns_df = pd.DataFrame({"A": rng.normal(0, 0.01, n), "B": rng.normal(0, 0.01, n)}, index=dates)
    cov_dict = OrderedDict((d, _identity_cov()) for d in dates)
    params = dict(
        name="momentum", alpha_df=alpha_df, cov_dict=cov_dict,
        gamma=1.0, kappa=0.1, lambd=0.0, max_weight=0.5,
        target_vol=0.10, ewma_halflife=5, scale_min=0.1, scale_max=5.0,
        periods_per_year=252, dollar_neutral=False,
    )
    params.update(overrides)
    return Book(**params), returns_df


def test_single_active_book_no_regime_matches_book_run_directly():
    book, returns_df = _real_book()
    allocator = Allocator([book])
    combined = allocator.run(returns_df)
    direct = book.run(returns_df)
    pd.testing.assert_series_equal(combined["pnl"], direct["pnl"])


def test_inactive_book_is_skipped_silently():
    book, returns_df = _real_book()
    book.is_active = False
    allocator = Allocator([book])
    combined = allocator.run(returns_df)
    assert combined["book_results"] == {}
    assert combined["pnl"] is None


def test_multiple_books_combined_by_addition_with_fill_value_zero():
    dates_a = pd.date_range("2020-01-01", periods=5, freq="D")
    dates_b = pd.date_range("2020-01-03", periods=5, freq="D")  # partial overlap
    alpha_a = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates_a)
    alpha_b = pd.DataFrame({"A": [2.0] * 5, "B": [2.0] * 5}, index=dates_b)

    book_a = FakeBook("a", alpha_a, pnl_index=dates_a)
    book_b = FakeBook("b", alpha_b, pnl_index=dates_b)

    allocator = Allocator([book_a, book_b])
    combined = allocator.run(returns_df=None)

    full_index = dates_a.union(dates_b)
    expected = pd.Series(0.0, index=full_index)
    expected.loc[dates_a] += 1.0
    expected.loc[dates_b] += 2.0
    pd.testing.assert_series_equal(combined["pnl"].sort_index(), expected.sort_index(), check_names=False)


def test_regime_lookup_scales_alpha_before_book_runs():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    book = FakeBook("momentum", alpha, pnl_index=dates)

    def regime_lookup(date):
        return {"momentum": {"active": True, "alpha_multiplier": 2.0}}

    allocator = Allocator([book], regime_lookup=regime_lookup)
    combined = allocator.run(returns_df=None)
    # The FakeBook's own run() reflects whatever alpha it actually received -
    # a 2x multiplier should show up directly in the resulting pnl.
    assert (combined["pnl"] == 2.0).all()


def test_regime_lookup_deactivates_book_for_all_dates():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    book = FakeBook("momentum", alpha, pnl_index=dates)

    def regime_lookup(date):
        return {"momentum": {"active": False, "alpha_multiplier": 1.0}}

    allocator = Allocator([book], regime_lookup=regime_lookup)
    combined = allocator.run(returns_df=None)
    assert combined["book_results"] == {}


def test_regime_lookup_does_not_mutate_original_book():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    book = FakeBook("momentum", alpha, pnl_index=dates)
    original_alpha = book.alpha_df.copy()

    def regime_lookup(date):
        return {"momentum": {"active": True, "alpha_multiplier": 3.0}}

    Allocator([book], regime_lookup=regime_lookup).run(returns_df=None)
    pd.testing.assert_frame_equal(book.alpha_df, original_alpha)


def test_regime_lookup_defaults_to_neutral_for_unmentioned_book():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    book = FakeBook("carry", alpha, pnl_index=dates)

    def regime_lookup(date):
        return {}  # doesn't mention "carry" at all

    allocator = Allocator([book], regime_lookup=regime_lookup)
    combined = allocator.run(returns_df=None)
    assert (combined["pnl"] == 1.0).all()  # unscaled, still active


def test_book_weights_scale_each_books_pnl_before_combining():
    dates_a = pd.date_range("2020-01-01", periods=5, freq="D")
    dates_b = pd.date_range("2020-01-03", periods=5, freq="D")  # partial overlap
    alpha_a = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates_a)
    alpha_b = pd.DataFrame({"A": [2.0] * 5, "B": [2.0] * 5}, index=dates_b)

    book_a = FakeBook("a", alpha_a, pnl_index=dates_a)
    book_b = FakeBook("b", alpha_b, pnl_index=dates_b)

    allocator = Allocator([book_a, book_b], book_weights={"a": 0.5, "b": 1.5})
    combined = allocator.run(returns_df=None)

    full_index = dates_a.union(dates_b)
    expected = pd.Series(0.0, index=full_index)
    expected.loc[dates_a] += 1.0 * 0.5
    expected.loc[dates_b] += 2.0 * 1.5
    pd.testing.assert_series_equal(combined["pnl"].sort_index(), expected.sort_index(), check_names=False)


def test_book_weights_missing_name_defaults_to_one():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    book = FakeBook("momentum", alpha, pnl_index=dates)

    # book_weights mentions a different book entirely - "momentum" should stay at 1.0
    allocator = Allocator([book], book_weights={"other_book": 5.0})
    combined = allocator.run(returns_df=None)
    assert (combined["pnl"] == 1.0).all()


def test_book_weights_none_matches_prior_equal_sum_behavior():
    book, returns_df = _real_book()
    allocator_default = Allocator([book])
    allocator_explicit_none = Allocator([book], book_weights=None)
    pd.testing.assert_series_equal(
        allocator_default.run(returns_df)["pnl"], allocator_explicit_none.run(returns_df)["pnl"],
    )


def test_asset_contributions_combined_with_same_weights_as_pnl():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    assets = ["A", "B"]
    contrib_a = pd.DataFrame({"A": [1.0] * 5, "B": [0.5] * 5}, index=dates)
    contrib_b = pd.DataFrame({"A": [2.0] * 5, "B": [1.0] * 5}, index=dates)

    class BookWithContributions:
        def __init__(self, name, contributions, weight):
            self.name = name
            self.is_active = True
            self._contributions = contributions
            self._weight = weight  # only used to build a matching "pnl" for the assertion

        def run(self, returns_df):
            return {"pnl": self._contributions.sum(axis=1), "asset_contributions": self._contributions}

    book_a = BookWithContributions("a", contrib_a, 1.0)
    book_b = BookWithContributions("b", contrib_b, 1.0)
    allocator = Allocator([book_a, book_b], book_weights={"a": 0.5, "b": 1.5})
    combined = allocator.run(returns_df=None)

    expected_contrib = contrib_a * 0.5 + contrib_b * 1.5
    pd.testing.assert_frame_equal(combined["asset_contributions"].sort_index(), expected_contrib.sort_index())
    # Invariant Book._compute_pnl's own docstring establishes per-Book must still hold combined:
    # summing asset_contributions across columns reproduces the combined pnl exactly.
    pd.testing.assert_series_equal(
        combined["asset_contributions"].sum(axis=1).sort_index(), combined["pnl"].sort_index(), check_names=False,
    )


def test_asset_contributions_none_when_no_book_provides_them():
    book, returns_df = _real_book()  # real Book.run() DOES include asset_contributions
    # FakeBook's run() does not include "asset_contributions" at all
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alpha = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates)
    fake_book = FakeBook("fake", alpha, pnl_index=dates)
    allocator = Allocator([fake_book])
    combined = allocator.run(returns_df=None)
    assert combined["asset_contributions"] is None
