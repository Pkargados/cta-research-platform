import numpy as np
import pandas as pd

from regime.interface import get_actions_for_date

BOOK_NAMES = ["momentum", "carry"]


def _action_fn(label):
    table = {
        "risk_on": {"momentum": {"active": True, "alpha_multiplier": 1.5}},
        "risk_off": {"momentum": {"active": False, "alpha_multiplier": 1.0},
                     "carry": {"active": True, "alpha_multiplier": 0.5}},
    }
    return table.get(label, {})


def _regime_df():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame({"regime": ["risk_on", "risk_on", "risk_off", "risk_off", "risk_on"]}, index=dates)


def test_get_actions_for_date_uses_most_recent_label_on_or_before_date():
    regime_df = _regime_df()
    actions = get_actions_for_date(regime_df, pd.Timestamp("2020-01-03"), _action_fn, BOOK_NAMES)
    assert actions["momentum"] == {"active": False, "alpha_multiplier": 1.0}
    assert actions["carry"] == {"active": True, "alpha_multiplier": 0.5}


def test_get_actions_for_date_no_lookahead():
    regime_df = _regime_df()
    # 2020-01-02 is still "risk_on" - the "risk_off" flip on 01-03 must not leak
    # backward into an earlier date's lookup.
    actions = get_actions_for_date(regime_df, pd.Timestamp("2020-01-02"), _action_fn, BOOK_NAMES)
    assert actions["momentum"]["alpha_multiplier"] == 1.5


def test_get_actions_for_date_before_any_history_returns_neutral():
    regime_df = _regime_df()
    actions = get_actions_for_date(regime_df, pd.Timestamp("2019-12-31"), _action_fn, BOOK_NAMES)
    for name in BOOK_NAMES:
        assert actions[name] == {"active": True, "alpha_multiplier": 1.0}


def test_get_actions_for_date_unrecognized_label_returns_neutral():
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    regime_df = pd.DataFrame({"regime": ["mystery_regime", "mystery_regime"]}, index=dates)
    actions = get_actions_for_date(regime_df, dates[-1], _action_fn, BOOK_NAMES)
    for name in BOOK_NAMES:
        assert actions[name] == {"active": True, "alpha_multiplier": 1.0}


def test_get_actions_for_date_nan_label_returns_neutral():
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    regime_df = pd.DataFrame({"regime": [np.nan, np.nan]}, index=dates)
    actions = get_actions_for_date(regime_df, dates[-1], _action_fn, BOOK_NAMES)
    for name in BOOK_NAMES:
        assert actions[name] == {"active": True, "alpha_multiplier": 1.0}


def test_get_actions_for_date_custom_default_multiplier():
    regime_df = _regime_df()
    actions = get_actions_for_date(
        regime_df, pd.Timestamp("2019-12-31"), _action_fn, BOOK_NAMES, default_multiplier=0.75,
    )
    assert actions["momentum"] == {"active": True, "alpha_multiplier": 0.75}


def test_get_actions_for_date_guarantees_entry_for_every_book_name_even_if_unmentioned():
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    regime_df = pd.DataFrame({"regime": ["risk_on"]}, index=dates)

    def sparse_action_fn(label):
        return {"momentum": {"active": True, "alpha_multiplier": 2.0}}  # doesn't mention "new_book"

    actions = get_actions_for_date(regime_df, dates[0], sparse_action_fn, ["momentum", "new_book"])
    assert actions["new_book"] == {"active": True, "alpha_multiplier": 1.0}
