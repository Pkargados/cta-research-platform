import numpy as np
import pandas as pd

from data.ewma_volatility import ewma_volatility, ANNUALIZATION, DEFAULT_COM


def test_constant_zero_returns_gives_zero_vol():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    returns = pd.DataFrame({"A": 0.0}, index=dates)
    vol = ewma_volatility(returns)
    assert np.allclose(vol["A"], 0.0)


def test_annualize_true_scales_by_sqrt_252_vs_false():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 30)}, index=dates)

    annualized = ewma_volatility(returns, annualize=True)
    raw = ewma_volatility(returns, annualize=False)
    ratio = (annualized / raw).dropna()
    assert np.allclose(ratio, np.sqrt(ANNUALIZATION), atol=1e-8)


def test_matches_manual_ewm_formula():
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({"A": rng.normal(0, 0.02, 15)}, index=dates)

    result = ewma_volatility(returns, com=10, annualize=False)
    expected = np.sqrt(returns.pow(2).ewm(com=10, adjust=False).mean())
    pd.testing.assert_frame_equal(result, expected)


def test_no_lookahead_first_value_uses_only_itself():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame({"A": [0.1, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    result = ewma_volatility(returns, com=DEFAULT_COM, annualize=False)
    # ewm(adjust=False)'s first value is exactly the first observation's own
    # squared value - sqrt(0.1^2) = 0.1.
    assert np.isclose(result["A"].iloc[0], 0.1)


def test_higher_vol_regime_produces_higher_estimate():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.001, 20)
    stormy = rng.normal(0, 0.05, 20)
    returns = pd.DataFrame({"A": np.concatenate([calm, stormy])}, index=dates)

    vol = ewma_volatility(returns, com=5)
    assert vol["A"].iloc[-1] > vol["A"].iloc[10]
