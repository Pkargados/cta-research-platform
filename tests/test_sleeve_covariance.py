import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio.sleeve_covariance import rolling_covariance, ewma_covariance, dcc_garch_covariance

_DCC_GARCH_SRC = Path(__file__).resolve().parent.parent.parent / "DCC Garch Rompolis" / "src"
_dcc_garch_available = (_DCC_GARCH_SRC / "dcc_garch").exists()


def _correlated_sleeve_returns(n=300, rho=-0.4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    a = rng.normal(0, 0.01, n)
    b = rho * a + rng.normal(0, 0.009, n) * np.sqrt(1 - rho ** 2)
    return pd.DataFrame({"sleeve_a": a, "sleeve_b": b}, index=dates)


def _is_psd(matrix, tol=-1e-10):
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= tol)


# --- rolling_covariance ---

def test_rolling_covariance_shape_and_symmetry():
    returns = _correlated_sleeve_returns()
    cov = rolling_covariance(returns, window=100)
    assert cov.shape == (2, 2)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    assert np.allclose(cov.values, cov.values.T)
    assert _is_psd(cov.values)


def test_rolling_covariance_uses_only_most_recent_window():
    returns = _correlated_sleeve_returns(n=300)
    # Corrupt the early part of the series with huge outliers - a window
    # covariance should be unaffected since it only looks at the tail.
    corrupted = returns.copy()
    corrupted.iloc[:50] = 100.0
    cov_clean = rolling_covariance(returns, window=100)
    cov_from_corrupted = rolling_covariance(corrupted, window=100)
    assert np.allclose(cov_clean.values, cov_from_corrupted.values, atol=1e-8)


def test_rolling_covariance_drops_nan_rows_first():
    returns = _correlated_sleeve_returns(n=150)
    gapped = returns.copy()
    gapped.iloc[10:20, 0] = np.nan
    cov = rolling_covariance(gapped, window=100)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov.values))


def test_rolling_covariance_raises_on_insufficient_observations():
    returns = _correlated_sleeve_returns(n=1)
    with pytest.raises(ValueError):
        rolling_covariance(returns, window=100)


# --- ewma_covariance ---

def test_ewma_covariance_shape_and_symmetry():
    returns = _correlated_sleeve_returns()
    cov = ewma_covariance(returns, halflife=26)
    assert cov.shape == (2, 2)
    assert list(cov.index) == list(returns.columns)
    assert np.allclose(cov.values, cov.values.T, atol=1e-12)
    assert _is_psd(cov.values)


def test_ewma_covariance_negative_correlation_sign_preserved():
    returns = _correlated_sleeve_returns(rho=-0.6)
    cov = ewma_covariance(returns, halflife=26)
    assert cov.loc["sleeve_a", "sleeve_b"] < 0


def test_ewma_covariance_raises_on_insufficient_observations():
    returns = _correlated_sleeve_returns(n=1)
    with pytest.raises(ValueError):
        ewma_covariance(returns, halflife=26)


# --- dcc_garch_covariance ---
# Requires the local, private "DCC Garch Rompolis" sibling repo (see module
# docstring) - not a pip package, and not guaranteed to exist in every
# environment this test suite runs in.

@pytest.mark.skipif(not _dcc_garch_available, reason="local DCC Garch Rompolis sibling repo not found")
def test_dcc_garch_covariance_shape_symmetry_and_convergence_flag():
    returns = _correlated_sleeve_returns(n=600)
    result = dcc_garch_covariance(returns)
    assert set(result.keys()) == {"cov", "converged"}
    assert isinstance(result["converged"], bool)
    cov = result["cov"]
    assert cov.shape == (2, 2)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    assert np.allclose(cov.values, cov.values.T, atol=1e-10)


@pytest.mark.skipif(not _dcc_garch_available, reason="local DCC Garch Rompolis sibling repo not found")
def test_dcc_garch_covariance_negative_correlation_sign_preserved():
    returns = _correlated_sleeve_returns(n=600, rho=-0.5)
    result = dcc_garch_covariance(returns)
    assert result["cov"].loc["sleeve_a", "sleeve_b"] < 0


@pytest.mark.skipif(not _dcc_garch_available, reason="local DCC Garch Rompolis sibling repo not found")
def test_dcc_garch_covariance_same_order_of_magnitude_as_simple_sample_covariance():
    # Not expected to match exactly (DCC is conditional, not unconditional),
    # but the x100/10000 unit conversion must land in the right ballpark -
    # this is the check that would have caught a missed or doubled rescale.
    returns = _correlated_sleeve_returns(n=600, rho=-0.4)
    dcc_cov = dcc_garch_covariance(returns)["cov"]
    simple_cov = returns.cov()
    ratio = dcc_cov.values / simple_cov.values
    assert np.all(ratio > 0.1) and np.all(ratio < 10.0)
