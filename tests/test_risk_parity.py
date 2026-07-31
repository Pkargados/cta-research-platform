import numpy as np

from portfolio.risk_parity import risk_contributions, risk_parity_weights


def _corr_to_cov(vols, corr):
    return np.outer(vols, vols) * corr


def test_risk_contributions_sums_to_portfolio_vol_for_arbitrary_weights():
    # Euler's theorem for a positively-homogeneous-degree-1 function: this must
    # hold for ANY weight vector, not just a risk-parity solution.
    vols = np.array([0.10, 0.20, 0.15])
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    Sigma = _corr_to_cov(vols, corr)
    w = np.array([0.5, 0.3, 0.2])

    rc = risk_contributions(w, Sigma)
    port_vol = np.sqrt(w @ Sigma @ w)
    assert np.isclose(rc.sum(), port_vol, atol=1e-10)


def test_risk_contributions_zero_weights_returns_zero_not_nan():
    Sigma = np.eye(3) * 0.04
    rc = risk_contributions(np.zeros(3), Sigma)
    assert np.allclose(rc, 0.0)


def test_risk_parity_weights_two_sleeve_matches_closed_form_inverse_vol():
    # For n=2, equal-risk-contribution has a closed form independent of
    # correlation: w_1/w_2 = sigma_2/sigma_1.
    vols = np.array([0.10, 0.25])
    for rho in (-0.5, 0.0, 0.3, 0.8):
        Sigma = _corr_to_cov(vols, np.array([[1.0, rho], [rho, 1.0]]))
        w = risk_parity_weights(Sigma)

        assert np.isclose(w[0] / w[1], vols[1] / vols[0], atol=1e-6)
        rc = risk_contributions(w, Sigma)
        assert np.isclose(rc[0], rc[1], atol=1e-9)


def test_risk_parity_weights_n3_gives_exactly_equal_risk_contributions():
    vols = np.array([0.10, 0.20, 0.15])
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    Sigma = _corr_to_cov(vols, corr)

    w = risk_parity_weights(Sigma)
    rc = risk_contributions(w, Sigma)

    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(rc, rc[0], atol=1e-9)


def test_risk_parity_weights_unequal_budgets_match_target_risk_shares():
    vols = np.array([0.10, 0.20, 0.15])
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    Sigma = _corr_to_cov(vols, corr)
    budgets = np.array([0.6, 0.3, 0.1])

    w = risk_parity_weights(Sigma, risk_budgets=budgets)
    rc = risk_contributions(w, Sigma)
    rc_shares = rc / rc.sum()

    assert np.allclose(rc_shares, budgets, atol=1e-6)


def test_risk_parity_weights_budgets_are_normalized_before_solving():
    # Passing un-normalized budgets (e.g. [3, 1]) should give the same result
    # as their normalized equivalent ([0.75, 0.25]).
    vols = np.array([0.10, 0.25])
    Sigma = _corr_to_cov(vols, np.array([[1.0, 0.2], [0.2, 1.0]]))

    w_raw = risk_parity_weights(Sigma, risk_budgets=np.array([3.0, 1.0]))
    w_normalized = risk_parity_weights(Sigma, risk_budgets=np.array([0.75, 0.25]))

    assert np.allclose(w_raw, w_normalized, atol=1e-8)


def test_risk_parity_weights_sum_to_one():
    vols = np.array([0.08, 0.12, 0.30, 0.05])
    corr = np.eye(4)
    Sigma = _corr_to_cov(vols, corr)
    w = risk_parity_weights(Sigma)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)


def test_risk_parity_weights_diagonal_cov_matches_inverse_variance_not_inverse_vol():
    # For a diagonal covariance matrix (zero correlation), equal-risk-contribution
    # weights are proportional to 1/vol (not 1/variance) - same closed form as
    # the 2-sleeve case, generalized to n sleeves.
    vols = np.array([0.05, 0.10, 0.20])
    Sigma = np.diag(vols ** 2)
    w = risk_parity_weights(Sigma)
    expected = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(w, expected, atol=1e-6)
