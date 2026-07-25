import numpy as np

from portfolio.optimizer import solve_weights


def test_solve_weights_matches_closed_form():
    alpha = np.array([0.1, -0.05])
    Sigma = np.array([[0.04, 0.0], [0.0, 0.09]])
    x_prev = np.array([0.0, 0.0])
    gamma, kappa, lambd, max_weight = 1.0, 0.0, 0.0, 10.0

    result = solve_weights(alpha, Sigma, x_prev, gamma, kappa, lambd, max_weight)

    # With kappa=lambd=0: A = gamma*Sigma, b = alpha -> x = alpha / (gamma*diag(Sigma))
    # for a diagonal Sigma.
    expected = alpha / (gamma * np.diag(Sigma))
    assert np.allclose(result, expected)


def test_solve_weights_respects_max_weight_clip():
    alpha = np.array([100.0, -100.0])
    Sigma = np.eye(2) * 0.01
    x_prev = np.zeros(2)
    result = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=0.0, lambd=0.0, max_weight=0.3)
    assert np.all(np.abs(result) <= 0.3 + 1e-12)


def test_solve_weights_dollar_neutral_sums_to_zero():
    # max_weight set generously wide (10.0) relative to the unclipped solution's
    # own scale, so the clip (applied AFTER recentering in solve_weights) doesn't
    # bind and reintroduce net exposure - isolating the recentering behavior itself.
    alpha = np.array([0.2, -0.05, 0.1])
    Sigma = np.eye(3) * 0.02
    x_prev = np.zeros(3)
    result = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=0.0, lambd=0.0, max_weight=10.0, dollar_neutral=True)
    assert np.isclose(result.sum(), 0.0, atol=1e-10)


def test_solve_weights_not_dollar_neutral_by_default():
    alpha = np.array([0.2, 0.2])  # both positive -> no reason to sum to zero
    Sigma = np.eye(2) * 0.01
    x_prev = np.zeros(2)
    result = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=0.0, lambd=0.0, max_weight=10.0)
    assert result.sum() > 0


def test_solve_weights_kappa_pulls_toward_x_prev():
    alpha = np.zeros(2)  # no alpha signal at all
    Sigma = np.eye(2) * 0.01
    x_prev = np.array([0.05, -0.05])

    no_inertia = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=0.0, lambd=0.0, max_weight=1.0)
    with_inertia = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=1e6, lambd=0.0, max_weight=1.0)

    # With alpha=0 and no inertia, the optimal position is flat (0).
    assert np.allclose(no_inertia, 0.0, atol=1e-8)
    # With overwhelming inertia (kappa >> gamma*Sigma), the solution should stay
    # very close to x_prev instead of collapsing to zero.
    assert np.allclose(with_inertia, x_prev, atol=1e-3)


def test_solve_weights_lambd_penalizes_in_direction_of_x_prev_sign():
    alpha = np.array([0.0])
    Sigma = np.array([[0.01]])
    x_prev = np.array([0.5])  # holding a long position

    no_penalty = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=1.0, lambd=0.0, max_weight=10.0)
    with_penalty = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=1.0, lambd=0.1, max_weight=10.0)

    # lambd subtracts lambd*sign(x_prev) from b, so a positive x_prev's solution
    # should shrink relative to the no-penalty case.
    assert with_penalty[0] < no_penalty[0]


def test_solve_weights_returns_correct_shape():
    n = 5
    alpha = np.random.default_rng(0).normal(size=n)
    Sigma = np.eye(n) * 0.02
    x_prev = np.zeros(n)
    result = solve_weights(alpha, Sigma, x_prev, gamma=1.0, kappa=0.5, lambd=0.01, max_weight=0.5)
    assert result.shape == (n,)
