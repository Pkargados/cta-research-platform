import numpy as np

from signals.ipca import (
    build_managed_portfolios, fit_ipca_restricted, total_r2,
    managed_portfolio_residuals, estimate_gamma_alpha, bootstrap_alpha_test,
    alpha_signal,
)


def test_build_managed_portfolios_matches_hand_computation():
    # T=1, N=2, L=1 (a single characteristic) - x_t = Z_t' r_t is just a dot product.
    z_panel = np.array([[[2.0], [3.0]]])  # (1, 2, 1)
    r_next = np.array([[0.1, 0.2]])       # (1, 2)

    X, ZZ = build_managed_portfolios(z_panel, r_next)

    assert np.isclose(X[0, 0], 2.0 * 0.1 + 3.0 * 0.2)
    assert np.isclose(ZZ[0, 0, 0], 2.0 ** 2 + 3.0 ** 2)


def test_build_managed_portfolios_excludes_assets_missing_return_or_characteristic():
    # N=3: asset 1 has a NaN return, asset 2 has a NaN characteristic - both
    # should be excluded from the date's managed portfolio, asset 0 alone remains.
    z_panel = np.array([[[1.0], [1.0], [np.nan]]])
    r_next = np.array([[0.05, np.nan, 0.05]])

    X, ZZ = build_managed_portfolios(z_panel, r_next)

    assert np.isclose(X[0, 0], 1.0 * 0.05)
    assert np.isclose(ZZ[0, 0, 0], 1.0)


def test_build_managed_portfolios_all_missing_date_is_nan():
    z_panel = np.array([[[1.0], [1.0]]])
    r_next = np.array([[np.nan, np.nan]])

    X, ZZ = build_managed_portfolios(z_panel, r_next)
    assert np.isnan(X[0, 0])
    assert np.isnan(ZZ[0, 0, 0])


def _single_factor_dgp(n_assets=8, n_dates=400, noise_std=0.02, seed=0):
    """r_{i,t+1} = z_{i,t} * beta_true * f_{t+1} + noise - a genuine
    single-factor structure with NO alpha, used to check that fit_ipca_restricted
    recovers a factor highly correlated with the true one and explains most of
    the managed-portfolio variance."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_dates, n_assets))  # one characteristic, static-ish per date
    f_true = rng.normal(scale=1.0, size=n_dates)
    beta_true = 1.0

    r_next = np.full((n_dates, n_assets), np.nan)
    for t in range(n_dates - 1):
        r_next[t] = z[t] * beta_true * f_true[t + 1] + rng.normal(scale=noise_std, size=n_assets)

    z_panel = z[:, :, None]  # (T, N, L=1)
    return z_panel, r_next, f_true


def test_fit_ipca_restricted_recovers_single_factor_structure():
    z_panel, r_next, f_true = _single_factor_dgp()
    fit = fit_ipca_restricted(z_panel, r_next, k=1)

    r2 = total_r2(fit)
    assert r2 > 0.5  # a real single-factor DGP should be explained well, not perfectly (noise present)

    # r_next[t] is driven by f_true[t+1] (see _single_factor_dgp), and is NaN
    # for the last date - so valid_t is True for t=0..n_dates-2, and the
    # estimated F[t] should correlate with f_true[t+1] for those same t.
    valid = fit["valid_t"]
    est_f = fit["F"][valid, 0]
    true_f_aligned = f_true[1:][valid[:-1]]
    corr = np.corrcoef(est_f, true_f_aligned)[0, 1]
    assert abs(corr) > 0.8


def test_bootstrap_alpha_test_p_value_in_unit_interval():
    z_panel, r_next, _ = _single_factor_dgp(seed=1)
    fit = fit_ipca_restricted(z_panel, r_next, k=1)
    test = bootstrap_alpha_test(fit, n_boot=200, seed=2)

    assert 0.0 <= test["p_value"] <= 1.0
    assert test["w_alpha"] >= 0.0
    assert test["gamma_alpha"].shape == (1,)


def test_bootstrap_alpha_test_detects_a_large_injected_alpha():
    """Two INDEPENDENT characteristics this time (L=2, not L=1) - z1 drives the
    true single-factor structure (beta), z2 is a separate, independent
    characteristic that gets a large constant alpha injected on top
    (r += z2 * alpha_true). With only one characteristic (L=1, as in
    _single_factor_dgp alone) the restricted model is under-identified -
    a constant alpha and an unusual factor loading are indistinguishable
    when there's only one shared cross-sectional shape to explain the
    managed portfolio with (checked directly: the single-characteristic
    version of this test recovers a genuine nonzero effect but not
    necessarily the injected effect's own sign, since there's nothing to
    separate them). Two independent characteristics give the genuine
    separating identification the real Curve IPCA driver relies on
    (L=4: duration, carry, vol, const)."""
    z_panel, r_next, f_true = _single_factor_dgp(seed=3, noise_std=0.01)
    n_dates, n_assets = r_next.shape

    rng = np.random.default_rng(5)
    z2 = rng.normal(size=(n_dates, n_assets))
    alpha_true = 0.5
    r_next_with_alpha = r_next + z2 * alpha_true

    z_panel_2 = np.concatenate([z_panel, z2[:, :, None]], axis=2)  # (T, N, 2)

    fit = fit_ipca_restricted(z_panel_2, r_next_with_alpha, k=1)
    test = bootstrap_alpha_test(fit, n_boot=300, seed=4)

    assert test["p_value"] < 0.05
    assert test["gamma_alpha"][1] > 0.0  # recovers the right SIGN on z2, the characteristic the alpha was injected on


def test_managed_portfolio_residuals_are_zero_when_k_equals_l():
    """No dimension reduction (K=L): Gamma_beta spans the full characteristic
    space, so the restricted model should fit the managed portfolios EXACTLY -
    a real bug was caught exactly this way (an earlier version of
    managed_portfolio_residuals multiplied by ZZ_t, inconsistent with how F
    is derived, and gave residuals in the hundreds/thousands instead of ~0)."""
    z_panel, r_next, _ = _single_factor_dgp(seed=7)
    rng = np.random.default_rng(8)
    z2 = rng.normal(size=r_next.shape)
    z_panel_2 = np.concatenate([z_panel, z2[:, :, None]], axis=2)  # L=2

    fit = fit_ipca_restricted(z_panel_2, r_next, k=2)  # K=L=2
    d = managed_portfolio_residuals(fit)

    assert np.nanmax(np.abs(d)) < 1e-8


def test_alpha_signal_is_zpanel_dot_gamma_alpha():
    z_panel = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (T=1, N=2, L=2)
    gamma_alpha = np.array([0.5, -1.0])

    signal = alpha_signal(z_panel, gamma_alpha)

    assert np.allclose(signal[0], [1.0 * 0.5 + 2.0 * -1.0, 3.0 * 0.5 + 4.0 * -1.0])
