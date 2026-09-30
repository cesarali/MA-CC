"""Numerical checks of the equations: bounds, directions, units, identities."""

import numpy as np
import pytest

from bound_estimators import objectives as O
from bound_estimators.synthetic import exact_quantities, scenarios


def _random_pmfs(rng, n=30):
    p = rng.dirichlet(np.ones(n)); r = rng.dirichlet(np.ones(n))
    return p, r


def test_nwj_and_dv_lower_bound_kl_and_are_tight_for_true_log_ratio():
    rng = np.random.default_rng(0)
    p, r = _random_pmfs(rng)
    kl = O.exact_discrete_kl(p, r)
    # population objectives with an arbitrary critic (evaluate expectations exactly)
    for _ in range(20):
        f = rng.normal(size=len(p))
        nwj = float(p @ f - r @ np.exp(f) + 1)
        dv = float(p @ f - np.log(r @ np.exp(f)))
        assert nwj <= dv + 1e-12 <= kl + 1e-12
    f_true = np.log(p) - np.log(r)
    assert abs(float(p @ f_true - r @ np.exp(f_true) + 1) - kl) < 1e-10
    assert abs(float(p @ f_true - np.log(r @ np.exp(f_true))) - kl) < 1e-10


def test_information_score_is_tight_for_true_posterior_and_below_for_others():
    rng = np.random.default_rng(1)
    p0, p2 = _random_pmfs(rng)
    mi = O.exact_discrete_mi({0: p0, 2: p2})
    q = 0.5 * p0 + 0.5 * p2
    post0 = 0.5 * p0 / q
    # population expectation of log(g(z|y)/w_z)
    def score(g0):
        return float(0.5 * p0 @ np.log(g0 / 0.5) + 0.5 * p2 @ np.log((1 - g0) / 0.5))
    assert abs(score(post0) - mi) < 1e-10
    assert score(np.full_like(post0, 0.5)) == pytest.approx(0.0)
    for _ in range(10):
        g = np.clip(post0 + rng.normal(scale=0.1, size=len(p0)), 1e-6, 1 - 1e-6)
        assert score(g) <= mi + 1e-9
    assert mi <= np.log(2) + 1e-12  # two balanced targets: at most one bit


def test_information_score_function_matches_hand_formula():
    prob = np.array([0.9, 0.2, 0.5]); labels = np.array([0, 2, 0])
    contrib, sat = O.information_score(prob, labels, eps=1e-6)
    np.testing.assert_allclose(contrib, [np.log(0.9 / 0.5), np.log(0.8 / 0.5), 0.0])
    assert sat == 0.0
    _, sat = O.information_score(np.array([1.0, 0.0]), np.array([0, 2]), eps=1e-6)
    assert sat == 1.0


def test_frequency_mi_matches_exact_mi_on_counts_and_smoothing_sums_to_one():
    rng = np.random.default_rng(2)
    sup = np.array(O.endpoint_support(24))
    e0 = sup[rng.integers(0, len(sup), 40)]; e2 = sup[rng.integers(0, len(sup), 40)]
    p0 = O.smoothed_endpoint_pmf(e0, 0.0); p2 = O.smoothed_endpoint_pmf(e2, 0.0)
    assert abs(O.frequency_mi({0: e0, 2: e2}) - O.exact_discrete_mi({0: p0, 2: p2})) < 1e-12
    for a in (1.0, 10.0, 100.0):
        assert abs(O.smoothed_endpoint_pmf(e0, a).sum() - 1) < 1e-12
    assert len(sup) == 325


def test_prior_correction_and_cap_and_tail_diagnostics():
    logit = np.array([0.0, 2.0])
    np.testing.assert_allclose(O.prior_corrected_logit(logit, 0.5), logit)
    np.testing.assert_allclose(O.prior_corrected_logit(logit, 0.25), logit + np.log(3.0))
    f = np.array([-100.0, 0.0, 100.0])
    capped = O.cap_critic(f, 5.0)
    assert capped[0] >= -5 and capped[2] <= 5 and capped[1] == 0.0 and abs(capped[0]) < abs(f[0])
    assert O.cap_critic(f, None) is not None and np.array_equal(O.cap_critic(f, None), f)
    d = O.baseline_tail_diagnostics(np.array([0.0, 0.0, 0.0, 0.0]))
    assert d["ess"] == pytest.approx(4.0) and d["max_weight_share"] == pytest.approx(0.25)
    d = O.baseline_tail_diagnostics(np.array([0.0, 20.0]))
    assert d["ess"] < 1.01


def test_exact_decomposition_and_coarse_graining_on_synthetic_laws():
    for sc in scenarios():
        ex = exact_quantities(sc)
        # C = I(Z; Gamma) + KL(Q || P_base)
        assert ex["cost"] == pytest.approx(ex["info_path"] + ex["kl_mixture"], abs=1e-9)
        # 0 <= I(Z; Y_h) <= I(Z; Gamma) <= C
        for h in range(1, sc.horizon + 1):
            assert -1e-12 <= ex[f"info_end_h{h}"] <= ex["info_path"] + 1e-9
        assert ex["info_path"] <= ex["cost"] + 1e-9
    by = {s.name: exact_quantities(s) for s in scenarios()}
    assert by["identical"]["cost"] == pytest.approx(0.0, abs=1e-12)
    assert by["generic"]["info_path"] == pytest.approx(0.0, abs=1e-9) and by["generic"]["cost"] > 0.5
    assert by["transient"]["info_end_h3"] == pytest.approx(0.0, abs=1e-9) and by["transient"]["info_path"] > 0.1
    assert by["reversed"]["cost"] == pytest.approx(by["opposite"]["cost"])
