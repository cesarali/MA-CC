"""Synthetic checks for frontier.py (handoff §7, tests 1-5 and 7; tie rule). Run from the repository root:
    .venv/bin/python -m pytest analysis/task003_llm_free/efficiency_estimates/tests -q
"""
import pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from frontier import best_gain, efficiencies, hull, hull_best_gain, least_cost, plurality_success  # noqa: E402
from efficiency import measures  # noqa: E402


def test_1_old_ratio_is_one_for_mirror_steering():
    p0 = np.array([0.5, 0.0, 0.5])
    Is = []
    for d in (0.05, 0.1, 0.2, 0.4):
        m = measures(np.array([0.5 + d, 0, 0.5 - d]), np.array([0.5 - d, 0, 0.5 + d]), p0)
        assert abs(m["R"]) < 1e-12 and abs(m["C"] - m["I"]) < 1e-12 and abs(m["eta"] - 1) < 1e-9
        Is.append(float(m["I"]))
    assert all(a < b for a, b in zip(Is, Is[1:]))
    assert np.isnan(measures(p0, p0, p0)["eta"])


def test_2_dominated_policies():
    e = efficiencies(np.array([0.2, 0.2, 0.2]), np.array([10.0, 5.0, 4.0]))
    assert np.allclose(e["eta_cost"], [0.4, 0.8, 1.0])


def test_3_all_on_frontier_despite_diminishing_returns():
    d, c = np.array([0.10, 0.15, 0.18, 0.19]), np.array([1.0, 2.0, 3.0, 4.0])
    e = efficiencies(d, c)
    assert np.allclose(e["eta_cost"], 1.0)
    g = e["gain_per_fact"]
    assert all(a > b for a, b in zip(g, g[1:]))
    marginal = np.diff(d) / np.diff(c)
    assert all(a > b for a, b in zip(marginal, marginal[1:]))


def test_4_harmful_policy_has_no_efficiency():
    e = efficiencies(np.array([-0.10, 0.30]), np.array([5.0, 5.0]))
    assert np.isnan(e["eta_cost"][0]) and e["eta_cost"][1] == 1.0


def test_5_zero_and_unresolved():
    e = efficiencies(np.array([0.0, 0.2, 0.01]), np.array([3.0, 0.0, 2.0]), resolved=np.array([True, True, False]))
    assert np.isnan(e["eta_cost"]).all()
    assert least_cost(np.array([0.1]), np.array([2.0]), 0.5) == np.inf


def test_7_best_gain_nondecreasing_in_budget():
    d, c = np.array([0.3, 0.1, 0.25]), np.array([2.0, 5.0, 8.0])
    vals = [best_gain(d, c, b) for b in np.linspace(0, 10, 21)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    xs, ys = hull(d, c)
    hv = [hull_best_gain(xs, ys, b) for b in np.linspace(0, 10, 21)]
    assert all(a <= b + 1e-12 for a, b in zip(hv, hv[1:]))
    assert all(h >= v - 1e-12 for h, v in zip(hv, vals))        # mixtures never worse


def test_mixtures_lower_cost():
    # (2, 0.2) and (10, 0.4): reaching 0.3 needs cost 10 with pure policies, 6 with a 50/50 mixture
    e = efficiencies(np.array([0.2, 0.4, 0.3]), np.array([2.0, 10.0, 9.0]))
    assert np.isclose(e["eta_cost"][2], 9.0 / 9.0) and np.isclose(e["eta_cost_mix"][2], 6.0 / 9.0)


def test_plurality_and_ties():
    v = np.array([[12, 2, 10], [8, 8, 8], [10, 4, 10], [0, 24, 0]])
    assert np.allclose(plurality_success(v, 0), [1, 1 / 3, 0.5, 0])
    assert np.allclose(plurality_success(v, 2), [0, 1 / 3, 0.5, 0])
    assert np.allclose(plurality_success(v, 0, "majority"), [0, 0, 0, 0])
    assert np.allclose(plurality_success(np.array([[13, 1, 10]]), 0, "majority"), [1])
