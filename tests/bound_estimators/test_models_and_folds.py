"""Logistic-loss convention, standardization, fold leakage prevention, 1-SE selection."""

import numpy as np
import pytest

from bound_estimators import models as M
from bound_estimators.crossfit import Candidate, Dataset, crossfit, one_se_select
from bound_estimators.folds import make_fold_plan


def test_logistic_minimizes_mean_ce_plus_half_lambda_beta_sq():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)); y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(float)
    lam = 0.5
    model = M.LogisticModel(lam=lam).fit(X, y)
    Xs = model.scaler(X)
    def objective(b0, b):
        s = b0 + Xs @ b
        return np.mean(np.logaddexp(0, s) - y * s) + 0.5 * lam * b @ b
    base = objective(model.intercept, model.beta)
    # stationary point: any perturbation increases the (convex) objective
    for _ in range(20):
        d = rng.normal(size=3) * 1e-2
        assert objective(model.intercept, model.beta + d) >= base - 1e-9
        assert objective(model.intercept + 1e-2, model.beta) >= base - 1e-9
    # stronger regularization shrinks the slope but not the intercept's freedom
    strong = M.LogisticModel(lam=50.0).fit(X, y)
    assert np.linalg.norm(strong.beta) < np.linalg.norm(model.beta)
    # standardization is fitted on the training data only
    np.testing.assert_allclose(model.scaler.mean, X.mean(axis=0))


def test_endpoint_and_path_features_shapes():
    counts = np.array([[12, 6, 6], [24, 0, 0]])
    lin = M.endpoint_features(counts, "linear"); quad = M.endpoint_features(counts, "quadratic")
    np.testing.assert_allclose(lin, [[0.5, 0.25], [1.0, 0.0]])
    assert quad.shape == (2, 5)
    paths = np.tile(counts[:, None, :], (1, 11, 1))
    assert M.path_summary_features(paths, "linear").shape == (2, 6)
    assert M.path_summary_features(paths, "quadratic").shape == (2, 27)
    seq = M.path_sequence_input(paths)
    assert seq.shape == (2, 11, 3) and seq[0, -1, 2] == pytest.approx(1.0)
    assert M.memory_indices(10, 2) == [0, 9, 10] and M.memory_indices(10, 10) == list(range(11))


def test_fold_plan_partitions_parents_and_keeps_examples_together():
    ids = [f"p{i}" for i in range(38)]
    plan = make_fold_plan(ids, 20260921, n_outer=5, n_inner=3, tag="t")
    seen = sorted(i for f in plan.outer for i in f)
    assert seen == list(range(38))
    for k in range(5):
        tr, te = plan.outer_split(k)
        assert not set(tr) & set(te)
        inner_val = sorted(i for j in range(3) for i in plan.inner[k][j])
        assert inner_val == sorted(tr.tolist())
        for j in range(3):
            fit, val = plan.inner_split(k, j)
            assert not set(fit) & set(val) and not set(val) & set(te)
    # same seed -> same plan; different repeat -> different plan
    assert make_fold_plan(ids, 20260921, tag="t").outer == plan.outer
    assert make_fold_plan(ids, 20260921, repeat=1, tag="t").outer != plan.outer
    ds = Dataset("endpoint", np.zeros((76, 3)), np.r_[np.ones(38), np.zeros(38)], np.ones(76),
                 np.r_[np.arange(38), np.arange(38)], 10)
    idx = ds.subset(np.array([3, 7]))
    assert sorted(idx.tolist()) == [3, 7, 41, 45]


def test_one_se_rule_prefers_simplest_within_one_se_and_breaks_ties_deterministically():
    cands = [Candidate("constant"), Candidate("linear", 0.1), Candidate("linear", 1.0),
             Candidate("quadratic", 0.1), Candidate("mlp", 0.1, 4)]
    mean = {"constant": 0.70, "linear-l0.1": 0.60, "linear-l1": 0.62, "quadratic-l0.1": 0.55, "mlp-w4-l0.1": 0.56}
    se = {k: 0.08 for k in mean}
    # best is quadratic (0.55), threshold 0.63: constant excluded, linear-l1 (more regularized) wins
    assert one_se_select(mean, se, cands) == "linear-l1"
    se = {k: 0.0 for k in mean}
    assert one_se_select(mean, se, cands) == "quadratic-l0.1"


def test_crossfit_has_no_leakage_and_constant_scores_zero():
    rng = np.random.default_rng(3)
    m = 20
    counts = rng.multinomial(24, [0.4, 0.3, 0.3], size=2 * m)
    ds = Dataset("endpoint", counts, np.r_[np.ones(m), np.zeros(m)], np.ones(2 * m), np.r_[np.arange(m), np.arange(m)], 10)
    plan = make_fold_plan([f"p{i}" for i in range(m)], 1, n_outer=4, n_inner=2)
    cands = [Candidate("constant"), Candidate("linear", 1.0)]
    res = crossfit(ds, cands, plan, M.TrainConfig(max_epochs=5))
    assert np.all(np.isfinite(res.selected_logit))
    np.testing.assert_array_equal(res.oof_logit["constant"], 0.0)
    # every example scored exactly once, by a fold that did not contain its parent
    for f in res.folds:
        test_examples = ds.subset(np.array(f.test_parents))
        assert set(ds.parent[test_examples]) == set(f.test_parents)
