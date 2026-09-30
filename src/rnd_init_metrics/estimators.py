"""Fitted lower-bound estimators for the information and KL terms (sections 6, 11).

Four estimation problems are built from one Comparison at one horizon h:

* ``endpoint``  target 0 vs target 2 from Y_h alone  -> lower bound on I(Z; Y_h)
* ``pathinfo``  target 0 vs target 2 from Gamma_h    -> lower bound on I(Z; Gamma_h)
* ``cost0`` / ``cost2``  controlled target z vs silence on Gamma_h
                                                     -> lower bounds on K_pi^z
* ``mixture``   the balanced target mixture vs silence -> lower bound on KL(Q || P_base)

The cross-fitting machinery, candidate models and variational objectives are
reused from :mod:`bound_estimators`, which already implements the training
protocol the reference document asks for (grouped folds, inner selection by log
loss, NWJ/DV critics, capped-critic and baseline-tail diagnostics).

Every split is grouped by initialization, so a model is never scored on an
initialization it was trained on, and all arms of an initialization stay together.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bound_estimators import objectives as O
from bound_estimators.crossfit import Candidate, Dataset, crossfit
from bound_estimators.folds import FoldPlan
from bound_estimators.models import TrainConfig

from .data import TARGETS, Comparison

PROBLEMS = ("endpoint", "pathinfo", "cost0", "cost2", "mixture")
CAPS = (2.0, 5.0, 10.0)
SMOOTHING_ALPHAS = (0.0, 1.0, 10.0)


def build_dataset(comp: Comparison, problem: str, h: int) -> Dataset:
    """Assemble the labelled array for one problem at horizon h.

    Class 1 is always the distribution whose divergence or target identity we are
    scoring; class 0 is the reference.  Weights are per-initialization so the two
    classes are balanced even when the mixture contributes two rows per unit.
    """
    m, T = comp.m, h + 1
    par = np.arange(m)
    if problem == "endpoint":
        X = np.concatenate([comp.controlled[0][:, h], comp.controlled[2][:, h]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("endpoint", X, y, np.ones(2 * m), np.concatenate([par, par]), h)
    if problem == "pathinfo":
        X = np.concatenate([comp.controlled[0][:, :T], comp.controlled[2][:, :T]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("path", X, y, np.ones(2 * m), np.concatenate([par, par]), h)
    if problem in ("cost0", "cost2"):
        z = 0 if problem == "cost0" else 2
        X = np.concatenate([comp.controlled[z][:, :T], comp.silent[:, :T]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("path", X, y, np.ones(2 * m), np.concatenate([par, par]), h)
    if problem == "mixture":
        X = np.concatenate([comp.controlled[0][:, :T], comp.controlled[2][:, :T], comp.silent[:, :T]])
        y = np.concatenate([np.ones(2 * m), np.zeros(m)])
        w = np.concatenate([np.full(2 * m, 0.5), np.ones(m)])
        return Dataset("path", X, y, w, np.concatenate([par, par, par]), h)
    raise ValueError(problem)


def _info_from_logits(logit: np.ndarray, ds: Dataset, m: int) -> dict[str, Any]:
    """Balanced held-out information score E log2[2 g(Z|.)], in bits and nats.

    The population gap from the true mutual information is the expected KL between
    the true and fitted target posteriors, so this is a lower bound in expectation
    and negative realized values are permitted.
    """
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = np.where(ds.y == 1, 0, 2)
    out: dict[str, Any] = {}
    for eps in (1e-6, 1e-4):
        contrib, sat = O.information_score(prob, labels, eps=eps)
        per_init = np.zeros(m)
        np.add.at(per_init, ds.parent, 0.5 * contrib)
        out[f"per_init_eps{eps:g}"] = per_init
        out[f"saturation_eps{eps:g}"] = sat
    out["score_bits"] = float(out["per_init_eps1e-06"].mean())
    out["score_nats"] = out["score_bits"] * np.log(2.0)
    out["per_init"] = out["per_init_eps1e-06"]
    out["log_loss"] = O.log_loss(prob, labels)
    out["accuracy"] = float(np.mean((prob > 0.5) == (labels == 0)))
    return out


def _cost_from_logits(logit: np.ndarray, ds: Dataset, m: int) -> dict[str, Any]:
    """NWJ / DV / plug-in divergence scores in nats, with baseline-tail diagnostics.

    NWJ is E_P[f] - E_base[e^f] + 1 and DV is E_P[f] - log E_base[e^f]; both are
    population lower bounds on the KL divergence.  The effective sample size of
    e^f over the silent paths is reported because a handful of baseline paths can
    dominate the exponential average.
    """
    ctl, sil = ds.y == 1, ds.y == 0
    out: dict[str, Any] = {}
    for cap in (None, *CAPS):
        f = O.cap_critic(logit, cap)
        fc = np.zeros(m); wc = np.zeros(m)
        np.add.at(fc, ds.parent[ctl], f[ctl] * ds.w[ctl])
        np.add.at(wc, ds.parent[ctl], ds.w[ctl])
        fc = np.divide(fc, wc, out=np.zeros_like(fc), where=wc > 0)
        fs = np.zeros(m); fs[ds.parent[sil]] = f[sil]
        tag = "raw" if cap is None else f"cap{cap:g}"
        contrib = O.nwj_contributions(fc, fs)
        out[f"per_init_nwj_{tag}"] = contrib
        out[f"nwj_{tag}"] = float(contrib.mean())
        out[f"dv_{tag}"] = O.dv_score(fc, fs)
        out[f"plugin_{tag}"] = O.plugin_score(fc)
        out.update({f"{k}_{tag}": v for k, v in O.baseline_tail_diagnostics(fs).items()})
    out["score_nats"] = out["nwj_cap5"]
    out["per_init"] = out["per_init_nwj_cap5"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    out["accuracy"] = float(np.average((prob > 0.5) == ctl, weights=ds.w))
    return out


def frequency_endpoint(comp: Comparison, h: int, plan: FoldPlan) -> dict[str, float]:
    """In-sample frequency / smoothed plug-in MI against a held-out smoothed posterior.

    The raw frequency estimator is strongly upward biased when endpoints are
    nearly unique; the held-out column is what should be quoted.
    """
    ends = {z: comp.controlled[z][:, h] for z in TARGETS}
    out: dict[str, float] = {}
    for a in SMOOTHING_ALPHAS:
        out[f"insample_mi_alpha{a:g}"] = O.frequency_mi(ends, alpha=a)
        if a == 0.0:
            continue
        per = np.zeros(comp.m)
        for k in range(plan.n_outer):
            tr, te = plan.outer_split(k)
            train = {z: ends[z][tr] for z in TARGETS}
            for z in TARGETS:
                p0 = O.smoothed_posterior(train, ends[z][te], alpha=a)
                contrib, _ = O.information_score(p0, np.full(len(te), z), eps=1e-12)
                per[te] += 0.5 * contrib
        out[f"heldout_score_alpha{a:g}"] = float(per.mean())
    return out


def run_problem(comp: Comparison, problem: str, h: int, cands: list[Candidate],
                plan: FoldPlan, cfg: TrainConfig, keep_curves: bool = False) -> dict[str, Any]:
    """Cross-fit every candidate, select by inner log loss, score out of fold."""
    ds = build_dataset(comp, problem, h)
    direct = all(c.objective == "nwj" for c in cands if c.family != "constant")
    res = crossfit(ds, cands, plan, cfg,
                   selection_objective="nwj" if direct else "bce", keep_curves=keep_curves)
    scorer = _info_from_logits if problem in ("endpoint", "pathinfo") else _cost_from_logits
    out: dict[str, Any] = {
        "comparison": comp.key, "profile": comp.profile, "rho": comp.rho, "budget": comp.budget,
        "problem": problem, "horizon": h, "m": comp.m,
        "selected": scorer(res.selected_logit, ds, comp.m),
        "by_candidate": {n: scorer(lg, ds, comp.m) for n, lg in res.oof_logit.items()},
        "selected_by_fold": res.selected_by_fold,
        "selection": res.summary(),
    }
    if keep_curves:
        out["curves"] = {f.fold: f.curves for f in res.folds}
    if problem == "endpoint":
        out["frequency"] = frequency_endpoint(comp, h, plan)
    return out
