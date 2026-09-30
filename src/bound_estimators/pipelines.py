"""Problem builders and per-job evaluation.

Four problem types are built from one ``Comparison`` at one horizon ``h``:

* ``endpoint``  target 0 vs target 2 from the endpoint Y_h only -> information score
                (lower bound on I(Z; Y_h)); plus frequency / smoothing comparisons.
* ``cost0`` / ``cost2``  controlled (target z) vs silent on the vote path (Y_0..Y_h)
                -> density-ratio critic scored with NWJ / DV (lower bounds on KL(p_z||p_base)).
* ``pathinfo``  target 0 vs target 2 from the full path -> lower bound on I(Z; Gamma).
* ``mixture``   target mixture Q vs silent -> lower bound on KL(Q || p_base).

Each job returns per-parent held-out contributions so uncertainty can be computed by
resampling parents (the only independent unit).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from . import objectives as O
from .crossfit import Candidate, Dataset, crossfit
from .data import Comparison, TARGETS, TARGET_WEIGHTS
from .folds import FoldPlan
from .models import TrainConfig

CAPS = (2.0, 5.0, 10.0)
CLIP_EPS = (1e-6, 1e-4, 1e-8)
SMOOTHING_ALPHAS = (0.0, 1.0, 10.0, 100.0)


def build_dataset(comp: Comparison, problem: str, horizon: int) -> Dataset:
    m, T = comp.m, horizon + 1
    par = np.arange(m)
    if problem == "endpoint":
        X = np.concatenate([comp.controlled[0][:, horizon], comp.controlled[2][:, horizon]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("endpoint", X, y, np.ones(2 * m), np.concatenate([par, par]), horizon)
    if problem in ("cost0", "cost2"):
        z = 0 if problem == "cost0" else 2
        X = np.concatenate([comp.controlled[z][:, :T], comp.silent[:, :T]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("path", X, y, np.ones(2 * m), np.concatenate([par, par]), horizon)
    if problem == "pathinfo":
        X = np.concatenate([comp.controlled[0][:, :T], comp.controlled[2][:, :T]])
        y = np.concatenate([np.ones(m), np.zeros(m)])
        return Dataset("path", X, y, np.ones(2 * m), np.concatenate([par, par]), horizon)
    if problem == "mixture":
        X = np.concatenate([comp.controlled[0][:, :T], comp.controlled[2][:, :T], comp.silent[:, :T]])
        y = np.concatenate([np.ones(2 * m), np.zeros(m)])
        w = np.concatenate([np.full(2 * m, 0.5), np.ones(m)])
        return Dataset("path", X, y, w, np.concatenate([par, par, par]), horizon)
    raise ValueError(problem)


def _info_from_logits(logit: np.ndarray, ds: Dataset, m: int) -> dict[str, Any]:
    """Information score per parent from P(class 1 | x) logits, at several clip levels."""
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = np.where(ds.y == 1, 0, 2)  # class 1 <-> target 0
    out: dict[str, Any] = {}
    for eps in CLIP_EPS:
        contrib, sat = O.information_score(prob, labels, eps=eps)
        per_parent = np.zeros(m)
        np.add.at(per_parent, ds.parent, 0.5 * contrib)   # w_z = 1/2 per target
        out[f"per_parent_eps{eps:g}"] = per_parent
        out[f"saturation_eps{eps:g}"] = sat
    out["score"] = float(out["per_parent_eps1e-06"].mean())
    out["log_loss"] = O.log_loss(prob, labels)
    out["brier"] = O.brier(prob, labels)
    out["accuracy"] = float(np.mean((prob > 0.5) == (labels == 0)))
    return out


def _cost_from_logits(logit: np.ndarray, ds: Dataset, m: int) -> dict[str, Any]:
    """NWJ / DV / plug-in scores per cap from controlled-vs-silent logits (paired by parent)."""
    ctl, sil = ds.y == 1, ds.y == 0
    out: dict[str, Any] = {}
    # per-parent controlled critic (mean over controlled examples of the parent) and silent critic
    for cap in (None, *CAPS):
        f = O.cap_critic(logit, cap)
        fc = np.zeros(m); wc = np.zeros(m)
        np.add.at(fc, ds.parent[ctl], f[ctl] * ds.w[ctl]); np.add.at(wc, ds.parent[ctl], ds.w[ctl])
        fc = fc / wc
        fs = np.zeros(m); fs[ds.parent[sil]] = f[sil]
        tag = "raw" if cap is None else f"cap{cap:g}"
        out[f"per_parent_nwj_{tag}"] = O.nwj_contributions(fc, fs)
        out[f"nwj_{tag}"] = float(np.mean(out[f"per_parent_nwj_{tag}"]))
        out[f"dv_{tag}"] = O.dv_score(fc, fs)
        out[f"plugin_{tag}"] = O.plugin_score(fc)
        out[f"tail_{tag}"] = O.baseline_tail_diagnostics(fs)
        out[f"per_parent_fc_{tag}"] = fc
        out[f"per_parent_fs_{tag}"] = fs
    out["score"] = out["nwj_raw"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    out["log_loss"] = O.log_loss(prob, np.where(ds.y == 1, 0, 2), ds.w)
    out["accuracy"] = float(np.average((prob > 0.5) == (ds.y == 1), weights=ds.w))
    return out


def frequency_comparisons(comp: Comparison, horizon: int, plan: FoldPlan) -> dict[str, Any]:
    """In-sample (smoothed) frequency MI and held-out smoothed-posterior information scores."""
    ends = {z: comp.controlled[z][:, horizon] for z in TARGETS}
    out: dict[str, Any] = {}
    for a in SMOOTHING_ALPHAS:
        out[f"insample_mi_alpha{a:g}"] = O.frequency_mi(ends, alpha=a)
        if a == 0.0:
            continue
        per_parent = np.zeros(comp.m)
        for k in range(plan.n_outer):
            tr, te = plan.outer_split(k)
            train = {z: ends[z][tr] for z in TARGETS}
            for z in TARGETS:
                p0 = O.smoothed_posterior(train, ends[z][te], alpha=a)
                contrib, _ = O.information_score(p0, np.full(len(te), z), eps=1e-12)
                per_parent[te] += TARGET_WEIGHTS[z] * contrib
        out[f"heldout_score_alpha{a:g}"] = float(per_parent.mean())
        out[f"per_parent_heldout_alpha{a:g}"] = per_parent
    return out


def run_problem(comp: Comparison, problem: str, horizon: int, cands: list[Candidate],
                plan: FoldPlan, cfg: TrainConfig, keep_curves: bool = False) -> dict[str, Any]:
    ds = build_dataset(comp, problem, horizon)
    direct = all(c.objective == "nwj" for c in cands if c.family != "constant")
    res = crossfit(ds, cands, plan, cfg, selection_objective="nwj" if direct else "bce", keep_curves=keep_curves)
    scorer = _info_from_logits if problem in ("endpoint", "pathinfo") else _cost_from_logits
    out: dict[str, Any] = {
        "comparison": comp.key, "q": comp.q, "rho": comp.rho, "schedule": comp.schedule,
        "budget": comp.budget, "problem": problem, "horizon": horizon, "m": comp.m,
        "repeat": plan.repeat, "selection": res.summary(),
        "selected": scorer(res.selected_logit, ds, comp.m),
        "by_candidate": {name: scorer(lg, ds, comp.m) for name, lg in res.oof_logit.items()},
        "oof_logit": {name: lg for name, lg in res.oof_logit.items()},
        "labels": ds.y, "parent": ds.parent,
    }
    if keep_curves:
        out["curves"] = {f.fold: f.curves for f in res.folds}
    if problem == "endpoint":
        out["frequency"] = frequency_comparisons(comp, horizon, plan)
    return out
