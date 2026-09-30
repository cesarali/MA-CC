"""Statistical objectives: information log-score, NWJ / DV variational KL, frequency MI.

All logarithms are natural (nats).  Exponentials and log-mean-exp use float64.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp

from .data import POPULATION, TARGETS, TARGET_WEIGHTS


# ----------------------------------------------------------------------------- endpoint
def clip_probability(p: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """Clip binary probabilities into [eps, 1-eps] with renormalization; return saturation fraction."""
    p = np.asarray(p, dtype=np.float64)
    saturated = float(np.mean((p < eps) | (p > 1 - eps))) if p.size else 0.0
    q = np.clip(p, eps, 1 - eps)
    # binary: clipping p and 1-p symmetrically already renormalizes to one
    return q, saturated


def information_score(prob_target0: np.ndarray, labels: np.ndarray, eps: float = 1e-6,
                      w0: float = TARGET_WEIGHTS[0]) -> tuple[np.ndarray, float]:
    """Per-example log(g(z|y)/w_z) and the saturation fraction.

    ``prob_target0`` is the model's P(Z=0 | y); ``labels`` in {0, 2}.  The mean of the
    returned array over held-out examples is the predictive information score
    L_I, a population lower bound on I(Z; Y_h) (tight for the true posterior).
    """
    p0, sat = clip_probability(prob_target0, eps)
    lab0 = np.asarray(labels) == 0
    g = np.where(lab0, p0, 1.0 - p0)
    w = np.where(lab0, w0, 1.0 - w0)
    return np.log(g) - np.log(w), sat


def log_loss(prob_target0: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
             eps: float = 1e-12) -> float:
    p0, _ = clip_probability(prob_target0, eps)
    lab0 = np.asarray(labels) == 0
    g = np.where(lab0, p0, 1.0 - p0)
    w = np.ones_like(g) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(-np.sum(w * np.log(g)) / np.sum(w))


def brier(prob_target0: np.ndarray, labels: np.ndarray) -> float:
    y = (np.asarray(labels) == 0).astype(np.float64)
    return float(np.mean((np.asarray(prob_target0) - y) ** 2))


# ---------------------------------------------------------------- frequency / smoothing
def endpoint_support(population: int = POPULATION) -> list[tuple[int, int, int]]:
    """All three-count vectors summing to ``population`` (325 of them for 24 agents)."""
    out = []
    for n0 in range(population + 1):
        for n1 in range(population + 1 - n0):
            out.append((n0, n1, population - n0 - n1))
    return out


_SUPPORT_CACHE: dict[int, tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int]]] = {}


def _support(population: int):
    if population not in _SUPPORT_CACHE:
        sup = endpoint_support(population)
        _SUPPORT_CACHE[population] = (sup, {s: i for i, s in enumerate(sup)})
    return _SUPPORT_CACHE[population]


def counts_to_support_index(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    population = int(counts[0].sum())
    _, index = _support(population)
    return np.array([index[tuple(int(v) for v in c)] for c in counts], dtype=int)


def smoothed_endpoint_pmf(counts: np.ndarray, alpha: float) -> np.ndarray:
    """p(y) = (c(y) + alpha/325) / (m + alpha): total pseudocount mass alpha."""
    counts = np.asarray(counts)
    sup, _ = _support(int(counts[0].sum()))
    idx = counts_to_support_index(counts)
    c = np.bincount(idx, minlength=len(sup)).astype(np.float64)
    m = len(idx)
    return (c + alpha / len(sup)) / (m + alpha)


def frequency_mi(endpoints_by_target: dict[int, np.ndarray], alpha: float = 0.0) -> float:
    """In-sample plug-in MI (nats) between target and endpoint from (smoothed) frequencies."""
    pz = {z: smoothed_endpoint_pmf(endpoints_by_target[z], alpha) for z in TARGETS}
    q = sum(TARGET_WEIGHTS[z] * pz[z] for z in TARGETS)
    mi = 0.0
    for z in TARGETS:
        mask = pz[z] > 0
        mi += TARGET_WEIGHTS[z] * float(np.sum(pz[z][mask] * (np.log(pz[z][mask]) - np.log(q[mask]))))
    return mi


def smoothed_posterior(train_by_target: dict[int, np.ndarray], test_counts: np.ndarray,
                       alpha: float) -> np.ndarray:
    """Bayes posterior P(Z=0 | y) from smoothed training pmfs, evaluated on test endpoints."""
    pz = {z: smoothed_endpoint_pmf(train_by_target[z], alpha) for z in TARGETS}
    idx = counts_to_support_index(test_counts)
    num = TARGET_WEIGHTS[0] * pz[0][idx]
    den = sum(TARGET_WEIGHTS[z] * pz[z][idx] for z in TARGETS)
    return num / den


# ------------------------------------------------------------------- variational KL
def cap_critic(f: np.ndarray, cap: float | None) -> np.ndarray:
    """Bounded critic f = M tanh(s / M); ``cap=None`` leaves the raw logit."""
    f = np.asarray(f, dtype=np.float64)
    if cap is None:
        return f
    return cap * np.tanh(f / cap)


def prior_corrected_logit(logit: np.ndarray, prior_controlled: float) -> np.ndarray:
    """log p_ctl/p_base = logit(d) + log((1-a)/a) for classifier trained with P(B=1)=a."""
    return np.asarray(logit, dtype=np.float64) + math.log((1 - prior_controlled) / prior_controlled)


def nwj_contributions(f_controlled: np.ndarray, f_silent: np.ndarray) -> np.ndarray:
    """Per-parent NWJ terms f(G_ctl) - exp f(G_base) + 1 (paired by parent)."""
    fc = np.asarray(f_controlled, dtype=np.float64)
    fs = np.asarray(f_silent, dtype=np.float64)
    return fc - np.exp(fs) + 1.0


def nwj_score(f_controlled: np.ndarray, f_silent: np.ndarray) -> float:
    return float(np.mean(nwj_contributions(f_controlled, f_silent)))


def dv_score(f_controlled: np.ndarray, f_silent: np.ndarray) -> float:
    """Donsker-Varadhan: mean f on controlled - log mean exp f on silent (stable log-mean-exp)."""
    fc = np.asarray(f_controlled, dtype=np.float64)
    fs = np.asarray(f_silent, dtype=np.float64)
    return float(np.mean(fc) - (logsumexp(fs) - math.log(len(fs))))


def plugin_score(f_controlled: np.ndarray) -> float:
    """Mean held-out log ratio on controlled paths; equals KL only if the ratio is exact."""
    return float(np.mean(np.asarray(f_controlled, dtype=np.float64)))


def baseline_tail_diagnostics(f_silent: np.ndarray) -> dict[str, float]:
    v = np.exp(np.asarray(f_silent, dtype=np.float64))
    s = v.sum()
    return {
        "max_weight_share": float(v.max() / s) if s > 0 else float("nan"),
        "ess": float(s ** 2 / np.sum(v ** 2)) if s > 0 else float("nan"),
        "n": int(len(v)),
    }


def exact_discrete_kl(p: np.ndarray, r: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    mask = p > 0
    if np.any(r[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(r[mask]))))


def exact_discrete_mi(p_by_target: dict[int, np.ndarray]) -> float:
    q = sum(TARGET_WEIGHTS[z] * np.asarray(p_by_target[z], dtype=np.float64) for z in p_by_target)
    return float(sum(TARGET_WEIGHTS[z] * exact_discrete_kl(p_by_target[z], q) for z in p_by_target))
