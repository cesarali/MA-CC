"""Mean-field (noisy-voter) kernel fitted to the observed round transitions.

The analytical blackboard reference (BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES.md,
section 7.1) posits a homogeneous one-round update with

    E[x' | x, u] = p_u + (x - p_u) * (1 - gamma_u / N)^L

and branch kernels Q_u(m|n) satisfying a detailed-balance-like ratio whose
stationary law is Binomial(N, p_u).  The concrete kernel with both properties is
the mean-field noisy voter: at each round every agent independently keeps its
vote with probability lambda_u, otherwise redraws it as Bernoulli(p_u).  Hence

    m | n  ~  Binomial(n, lambda + (1-lambda) p)  +  Binomial(N - n, (1-lambda) p)

which is a convolution of two binomials, so the full 25x25 transition matrix is
cheap to build and the two parameters are fitted by exact maximum likelihood.

Interpretation.  ``p`` is the attractor the round-dynamics pull toward and
``lambda`` is per-round vote retention; ``gamma = N (1 - lambda^{1/L})`` is the
implied per-update compliance rate of the reference's parameterization.

This is a *mean-field* model in the strict sense: agents are exchangeable and
interact only through the population fraction.  Fitting it is therefore also a
test of whether the board dynamics are well described at that level of
resolution, which the reference explicitly leaves open for board mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom

from .data import POPULATION as N


def transition_matrix(lam: float, p: float, n_pop: int = N) -> np.ndarray:
    """T[n, m] = P(m target voters next round | n now) under the noisy-voter kernel."""
    a = lam + (1.0 - lam) * p          # a currently-target agent stays target
    b = (1.0 - lam) * p                # a currently-non-target agent becomes target
    T = np.zeros((n_pop + 1, n_pop + 1))
    for n in range(n_pop + 1):
        keep = binom.pmf(np.arange(n + 1), n, a)
        gain = binom.pmf(np.arange(n_pop - n + 1), n_pop - n, b)
        T[n, : n_pop + 1] = np.convolve(keep, gain)[: n_pop + 1]
    return T


def _nll(par: np.ndarray, counts: np.ndarray) -> float:
    lam = 1.0 / (1.0 + np.exp(-par[0]))
    p = 1.0 / (1.0 + np.exp(-par[1]))
    T = transition_matrix(lam, p)
    return float(-(counts * np.log(np.maximum(T, 1e-300))).sum())


@dataclass
class Fit:
    p: float
    lam: float
    gamma: float
    nll_per_obs: float
    n: int


def fit_kernel(n_now: np.ndarray, n_next: np.ndarray, l_updates: int = N) -> Fit | None:
    """Exact maximum likelihood for (lambda, p) from observed round transitions."""
    n_now = np.asarray(n_now, dtype=int).ravel()
    n_next = np.asarray(n_next, dtype=int).ravel()
    if len(n_now) < 15:
        return None
    counts = np.zeros((N + 1, N + 1))
    np.add.at(counts, (n_now, n_next), 1.0)
    best = None
    for start in ([1.0, 0.0], [2.0, 2.0], [0.5, -1.0], [0.0, 1.0]):
        r = minimize(_nll, start, args=(counts,), method="Nelder-Mead",
                     options=dict(maxiter=1500, fatol=1e-7, xatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    lam = 1.0 / (1.0 + np.exp(-best.x[0]))
    p = 1.0 / (1.0 + np.exp(-best.x[1]))
    gamma = N * (1.0 - lam ** (1.0 / l_updates)) if 0 < lam < 1 else float("nan")
    return Fit(p=p, lam=lam, gamma=gamma, nll_per_obs=best.fun / len(n_now), n=len(n_now))


def fit_all(comparisons: dict, coordinate: int = 0) -> pd.DataFrame:
    """Fit the passive and active kernels for every comparison and arm.

    The silent arm gives one kernel.  A controlled arm gives two, because the
    gate is randomized: rounds with U=0 should follow the *same* kernel as the
    silent arm if the mean-field description is adequate, which is a falsifiable
    internal check rather than a fitted quantity.
    """
    rows = []
    for key, c in comparisons.items():
        f = fit_kernel(c.silent[:, :-1, coordinate], c.silent[:, 1:, coordinate])
        if f:
            rows.append(dict(comparison=key, profile=c.profile, rho=c.rho, budget=c.budget,
                             arm="silent", u=-1, **f.__dict__))
        for z, arm in ((0, "t0"), (2, "t2")):
            arr, u = c.controlled[z], c.u[z]
            now, nxt = arr[:, :-1, coordinate], arr[:, 1:, coordinate]
            for uu in (0, 1):
                m = u == uu
                f = fit_kernel(now[m], nxt[m])
                if f:
                    rows.append(dict(comparison=key, profile=c.profile, rho=c.rho,
                                     budget=c.budget, arm=arm, u=uu, **f.__dict__))
    return pd.DataFrame(rows)


def simulate(fits: pd.DataFrame, comparisons: dict, key: str, arm: str,
             y0: np.ndarray, horizon: int, activation: float,
             seed: int = 0, n_paths: int = 2000) -> np.ndarray:
    """Forward-simulate the fitted kernel from observed initial counts.

    Each round draws the gate with probability ``activation`` and applies the
    corresponding fitted kernel.  Returns the mean target fraction per horizon,
    which is what the data figures plot.
    """
    f = fits[(fits.comparison == key) & (fits.arm == arm)]
    # the silent arm stores its single passive kernel under u = -1
    k = {(0 if int(r.u) < 0 else int(r.u)): transition_matrix(r.lam, r.p) for _, r in f.iterrows()}
    if not k:
        return np.full(horizon + 1, np.nan)
    rng = np.random.default_rng(seed)
    n = rng.choice(np.asarray(y0, dtype=int), size=n_paths)
    out = np.zeros(horizon + 1)
    out[0] = n.mean() / N
    for t in range(1, horizon + 1):
        act = rng.random(n_paths) < activation if 1 in k else np.zeros(n_paths, bool)
        nxt = np.empty_like(n)
        for uu in (0, 1):
            sel = act if uu == 1 else ~act
            if not sel.any() or uu not in k:
                if sel.any():
                    nxt[sel] = n[sel]
                continue
            probs = k[uu][n[sel]]
            cum = probs.cumsum(axis=1)
            r = rng.random(sel.sum())[:, None]
            nxt[sel] = (r > cum).sum(axis=1)
        n = nxt
        out[t] = n.mean() / N
    return out


# --------------------------------------------------------------- q-voter family
# The noisy-voter kernel above assumes the redraw probability is a constant p, so
# the drift (1-lambda)(p - x) is linear in x.  The observed passive drift is not
# linear: it is hump shaped, rising out of mid-range and vanishing at both ends,
# which is the signature of nonlinear reinforcement.  The q-voter family supplies
# exactly that.  Writing the per-round adoption probability as
#
#     logit pi(x) = alpha + q * logit(x)
#
# gives a one-parameter nest of the models of interest:
#
#     q = 0   pi(x) = sigmoid(alpha) is constant  -> the noisy voter fitted above
#     q = 1   pi(x) = x under alpha = 0           -> the linear (standard) voter
#     q > 1   conformity: an agent is disproportionately swayed by a majority
#     q < 1   anti-conformity
#
# The interpretation of q is the effective number of concordant observations an
# agent needs; the board reading allowance is 12, which bounds what is plausible.
# x is smoothed as (n + 1/2)/(N + 1) so the logit is finite at n = 0 and n = N.

def _pi_of_n(alpha: float, q: float, n_pop: int = N) -> np.ndarray:
    """Adoption probability per count.  alpha and q are clipped, and pi is kept
    strictly inside (0, 1): a binomial with p exactly 0 or 1 overflows the
    library routine, and the optimizer does visit those corners."""
    alpha = float(np.clip(alpha, -25.0, 25.0))
    q = float(np.clip(q, -5.0, 15.0))
    x = (np.arange(n_pop + 1) + 0.5) / (n_pop + 1.0)
    z = np.clip(alpha + q * np.log(x / (1.0 - x)), -30.0, 30.0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1.0 - 1e-9)


def qvoter_matrix(lam: float, alpha: float, q: float, n_pop: int = N) -> np.ndarray:
    """T[n, m] for the q-voter kernel: keep w.p. lambda, else adopt w.p. pi(x)."""
    pi = _pi_of_n(alpha, q, n_pop)
    lam = float(np.clip(lam, 0.0, 1.0 - 1e-9))
    T = np.zeros((n_pop + 1, n_pop + 1))
    for n in range(n_pop + 1):
        a = np.clip(lam + (1.0 - lam) * pi[n], 1e-9, 1.0 - 1e-9)
        b = np.clip((1.0 - lam) * pi[n], 1e-9, 1.0 - 1e-9)
        keep = binom.pmf(np.arange(n + 1), n, a)
        gain = binom.pmf(np.arange(n_pop - n + 1), n_pop - n, b)
        T[n, : n_pop + 1] = np.convolve(keep, gain)[: n_pop + 1]
    return T


def _qnll(par: np.ndarray, counts: np.ndarray, q_fixed: float | None) -> float:
    lam = 1.0 / (1.0 + np.exp(-par[0]))
    alpha = par[1]
    q = par[2] if q_fixed is None else q_fixed
    T = qvoter_matrix(lam, alpha, q)
    return float(-(counts * np.log(np.maximum(T, 1e-300))).sum())


def fit_qvoter(n_now: np.ndarray, n_next: np.ndarray,
               q_fixed: float | None = None) -> dict | None:
    """Maximum likelihood for (lambda, alpha, q).  ``q_fixed`` pins q for nested tests."""
    n_now = np.asarray(n_now, dtype=int).ravel()
    n_next = np.asarray(n_next, dtype=int).ravel()
    if len(n_now) < 30:
        return None
    counts = np.zeros((N + 1, N + 1))
    np.add.at(counts, (n_now, n_next), 1.0)
    best = None
    starts = ([0.5, 0.0, 1.0], [1.0, 1.0, 2.0], [0.0, -1.0, 0.5], [1.5, 0.5, 4.0])
    for s in starts:
        p0 = np.array(s if q_fixed is None else s[:2])
        r = minimize(_qnll, p0, args=(counts, q_fixed), method="Nelder-Mead",
                     options=dict(maxiter=4000, fatol=1e-7, xatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    lam = 1.0 / (1.0 + np.exp(-best.x[0]))
    alpha = float(best.x[1])
    q = float(best.x[2]) if q_fixed is None else float(q_fixed)
    k = 3 if q_fixed is None else 2
    return {"lam": lam, "alpha": alpha, "q": q, "nll": float(best.fun),
            "aic": 2 * k + 2 * float(best.fun), "k": k, "n": int(len(n_now))}


def drift_curve(lam: float, alpha: float, q: float, n_pop: int = N) -> np.ndarray:
    """Mean one-round change in the target count, E[m - n | n], under the fit."""
    pi = _pi_of_n(alpha, q, n_pop)
    n = np.arange(n_pop + 1)
    return (1.0 - lam) * (pi * n_pop - n)


def simulate_qvoter(fits: pd.DataFrame, key: str, arm: str, y0: np.ndarray,
                    horizon: int, activation: float, seed: int = 0,
                    n_paths: int = 4000, prefix: str = "M2") -> np.ndarray:
    """Forward-simulate the fitted q-voter kernels, drawing the gate each round."""
    f = fits[(fits.comparison == key) & (fits.arm == arm)]
    k = {}
    for _, r in f.iterrows():
        u = 0 if int(r.u) < 0 else int(r.u)
        k[u] = qvoter_matrix(r[f"{prefix}_lam"], r[f"{prefix}_alpha"], r[f"{prefix}_q"])
    if not k:
        return np.full(horizon + 1, np.nan)
    rng = np.random.default_rng(seed)
    n = rng.choice(np.asarray(y0, dtype=int), size=n_paths)
    out = np.zeros(horizon + 1); out[0] = n.mean() / N
    for t in range(1, horizon + 1):
        act = rng.random(n_paths) < activation if 1 in k else np.zeros(n_paths, bool)
        nxt = n.copy()
        for uu in (0, 1):
            sel = act if uu == 1 else ~act
            if not sel.any() or uu not in k:
                continue
            cum = k[uu][n[sel]].cumsum(axis=1)
            r = rng.random(sel.sum())[:, None]
            nxt[sel] = (r > cum).sum(axis=1)
        n = nxt; out[t] = n.mean() / N
    return out


# ------------------------------------------------- trajectory-matched fitting
# One-step maximum likelihood cannot separate lambda (vote retention) from q
# (social coupling): both govern how fast x moves in a single round, and the
# likelihood of a single transition is nearly flat along their trade-off.  The
# symptom is that lambda collapses from 0.61 to 0.01 when q is freed while the
# one-step fit improves and the 15-round trajectory gets worse.
#
# Matching the trajectory identifies them, because they leave different
# signatures over many rounds: lambda controls how much of the population's
# spread survives round to round, q controls where the mean is pulled.  We
# therefore match both the mean AND the across-initialization standard deviation
# of the target count at every horizon.
#
# The distribution over counts is propagated exactly rather than simulated:
# v_{t+1} = v_t T, with T the gate-averaged kernel.  No Monte Carlo noise enters
# the objective, so the optimizer sees a smooth surface.

def propagate(v0: np.ndarray, kernels: dict[int, np.ndarray], activation: float,
              horizon: int) -> np.ndarray:
    """Exact law of the target count at each horizon, as a (horizon+1, N+1) array."""
    T = kernels[0] if 1 not in kernels else (1 - activation) * kernels[0] + activation * kernels[1]
    out = np.zeros((horizon + 1, len(v0)))
    out[0] = v0
    for t in range(1, horizon + 1):
        out[t] = out[t - 1] @ T
    return out


def _moments(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(dist.shape[1])
    mean = dist @ n
    var = dist @ (n ** 2) - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0))


def empirical_start(counts: np.ndarray, n_pop: int = N) -> np.ndarray:
    v = np.bincount(np.asarray(counts, dtype=int).ravel(), minlength=n_pop + 1).astype(float)
    return v / v.sum()


def fit_trajectory(v0: np.ndarray, obs_mean: np.ndarray, obs_sd: np.ndarray,
                   activation: float, passive: np.ndarray | None = None,
                   sd_weight: float = 1.0, start: tuple = (0.5, 0.5, 1.0)) -> dict:
    """Fit (lambda, alpha, q) of one kernel by matching the trajectory moments.

    ``passive`` is the gate-OFF kernel, held fixed; when it is None the fitted
    kernel is itself the passive one (the silent arm).
    """
    horizon = len(obs_mean) - 1

    def obj(par):
        lam = 1.0 / (1.0 + np.exp(-par[0]))
        T = qvoter_matrix(lam, par[1], par[2])
        k = {0: passive, 1: T} if passive is not None else {0: T}
        a = activation if passive is not None else 0.0
        m, s = _moments(propagate(v0, k, a, horizon))
        return float(((m - obs_mean) ** 2).sum() + sd_weight * ((s - obs_sd) ** 2).sum())

    best = None
    for s0 in (start, (1.0, 0.0, 1.0), (0.0, 1.0, 0.5), (1.5, -0.5, 2.0)):
        r = minimize(obj, np.array(s0), method="Nelder-Mead",
                     options=dict(maxiter=3000, fatol=1e-9, xatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    lam = 1.0 / (1.0 + np.exp(-best.x[0]))
    return {"lam": lam, "alpha": float(best.x[1]), "q": float(best.x[2]),
            "objective": float(best.fun)}


# --------------------------------------------------- combined (composite) objective
# Neither criterion alone identifies the kernel.  One-step likelihood pins the
# SHAPE of pi(x) but cannot separate lambda from q, because both set how far x
# moves in one round.  Trajectory matching separates them in principle but leaves
# the shape free, so the optimizer runs to boundary solutions (q at its clip).
#
# We therefore add the two as proper log-likelihoods rather than weighting them
# by hand.  The observed mean count at horizon t has standard error sd_t/sqrt(m)
# and the observed spread has standard error roughly sd_t/sqrt(2(m-1)), so each
# trajectory moment contributes a Gaussian term with a known scale and the
# relative weight is fixed by the data, not chosen.
#
# This is a COMPOSITE likelihood: the same observations enter both terms, so it
# is a consistent estimating equation but its curvature is not a valid
# information matrix.  Uncertainty must come from resampling initializations.

def onestep_nll(counts: np.ndarray, lam: float, alpha: float, q: float) -> float:
    T = qvoter_matrix(lam, alpha, q)
    return float(-(counts * np.log(np.maximum(T, 1e-300))).sum())


def trajectory_nll(v0: np.ndarray, kernels: dict, activation: float,
                   obs_mean: np.ndarray, obs_sd: np.ndarray, m_init: int) -> float:
    horizon = len(obs_mean) - 1
    mm, ss = _moments(propagate(v0, kernels, activation, horizon))
    se_m = np.maximum(obs_sd, 1e-3) / np.sqrt(m_init)
    se_s = np.maximum(obs_sd, 1e-3) / np.sqrt(2.0 * max(m_init - 1, 1))
    t = slice(1, horizon + 1)          # t=0 is the shared initial condition
    return 0.5 * float((((mm[t] - obs_mean[t]) / se_m[t]) ** 2).sum()
                       + (((ss[t] - obs_sd[t]) / se_s[t]) ** 2).sum())


def fit_combined(counts: np.ndarray, v0: np.ndarray, obs_mean: np.ndarray,
                 obs_sd: np.ndarray, m_init: int, activation: float = 0.0,
                 passive: np.ndarray | None = None, q_fixed: float | None = None,
                 q_max: float = 12.0) -> dict:
    """Minimise onestep_nll + trajectory_nll.

    ``q_max`` is 12 because an agent reads at most 12 board messages, so an
    effective coupling above that has no mechanistic reading; a fit that lands
    on the bound is reported rather than silently accepted.
    """
    def unpack(par):
        lam = 1.0 / (1.0 + np.exp(-par[0]))
        alpha = float(np.clip(par[1], -12.0, 12.0))
        q = q_fixed if q_fixed is not None else float(np.clip(par[2], -2.0, q_max))
        return lam, alpha, q

    def obj(par):
        lam, alpha, q = unpack(par)
        T = qvoter_matrix(lam, alpha, q)
        k = {0: passive, 1: T} if passive is not None else {0: T}
        a = activation if passive is not None else 0.0
        return (onestep_nll(counts, lam, alpha, q)
                + trajectory_nll(v0, k, a, obs_mean, obs_sd, m_init))

    best = None
    starts = [(0.5, 0.5, 1.0), (1.0, 0.0, 0.5), (0.0, 1.0, 2.0), (1.5, -0.5, 0.2)]
    for s in starts:
        p0 = np.array(s if q_fixed is None else s[:2])
        r = minimize(obj, p0, method="Nelder-Mead",
                     options=dict(maxiter=4000, fatol=1e-8, xatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    lam, alpha, q = unpack(best.x if q_fixed is None else np.append(best.x, 0.0))
    T = qvoter_matrix(lam, alpha, q)
    k = {0: passive, 1: T} if passive is not None else {0: T}
    a = activation if passive is not None else 0.0
    return {"lam": lam, "alpha": alpha, "q": q, "total": float(best.fun),
            "onestep_nll": onestep_nll(counts, lam, alpha, q),
            "traj_nll": trajectory_nll(v0, k, a, obs_mean, obs_sd, m_init),
            "at_q_bound": bool(abs(q - q_max) < 1e-3 or abs(q + 2.0) < 1e-3)}


def counts_matrix(n_now: np.ndarray, n_next: np.ndarray, n_pop: int = N) -> np.ndarray:
    c = np.zeros((n_pop + 1, n_pop + 1))
    np.add.at(c, (np.asarray(n_now, int).ravel(), np.asarray(n_next, int).ravel()), 1.0)
    return c
