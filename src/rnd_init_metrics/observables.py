"""Closed-form observables from BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES.md.

Everything here is an exact plug-in calculation on the vote counts: no model is
fitted.  Each function returns per-initialization contributions wherever the
quantity is an average over initializations, so that uncertainty can be obtained
by resampling whole initializations (the only independent unit).

Section numbers in the docstrings refer to the reference document.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import hypergeom

from .data import POPULATION, TARGETS, Comparison

LOG2 = np.log(2.0)


# ----------------------------------------------------------------- section 3-4
def fractions(counts: np.ndarray) -> np.ndarray:
    """(..., 3) counts -> (..., 3) fractions x_a."""
    return np.asarray(counts, dtype=float) / POPULATION


def paired_gain(comp: Comparison, a: int, h: int) -> dict[int, np.ndarray]:
    """[P] G_H^{(a)}: per-initialization x_a(h) under each target minus silence."""
    base = fractions(comp.silent[:, h, a])
    return {z: fractions(comp.controlled[z][:, h, a]) - base for z in TARGETS}


def md_decomposition(comp: Comparison, a: int, h: int) -> dict[str, np.ndarray]:
    """[C] Shared component M and directional half-contrast D, per initialization.

    M = (G+ + G-)/2 and D = (G+ - G-)/2, so G+ = M+D and G- = M-D.  The full
    effect of switching the requested target is 2D; both are exported because the
    two conventions differ by a factor of two.
    """
    g = paired_gain(comp, a, h)
    return {"g_plus": g[0], "g_minus": g[2],
            "M": 0.5 * (g[0] + g[2]), "D": 0.5 * (g[0] - g[2]),
            "switch": g[0] - g[2]}


def state_matched_susceptibility(comp: Comparison, a: int, z: int) -> dict[str, Any]:
    """[R] chi_a(n) = E[dx_a | U=1, n_a=n] - E[dx_a | U=0, n_a=n], pooled over rounds.

    Rounds 1..14 carry a randomized gate; round 0 has none and is excluded.  The
    occupancy summary weights states by their visit frequency and reports the
    support mass over states where both arms were actually observed.
    """
    x = fractions(comp.controlled[z])                      # (m, H+1, 3)
    u = comp.u[z]                                          # (m, N_ROUNDS)
    n_t = comp.controlled[z][:, :-1, a]                    # count before the round
    dx = x[:, 1:, a] - x[:, :-1, a]                        # change across the round
    ok = ~np.isnan(u)
    n_flat, u_flat, d_flat = n_t[ok].astype(int), u[ok].astype(int), dx[ok]
    states, chi, w, both = [], [], [], []
    for n in np.unique(n_flat):
        sel = n_flat == n
        d1, d0 = d_flat[sel & (u_flat == 1)], d_flat[sel & (u_flat == 0)]
        states.append(int(n)); w.append(int(sel.sum())); both.append(bool(len(d1) and len(d0)))
        chi.append(float(d1.mean() - d0.mean()) if len(d1) and len(d0) else np.nan)
    states, chi, w, both = map(np.asarray, (states, chi, w, both))
    mass = w[both].sum() / max(w.sum(), 1)
    wn = w[both] / max(w[both].sum(), 1)
    return {"states": states, "chi": chi, "visits": w, "both_arms": both,
            "chi_bar": float((wn * chi[both]).sum()) if both.any() else np.nan,
            "support_mass": float(mass),
            "unconditioned": float(d_flat[u_flat == 1].mean() - d_flat[u_flat == 0].mean()),
            "activation_rate": float(u_flat.mean())}


def ipw_causal_response(comp: Comparison, a: int, z: int, lags: tuple[int, ...] = (1, 2, 5)) -> dict[str, Any]:
    """[R] Inverse-probability-weighted lag-l causal response tau_{a,l}.

    W = U/e - (1-U)/(1-e) with e the logged activation probability.  This study
    randomizes the gate with e bounded away from 0 and 1, so the estimator is
    available here; it is not available for a deterministic always-policy.
    Contributions are summed per initialization to keep the resampling unit intact.
    """
    x = fractions(comp.controlled[z])
    u, e = comp.u[z], comp.e[z]
    ok = ~np.isnan(u) & ~np.isnan(e)
    w = np.where(ok, u / np.where(ok, e, 1.0) - (1 - u) / np.where(ok, 1 - e, 1.0), 0.0)
    out: dict[str, Any] = {"max_abs_weight": float(np.abs(w[ok]).max()),
                           "e_min": float(np.nanmin(e)), "e_max": float(np.nanmax(e))}
    m, T = u.shape
    for lag in lags:
        num = np.zeros(m); cnt = np.zeros(m)
        for t in range(T):
            if t + lag > x.shape[1] - 1:
                break
            sel = ok[:, t]
            num[sel] += w[sel, t] * (x[sel, t + lag, a] - x[sel, t, a])
            cnt[sel] += 1
        per_init = np.divide(num, cnt, out=np.zeros_like(num), where=cnt > 0)
        out[f"tau_lag{lag}"] = float(num.sum() / max(cnt.sum(), 1))
        out[f"per_init_lag{lag}"] = per_init
        out[f"n_lag{lag}"] = int(cnt.sum())
    return out


def available_susceptibility(comp: Comparison, a: int, z: int) -> dict[str, float]:
    """[R] Row-ratio and cell-ratio normalizations by the not-yet-supporting fraction.

    Saturated rows (x_a = 1) are excluded from both numerator and denominator, as
    the reference requires; the two ratios are different quantities and both are
    reported.
    """
    x = fractions(comp.controlled[z])
    u, e = comp.u[z], comp.e[z]
    ok = ~np.isnan(u) & ~np.isnan(e)
    w = np.where(ok, u / np.where(ok, e, 1.0) - (1 - u) / np.where(ok, 1 - e, 1.0), 0.0)
    xt = x[:, :-1, a]
    dx = x[:, 1:, a] - xt
    keep = ok & (xt < 1.0)
    denom = 1.0 - xt[keep]
    return {"row_ratio_mean": float(np.mean(w[keep] * dx[keep] / denom)),
            "cell_ratio": float(np.sum(w[keep] * dx[keep]) / np.sum(denom)),
            "n_rows": int(keep.sum()), "n_saturated": int((ok & (xt >= 1.0)).sum())}


# ------------------------------------------------------------------- section 8
def posting_cost(comp: Comparison, z: int, h: int) -> np.ndarray:
    """[P] C_H = sum of realized controller posts over rounds 1..h, per initialization."""
    return comp.posts[z][:, :h].sum(axis=1)


def sensing_cost(comp: Comparison, z: int, h: int) -> np.ndarray:
    """Sampled votes read by the controller over rounds 1..h, per initialization.

    The sensor reads on every round with a gate, whether or not it posts, so this
    is charged to both arms and is not a sensing-policy-only expense.
    """
    s = comp.sensor[z][:, :h, :]
    per_round = np.where(np.isnan(s).all(axis=2), 0.0, np.nansum(s, axis=2))
    return per_round.sum(axis=1)


def resource(comp: Comparison, z: int, h: int, lam: float) -> np.ndarray:
    """[P/C] R_H = posts + lambda * sensed votes.  lambda is a declared valuation."""
    return posting_cost(comp, z, h) + lam * sensing_cost(comp, z, h)


def budget_efficiency(comp: Comparison, a: int, h: int) -> dict[str, float]:
    """[P] eta_budget = N * G_H / C_H, a ratio of means on the same initializations."""
    g = paired_gain(comp, a, h)
    out = {}
    for z in TARGETS:
        c = posting_cost(comp, z, h).mean()
        out[f"eta_budget_t{z}"] = float(POPULATION * g[z].mean() / c) if c > 0 else np.nan
        out[f"posts_t{z}"] = float(c)
    return out


# ------------------------------------------------------------------- section 9
def command_channel(comp: Comparison, h: int, alpha: float = 0.0) -> dict[str, Any]:
    """[C] The single-agent command channel p_h(v|z) = E[x_v(h) | Z=z].

    Marginalizing a uniformly chosen agent is exact from the vote fractions, so no
    classifier is needed and none is used.  alpha adds pseudocount mass at the
    initialization level as a regularization sensitivity, not as extra data.
    """
    m = comp.m
    rows = {z: fractions(comp.controlled[z][:, h, :]).mean(axis=0) for z in TARGETS}
    if alpha > 0:
        rows = {z: (m * r + alpha / 3.0) / (m + alpha) for z, r in rows.items()}
    bar = 0.5 * (rows[0] + rows[2])
    mi = 0.0
    for z in TARGETS:
        p = rows[z]
        nz = (p > 0) & (bar > 0)
        mi += 0.5 * float(np.sum(p[nz] * np.log2(p[nz] / bar[nz])))
    base = fractions(comp.silent[:, h, :]).mean(axis=0)
    f = 0.5 * (rows[0][0] + rows[2][2])
    f_base = 0.5 * (base[0] + base[2])
    return {"p_v_given_0": rows[0], "p_v_given_2": rows[2], "p_bar": bar, "p_base": base,
            "mi_bits": mi, "eta_command": mi,          # H(Z) = 1 bit, so these coincide
            "following": float(f), "following_base": float(f_base),
            "following_gain": float(f - f_base)}


def command_per_init(comp: Comparison, h: int) -> dict[str, np.ndarray]:
    """Per-initialization channel rows, for resampling whole initializations."""
    return {f"row{z}_{v}": fractions(comp.controlled[z][:, h, v]) for z in TARGETS for v in range(3)}


def info_per_resource(comp: Comparison, h: int, lam: float) -> dict[str, float]:
    """[C] E_info = I(Z;V_h) / mean resource, in bits per post-equivalent."""
    ch = command_channel(comp, h)
    r = 0.5 * (resource(comp, 0, h, lam).mean() + resource(comp, 2, h, lam).mean())
    return {"resource_mean": float(r),
            "bits_per_resource": float(ch["mi_bits"] / r) if r > 0 else np.nan,
            "following_gain_per_resource": float(ch["following_gain"] / r) if r > 0 else np.nan}


# ------------------------------------------------------------------- section 6
def sensing_information(comp: Comparison, z: int) -> dict[str, float]:
    """[R] I(n_Z; S) with the known hypergeometric sampling kernel.

    The kernel P(S=s|n) is exact for uniform sampling without replacement, so only
    the occupancy law over n is estimated.  This removes sampling noise from the
    channel but does not make the nonlinear information estimate unbiased.
    """
    q_c = 12
    n_obs = comp.controlled[z][:, :-1, z].astype(int).ravel()
    occ = np.bincount(n_obs, minlength=POPULATION + 1).astype(float)
    occ /= occ.sum()
    s_grid = np.arange(q_c + 1)
    kernel = np.stack([hypergeom.pmf(s_grid, POPULATION, n, q_c) for n in range(POPULATION + 1)])
    joint = occ[:, None] * kernel
    ps = joint.sum(axis=0)
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log2(joint[nz] / (occ[:, None] * ps[None, :])[nz])))
    return {"I_scalar_sensing_bits": mi, "H_n_bits": float(-np.sum(occ[occ > 0] * np.log2(occ[occ > 0]))),
            "n_round_observations": int(n_obs.size)}


def _cmi_bits(label: np.ndarray, outcome: np.ndarray, cond: np.ndarray) -> tuple[float, float]:
    """Plug-in I(L;O|W) in bits and the conditional label entropy H(L|W) in bits."""
    mi = 0.0; hl = 0.0
    n = len(label)
    for c in np.unique(cond):
        s = cond == c
        pc = s.mean()
        lab, out = label[s], outcome[s]
        for l in np.unique(lab):
            pl = (lab == l).mean()
            if pl > 0:
                hl -= pc * pl * np.log2(pl)
        for l in np.unique(lab):
            pl = (lab == l).mean()
            for o in np.unique(out):
                po = (out == o).mean()
                plo = ((lab == l) & (out == o)).mean()
                if plo > 0:
                    mi += pc * plo * np.log2(plo / (pl * po))
    return float(mi), float(hl)


def activation_information(comp: Comparison, a: int, z: int, h: int) -> dict[str, float]:
    """[R] T_act = I(U_h; n_a(h) | n_a(h-1)) at one horizon, across initializations.

    Estimated at a single transition rather than pooled over rounds, because
    pooling defines a different (occupancy-mixed) law.  eta_IF normalizes by the
    conditional gate entropy; a deterministic gate gives a zero denominator and an
    undefined ratio, not perfect efficiency.
    """
    u = comp.u[z][:, h - 1]
    if np.isnan(u).all():
        return {"T_act_bits": np.nan, "H_U_given_n_bits": np.nan, "eta_IF": np.nan, "n": 0}
    ok = ~np.isnan(u)
    mi, hl = _cmi_bits(u[ok].astype(int), comp.controlled[z][ok, h, a].astype(int),
                       comp.controlled[z][ok, h - 1, a].astype(int))
    return {"T_act_bits": mi, "H_U_given_n_bits": hl,
            "eta_IF": float(mi / hl) if hl > 0 else np.nan, "n": int(ok.sum())}


def information_response_bound(comp: Comparison, a: int, z: int, h: int) -> dict[str, float]:
    """[R] Pinsker lower bound B_IR = 2 a_n (1-a_n) chi(n)^2 / ln2 and eta_IR.

    The bound squares the response, so beneficial and harmful effects of equal
    magnitude are indistinguishable here.  It is not a resource efficiency.
    """
    u = comp.u[z][:, h - 1]
    ok = ~np.isnan(u)
    if not ok.any():
        return {"B_IR_bits": np.nan, "eta_IR": np.nan}
    n_prev = comp.controlled[z][ok, h - 1, a].astype(int)
    dx = fractions(comp.controlled[z][ok, h, a]) - fractions(comp.controlled[z][ok, h - 1, a])
    uu = u[ok].astype(int)
    bound = 0.0; total = 0
    for n in np.unique(n_prev):
        s = n_prev == n
        d1, d0 = dx[s & (uu == 1)], dx[s & (uu == 0)]
        if len(d1) == 0 or len(d0) == 0:
            continue
        an = uu[s].mean()
        chi = d1.mean() - d0.mean()
        bound += s.sum() * 2.0 * an * (1 - an) * chi ** 2 / LOG2
        total += s.sum()
    b = bound / total if total else np.nan
    act = activation_information(comp, a, z, h)
    return {"B_IR_bits": float(b), "eta_IR": float(b / act["T_act_bits"])
            if act["T_act_bits"] and act["T_act_bits"] > 0 else np.nan}


def assigned_policy_information(comp: Comparison, a: int, z: int, h: int,
                                n_bins: int = 4) -> dict[str, float]:
    """[P] T = I(B; n_a(h) | n_a(0)) with a balanced controlled-versus-silent label B.

    The initial count is binned because with 60 initializations an unbinned
    conditioning variable would leave one observation per cell.
    """
    n0 = np.concatenate([comp.controlled[z][:, 0, a], comp.silent[:, 0, a]]).astype(int)
    nh = np.concatenate([comp.controlled[z][:, h, a], comp.silent[:, h, a]]).astype(int)
    b = np.concatenate([np.ones(comp.m, int), np.zeros(comp.m, int)])
    edges = np.quantile(n0, np.linspace(0, 1, n_bins + 1)[1:-1])
    cond = np.digitize(n0, edges)
    out_bins = np.digitize(nh, np.quantile(nh, np.linspace(0, 1, 7)[1:-1]))
    mi, _ = _cmi_bits(b, out_bins, cond)
    return {"T_assigned_bits": mi, "m": comp.m}


# ------------------------------------------------------------------ section 11
LAMBDA_GRID = np.linspace(-500, 500, 100001)


def k_min(comp: Comparison, a: int, h: int, z: int) -> dict[str, float]:
    """[T] K_min(delta) = sup_lambda [lambda*delta - psi(lambda)] for reward x_a(h).

    psi is the log moment generating function of the reward under the *silent*
    law, estimated from the silent initializations.

    The empirical transform is only identified when the requested mean lies
    strictly inside the observed baseline support: the empirical MGF puts no mass
    beyond the sample range, so a shift past it drives the optimal tilt to
    infinity and returns whatever the lambda grid happens to end at.  That is a
    missing-baseline-event artifact, not a truly infinite cost, so it is reported
    as unidentified rather than as a number.  ``K_min_identified`` records this.
    """
    r_base = fractions(comp.silent[:, h, a])
    r_mean = float(r_base.mean())
    delta = float(fractions(comp.controlled[z][:, h, a]).mean() - r_mean)
    centred = r_base - r_mean
    lam = LAMBDA_GRID
    psi = np.log(np.mean(np.exp(np.clip(np.outer(lam, centred), -700, 700)), axis=1))
    obj = lam * delta - psi
    j = int(np.nanargmax(obj))
    lam_star = float(lam[j])
    var = float(centred.var(ddof=1))
    target = r_mean + delta
    inside = bool(r_base.min() < target < r_base.max())
    at_edge = j in (0, len(lam) - 1)
    identified = inside and not at_edge
    return {"delta": delta,
            "K_min_nats": float(obj[j]) if identified else np.nan,
            "K_min_raw_nats": float(obj[j]),
            "K_min_gaussian_nats": float(delta ** 2 / (2 * var)) if var > 0 else np.nan,
            "lambda_star": lam_star, "var_base": var,
            "K_min_identified": identified,
            "target_in_support": inside, "lambda_at_grid_edge": at_edge,
            "base_min": float(r_base.min()), "base_max": float(r_base.max())}


def evidence_observables(comp: Comparison, h: int) -> dict[str, float]:
    """[R] Mean proof coverage kappa and full-proof ownership phi, by arm."""
    out = {}
    for arm in ("silent", "t0", "t2"):
        k, p = comp.kappa.get(arm), comp.phi.get(arm)
        idx = min(h, k.shape[1]) - 1
        out[f"kappa_{arm}"] = float(np.nanmean(k[:, idx]))
        out[f"phi_{arm}"] = float(np.nanmean(p[:, idx]))
    return out
