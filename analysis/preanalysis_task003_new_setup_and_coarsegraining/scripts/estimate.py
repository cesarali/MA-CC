"""Information quantities on the epistemic coarse-grained chain.

Why this differs from the earlier attempt.  Previously these quantities were
estimated with fitted neural critics on the 325-state vote vector; the estimates
failed every reliability test (a third of horizon steps violated a monotonicity
the quantity must obey).  On a 4-state Markov chain the same quantities are
*exact* given the fitted kernels: path KL factorises over time, so no variational
bound and no critic is needed.

    D(P||Q) over paths = D(p0||q0) + sum_t E_P[ D( P(.|s_t) || Q(.|s_t) ) ]

Kernels are fitted separately per (persistence, arm), never pooled, and
uncertainty is a cluster bootstrap over episodes.

Scope limit: the A0-target arm is the confounded configuration (its controller
recommends ALLOCATION_0 while quoting ALLOCATION_2-favouring evidence).  Target
information I(Z;.) therefore measures the CURRENT design, not truth-vs-false
steering, and is labelled as such.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd

SC = "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/"
REC = ("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment/"
       "simulation_data/records/21-09-2026-full-vs-report-v1_analysis/")
EDGES = [0.33, 0.50, 0.75]          # 4 states
NS = 4
EPS = 1e-12


def load():
    d = pd.read_parquet(SC + "results/data/states_all.parquet")
    d["S"] = np.digitize(d.e, EDGES)
    return d


def fit(df):
    """Initial distribution and transition kernel, Laplace-smoothed."""
    p0 = np.full(NS, EPS)
    T = np.full((NS, NS), EPS)
    for _, g in df.sort_values("round_index").groupby(["cell_id", "episode_id"]):
        s = g.S.to_numpy()
        p0[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1
    p0 /= p0.sum()
    T /= T.sum(axis=1, keepdims=True)
    return p0, T


def occupancy(p0, T, h):
    """State distribution at each horizon under the chain."""
    out = [p0]
    for _ in range(h):
        out.append(out[-1] @ T)
    return np.array(out)


def path_kl(pP, TP, pQ, TQ, h):
    """Exact KL between path laws of two Markov chains over h transitions."""
    kl = float(np.sum(pP * np.log((pP + EPS) / (pQ + EPS))))
    occ = occupancy(pP, TP, h)
    for t in range(h):
        row = np.sum(TP * np.log((TP + EPS) / (TQ + EPS)), axis=1)
        kl += float(occ[t] @ row)
    return kl


def mi_label(pA, TA, pB, TB, h, wA=0.5):
    """I(label; S_h) for a binary label with prior wA, in bits."""
    a = occupancy(pA, TA, h)[h]
    b = occupancy(pB, TB, h)[h]
    m = wA * a + (1 - wA) * b
    f = lambda p, w: w * np.sum(p * np.log2((p + EPS) / (m + EPS)))
    return float(f(a, wA) + f(b, 1 - wA))


def entropy_production(p, T, h):
    """Time-reversal EP rate of the chain, nats per step.

    This is the Markov-chain quantity sum_ij p_i T_ij ln(p_i T_ij / p_j T_ji).
    It is a statistical irreversibility diagnostic of the coarse-grained process,
    NOT a physical dissipation: no reverse protocol has been justified, and the
    reference document is explicit that a sequence-reversal score must not be
    relabelled entropy production.
    """
    occ = occupancy(p, T, h)
    rates = []
    for t in range(h):
        pt = occ[t]
        J = pt[:, None] * T
        rates.append(float(np.sum(J * np.log((J + EPS) / (J.T + EPS)))))
    return float(np.mean(rates)), rates


def boot(df, fn, n=300, seed=0, point=None):
    """Cluster bootstrap over episodes, bias-corrected.

    Resampling episodes with replacement duplicates them, which smooths the
    refitted transition matrix and pulls a KL systematically toward zero.  The
    raw percentile interval can therefore exclude the point estimate.  We shift
    the interval by the measured bias (bootstrap mean minus point estimate),
    which is the standard first-order correction.
    """
    eps = df[["cell_id", "episode_id"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    idx = {tuple(k): g for k, g in df.groupby(["cell_id", "episode_id"])}
    for _ in range(n):
        pick = eps[rng.integers(0, len(eps), len(eps))]
        s = pd.concat([idx[tuple(k)] for k in pick], copy=False)
        try:
            out.append(fn(s))
        except Exception:
            pass
    if not out:
        return (np.nan, np.nan)
    lo, hi = np.percentile(out, [2.5, 97.5])
    if point is not None:
        bias = float(np.mean(out)) - point
        lo, hi = lo - bias, hi - bias
    return (lo, hi)


if __name__ == "__main__":
    d = load()
    H = 14
    rows = []
    for rho in (0.75, 1.00):
        sub = d[d.epistemic_persistence == rho]
        sil = sub[sub.arm == "silent"]
        a2 = sub[sub.arm == "A2"]
        a0 = sub[sub.arm == "A0"] if "A0" in set(sub.arm) else None
        if len(sil) < 100 or len(a2) < 100:
            continue
        p0s, Ts = fit(sil)
        p02, T2 = fit(a2)
        # K_pi: controlled-vs-silent path divergence, exact on the chain
        K2 = path_kl(p02, T2, p0s, Ts, H)
        ci = boot(a2, lambda s: path_kl(*fit(s), p0s, Ts, H), point=K2)
        # assigned-policy information I(B; S_h), B = controlled vs silent
        T_ap = mi_label(p02, T2, p0s, Ts, H)
        ci_ap = boot(a2, lambda s: mi_label(*fit(s), p0s, Ts, H), point=T_ap)
        # entropy production rate of each arm
        ep_s, _ = entropy_production(p0s, Ts, H)
        ep_2, _ = entropy_production(p02, T2, H)
        # resource: mean posts per episode in the controlled arm
        posts = a2.groupby(["cell_id", "episode_id"]).posts.sum().mean()
        rows.append(dict(rho=rho, K_A2_nats=K2, K_lo=ci[0], K_hi=ci[1],
                         I_assigned_bits=T_ap, I_lo=ci_ap[0], I_hi=ci_ap[1],
                         EP_silent=ep_s, EP_A2=ep_2, posts_per_episode=posts,
                         n_silent=len(sil), n_A2=len(a2)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(SC + "40_information_estimates/estimates_chain.csv", index=False)

    # horizon curves and the confounded target-information term
    curves = []
    for rho in (0.75, 1.00):
        sub = d[d.epistemic_persistence == rho]
        sil, a2 = sub[sub.arm == "silent"], sub[sub.arm == "A2"]
        a0 = sub[sub.arm == "A0"]
        if len(sil) < 100 or len(a2) < 100:
            continue
        p0s, Ts = fit(sil); p02, T2 = fit(a2)
        has0 = len(a0) > 100
        if has0:
            p00, T0 = fit(a0)
        for h in range(1, H + 1):
            r = dict(rho=rho, h=h,
                     K_A2=path_kl(p02, T2, p0s, Ts, h),
                     I_assigned=mi_label(p02, T2, p0s, Ts, h),
                     EP_A2=entropy_production(p02, T2, h)[0],
                     EP_silent=entropy_production(p0s, Ts, h)[0])
            if has0:
                r["K_A0"] = path_kl(p00, T0, p0s, Ts, h)
                r["I_target_confounded"] = mi_label(p00, T0, p02, T2, h)
                r["C_pi"] = 0.5 * (r["K_A0"] + r["K_A2"])
                r["eta_ctl_confounded"] = (r["I_target_confounded"] * np.log(2)) / r["C_pi"] if r["C_pi"] > 0 else np.nan
            curves.append(r)
    pd.DataFrame(curves).to_csv(SC + "40_information_estimates/estimates_curves.csv", index=False)
    print("DONE", flush=True)
