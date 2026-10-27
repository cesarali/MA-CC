"""Phenomenological fact-flow model (researcher's plan of 27 Oct 2026, Eqs. 1-3, 5, 6). Self-contained: no import of
the Bayesian engine, task worlds or M1-M4 code.

State y = (m_A, m_C, b_A, b_C): per-fact memory probabilities and board shares of the two evidence classes.
  dm_a/dt = (1 - m_a) r_a - gamma m_a,              r_a = 1 - (1 - b_a / F_a)^q              (1)
  db_a/dt = (N/W) [S_a - b_a] + (lam(t)/W) [u_a - b_a],   u_C = 1, u_A = 0 during control      (2)
  Phi2    = sigma(h0 + beta_C F_C m_C - beta_A F_A m_A)                                         (3)
  S_C = w_C / (w_A + w_C + k0), S_A = w_A / (...), w_C = kC F_C m_C Phi2, w_A = kA F_A m_A (1 - Phi2)   (5)
  neutral variant (switch): w_C = kC F_C m_C, w_A = kA F_A m_A
  revision 3: w_C = kC F_C m_C [eps + (1 - eps) Phi2], w_A = kA F_A m_A [eps + (1 - eps)(1 - Phi2)]
Continuations from T: normal, no_comm (r = 0, ordinary posting off), frozen (b fixed), purge (m_C = b_C = 0 at T).
"""
from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class P:
    N_over_W: float = 0.5
    F_A: float = 16.0
    F_C: float = 4.0
    beta_A: float = 0.5
    beta_C: float = 0.5
    h0: float = float(np.log(0.5))
    kA: float = 1.0
    kC: float = 1.0
    k0: float = 0.2
    lam0: float = 1.0
    W: float = 48.0
    T: float = 30.0
    selective: bool = True
    eps: float = 0.0
    proof_l: int = 0          # Stage C: proof size l (0 = no proof variable)
    proof_P: float = 0.0      # Stage C: effective number of alternative proofs P
    coretention: bool = False  # Step 1: z from the co-retention moment h_l instead of m_A^l


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def zproof(p: P, mA):
    """Stage C (plan Eq. 4): chance that an agent holds a complete proof of the correct answer."""
    if p.proof_l <= 0:
        return 0.0 * mA
    return 1 - (1 - np.clip(mA, 0, 1) ** p.proof_l) ** p.proof_P


def phi2(p: P, mA, mC, hl=None):
    z = (1 - (1 - np.clip(hl, 0, 1)) ** p.proof_P) if (hl is not None and p.coretention) else zproof(p, mA)
    return (1 - z) * sigma(p.h0 + p.beta_C * p.F_C * mC - p.beta_A * p.F_A * mA)


def p_exact(k, j, pf, q):
    """Chance that one read of q posts (with replacement) brings exactly a given j of k interchangeable facts, each on a
    share pf of the board."""
    from math import comb
    return sum((-1) ** i * comb(j, i) * max(1 - (k - j + i) * pf, 0.0) ** q for i in range(j + 1)) * comb(k, j)


def dh(h, pf, q, gamma):
    """Co-retention moments h_1..h_l (h_0 = 1): dh_k = -k gamma h_k + sum_j C(k,j) p_k(j) (h_{k-j} - h_k)."""
    hh = np.concatenate([[1.0], h])
    out = np.zeros(len(h))
    for k in range(1, len(h) + 1):
        acc = -k * gamma * hh[k]
        for j in range(1, k + 1):
            acc += p_exact(k, j, pf, q) * (hh[k - j] - hh[k])
        out[k - 1] = acc
    return out


def rhs(t, y, p: P, q, gamma, controlled, mode="normal", frozen_b=None):
    mA, mC, bA, bC = y[:4]
    hl = y[3 + p.proof_l] if (p.coretention and p.proof_l > 0) else None
    after = t >= p.T
    if mode == "frozen" and after:
        bA, bC = frozen_b
    read = not (after and mode == "no_comm")
    rA = 1 - (1 - min(max(bA, 0.0), 1.0) / p.F_A) ** q if read else 0.0
    rC = 1 - (1 - min(max(bC, 0.0), 1.0) / p.F_C) ** q if read else 0.0
    dmA = (1 - mA) * rA - gamma * mA
    dmC = (1 - mC) * rC - gamma * mC
    ph = phi2(p, mA, mC, hl)
    if p.selective:            # revision 3: eps = vote-independent eligibility (eps = 0: plan's closure (5))
        wA = p.kA * p.F_A * mA * (p.eps + (1 - p.eps) * (1 - ph))
        wC = p.kC * p.F_C * mC * (p.eps + (1 - p.eps) * ph)
    else:
        wA, wC = p.kA * p.F_A * mA, p.kC * p.F_C * mC
    tot = wA + wC + p.k0
    SA, SC = wA / tot, wC / tot
    lam = p.lam0 if (controlled and t < p.T) else 0.0
    post = 0.0 if (after and mode in ("no_comm", "frozen")) else 1.0      # ordinary posting on/off
    nw = p.N_over_W * post
    if mode == "frozen" and after:
        dbA = dbC = 0.0
    else:
        dbA = nw * (SA - bA) + lam / p.W * (0.0 - bA)
        dbC = nw * (SC - bC) + lam / p.W * (1.0 - bC)
    if not controlled:
        dmC, dbC = 0.0, 0.0
    out = [dmA, dmC, dbA, dbC]
    if p.coretention and p.proof_l > 0:
        pf = min(max(bA, 0.0), 1.0) / p.F_A if read else 0.0
        out += list(dh(np.asarray(y[4:4 + p.proof_l]), pf, q, gamma))
    return out


def y0(p: P):
    base = [1.0 / p.F_A, 0.0, 0.0, 0.0]
    if p.coretention and p.proof_l > 0:
        base += [1.0 / p.F_A] + [0.0] * (p.proof_l - 1)     # each agent starts with one own fact
    return base


def run(p: P, q, rho, controlled, t_end=None, mode="normal", t_eval=None):
    gamma = -np.log(rho)
    t_end = p.T if t_end is None else t_end
    if t_end <= p.T:
        sol = solve_ivp(rhs, (0, t_end), y0(p), args=(p, q, gamma, controlled, "normal"), t_eval=t_eval,
                        rtol=1e-8, atol=1e-10, max_step=0.5)
        return sol
    t1 = None if t_eval is None else t_eval[t_eval <= p.T]
    first = solve_ivp(rhs, (0, p.T), y0(p), args=(p, q, gamma, controlled, "normal"), t_eval=t1, rtol=1e-8, atol=1e-10,
                      max_step=0.5)
    yT = first.y[:, -1].copy()
    if mode == "purge" and controlled:
        yT[1] = yT[3] = 0.0
    m2 = "normal" if mode == "purge" else mode
    te = None if t_eval is None else t_eval[t_eval >= p.T]
    second = solve_ivp(rhs, (p.T, t_end), yT, args=(p, q, gamma, controlled, m2, (yT[2], yT[3])), t_eval=te,
                       rtol=1e-8, atol=1e-10, max_step=0.5)
    return first, second


def _hl(p, y):
    return y[3 + p.proof_l] if (p.coretention and p.proof_l > 0) else None


def G_at_T(p: P, q, rho):
    c = run(p, q, rho, True).y[:, -1]
    s = run(p, q, rho, False).y[:, -1]
    return phi2(p, c[0], c[1], _hl(p, c)) - phi2(p, s[0], s[1], _hl(p, s)), c, s


def D_curve(p: P, rhos, q1, q2):
    return np.array([G_at_T(p, q2, r)[0] - G_at_T(p, q1, r)[0] for r in rhos])
