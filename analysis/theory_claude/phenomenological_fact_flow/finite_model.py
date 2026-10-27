"""Model F: finite, stochastic version of the fact-flow model (decision_log.md, 2026-10-28, step 3). Individual agents
(memories over F_A agent facts + F_C controller facts), a board of whole posts (W newest readable, own posts excluded),
forgetting at rate gamma, sigmoid votes on fact counts, eligibility-based reposting, one controller posting random
controller facts until T. Time step dt. No Bayesian engine, no task world.

Fact ids: 0..F_A-1 agent facts, F_A..F_A+F_C-1 controller facts, -1 background post.
"""
from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class FP:
    N: int = 24
    W: int = 48
    F_A: int = 16
    F_C: int = 4
    beta_A: float = 0.5
    beta_C: float = 0.5
    h0: float = float(np.log(0.5))
    kA: float = 1.0
    kC: float = 1.0
    k0: float = 0.2
    eps: float = 1.0
    lam0: float = 1.0
    T: float = 30.0
    dt: float = 0.05


def episode(p: FP, q: int, rho: float, controlled: bool, seed: int, t_end: float = 30.0, mode: str = "normal",
            record=(30.0,)):
    """One episode. Returns dict t -> (x_A2 among voters, distinct agent facts circulating, mean copies per circulating
    agent fact on the board, distinct controller facts circulating)."""
    rng = np.random.default_rng(seed)
    F = p.F_A + p.F_C
    mem = np.zeros((p.N, F), dtype=bool)
    for i in range(p.N):
        mem[i, i % p.F_A] = True
    vote = np.full(p.N, -1)
    board_f: list = []
    board_a: list = []
    pf = 1 - rho ** p.dt if rho < 1 else 0.0
    steps = int(round(t_end / p.dt))
    rec = {}
    frozen = None
    stopped = False
    rec_steps = {int(round(t / p.dt)): t for t in record}
    for k in range(steps + 1):
        t = k * p.dt
        if k in rec_steps:
            rec[rec_steps[k]] = measure(p, mem, vote, board_f)
        if k == steps:
            break
        if not stopped and t >= p.T:
            stopped = True
            if mode == "purge" and controlled:
                mem[:, p.F_A:] = False
                keep = [j for j, f in enumerate(board_f) if f < p.F_A]
                board_f = [board_f[j] for j in keep]
                board_a = [board_a[j] for j in keep]
            if mode == "frozen":
                frozen = (list(board_f), list(board_a))
        # forgetting
        if pf > 0:
            mem &= rng.random(mem.shape) >= pf
        # controller
        if controlled and t < p.T and rng.random() < p.lam0 * p.dt:
            board_f.append(p.F_A + int(rng.integers(p.F_C)))
            board_a.append(-2)
        # agents
        acting = np.nonzero(rng.random(p.N) < p.dt)[0]
        no_read = stopped and mode == "no_comm"
        no_post = stopped and mode in ("no_comm", "frozen")
        bf, ba = (frozen if (stopped and mode == "frozen") else (board_f, board_a))
        for i in acting:
            if not no_read and bf:
                win = [f for f, a in zip(bf[::-1], ba[::-1]) if a != i][: p.W]
                if win:
                    for j in rng.integers(len(win), size=q):
                        f = win[j]
                        if f >= 0:
                            mem[i, f] = True
            a = int(mem[i, : p.F_A].sum())
            c = int(mem[i, p.F_A:].sum())
            phi = 1 / (1 + np.exp(-(p.h0 + p.beta_C * c - p.beta_A * a)))
            v2 = rng.random() < phi
            vote[i] = 2 if v2 else 0
            if no_post:
                continue
            wA = p.kA * a * (p.eps + (1 - p.eps) * (not v2))
            wC = p.kC * c * (p.eps + (1 - p.eps) * v2)
            u = rng.random() * (wA + wC + p.k0)
            if u < wA:
                f = int(rng.choice(np.nonzero(mem[i, : p.F_A])[0]))
            elif u < wA + wC:
                f = p.F_A + int(rng.choice(np.nonzero(mem[i, p.F_A:])[0]))
            else:
                f = -1
            board_f.append(f)
            board_a.append(i)
        if len(board_f) > 20 * p.W:
            board_f, board_a = board_f[-4 * p.W:], board_a[-4 * p.W:]
    return rec


def measure(p: FP, mem, vote, board_f):
    voted = vote >= 0
    x2 = float((vote[voted] == 2).mean()) if voted.any() else np.nan
    win = np.array(board_f[-p.W:], dtype=int)
    held = mem.any(axis=0)
    on_board = np.zeros(p.F_A + p.F_C, dtype=bool)
    counts = np.zeros(p.F_A + p.F_C)
    for f in win[win >= 0]:
        on_board[f] = True
        counts[f] += 1
    circ = held | on_board
    a_circ = circ[: p.F_A]
    return dict(x2=x2, distinct_A=int(a_circ.sum()), copies_A=float(counts[: p.F_A][on_board[: p.F_A]].mean()) if on_board[: p.F_A].any() else 0.0,
                distinct_C=int(circ[p.F_A:].sum()), holders_A=float(mem[:, : p.F_A].sum(axis=0)[a_circ].mean()) if a_circ.any() else 0.0)
