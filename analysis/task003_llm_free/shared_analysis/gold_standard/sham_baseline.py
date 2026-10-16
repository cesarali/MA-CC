"""Sham-baseline measures (handoff `blackboard_efficiency_info_goal_and_resource_handoff.md`) for Simulation 2,
task003-symmetric, from the 15 October run.

Treatment Z: informed (Z = 1, the controller targeting A0, the correct answer) against sham (Z = 0, a controller
whose target is drawn by a fair coin from {A0, A2}, fixed for the episode, independently of the truth). There is no
A1-target pool in this setup, so the sham draws from the two targets that exist. Because the vote distribution of an
arm is an average over its episodes, the sham's distribution is exactly the average of the A0-target and A2-target
arms, and every episode was run in all three arms with the same random draws (paired).

For every setting (q, qc, rate, rho, gate) and time t, writes results/gold_standard/<date>/sham_table.parquet with:
  chi, chi_se   paired correctness susceptibility: mean over episodes of x_A0(informed) - x_A0(sham)
                = 1/2 [x_A0(target A0) - x_A0(target A2)], same episode; standard error over episodes
  F1            facts posted by the informed controller up to t (mean per episode)
  eta_post      chi / F1 (undefined below half a fact)
  I_X, I_Y      I(Z; X) and I(Z; Y) in bits, X = one agent's vote, Y = 1{X = A0}, Z informed/sham (1/2 each)
  C_X           1/2 KL(p_1 || q) + 1/2 KL(p_0 || q), q = no control (silent arm)
  R_X           KL(pbar || q), the target-availability-independent part: C_X = I_X + R_X
  eta_info      I_X / C_X,  eta_goal = I_Y / C_X (undefined when C_X < 0.01 bits)
Every distribution gets the same pseudo-count of 0.5 episodes per value before any KL (as in the 15/16 Oct reports).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/sham_baseline.py [--date 2026-10-15] [--workers 12]
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CELLS = ROOT / "results/simulation_2/2026-10-15_sim2_symmetric_fine/cells"
RES = ROOT / "results/gold_standard"
TIMES = [0.25, 0.5, 0.75] + [float(t) for t in range(1, 41)]
ALPHA, C_MIN = 0.5, 0.01


def shares(folder):
    d = pd.read_parquet(folder / "snapshots.parquet", columns=["episode", "t", "share_A0", "share_A1", "share_A2",
                                                               "share_voted", "controller_messages"])
    d = d[d.t.isin(TIMES) & (d.share_voted > 0)].copy()
    for k in range(3):
        d[f"x{k}"] = d[f"share_A{k}"] / d.share_voted
    return d[["episode", "t", "x0", "x1", "x2", "controller_messages"]]


def smooth(p, n):
    return (n * p + ALPHA) / (n + ALPHA * len(p))


def kl(p, q):
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))


def pair(job):
    f_inf, f_false, f_sil, prm = job
    a, b, s = shares(f_inf), shares(f_false), shares(f_sil)
    j = a.merge(b, on=["episode", "t"], suffixes=("_i", "_f"))
    j["chi"] = 0.5 * (j.x0_i - j.x0_f)
    rows = []
    for t, g in j.groupby("t"):
        st = s[s.t == t]
        n = len(g)
        p1 = g[["x0_i", "x1_i", "x2_i"]].mean().to_numpy()
        pf = g[["x0_f", "x1_f", "x2_f"]].mean().to_numpy()
        q = st[["x0", "x1", "x2"]].mean().to_numpy()
        p1, pf, q = smooth(p1, n), smooth(pf, n), smooth(q, len(st))
        p0 = 0.5 * (p1 + pf)                      # sham: target A0 or A2 by a fair coin
        pbar = 0.5 * (p1 + p0)
        I_X = 0.5 * kl(p1, pbar) + 0.5 * kl(p0, pbar)
        C_X = 0.5 * kl(p1, q) + 0.5 * kl(p0, q)
        R_X = kl(pbar, q)
        r1, r0, rb = np.array([p1[0], 1 - p1[0]]), np.array([p0[0], 1 - p0[0]]), np.array([pbar[0], 1 - pbar[0]])
        I_Y = 0.5 * kl(r1, rb) + 0.5 * kl(r0, rb)
        chi, chi_se = g.chi.mean(), g.chi.std() / np.sqrt(n)
        F1 = g.controller_messages_i.mean()
        rows.append(dict(**prm, t=t, chi=chi, chi_se=chi_se, F1=F1, eta_post=chi / F1 if F1 >= 0.5 else np.nan,
                         I_X=I_X, I_Y=I_Y, C_X=C_X, R_X=R_X,
                         eta_info=I_X / C_X if C_X >= C_MIN else np.nan, eta_goal=I_Y / C_X if C_X >= C_MIN else np.nan))
    return rows


def jobs():
    out = []
    for f in sorted(CELLS.iterdir()):
        prm = json.loads((f / "params.json").read_text())
        if prm["arm"] != "truth":
            continue
        ff = CELLS / f.name.replace("__truth__", "__false__")
        out.append((f, ff, CELLS / prm["silent_cell"],
                    dict(q=prm["q"], qc=prm["qc"], x=prm["lambda_c"], rho=float(prm["rho"]),
                         gate="on" if prm["controller_gate"] == "votes" else "off")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(pair, jobs(), chunksize=4)
    t = pd.DataFrame([r for p in parts for r in p])
    out = RES / a.date / "sham_table.parquet"
    t.to_parquet(out, index=False)
    print(len(t), "rows ->", out)


if __name__ == "__main__":
    main()
