"""Phase A1 and A3 (00_preregistration.md): belief versus vote, and the marginal and evidence-state predictors of the
sign of D(3->6), from the agent memories stored at t = 30.

Per (task, setup, rho, q, arm) cell and episode:
  x_vote   share of A2 among agents who have voted (the data)
  belief   mean over agents of P(A2 | M_i)                                 (A1; exact identity, reference)
  marg     mean over agents of sum_{f in M_i, f in U} (P(A2 | {f}) - 1/3)   (A3 predictor 1)
  state    mean over agents of Phi_hat(s(M_i)), map from evidence_map.csv  (A3 predictor 2)
  err      mean over agents of |Phi_hat(s(M_i)) - P(A2 | M_i)|             (A2 approximation error)
Each quantity's G = false - silent, and D = G(q 6) - G(q 3), per episode (episodes are paired across arms and q).
95% intervals: 200 bootstrap resamples of episodes. Writes memory_quantities.csv (cell level) and
memory_predictors.csv (16 points, with signs and the scoring of 00_preregistration.md).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/phaseA/memory_predictors.py --workers 7
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_map import WORLDS, C_CAP, A_CAP, facts  # noqa: E402

A = HERE.parents[1]
RUNS = {"task003": A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_reading_mechanism",
        "Object Placements": A / "object_placements_llm_free/results/simulation_2/2026-10-22_sim2_op_main"}
RHOS, QS, GI = [0.25, 0.5, 0.75, 1.0], [3, 6], 120
TOL, BOOT = 0.005, 200


def cell_quantities(args):
    task, cell = args
    p = json.loads((cell / "params.json").read_text())
    w, agent, ctrl = facts(task, p["setup"])
    U = agent + ctrl
    wts = {f: w.posterior(1 << f)[2] - 1 / 3 for f in U}
    amask, cmask = sum(1 << f for f in agent), sum(1 << f for f in ctrl)
    emap = pd.read_csv(HERE / "evidence_map.csv")
    emap = emap[(emap.task == task) & (emap.setup == p["setup"])]
    phi = {(int(r.z), int(r.c), int(r.a)): r.phi_hat for r in emap.itertuples()}
    phi_ca = emap.groupby(["c", "a"]).apply(lambda g: np.average(g.phi_hat, weights=g.n_draws), include_groups=False).to_dict()
    cache = {}
    ag = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                         filters=[("grid_index", "==", GI)])
    rows = []
    fallback = 0
    for e, g in ag.groupby("episode"):
        b = mg = st = er = 0.0
        for m in g.active.to_numpy():
            m = int(m)
            if m not in cache:
                P = w.posterior(m)
                z = int(P[0] >= 1 - 1e-12)
                c = min(bin(m & cmask).count("1"), C_CAP)
                a = min(bin(m & amask).count("1"), A_CAP)
                ph = phi.get((z, c, a))
                fb = ph is None
                if fb:
                    ph = phi_ca[(c, a)]
                cache[m] = (P[2], sum(wts[f] for f in U if (m >> f) & 1), ph, fb)
            p2, mm, ph, fb = cache[m]
            fallback += fb
            b += p2; mg += mm; st += ph; er += abs(ph - p2)
        n = len(g)
        rows.append(dict(episode=e, belief=b / n, marg=mg / n, state=st / n, err=er / n))
    q = pd.DataFrame(rows)
    sn = pd.read_parquet(cell / "snapshots.parquet", columns=["episode", "grid_index", "share_A2", "share_voted"])
    sn = sn[sn.grid_index == GI]
    q = q.merge(sn.assign(x_vote=sn.share_A2 / sn.share_voted.where(sn.share_voted > 0))[["episode", "x_vote"]], on="episode")
    q = q.assign(task=task, setup=p["setup"], rho=float(p["rho"]), q=int(p["q"]), arm=p["arm"], fallback=fallback)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    jobs = []
    for task, run in RUNS.items():
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if float(p["rho"]) in RHOS and int(p["q"]) in QS:
                jobs.append((task, c))
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(cell_quantities, jobs, chunksize=1)
    d = pd.concat(parts)
    d.to_parquet(HERE / "memory_quantities.parquet", index=False)
    cols = ["x_vote", "belief", "marg", "state"]
    rng = np.random.default_rng(7)
    out = []
    for (task, setup, rho), g in d.groupby(["task", "setup", "rho"]):
        piv = g.pivot_table(index="episode", columns=["q", "arm"], values=cols)
        Dep = {c: (piv[(c, 6, "false")] - piv[(c, 6, "silent")]) - (piv[(c, 3, "false")] - piv[(c, 3, "silent")])
               for c in cols}
        Dep = pd.DataFrame(Dep).dropna()
        row = dict(task=task, setup=setup.split("_", 1)[1], rho=rho, episodes=len(Dep),
                   approx_err=g.err.mean(), fallback_memories=int(g.fallback.max()))
        for c in cols:
            v = Dep[c].to_numpy()
            boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row[f"D_{c}"], row[f"D_{c}_lo"], row[f"D_{c}_hi"] = v.mean(), lo, hi
        for c in ("x_vote", "belief"):
            row[f"G6_{c}"] = (piv[(c, 6, "false")] - piv[(c, 6, "silent")]).mean()
        row["data_sign"] = "+" if row["D_x_vote_lo"] > 0 else ("-" if row["D_x_vote_hi"] < 0 else "0")
        for c in ("belief", "marg", "state"):
            row[f"sign_{c}"] = "+" if row[f"D_{c}"] >= TOL else ("-" if row[f"D_{c}"] <= -TOL else "0")
        out.append(row)
    r = pd.DataFrame(out)
    r.round(4).to_csv(HERE / "memory_predictors.csv", index=False)
    pd.set_option("display.width", 250)
    print(r[["task", "setup", "rho", "G6_x_vote", "G6_belief", "D_x_vote", "data_sign", "D_belief", "sign_belief",
             "D_marg", "sign_marg", "D_state", "sign_state", "approx_err", "fallback_memories"]].round(3).to_string(index=False))
    res = r[r.data_sign != "0"]
    for c in ("belief", "marg", "state"):
        print(c, {k: f"{int((g.data_sign == g[f'sign_{c}']).sum())}/{len(g)}" for k, g in res.groupby("task")})


if __name__ == "__main__":
    main()
