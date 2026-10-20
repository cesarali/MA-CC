"""Step 2 (Campaign A): the reading reversal's mechanism, from the full-record run 2026-10-20_sim2_reading_mechanism
(both setups, 8 rho, q = 3, 6, 9, 12, false and silent arms, rate 1, qc 12, gate on).

1. Validation: for q = 3, 6, 12 the light tables (snapshots, controller, episodes) must equal the fine runs'.
2. G(q) and paired D for neighbouring q (3->6, 6->9, 9->12) at every time, as in s1_reversal.py.
3. Memory and board composition at TIMES for every (setup, rho, q, arm), as in s1_mechanism.py:
   mem_agent, mem_ctrl, share_ctrl_any, cover_agent, p_A0, p_A2, entropy, proof, board_ctrl, board_agent_distinct,
   p_A2_agentonly (each agent's belief in A2 from its agent-origin facts alone). With it the controller's belief shift
   splits exactly into a direct part, p_A2 - p_A2_agentonly (false arm; what the controller's facts add), and an
   indirect part, p_A2_agentonly(false) - p_A2(silent) (which agent facts the swarm keeps; silent memories hold no
   controller-only facts, so p_A2_agentonly = p_A2 there).
Writes results/key_phenomena/s2_validation.csv, s2_G.parquet, s2_D.parquet, s2_composition.parquet.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s2_mechanism.py
"""
from __future__ import annotations
import json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "shared_analysis" / "gold_standard"))
from build_tables import load  # noqa: E402
from s1_mechanism import board_stats, classes, entropy  # noqa: E402

RES = ROOT / "results" / "simulation_2"
OUT = ROOT / "results" / "key_phenomena"
QS = [3, 6, 9, 12]
TIMES = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
FINE = {"task003_symmetric": ["2026-10-15_sim2_symmetric_fine", "2026-10-16_sim2_symmetric_fine_extra_rho"],
        "task003_nosolution": ["2026-10-16_sim2_nosolution_fine"]}
LIGHT = ["snapshots.parquet", "controller.parquet", "episodes.parquet"]


def run_dir():
    runs = sorted(RES.glob("*_sim2_reading_mechanism"))
    assert runs, "run sim2_reading_mechanism first"
    return runs[-1]


def cells():
    out = {}
    for c in (run_dir() / "cells").iterdir():
        prm = json.loads((c / "params.json").read_text())
        out[(prm["setup"], float(prm["rho"]), prm["q"], prm["arm"])] = c
    return out


def fine_counterpart(setup, rho, q, arm):
    for r in FINE[setup]:
        base = RES / r / "cells"
        name = (f"{setup}__silent__q{q}__rho{rho:g}" if arm == "silent"
                else f"{setup}__false__q{q}__qc12__rho{rho:g}__lc1__gateon")
        if (base / name).exists():
            return base / name
    return None


def validate(item):
    key, cell = item
    old = fine_counterpart(*key)
    if old is None:
        return None
    same = all(pd.read_parquet(cell / t).equals(pd.read_parquet(old / t)) for t in LIGHT)
    return (*key, same)


def composition(item):
    (setup, rho, q, arm), cell = item
    world, agent, ctrl = classes(setup.replace("task003_", ""))
    grid = {round(t / 0.25): t for t in TIMES}
    snaps = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"])
    snaps = snaps[snaps.grid_index.isin(grid)]
    post = {}
    for m in snaps.active.unique():
        m = int(m)
        p = world.posterior(m)
        pa = world.posterior(m & agent)                  # the belief from the agent-origin facts alone
        post[m] = (bin(m & agent).count("1"), bin(m & ctrl).count("1"), p[0], p[2], entropy(p), float(p[0] == 1.0),
                   pa[2])
    v = np.array([post[int(m)] for m in snaps.active.to_numpy()])
    a = pd.DataFrame(v, columns=["mem_agent", "mem_ctrl", "p_A0", "p_A2", "entropy", "proof", "p_A2_agentonly"])
    a["share_ctrl_any"] = (a.mem_ctrl > 0).astype(float)
    a["episode"], a["t"] = snaps.episode.to_numpy(), snaps.grid_index.map(grid).to_numpy()
    a["active"] = snaps.active.to_numpy()
    g = a.groupby(["episode", "t"])
    per = g.mean(numeric_only=True).drop(columns="active")
    per["cover_agent"] = g.active.agg(lambda s: bin(int(np.bitwise_or.reduce(s.to_numpy())) & agent).count("1"))
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
    per = per.reset_index().merge(board_stats(posts, agent, ctrl, TIMES), on=["episode", "t"], how="left")
    m = per.drop(columns="episode").groupby("t").mean().reset_index()
    m.insert(0, "setup", setup.replace("task003_", ""))
    m.insert(1, "rho", rho)
    m.insert(2, "q", q)
    m.insert(3, "arm", arm)
    return m


def gains(item):
    (setup, rho), by_q = item
    ep = {}
    for q, (f, s) in by_q.items():
        c, z = load("sim2", f), load("sim2", s)
        j = c[["episode", "t", "x2"]].merge(z[["episode", "t", "x2"]], on=["episode", "t"], suffixes=("", "_s"))
        j["g"] = j.x2 - j.x2_s
        ep[q] = j[["episode", "t", "g"]].dropna()
    s = setup.replace("task003_", "")
    G = [(s, rho, q, t, r.g.mean(), r.g.std() / np.sqrt(len(r))) for q, d in ep.items() for t, r in d.groupby("t")]
    D = []
    for q1, q2 in zip(QS[:-1], QS[1:]):
        j = ep[q1].merge(ep[q2], on=["episode", "t"], suffixes=("1", "2"))
        j["d"] = j.g2 - j.g1
        for t, r in j.groupby("t"):
            D.append((s, rho, f"{q1}->{q2}", q1, q2, t, r.d.mean(), r.d.std() / np.sqrt(len(r))))
    return G, D


def main():
    cs = cells()
    with multiprocessing.Pool(12) as pool:
        val = [v for v in pool.map(validate, cs.items()) if v is not None]
        jobs = {}
        for (setup, rho, q, arm), c in cs.items():
            if arm == "false":
                jobs.setdefault((setup, rho), {})[q] = (c, cs[(setup, rho, q, "silent")])
        parts = pool.map(gains, jobs.items())
        comp = pool.map(composition, cs.items(), chunksize=2)
    v = pd.DataFrame(val, columns=["setup", "rho", "q", "arm", "identical"])
    v.to_csv(OUT / "s2_validation.csv", index=False)
    print(f"validation: {int(v.identical.sum())}/{len(v)} cells identical to the fine runs")
    pd.DataFrame([g for p in parts for g in p[0]], columns=["setup", "rho", "q", "t", "G", "G_se"]).to_parquet(
        OUT / "s2_G.parquet", index=False)
    pd.DataFrame([d for p in parts for d in p[1]],
                 columns=["setup", "rho", "step", "q1", "q2", "t", "D", "D_se"]).to_parquet(OUT / "s2_D.parquet",
                                                                                           index=False)
    pd.concat(comp, ignore_index=True).to_parquet(OUT / "s2_composition.parquet", index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
