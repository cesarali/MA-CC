"""D1 (pre-registered in ../../theory_blackboard/2026-11-08_claude.md): how many distinct facts does the simulator's
controller use, and how many controller-only facts circulate at its stop (t = 30)? 15-agent campaign, false arm,
standard reading, q in {2,4,8}, rho in {0.135,0.37,0.61,0.78,0.88}, all four setups, all 1,000 episodes.

Per episode:
  posts_used     controller posts made in [0, 30)
  distinct_used  distinct facts among them
  alive30        controller-only facts (pool minus agents' facts) on the 30 newest posts or in any memory at t = 30
Writes data_check.csv (cell means) and prints a summary. Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/fact_efficiency/data_check.py
"""
import json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "15_agent_simulation/results/2026-10-29_sim2_15agents_task003",
                    A / "task003_llm_free/llmfree_core/world_task003.json"),
        "op": (A / "15_agent_simulation/results/2026-10-29_sim2_15agents_op",
               A / "object_placements_llm_free/world/world_op1.json")}
SETUPS = {"task003": ["task003_symmetric_15", "task003_nosolution_uniform_15"],
          "op": ["op_symmetric_15", "op_nosolution_uniform_15"]}
CONTROLLER = -1


def cell(args):
    task, setup, q, rho = args
    run, wf = RUNS[task]
    w = load_world(wf)
    st = load_setup(setup, A / "15_agent_simulation", w)
    ctrl = set(st.pools[2]) - set(st.agent_facts)
    c = run / "cells" / f"{setup}__false__q{q}__rho{rho:g}__readposts"
    posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "author", "fact"])
    ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                         filters=[("grid_index", "==", 120)])
    mem = {e: g.active.to_numpy() for e, g in ag.groupby("episode")}
    rows = []
    for e, g in posts.groupby("episode"):
        g = g.sort_values("t")
        cp = g[(g.author == CONTROLLER) & (g.t < 30)]
        win = g[g.t <= 30].fact.to_numpy()[-30:]
        alive = {int(f) for f in win if int(f) in ctrl}
        for m in mem.get(e, []):
            alive |= {f for f in ctrl if (int(m) >> f) & 1}
        rows.append(dict(posts_used=len(cp), distinct_used=cp.fact.nunique(), alive30=len(alive)))
    d = pd.DataFrame(rows)
    return dict(task=task, setup=setup, q=q, rho=rho, pool_controller_only=len(ctrl), **d.mean().round(3).to_dict())


def main():
    jobs = [(t, s, q, r) for t in RUNS for s in SETUPS[t] for q in (2, 4, 8) for r in (0.135, 0.37, 0.61, 0.78, 0.88)]
    with multiprocessing.Pool(2) as pool:
        out = pd.DataFrame(pool.map(cell, jobs))
    out.to_csv(HERE / "data_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
