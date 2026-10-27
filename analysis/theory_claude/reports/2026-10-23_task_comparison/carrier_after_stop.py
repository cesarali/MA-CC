"""After the controller stops (t = 30), normal continuation: does the controller's imprint fade because its facts
stop circulating, or because they stop persuading? Both tasks, both setups, rho 0.25 / 0.75, q 6.

Per (task, setup, rho, arm) and t in {30, 40, 80, 130}, averaged over episodes:
  board_ctrl   share of the 48 newest posts that carry a controller-only fact (pool facts no agent starts with)
  mem_ctrl     mean number of controller-only facts per agent memory
  agent_facts  number of distinct agent-origin facts still held by some agent or on the board
  p2_belief    mean P(A2 | memory) over agents (exact posterior), and x2 the vote share of A2 among voters
  proof_A0     share of agents whose memory proves A0 (from the simulator's snapshots)
Writes carrier_after_stop.csv next to this script.

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-23_task_comparison/carrier_after_stop.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[2]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_persistence_branches",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup"),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-23_sim2_op_post_stop",
                              A / "object_placements_llm_free/world/world_op1.json",
                              A / "object_placements_llm_free/experimental_setup")}
TIMES = {120: 30.0, 160: 40.0, 320: 80.0, 520: 130.0}
W = 48


def popcount_mask(arr, mask):
    x = np.bitwise_and(arr.astype(np.uint64), np.uint64(mask))
    return np.array([bin(int(v)).count("1") for v in x])


def main():
    rows = []
    for task, (run, wf, sd) in RUNS.items():
        world = load_world(str(wf))
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] != "normal":
                continue
            st = load_setup(p["setup"], sd, world)
            agent = set(st.agent_facts)
            ctrl = set(st.pools[2]) - agent
            cmask = sum(1 << f for f in ctrl)
            amask = sum(1 << f for f in agent)
            posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
            ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                                 filters=[("grid_index", "in", list(TIMES))])
            sn = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "grid_index", "mean_p_A2", "share_A2",
                                                                    "share_voted", "proof_rate_A0"])
            sn = sn[sn.grid_index.isin(list(TIMES))]
            pe = {e: g for e, g in posts.groupby("episode")}
            for gi, t in TIMES.items():
                bc, af = [], []
                a_t = ag[ag.grid_index == gi]
                for e, g in a_t.groupby("episode"):
                    pp = pe.get(e)
                    win = pp[pp.t < t].fact.to_numpy()[-W:] if pp is not None else np.array([], dtype=int)
                    bc.append(np.isin(win, list(ctrl)).mean() if len(win) else 0.0)
                    held = int(np.bitwise_or.reduce(g.active.to_numpy().astype(np.uint64))) & amask
                    onb = {int(f) for f in win} & agent
                    af.append(len({f for f in agent if (held >> f) & 1} | onb))
                s = sn[sn.grid_index == gi]
                rows.append(dict(task=task, setup=p["setup"].replace("task003_", "").replace("op_", ""), rho=float(p["rho"]),
                                 arm=p["arm"], t=t, board_ctrl=np.mean(bc),
                                 mem_ctrl=popcount_mask(a_t.active.to_numpy(), cmask).mean(),
                                 agent_facts=np.mean(af), n_agent_facts=len(agent),
                                 p2_belief=s.mean_p_A2.mean(), x2=(s.share_A2 / s.share_voted.where(s.share_voted > 0)).mean(),
                                 proof_A0=s.proof_rate_A0.mean()))
            print(task, c.name, flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(HERE / "carrier_after_stop.csv", index=False)
    f = out[out.arm == "false"].set_index(["task", "setup", "rho", "t"])
    s = out[out.arm == "silent"].set_index(["task", "setup", "rho", "t"])
    view = f[["board_ctrl", "mem_ctrl", "agent_facts"]].join((f[["p2_belief", "x2", "proof_A0", "agent_facts"]] -
                                                            s[["p2_belief", "x2", "proof_A0", "agent_facts"]]).add_suffix("_diff"))
    pd.set_option("display.width", 250)
    print(view.round(3).to_string())


if __name__ == "__main__":
    main()
