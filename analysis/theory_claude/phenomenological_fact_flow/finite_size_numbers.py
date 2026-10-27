"""Effective finite sizes in the simulator at t = 30 (false arm, q 6, rho 0.25 / 0.75; stored data, read-only):
copies of each distinct fact on the board (48 newest posts) and holders per fact among the 24 agents, separately for
agent facts and controller-only facts. Writes findings/finite_size_numbers.csv."""
import json, pathlib, sys
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_reading_mechanism",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup"),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-22_sim2_op_main",
                              A / "object_placements_llm_free/world/world_op1.json", A / "object_placements_llm_free/experimental_setup")}
rows = []
for task, (run, wf, sd) in RUNS.items():
    w = load_world(str(wf))
    for c in sorted((run / "cells").iterdir()):
        p = json.loads((c / "params.json").read_text())
        if p["arm"] != "false" or int(p["q"]) != 6 or float(p["rho"]) not in (0.25, 0.75):
            continue
        st = load_setup(p["setup"], sd, w)
        agent, ctrl = set(st.agent_facts), set(st.pools[2]) - set(st.agent_facts)
        posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"])
        posts = posts[posts.t <= 30.0].sort_values(["episode", "t"])
        ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"], filters=[("grid_index", "==", 120)])
        stats = {"agent": [], "ctrl": []}
        for e, g in posts.groupby("episode"):
            win = g.fact.to_numpy()[-48:]
            vals, cnt = np.unique(win, return_counts=True)
            mem = ag[ag.episode == e].active.to_numpy().astype(np.int64)
            for kind, S in (("agent", agent), ("ctrl", ctrl)):
                on = [(int(v), int(k)) for v, k in zip(vals, cnt) if int(v) in S]
                holders = [int(np.sum((mem >> f) & 1)) for f in S]
                held = [h for h in holders if h > 0]
                stats[kind].append(dict(distinct_on_board=len(on), copies_per_board_fact=np.mean([k for _, k in on]) if on else 0.0,
                                        board_share=sum(k for _, k in on) / len(win), distinct_held=len(held),
                                        holders_per_held_fact=np.mean(held) if held else 0.0))
        for kind, lst in stats.items():
            d = pd.DataFrame(lst).mean()
            rows.append(dict(task=task, setup=p["setup"].split("_", 1)[1], rho=float(p["rho"]), kind=kind,
                             n_facts=len(agent if kind == "agent" else ctrl), **d.round(3).to_dict()))
r = pd.DataFrame(rows)
r.to_csv(HERE / "findings" / "finite_size_numbers.csv", index=False)
pd.set_option("display.width", 220)
print(r.to_string(index=False))
