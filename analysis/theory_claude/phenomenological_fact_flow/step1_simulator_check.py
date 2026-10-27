"""Step 1 diagnosis: the simulator's per-fact holding share of agent facts and share of agents holding a proof of A0, at
t = 30, symmetric setups, silent and false arms, q 6, rho 0.25 / 0.5 / 0.75 (stored memories; read-only). Compared with
the model's m_A and z. This reads simulator data (allowed for qualitative comparison); the model does not import it.
Writes findings/step1_simulator_check.csv."""
import json, pathlib, sys
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_reading_mechanism",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup",
                    "task003_symmetric"),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-22_sim2_op_main",
                              A / "object_placements_llm_free/world/world_op1.json",
                              A / "object_placements_llm_free/experimental_setup", "op_symmetric")}
rows = []
for task, (run, wf, sd, setup) in RUNS.items():
    w = load_world(str(wf))
    st = load_setup(setup, sd, w)
    agent = sorted(set(st.agent_facts))
    for c in sorted((run / "cells").iterdir()):
        p = json.loads((c / "params.json").read_text())
        if p["setup"] != setup or int(p["q"]) != 6 or float(p["rho"]) not in (0.25, 0.5, 0.75):
            continue
        ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["grid_index", "active"], filters=[("grid_index", "==", 120)])
        mem = ag.active.to_numpy().astype(np.int64)
        share = [float(np.mean((mem >> f) & 1)) for f in agent]
        sn = pd.read_parquet(c / "snapshots.parquet", columns=["grid_index", "proof_rate_A0", "mean_active_facts"])
        sn = sn[sn.grid_index == 120]
        rows.append(dict(task=task, arm=p["arm"], rho=float(p["rho"]), per_fact_share_mean=np.mean(share),
                         per_fact_share_min=np.min(share), per_fact_share_max=np.max(share),
                         agent_facts_per_memory=float(np.mean([bin(int(m) & sum(1 << f for f in agent)).count("1") for m in mem])),
                         facts_per_memory=float(sn.mean_active_facts.mean()), proof_share=float(sn.proof_rate_A0.mean())))
r = pd.DataFrame(rows).sort_values(["task", "arm", "rho"])
r.to_csv(HERE / "findings" / "step1_simulator_check.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
