"""Step 2: effective evidence strengths (h0, beta_A, beta_C) of each simulator setup, measured from stored memories
(false arm, t = 30, q 6, rho 0.25/0.5/0.75 pooled), then put into the fact-flow model to compare crossover orders.
See decision_log.md (2026-10-28, step 2). Writes findings/step2_strengths.csv and findings/step2_model_crossovers.csv."""
import json, pathlib, sys
from dataclasses import replace
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402
from fact_flow import P, G_at_T  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_reading_mechanism",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup",
                    {"symmetric": "task003_symmetric", "no-proof": "task003_nosolution"}),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-22_sim2_op_main",
                              A / "object_placements_llm_free/world/world_op1.json",
                              A / "object_placements_llm_free/experimental_setup",
                              {"symmetric": "op_symmetric", "no-proof": "op_nosolution"})}
SIM_CROSS = {("Object Placements", "symmetric"): "0.25-0.375", ("task003", "symmetric"): "0.5-0.625",
             ("Object Placements", "no-proof"): "0.625-0.875", ("task003", "no-proof"): "0.75-1"}
CLIP = 1e-3


def logit(p):
    p = min(max(p, CLIP), 1 - CLIP)
    return np.log(p / (1 - p))


def strengths(task, kind):
    run, wf, sd, names = RUNS[task]
    w = load_world(str(wf))
    st = load_setup(names[kind], sd, w)
    agent, ctrl = sorted(set(st.agent_facts)), sorted(set(st.pools[2]) - set(st.agent_facts))
    am, cm = sum(1 << f for f in agent), sum(1 << f for f in ctrl)
    X, y, drops = [], [], []
    cache = {}
    rng = np.random.default_rng(3)
    for c in sorted((run / "cells").iterdir()):
        p = json.loads((c / "params.json").read_text())
        if p["setup"] != names[kind] or p["arm"] != "false" or int(p["q"]) != 6 or float(p["rho"]) not in (0.25, 0.5, 0.75):
            continue
        ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["grid_index", "active"], filters=[("grid_index", "==", 120)])
        for m in ag.active.to_numpy().astype(np.int64):
            m = int(m)
            if m not in cache:
                cache[m] = logit(w.posterior(m)[2])
            X.append((1.0, bin(m & cm).count("1"), bin(m & am).count("1")))
            y.append(cache[m])
            missing = [f for f in agent if not (m >> f) & 1]
            if missing and rng.random() < 0.2:
                f = int(rng.choice(missing))
                mm = m | (1 << f)
                if mm not in cache:
                    cache[mm] = logit(w.posterior(mm)[2])
                drops.append(cache[m] - cache[mm])
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return dict(task=task, setup=kind, h0=coef[0], beta_C=coef[1], beta_A=-coef[2], mean_drop_per_agent_fact=np.mean(drops),
                memories=len(y), sim_cross_36=SIM_CROSS[(task, kind)])


def model_cross(row):
    p = replace(P(), eps=1.0, h0=float(row["h0"]), beta_A=float(row["beta_A"]), beta_C=float(row["beta_C"]))
    rh = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    D = np.array([G_at_T(p, 6, r)[0] - G_at_T(p, 3, r)[0] for r in rh])
    cross = next((f"{rh[i]:g}-{rh[i+1]:g}" for i in range(len(D) - 1) if D[i] > 0 >= D[i + 1]), "none")
    first_neg = next((rh[i] for i in range(len(D)) if D[i] < 0), np.nan)
    return cross, first_neg, D


rows = [strengths(t, k) for t in RUNS for k in ("symmetric", "no-proof")]
s = pd.DataFrame(rows)
s.to_csv(HERE / "findings" / "step2_strengths.csv", index=False)
pd.set_option("display.width", 220)
print(s.round(3).to_string(index=False))
mc = []
for _, r in s.iterrows():
    cross, fn, D = model_cross(r)
    mc.append(dict(task=r.task, setup=r.setup, model_cross_36=cross, model_first_negative_rho=fn, sim_cross_36=r.sim_cross_36,
                   D_at_005=D[0], D_min=D.min()))
m = pd.DataFrame(mc)
m.to_csv(HERE / "findings" / "step2_model_crossovers.csv", index=False)
print(m.round(4).to_string(index=False))
