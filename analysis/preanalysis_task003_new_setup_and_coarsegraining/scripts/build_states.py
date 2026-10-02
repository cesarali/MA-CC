"""Per-round epistemic state for ALL three arms, including the confounded A0 arm.

The A0 arm is retained here so that target information I(Z;.) can be reported for
the design as executed.  It is labelled confounded wherever it is used: that
controller recommends ALLOCATION_0 while quoting ALLOCATION_2-favouring evidence.
"""
import json, sys
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
import numpy as np, pandas as pd
from engine import World

REC = ("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment/"
       "simulation_data/records/21-09-2026-full-vs-report-v1_analysis/")
W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
FIDX = {f: i for i, f in enumerate(W.ids)}
_c = {}
def epi(p):
    k = frozenset(f for f in p if f in FIDX)
    if k not in _c:
        _c[k] = W.posterior([FIDX[f] for f in k])[0]
    return _c[k]

r = pd.read_parquet(REC + "rounds.parquet",
                    columns=["cell_id", "episode_id", "round_index", "U_k",
                             "active_fact_ids_by_agent_after", "population_state_after"])
c = pd.read_parquet(REC + "cells.parquet",
                    columns=["cell_id", "intervention_budget",
                             "controller_target_semantics", "epistemic_persistence"])
d = r.merge(c, on="cell_id")
d["arm"] = np.where(d.intervention_budget.isna(), "silent",
                    d.controller_target_semantics.map({"correct": "A0", "ALLOCATION_2": "A2"}))
e, x0 = [], []
for _, row in d.iterrows():
    inv = json.loads(row.active_fact_ids_by_agent_after)
    packs = [p for p in (inv.values() if isinstance(inv, dict) else inv) if isinstance(p, list)]
    e.append(float(np.mean([epi(p) for p in packs])) if packs else 1/3)
    v = json.loads(row.population_state_after)
    x0.append(v.count("ALLOCATION_0") / len(v))
d["e"], d["x0"] = e, x0
d["posts"] = d.U_k.fillna(0) * d.intervention_budget.fillna(0)
d.drop(columns=["active_fact_ids_by_agent_after", "population_state_after"]).to_parquet(
    "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/results/data/states_all.parquet")
print("rows %d  arms %s" % (len(d), sorted(d.arm.unique())), flush=True)
print("DONE", flush=True)
