"""Does adding controller-exposure to the state restore agreement?

markov_test.py showed that a kernel fitted on silent episodes fails to describe
gate-OFF rounds of controlled episodes under three candidate state variables.
The cause is now identified: a gate-OFF round inside a controlled episode is not
passive in STATE -- 90% of its agents already hold a controller-supplied fact,
against 54% under silence.  So the test's premise was wrong, not merely its state
variable.

This script asks the repaired question: once exposure is part of the state, do
the kernels agree?  If they do, the dynamics are Markovian in (epistemic,
exposure) and the coarse-graining is usable.
"""
import json, sys
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/scripts")
import numpy as np, pandas as pd
from engine import World
from markov_test import kernel, compare, REC, M_EDGES, V_EDGES

W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
FIDX = {f: i for i, f in enumerate(W.ids)}
POOL = set(pd.read_csv("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/task_003_facts.csv")
           .query("allocation in ['pool only','both']").fact)
E_EDGES = [0.70]                       # exposure: low / high

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
                    columns=["cell_id", "intervention_budget", "controller_target_semantics"])
d = r.merge(c, on="cell_id")
d["arm"] = np.where(d.intervention_budget.isna(), "silent",
                    d.controller_target_semantics.map({"correct": "A0", "ALLOCATION_2": "A2"}))
d = d[d.arm.isin(["silent", "A2"])].copy()
me, xe, ex = [], [], []
for _, row in d.iterrows():
    inv = json.loads(row.active_fact_ids_by_agent_after)
    packs = [p for p in (inv.values() if isinstance(inv, dict) else inv) if isinstance(p, list)]
    me.append(float(np.mean([epi(p) for p in packs])) if packs else 1/3)
    ex.append(float(np.mean([len(set(p) & POOL) > 0 for p in packs])) if packs else 0.0)
    v = json.loads(row.population_state_after)
    xe.append(v.count("ALLOCATION_0") / len(v))
d["e"], d["x0"], d["exp"] = me, xe, ex
d["M"] = np.digitize(d.e, M_EDGES)
d["V"] = np.digitize(d.x0, V_EDGES)
d["E"] = np.digitize(d["exp"], E_EDGES)
d["ME"] = d.M * 10 + d.E
d["MVE"] = d.M * 100 + d.V * 10 + d.E

silent = d[d.arm == "silent"]
off = d[(d.arm == "A2") & (d.U_k == 0)]
on = d[(d.arm == "A2") & (d.U_k == 1)]
out = []
for col in ("M", "ME", "MVE"):
    a = compare(silent, off, col); a["contrast"] = "silent -> A2 gate-OFF (should MATCH)"
    b = compare(silent, on, col);  b["contrast"] = "silent -> A2 gate-ON (should DIFFER)"
    out += [a, b]
    print(col, "OFF TV=%.3f  ON TV=%.3f" % (a["weighted_TV"], b["weighted_TV"]), flush=True)
json.dump(out, open("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/02_markov_tests/markov_results2.json", "w"), indent=1)
print("DONE", flush=True)
