"""Build the full per-fact classification table for task_003."""
import sys, json
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/src")
import pandas as pd
from engine import World
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.selective_design import (
    _greedy_decisive, _private_eligible, SelectiveThresholds)

V = (3, 1, 1, 2, 2, 1, 1, 1, 2)
w = World(V)
idx = TeamAllocationCompletionIndex()
th = SelectiveThresholds()
elig = _private_eligible(w.facts, idx, th)
elig_ids = {f.fact_id for f in elig}
dec_ids = {f.fact_id for f in _greedy_decisive(idx, elig, 0)}

# who actually holds each fact, read from the executed archive
B = ("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment/"
     "simulation_data/records/21-09-2026-full-vs-report-v1_analysis/")
r = pd.read_parquet(B + "rounds.parquet",
                    columns=["cell_id", "controller_report_fact_ids",
                             "initial_active_fact_ids_by_agent"])
c = pd.read_parquet(B + "cells.parquet", columns=["cell_id", "intervention_budget"])
d = r.merge(c, on="cell_id")
pool, packets = set(), set()
for v in d[d.intervention_budget.notna()].controller_report_fact_ids:
    if isinstance(v, str):
        try: pool.update(json.loads(v))
        except Exception: pass
for v in d.initial_active_fact_ids_by_agent:
    if isinstance(v, str):
        try:
            for a in json.loads(v):
                packets.update(a if isinstance(a, list) else [a])
        except Exception: pass

zero = w.posterior([])
rows = []
for i, f in enumerate(w.facts):
    post = w.posterior([i])
    m = idx.metrics_for_facts((f,))
    in_pool, in_pkt = f.fact_id in pool, f.fact_id in packets
    rows.append(dict(
        fact=f.fact_id, kind=f.kind,
        dP_A0=post[0]-zero[0], dP_A1=post[1]-zero[1], dP_A2=post[2]-zero[2],
        maxP=m.max_predictability, entropy=m.normalized_entropy,
        eligible=f.fact_id in elig_ids, decisive=f.fact_id in dec_ids,
        allocation=("both" if in_pool and in_pkt else "pool only" if in_pool
                    else "agent only" if in_pkt else "neither"),
        text=f.canonical_text))
df = pd.DataFrame(rows).sort_values("dP_A0", ascending=False)
df.to_csv("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/task_003_facts.csv", index=False)

# consistency checks
assert len(df) == 49
assert df.eligible.sum() == 41 and (~df.eligible).sum() == 8
assert df.decisive.sum() == 6 and df[df.decisive].eligible.all()
assert abs(df[["dP_A0", "dP_A1", "dP_A2"]].sum(axis=1)).max() < 1e-9, "deltas must sum to 0"
print("checks passed; 49 rows written")
print(df.allocation.value_counts().to_string())
