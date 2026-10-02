"""Add candidate hidden coordinates to the per-round state file.

These are the features the population mean discards.  If any of them predicts
the next macrostate *within* a macrostate, the mean is not a sufficient summary
and that feature is part of what is missing.

  spread    standard deviation of P(A0|K_j) ACROSS the 24 agents.  The mean says
            how informed the population is on average; the spread says whether
            that knowledge is concentrated in a few agents or shared out.  Two
            populations with identical means but different spreads have
            different abilities to broadcast a complete argument.
  dec_frac  fraction of agents holding at least one of the 6 decisive facts.
            Those six are the population's only complete route to the answer, so
            this measures whether the proof is still alive in the population.
  pool_frac fraction of agents holding at least one controller-pool fact.
            Measures how much of the controller's curated evidence is resident
            in memory -- the channel by which a finite-persistence population
            carries the controller's influence between active rounds.
"""
import json, sys
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
import numpy as np, pandas as pd
from engine import World

SC = "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/"
REC = ("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment/"
       "simulation_data/records/21-09-2026-full-vs-report-v1_analysis/")
W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
FIDX = {f: i for i, f in enumerate(W.ids)}
facts = pd.read_csv("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/task_003_facts.csv")
DEC = set(facts[facts.decisive].fact)
POOL = set(facts[facts.allocation.isin(["pool only", "both"])].fact)

_c = {}
def epi(p):
    k = frozenset(f for f in p if f in FIDX)
    if k not in _c:
        _c[k] = W.posterior([FIDX[f] for f in k])[0]
    return _c[k]

base = pd.read_parquet(SC + "results/data/states_all.parquet")
raw = pd.read_parquet(REC + "rounds.parquet",
                      columns=["cell_id", "episode_id", "round_index",
                               "active_fact_ids_by_agent_after"])
spread, dec, pool, nf = [], [], [], []
for _, row in raw.iterrows():
    inv = json.loads(row.active_fact_ids_by_agent_after)
    packs = [p for p in (inv.values() if isinstance(inv, dict) else inv) if isinstance(p, list)]
    if not packs:
        spread.append(0.0); dec.append(0.0); pool.append(0.0); nf.append(0.0); continue
    vals = [epi(p) for p in packs]
    spread.append(float(np.std(vals)))
    dec.append(float(np.mean([len(set(p) & DEC) > 0 for p in packs])))
    pool.append(float(np.mean([len(set(p) & POOL) > 0 for p in packs])))
    nf.append(float(np.mean([len(p) for p in packs])))
raw["spread"], raw["dec_frac"], raw["pool_frac"], raw["mean_nfacts"] = spread, dec, pool, nf
out = base.merge(raw[["cell_id", "episode_id", "round_index",
                      "spread", "dec_frac", "pool_frac", "mean_nfacts"]],
                 on=["cell_id", "episode_id", "round_index"], how="left")
out.to_parquet(SC + "results/data/states_full.parquet")
print("rows %d  columns %s" % (len(out), [c for c in out.columns if c in
      ("e", "spread", "dec_frac", "pool_frac", "mean_nfacts")]), flush=True)
print(out[["e", "spread", "dec_frac", "pool_frac", "mean_nfacts"]].describe().round(3).to_string(), flush=True)
print("DONE", flush=True)
