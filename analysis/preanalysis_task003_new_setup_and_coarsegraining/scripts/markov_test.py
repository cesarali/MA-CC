"""Is an epistemic coarse-graining Markovian where the vote count is not?

Test.  Under a sufficient state variable, rounds in which the controller did NOT
act are passive by definition, so the transition kernel estimated from the silent
arm must also describe the gate-OFF rounds of controlled episodes.  The vote-count
state fails this.  We ask whether a state built from agent MEMORY does better.

Three candidate state variables are compared:
    V      binned truth-vote share            (the known-failing baseline)
    M      binned population-mean P(A0 | K_j) (epistemic load alone)
    MV     the pair (M, V)

Only the silent and A2-target arms are used.  The A0-target arm is excluded: its
recorded dynamics are sound, but the controller there recommends ALLOCATION_0
while quoting ALLOCATION_2-favouring evidence, so it is the confounded condition.
"""
from __future__ import annotations
import json, sys, itertools
sys.path.insert(0, "/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts")
import numpy as np, pandas as pd
from engine import World

REC = ("/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment/"
       "simulation_data/records/21-09-2026-full-vs-report-v1_analysis/")
W = World((3, 1, 1, 2, 2, 1, 1, 1, 2))
FIDX = {f: i for i, f in enumerate(W.ids)}
M_EDGES = [0.33, 0.50, 0.75]            # -> 4 epistemic levels
V_EDGES = [0.33, 0.66]                  # -> 3 vote levels

_cache: dict[frozenset, float] = {}
def epistemic(fact_ids) -> float:
    """P(ALLOCATION_0 | this agent's active facts), exactly."""
    key = frozenset(f for f in fact_ids if f in FIDX)
    if key not in _cache:
        _cache[key] = W.posterior([FIDX[f] for f in key])[0]
    return _cache[key]


def load() -> pd.DataFrame:
    r = pd.read_parquet(REC + "rounds.parquet",
                        columns=["cell_id", "episode_id", "round_index", "U_k",
                                 "active_fact_ids_by_agent_after",
                                 "population_state_after"])
    c = pd.read_parquet(REC + "cells.parquet",
                        columns=["cell_id", "intervention_budget",
                                 "controller_target_semantics"])
    d = r.merge(c, on="cell_id")
    d["arm"] = np.where(d.intervention_budget.isna(), "silent",
                        d.controller_target_semantics.map(
                            {"correct": "A0", "ALLOCATION_2": "A2"}))
    d = d[d.arm.isin(["silent", "A2"])].copy()      # drop the confounded arm
    mean_e, vote = [], []
    for _, row in d.iterrows():
        inv = json.loads(row.active_fact_ids_by_agent_after)
        packs = list(inv.values()) if isinstance(inv, dict) else inv
        mean_e.append(float(np.mean([epistemic(p) for p in packs if isinstance(p, list)])))
        votes = json.loads(row.population_state_after)
        vote.append(votes.count("ALLOCATION_0") / len(votes))
    d["e"] = mean_e
    d["x0"] = vote
    d["M"] = np.digitize(d.e, M_EDGES)
    d["V"] = np.digitize(d.x0, V_EDGES)
    d["MV"] = d.M * 10 + d.V
    return d


def kernel(df: pd.DataFrame, col: str, states) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised transition matrix and raw counts over consecutive rounds."""
    pos = {s: i for i, s in enumerate(states)}
    n = len(states)
    cnt = np.zeros((n, n))
    for _, g in df.sort_values("round_index").groupby(["cell_id", "episode_id"]):
        s = g[col].to_numpy()
        for a, b in zip(s[:-1], s[1:]):
            if a in pos and b in pos:
                cnt[pos[a], pos[b]] += 1
    tot = cnt.sum(axis=1, keepdims=True)
    return np.divide(cnt, tot, out=np.zeros_like(cnt), where=tot > 0), cnt


def compare(train: pd.DataFrame, test: pd.DataFrame, col: str) -> dict:
    """How well does the silent-arm kernel describe held-out passive rounds?

    Reports mean total-variation distance between the two row distributions
    (weighted by how often each state is visited in the test set) and a
    likelihood-ratio statistic against the test set's own kernel.
    """
    states = sorted(set(train[col]) | set(test[col]))
    Ptr, _ = kernel(train, col, states)
    Pte, Cte = kernel(test, col, states)
    w = Cte.sum(axis=1)
    mask = (w > 0) & (Ptr.sum(axis=1) > 0)
    tv = 0.5 * np.abs(Ptr - Pte).sum(axis=1)
    wtv = float((tv[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() else np.nan
    eps = 1e-12
    ll_tr = float((Cte[mask] * np.log(Ptr[mask] + eps)).sum())
    ll_te = float((Cte[mask] * np.log(Pte[mask] + eps)).sum())
    dof = int(mask.sum() * (len(states) - 1))
    return dict(variable=col, n_states=len(states), n_test_transitions=int(w.sum()),
                weighted_TV=wtv, loglik_silent_kernel=ll_tr, loglik_own_kernel=ll_te,
                LR_statistic=2 * (ll_te - ll_tr), dof=dof,
                LR_per_dof=2 * (ll_te - ll_tr) / max(dof, 1))


if __name__ == "__main__":
    d = load()
    d.to_parquet("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/results/data/states.parquet")
    silent = d[d.arm == "silent"]
    a2_off = d[(d.arm == "A2") & (d.U_k == 0)]
    a2_on = d[(d.arm == "A2") & (d.U_k == 1)]
    print("silent rows %d | A2 gate-OFF %d | A2 gate-ON %d" % (len(silent), len(a2_off), len(a2_on)), flush=True)
    out = []
    for col in ("V", "M", "MV"):
        res = compare(silent, a2_off, col)
        res["contrast"] = "silent -> A2 gate-OFF (should MATCH)"
        out.append(res)
        res2 = compare(silent, a2_on, col)
        res2["contrast"] = "silent -> A2 gate-ON (should DIFFER)"
        out.append(res2)
        print(res, flush=True)
    json.dump(out, open("/Users/rsanchez/Projects/MA-CC/analysis/preanalysis_task003_new_setup_and_coarsegraining/02_markov_tests/markov_results.json", "w"), indent=1)
    print("DONE", flush=True)
