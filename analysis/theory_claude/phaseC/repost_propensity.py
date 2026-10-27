"""Phase C: how likely is a held controller-only fact to be reposted by its holder, under the simulator's posting rule?

For every agent memory M stored in the false arm of the normal continuation (t = 30, 40, 80, 130), with the agent's
last vote v: the candidate set C(M, v) of choose_post (support in context, else fallback). For each controller-only
fact f held, kappa = 1[f in C] / |C|: the chance that f is the one posted if the agent posted from this memory and vote
now (ignoring the prefer-not-just-read tie-break). Same for agent facts. Averaged over holders.
Writes repost_propensity.csv.

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/phaseC/repost_propensity.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "theory_claude/phaseA"))
sys.path.insert(0, str(A / "task003_llm_free/simulation_2/simulator"))
from evidence_map import facts  # noqa: E402
from solvable_branches import RUNS, bits  # noqa: E402

EPS = 1e-12
GRID = {120: 30.0, 160: 40.0, 320: 80.0, 520: 130.0}


def candidates(w, m, v):
    fs = bits(m)
    pk = w.posterior(m)[v]
    sup = [f for f in fs if pk > w.posterior(m & ~(1 << f))[v] + EPS]
    if not sup:
        prior = w.posterior(0)[v]
        sup = [f for f in fs if w.posterior(1 << f)[v] > prior + EPS]
    return sup


def main():
    rows = []
    for task, run in RUNS.items():
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] != "normal" or p["arm"] != "false":
                continue
            w, agent, ctrl = facts(task, p["setup"])
            ctrl, agent = set(ctrl), set(agent)
            ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["grid_index", "active", "last_vote"],
                                 filters=[("grid_index", "in", list(GRID))])
            cache = {}
            for gi, t in GRID.items():
                kc, ka, held_c = [], [], []
                for m, v in zip(ag[ag.grid_index == gi].active.to_numpy(), ag[ag.grid_index == gi].last_vote.to_numpy()):
                    m, v = int(m), int(v)
                    if v < 0 or not m:
                        continue
                    if (m, v) not in cache:
                        cache[(m, v)] = candidates(w, m, v)
                    C = cache[(m, v)]
                    for f in bits(m):
                        k = (f in C) / len(C) if C else 0.0
                        if f in ctrl:
                            kc.append(k)
                        elif f in agent:
                            ka.append(k)
                    held_c.append(sum(1 for f in bits(m) if f in ctrl))
                rows.append(dict(task=task, setup=p["setup"].split("_", 1)[1], rho=float(p["rho"]), t=t,
                                 kappa_ctrl=np.mean(kc) if kc else np.nan, eligible_ctrl=np.mean([k > 0 for k in kc]) if kc else np.nan,
                                 kappa_agent=np.mean(ka) if ka else np.nan, eligible_agent=np.mean([k > 0 for k in ka]) if ka else np.nan,
                                 ctrl_per_memory=np.mean(held_c)))
            print(task, c.name, flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(HERE / "repost_propensity.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
