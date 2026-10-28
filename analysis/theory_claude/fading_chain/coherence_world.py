"""Step 2, Q5 (pre-registered in ../../theory_blackboard/2026-10-28_claude_13.md): coherence of each controller fact,
read from the world on silent-arm memories (no controller, no votes).

  c_f = share of silent-arm memories at t = 30 (normal continuation, q 6, rho 0.25 and 0.75 pooled, 300 episodes per
        cell) in which adding f raises P(A2) (by more than 1e-12)
Silent memories never hold controller-exclusive facts. Writes coherence_facts.csv, coherence_summary.csv.

Also a post-hoc run of the selection model (Q4 as run did not put the majority on A2): h0 raised until the controlled
arm's Phi2 exceeds 1/2 at t = 30, ambivalent set, eps 0.5. Writes selection_majority_posthoc.csv.

Usage, from the repository root: .venv/bin/python analysis/theory_claude/fading_chain/coherence_world.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fading_chain as fc  # noqa: E402
import selection_model as sm  # noqa: E402


def coherence():
    rows = []
    for task, (run, wf, sd) in fc.RUNS.items():
        w = fc.load_world(str(wf))
        by_setup = {}
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] == "normal" and int(p["q"]) == 6 and p["arm"] == "silent":
                by_setup.setdefault(p["setup"], []).append(c)
        for setup, cells in by_setup.items():
            st = fc.load_setup(setup, sd, w)
            ctrl = sorted(set(st.pools[2]) - set(st.agent_facts))
            mems = []
            for c in cells:
                mem, _, _ = fc.load(c)
                mems += [m for (e, t), Ms in mem.items() if t == 30.0 for m in Ms]
            cache = {}

            def P2(m):
                if m not in cache:
                    cache[m] = w.posterior(m)[2]
                return cache[m]
            uniq, counts = np.unique(np.array(mems, dtype=object), return_counts=True)
            for f in ctrl:
                d = np.array([P2(int(m) | 1 << f) - P2(int(m)) for m in uniq])
                rows.append(dict(task=task, setup=setup.split("_", 1)[1], fact=f,
                                 c=float(np.sum(counts * (d > 1e-12)) / counts.sum()),
                                 c_neg=float(np.sum(counts * (d < -1e-12)) / counts.sum()),
                                 mean_dP=float(np.sum(counts * d) / counts.sum())))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "coherence_facts.csv", index=False)
    s = d.groupby(["task", "setup"]).agg(facts=("c", "size"), c_mean=("c", "mean"), c_sd=("c", "std"), c_min=("c", "min"),
                                         c_max=("c", "max"), below_half=("c", lambda x: int((x < 0.5).sum()))).reset_index()
    s.to_csv(HERE / "coherence_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print(s.round(3).to_string(index=False))


def majority_posthoc():
    rows = []
    c = sm.SETS["ambivalent"]
    for rho in (0.25, 0.75):
        for h0 in np.arange(0.0, 6.01, 0.25):
            p = dict(sm.P0, h0=float(h0))
            yc, ys = sm.solve(c, p, 0.5, 6, rho, True), sm.solve(c, p, 0.5, 6, rho, False)
            ob = sm.observables(yc, ys, c, p, 0.5)
            if ob[0]["Phi2_c"] > 0.5:
                rows.append(dict(rho=rho, h0=h0, Phi2_c30=ob[0]["Phi2_c"], cbar30=ob[0]["cbar"], cbar130=ob[-1]["cbar"],
                                 ret_delta=ob[-1]["delta"] / ob[0]["delta"], ret_bC=ob[-1]["b_C"] / ob[0]["b_C"]))
                break
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "selection_majority_posthoc.csv", index=False)
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    majority_posthoc()
    coherence()
