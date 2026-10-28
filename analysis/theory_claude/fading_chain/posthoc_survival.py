"""Post-hoc check (NOT pre-registered; prompted by fading_chain.py's result that Object Placements controller facts
push both ways). Within each episode: among controller-exclusive facts on the board at t = 30, do those still on the
board at t = 130 have a different persuasion at t = 30 than those lost?

  s_f = mean over controlled-arm agents at t = 30 not holding f of P(A2 | M + f) - P(A2 | M)
  share_pos_f = share of those agents for whom the push is positive
Paired within episode: mean over survivors minus mean over lost facts (episodes having both). Writes survival.csv.

Usage, from the repository root: .venv/bin/python analysis/theory_claude/fading_chain/posthoc_survival.py
"""
from __future__ import annotations
import json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fading_chain as fc  # noqa: E402


def cell(args):
    task, ccell = args
    run, wf, sd = fc.RUNS[task]
    p = json.loads((pathlib.Path(ccell) / "params.json").read_text())
    w = fc.load_world(str(wf))
    st = fc.load_setup(p["setup"], sd, w)
    ctrl = set(st.pools[2]) - set(st.agent_facts)
    cache = {}

    def P2(m):
        if m not in cache:
            cache[m] = w.posterior(m)[2]
        return cache[m]

    mem, facts, times = fc.load(pathlib.Path(ccell))
    rows = []
    for e in sorted(facts):
        if (e, 30.0) not in mem:
            continue
        f_all, t_all = facts[e], times[e]
        on30 = {int(f) for f in f_all[t_all <= 30.0][-fc.W:] if int(f) in ctrl}
        on130 = {int(f) for f in f_all[t_all <= 130.0][-fc.W:] if int(f) in ctrl}
        for f in on30:
            d = [P2(m | 1 << f) - P2(m) for m in mem[(e, 30.0)] if not m >> f & 1]
            if d:
                rows.append(dict(task=task, setup=p["setup"].split("_", 1)[1], rho=float(p["rho"]), episode=e, fact=f,
                                 s=np.mean(d), share_pos=np.mean(np.array(d) > 1e-12), survives=f in on130))
    return rows


def main():
    jobs = []
    for task, (run, _, _) in fc.RUNS.items():
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] == "normal" and int(p["q"]) == 6 and p["arm"] == "false":
                jobs.append((task, str(c)))
    with multiprocessing.Pool(8) as pool:
        d = pd.DataFrame([r for part in pool.map(cell, jobs, chunksize=1) for r in part])
    out = []
    for key, g in d.groupby(["task", "setup", "rho"]):
        diffs, dpos = [], []
        for _, ge in g.groupby("episode"):
            a, b = ge[ge.survives], ge[~ge.survives]
            if len(a) and len(b):
                diffs.append(a.s.mean() - b.s.mean())
                dpos.append(a.share_pos.mean() - b.share_pos.mean())
        diffs, dpos = np.array(diffs), np.array(dpos)
        out.append(dict(zip(["task", "setup", "rho"], key), facts=len(g), survive_share=g.survives.mean(),
                        s_survivors=g[g.survives].s.mean(), s_lost=g[~g.survives].s.mean(), episodes_paired=len(diffs),
                        paired_diff_s=diffs.mean(), paired_diff_s_se=diffs.std(ddof=1) / np.sqrt(len(diffs)),
                        paired_diff_sharepos=dpos.mean(), paired_diff_sharepos_se=dpos.std(ddof=1) / np.sqrt(len(dpos))))
    o = pd.DataFrame(out)
    o.to_csv(HERE / "survival.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
