"""How persuasive is the board itself, and which controller-only facts circulate, after the stop (normal continuation)?

  read_p2      E[P(A2 | facts of q = 6 posts drawn from the 48 newest)], 20 draws per episode, false and silent arms
  ctrl_mix     share of each controller-only fact among controller-only posts in the window, t = 30 and 130
  single_p2    P(A2 | {f}) for each controller-only fact alone (exact)
Both tasks, both setups, rho 0.25 / 0.75. Writes board_persuasion.csv and ctrl_fact_mix.csv.

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-23_task_comparison/board_persuasion.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from carrier_after_stop import RUNS, W, load_world, load_setup  # noqa: E402

TIMES = [30.0, 40.0, 80.0, 130.0]
Q, DRAWS, EPISODES = 6, 20, 300


def main():
    rng = np.random.default_rng(1)
    rows, mix = [], []
    for task, (run, wf, sd) in RUNS.items():
        world = load_world(str(wf))
        cache = {}

        def p2(facts):
            key = frozenset(facts)
            if key not in cache:
                m = 0
                for f in key:
                    m |= 1 << int(f)
                cache[key] = world.posterior(m)[2]
            return cache[key]

        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] != "normal":
                continue
            st = load_setup(p["setup"], sd, world)
            ctrl = sorted(set(st.pools[2]) - set(st.agent_facts))
            posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
            posts = posts[posts.episode < EPISODES]
            setup = p["setup"].replace("task003_", "").replace("op_", "")
            for t in TIMES:
                vals, counts = [], pd.Series(0.0, index=ctrl)
                for e, g in posts.groupby("episode"):
                    win = g[g.t < t].fact.to_numpy()[-W:]
                    if len(win) == 0:
                        continue
                    for _ in range(DRAWS):
                        vals.append(p2(win[rng.choice(len(win), size=min(Q, len(win)), replace=False)]))
                    cw = pd.Series(win[np.isin(win, ctrl)]).value_counts()
                    counts = counts.add(cw, fill_value=0)
                rows.append(dict(task=task, setup=setup, rho=float(p["rho"]), arm=p["arm"], t=t, read_p2=np.mean(vals)))
                if p["arm"] == "false" and t in (30.0, 130.0):
                    tot = counts.sum()
                    for f in ctrl:
                        mix.append(dict(task=task, setup=setup, rho=float(p["rho"]), t=t, fact=world.fact_ids[f],
                                        share=counts[f] / tot if tot else 0.0, single_p2=p2([f])))
            print(task, c.name, flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(HERE / "board_persuasion.csv", index=False)
    m = pd.DataFrame(mix)
    m.round(4).to_csv(HERE / "ctrl_fact_mix.csv", index=False)
    v = r.pivot_table(index=["task", "setup", "rho", "t"], columns="arm", values="read_p2")
    v["diff"] = v["false"] - v["silent"]
    pd.set_option("display.width", 250)
    print(v.round(3).to_string())
    w = m.assign(wp2=m.share * m.single_p2).groupby(["task", "setup", "rho", "t"]).wp2.sum().unstack("t")
    print("share-weighted single-fact P(A2) of circulating controller-only facts")
    print(w.round(3).to_string())
    print(m.groupby(["task", "setup"]).single_p2.describe()[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
