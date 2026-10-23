"""Within-episode diversity of the controller's facts after the stop (false arm, normal continuation): mean number of
distinct controller-only facts among the 48 newest posts and among agent memories (union over agents), at
t = 30, 40, 80, 130. Tests whether drift removes the complementary controller facts from each episode's board while the
average shares across episodes stay constant. Writes ctrl_diversity.csv.

Usage, from the repository root:
    .venv/bin/python analysis/theory/reports/2026-10-23_task_comparison/ctrl_diversity.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from carrier_after_stop import RUNS, W, TIMES, load_world, load_setup  # noqa: E402


def main():
    rows = []
    for task, (run, wf, sd) in RUNS.items():
        world = load_world(str(wf))
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] != "normal" or p["arm"] != "false":
                continue
            st = load_setup(p["setup"], sd, world)
            ctrl = sorted(set(st.pools[2]) - set(st.agent_facts))
            cmask = sum(1 << f for f in ctrl)
            posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
            ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                                 filters=[("grid_index", "in", list(TIMES))])
            pe = {e: g for e, g in posts.groupby("episode")}
            for gi, t in TIMES.items():
                nb, nm = [], []
                for e, g in ag[ag.grid_index == gi].groupby("episode"):
                    win = pe[e][pe[e].t < t].fact.to_numpy()[-W:]
                    nb.append(len(set(win.tolist()) & set(ctrl)))
                    u = int(np.bitwise_or.reduce(g.active.to_numpy().astype(np.uint64))) & cmask
                    nm.append(bin(u).count("1"))
                rows.append(dict(task=task, setup=p["setup"].replace("task003_", "").replace("op_", ""),
                                 rho=float(p["rho"]), t=t, distinct_ctrl_board=np.mean(nb), distinct_ctrl_memory=np.mean(nm),
                                 n_ctrl_facts=len(ctrl)))
            print(task, c.name, flush=True)
    r = pd.DataFrame(rows)
    r.round(3).to_csv(HERE / "ctrl_diversity.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
