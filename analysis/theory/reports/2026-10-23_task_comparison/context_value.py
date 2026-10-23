"""Hypothesis test: in Object Placements the controller's facts push A2 only in combination with particular other facts,
and these combinations dissolve after the controller stops; in task003 they push A2 on their own.

For reads S of q = 6 posts drawn from the 48 newest posts (false arm, normal continuation) that contain at least one
controller-only fact:
  value_in_context  = P(A2 | S) - P(A2 | S without its controller-only facts)
  value_alone       = P(A2 | controller-only facts of S) - 1/3          (the same facts with no other evidence)
averaged over reads (20 per episode, 300 episodes), at t = 30, 40, 80, 130. If the hypothesis holds, value_in_context
falls with t in Object Placements but not in task003, while value_alone stays flat in both.
Writes context_value.csv.

Usage, from the repository root:
    .venv/bin/python analysis/theory/reports/2026-10-23_task_comparison/context_value.py
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
    rng = np.random.default_rng(2)
    rows = []
    for task, (run, wf, sd) in RUNS.items():
        world = load_world(str(wf))
        cache = {}

        def p2(facts):
            key = frozenset(int(f) for f in facts)
            if key not in cache:
                m = 0
                for f in key:
                    m |= 1 << f
                cache[key] = world.posterior(m)[2]
            return cache[key]

        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] != "normal" or p["arm"] != "false":
                continue
            st = load_setup(p["setup"], sd, world)
            ctrl = set(st.pools[2]) - set(st.agent_facts)
            posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
            posts = posts[posts.episode < EPISODES]
            for t in TIMES:
                ctx, alone = [], []
                for e, g in posts.groupby("episode"):
                    win = g[g.t < t].fact.to_numpy()[-W:]
                    if len(win) == 0:
                        continue
                    for _ in range(DRAWS):
                        S = {int(f) for f in win[rng.choice(len(win), size=min(Q, len(win)), replace=False)]}
                        C = S & ctrl
                        if not C:
                            continue
                        ctx.append(p2(S) - p2(S - C))
                        alone.append(p2(C) - 1 / 3)
                rows.append(dict(task=task, setup=p["setup"].replace("task003_", "").replace("op_", ""),
                                 rho=float(p["rho"]), t=t, value_in_context=np.mean(ctx),
                                 se_in_context=np.std(ctx) / np.sqrt(len(ctx)), value_alone=np.mean(alone), reads=len(ctx)))
            print(task, c.name, flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(HERE / "context_value.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
