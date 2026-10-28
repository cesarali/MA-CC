"""Post-hoc (NOT pre-registered; prompted by the failure of Q5): mean world-coherence of the controller facts actually
on the board in the controlled arm, c_bar(t) = sum_f share_f(t) c_f, with c_f from coherence_facts.csv (silent memories,
no votes). Normal continuation, q 6, 300 episodes per cell; episodes with no controller post on the board are skipped
(the share skipped is reported). Writes board_coherence.csv.

Usage, from the repository root: .venv/bin/python analysis/theory_claude/fading_chain/posthoc_board_coherence.py
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fading_chain as fc  # noqa: E402

coh = pd.read_csv(HERE / "coherence_facts.csv")
rows = []
for task, (run, _, _) in fc.RUNS.items():
    for c in sorted((run / "cells").iterdir()):
        p = json.loads((c / "params.json").read_text())
        if not (p["post_stop_mode"] == "normal" and int(p["q"]) == 6 and p["arm"] == "false"):
            continue
        setup = p["setup"].split("_", 1)[1]
        cf = dict(coh[(coh.task == task) & (coh.setup == setup)][["fact", "c"]].itertuples(index=False))
        posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"])
        posts = posts[posts.episode < fc.EPISODES].sort_values(["episode", "t"])
        for t in (30.0, 130.0):
            vals = []
            for e, g in posts.groupby("episode"):
                win = [int(f) for f in g[g.t <= t].fact.to_numpy()[-fc.W:] if int(f) in cf]
                vals.append(np.mean([cf[f] for f in win]) if win else np.nan)
            v = np.array(vals)
            rows.append(dict(task=task, setup=setup, rho=float(p["rho"]), t=t, cbar=np.nanmean(v),
                             cbar_se=np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))), skipped=float(np.mean(np.isnan(v)))))
d = pd.DataFrame(rows)
d.to_csv(HERE / "board_coherence.csv", index=False)
w = d.pivot_table(index=["task", "setup", "rho"], columns="t", values="cbar").reset_index()
w["change"] = w[130.0] - w[30.0]
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
print(w.round(3).to_string(index=False))
