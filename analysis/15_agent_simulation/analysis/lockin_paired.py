"""Paired persistence of A2 lock-ins (correction of the 3 Nov metastability report, 8 Nov 2026).

The report divided two marginal frequencies, P(L130)/P(L30), which does not estimate the share of the SAME episodes
still locked in (ChatGPT, blackboard 2026-11-08_chatgpt_03). Here, per episode of the false arm (standard reading):
  L_t = A2 share among voters >= 2/3 at time t
  paired      P(L130 | L30)
  never_left  share of L30 episodes with A2 >= 2/3 at every whole time unit from 30 to 130
  hysteresis  share of L30 episodes never below 1/2 on the same grid
All cells with P(L30) >= 0.2. Writes ../reports/2026-11-03_metastability/table_lockin_paired.csv.
Usage, from the repository root: .venv/bin/python analysis/15_agent_simulation/analysis/lockin_paired.py
"""
import json, multiprocessing, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RUNS = [HERE.parent / "results/2026-10-29_sim2_15agents_task003", HERE.parent / "results/2026-10-29_sim2_15agents_op"]


def one(c):
    c = pathlib.Path(c)
    p = json.loads((c / "params.json").read_text())
    d = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "t", "share_A2", "share_voted"])
    d = d[(d.t >= 30) & np.isclose(d.t % 1, 0)]
    d["s"] = d.share_A2 / d.share_voted.where(d.share_voted > 0)
    w = d.pivot(index="episode", columns="t", values="s")
    L30, L130 = w[30.0] >= 2 / 3 - 1e-12, w[130.0] >= 2 / 3 - 1e-12
    if L30.mean() < 0.2:
        return None
    return dict(setup=p["setup"], q=p["q"], rho=p["rho"], P_L30=L30.mean(), P_L130=L130.mean(),
                marginal_ratio=L130.mean() / L30.mean(), paired=(L30 & L130).sum() / L30.sum(),
                never_left=(w.loc[L30] >= 2 / 3 - 1e-12).all(axis=1).mean(),
                hysteresis=(w.loc[L30] >= 0.5).all(axis=1).mean())


def main():
    cells = [str(c) for r in RUNS for c in (r / "cells").iterdir()
             if "__false__" in c.name and c.name.endswith("readposts")]
    with multiprocessing.Pool(12) as pool:
        rows = [r for r in pool.map(one, cells) if r]
    d = pd.DataFrame(rows).sort_values(["setup", "q", "rho"])
    out = HERE.parent / "reports/2026-11-03_metastability/table_lockin_paired.csv"
    d.to_csv(out, index=False)
    print(d.groupby("setup")[["marginal_ratio", "paired", "never_left", "hysteresis"]].median().round(2).to_string())
    print("cells:", len(d), "->", out)


if __name__ == "__main__":
    main()
