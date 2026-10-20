"""Step 1.5 (secondary): the no-proof backfire, the false controller's gain against messages and distinct facts
delivered, and mean beliefs against the full-evidence posterior (0.75, 0, 0.25).

For nosolution, q = 6, qc = 12, gate on, rho in {1, 0.75, 0.5}, every rate, budgets 30 and 100, at t = 30:
  G, messages (from the 2026-10-16 / 2026-10-18 tables), distinct facts delivered per episode (controller.parquet,
  posts before t = 30), mean beliefs m0 = P(A0), m2 = P(A2) in the false and silent arms.
Writes results/key_phenomena/backfire.parquet.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s1_backfire.py
"""
from __future__ import annotations
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results"
OUT = RES / "key_phenomena"
RUNS = {30: "2026-10-16_sim2_nosolution_fine", 100: "2026-10-18_sim2_nosolution_budget100"}
TABLES = {30: "2026-10-16", 100: "2026-10-18"}


def main():
    inv = pd.read_csv(OUT / "inventory_cells.csv")
    rows = []
    for B, run in RUNS.items():
        t = pd.read_parquet(RES / "gold_standard" / TABLES[B] / "cells_time.parquet")
        t = t[(t.sim == "sim2") & (t.setup == "task003_nosolution") & (t.q == 6) & (t.t == 30.0)]
        if "case" in t.columns:                       # the 2026-10-18 tables hold all three variants
            t = t[t.case == "budget100"]
        sil = t[t.arm == "silent"].set_index("rho")
        cells = inv[(inv.run == run) & (inv.arm == "false") & (inv.q == 6) & (inv.qc == 12) & (inv.gate == "on")
                    & inv.rho.isin([1.0, 0.75, 0.5])]
        for r in cells.itertuples():
            c = pd.read_parquet(RES / "simulation_2" / run / "cells" / r.cell / "controller.parquet",
                                columns=["episode", "t", "post_id", "posted_fact"])
            c = c[(c.post_id >= 0) & (c.t < 30)]
            distinct = c.groupby("episode").posted_fact.nunique().reindex(range(1000), fill_value=0).mean()
            g = t[(t.cell == r.cell)].iloc[0]
            rows.append(dict(budget=B, rho=r.rho, lambda_c=r.lambda_c, G=g.G, G_se=g.G_se, messages=g.facts,
                             distinct_facts=distinct, m0=g.m0, m2=g.m2, m0_silent=sil.loc[r.rho, "m0"],
                             m2_silent=sil.loc[r.rho, "m2"]))
    d = pd.DataFrame(rows).sort_values(["rho", "budget", "lambda_c"])
    d.to_parquet(OUT / "backfire.parquet", index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
