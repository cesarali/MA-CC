"""Models M3 and M4 against the K-projects simulations (post-hoc: these data were seen before the models were run).

Data: tables/scaling_K.csv (compare_K.py; G at t = 30 for q = 3, 6 and D(3 -> 6) with paired standard errors, K = 1,
2, 4). Models: analysis/theory/results/theory_runs/<tag>_k2.parquet, <tag>_k4.parquet (run_models.py --set k2/k4) and
<tag>_all.parquet for K = 1. A model increment counts as positive / negative when |D| >= 0.01.
Writes tables/scaling_K_models.csv and prints a summary per model and K.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/score_models_K.py
"""
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parents[3] / "theory" / "results" / "theory_runs"     # analysis/theory
TAB = HERE.parent / "tables"


def model_G(tag, K):
    f = RUNS / (f"{tag}_all.parquet" if K == 1 else f"{tag}_k{K}.parquet")
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d = d[np.isclose(d.t, 30.0) & d.q.isin([3, 6]) & d.rho.isin([0.25, 0.5, 0.75, 1.0])]
    p = d.pivot_table(index=["setup", "rho", "q"], columns="arm", values="x2")
    return (p["false"] - p["silent"]).unstack("q")


def main():
    data = pd.read_csv(TAB / "scaling_K.csv")
    rows = []
    for tag in ("M3", "M4"):
        for K in (1, 2, 4):
            g = model_G(tag, K)
            if g is None:
                continue
            for r in data[data.K == K].itertuples():
                if (r.setup, r.rho) not in g.index:
                    continue
                m3, m6 = g.loc[(r.setup, r.rho), 3], g.loc[(r.setup, r.rho), 6]
                dm = m6 - m3
                ds = "+" if r.D_3_6 > 1.96 * r.D_3_6_se else ("-" if r.D_3_6 < -1.96 * r.D_3_6_se else "0")
                ms = "+" if dm >= 0.01 else ("-" if dm <= -0.01 else "0")
                rows.append(dict(model=tag, K=K, setup=r.setup, rho=r.rho, G3_data=r.G_q3, G3_model=m3, G6_data=r.G_q6,
                                 G6_model=m6, D_data=r.D_3_6, D_model=dm, data_sign=ds, model_sign=ms))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TAB / "scaling_K_models.csv", index=False)
    for (tag, K), d in out.groupby(["model", "K"]):
        res = d[d.data_sign != "0"]
        err = pd.concat([(d.G3_model - d.G3_data).abs(), (d.G6_model - d.G6_data).abs()]).mean()
        print(f"{tag} K={K}: D(3->6) signs {int((res.data_sign == res.model_sign).sum())}/{len(res)} resolved; "
              f"mean |G error| {err:.3f}; misses: " +
              "; ".join(f"{x.setup} {x.rho:g} data {x.data_sign} model {x.model_sign}"
                        for x in res[res.data_sign != res.model_sign].itertuples()))


if __name__ == "__main__":
    main()
