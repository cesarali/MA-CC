"""Revised step 2: does vote-gated reposting (lower eps) make persistence fade in the finite model? Qualitative check.
Writes findings/step2_gating_finite.csv."""
import multiprocessing, pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from finite_model import FP, episode

HERE = pathlib.Path(__file__).resolve().parent
TS = (30.0, 50.0, 80.0, 130.0)


def job(args):
    eps, rho, s = args
    p = replace(FP(), eps=eps)
    out = []
    for controlled in (True, False):
        rec = episode(p, 6, rho, controlled, seed=10_000 * s + 7, t_end=130.0, record=TS)
        for t, m in rec.items():
            out.append(dict(eps=eps, rho=rho, seed=s, arm="false" if controlled else "silent", t=t, **m))
    return out


if __name__ == "__main__":
    jobs = [(e, r, s) for e in (1.0, 0.75, 0.5) for r in (0.25, 0.75) for s in range(300)]
    with multiprocessing.Pool(7) as pool:
        d = pd.DataFrame([x for part in pool.map(job, jobs, chunksize=20) for x in part])
    piv = d.pivot_table(index=["eps", "rho", "t", "seed"], columns="arm", values="x2")
    g = (piv["false"] - piv["silent"]).groupby(["eps", "rho", "t"]).agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    dc = d[d.arm == "false"].groupby(["eps", "rho", "t"]).distinct_C.mean()
    out = g.join(dc).reset_index()
    out.to_csv(HERE / "findings" / "step2_gating_finite.csv", index=False)
    w = out.pivot_table(index=["eps", "rho"], columns="t", values="mean")
    w["retained"] = w[130.0] / w[30.0]
    w["distinct_C_130"] = out[out.t == 130.0].set_index(["eps", "rho"]).distinct_C
    print(w.round(3).to_string())
