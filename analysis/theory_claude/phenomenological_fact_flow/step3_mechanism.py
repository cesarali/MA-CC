"""Step 3 mechanism check: spread the controller's evidence over F_C = 16 facts (as many as the agents') instead of 4.
Prediction (decision_log.md): G shrinks towards the mean field, and fewer controller facts per controller... survive.
Corrected control: beta_C scaled so F_C * beta_C = 2. Writes findings/step3_mechanism_equal_evidence.csv
(the uncorrected first run is in step3_mechanism.csv)."""
import multiprocessing, pathlib
from dataclasses import replace
import numpy as np, pandas as pd
from finite_model import FP, episode
from fact_flow import P, G_at_T

HERE = pathlib.Path(__file__).resolve().parent


def job(args):
    FC, rho, s = args
    p = replace(FP(), F_C=FC, beta_C=0.5 * 4 / FC)        # same total controller evidence F_C * beta_C = 2
    out = []
    for controlled in (True, False):
        m = episode(p, 6, rho, controlled, seed=10_000 * s + 7)[30.0]
        out.append(dict(F_C=FC, rho=rho, seed=s, arm="false" if controlled else "silent", **m))
    return out


if __name__ == "__main__":
    jobs = [(fc, r, s) for fc in (4, 16) for r in (0.25, 0.5) for s in range(400)]
    with multiprocessing.Pool(7) as pool:
        d = pd.DataFrame([r for part in pool.map(job, jobs, chunksize=50) for r in part])
    piv = d.pivot_table(index=["F_C", "rho", "seed"], columns="arm", values="x2")
    g = (piv["false"] - piv["silent"]).groupby(["F_C", "rho"]).agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    cf = d[d.arm == "false"].groupby(["F_C", "rho"])[["distinct_C", "distinct_A"]].mean()
    out = g.join(cf).reset_index()
    out["G_meanfield"] = [G_at_T(replace(P(), eps=1.0, F_C=float(fc), beta_C=0.5 * 4 / fc), 6, r)[0] for fc, r in zip(out.F_C, out.rho)]
    out["ratio"] = out["mean"] / out["G_meanfield"]
    out.to_csv(HERE / "findings" / "step3_mechanism_equal_evidence.csv", index=False)
    print(out.round(4).to_string(index=False))
