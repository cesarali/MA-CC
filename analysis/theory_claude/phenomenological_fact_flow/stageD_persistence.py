"""Stage D: the four continuations after T for model V1, same parameters as the reversal (P0; no retuning), with
selective and neutral reposting. delta(t) = Phi2(controlled) - Phi2(silent), each arm under the same continuation.
Prints delta at T, T+1, T+10, T+50, T+100 and the retained fraction; writes results/stageD_persistence.csv.
"""
import pathlib
from dataclasses import replace

import numpy as np
import pandas as pd

from fact_flow import P, run, phi2

HERE = pathlib.Path(__file__).resolve().parent
TS = np.array([30.0, 31.0, 40.0, 80.0, 130.0])


def delta_path(p, q, rho, mode):
    out = {}
    for arm in (True, False):
        f, s = run(p, q, rho, arm, t_end=130.0, mode=mode, t_eval=np.linspace(0, 130, 1301))
        y = np.concatenate([f.y[:, :-1], s.y], axis=1)
        t = np.concatenate([f.t[:-1], s.t])
        out[arm] = (t, phi2(p, y[0], y[1]), y)
    t, pc, yc = out[True]
    _, ps, _ = out[False]
    return t, pc - ps, yc


def main():
    rows = []
    for label, p in (("selective", P()), ("neutral", replace(P(), selective=False))):
        for rho in (0.25, 0.75):
            for mode in ("normal", "frozen", "no_comm", "purge"):
                t, d, yc = delta_path(p, 6, rho, mode)
                vals = {f"d{int(x)}": float(np.interp(x, t, d)) for x in TS}
                bC = {f"bC{int(x)}": float(np.interp(x, t, yc[3])) for x in (30.0, 130.0)}
                rows.append(dict(reposting=label, rho=rho, mode=mode, **vals, retained=vals["d130"] / vals["d30"], **bC))
    r = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    r.to_csv(HERE / "results" / "stageD_persistence.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
