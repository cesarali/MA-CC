"""How well can each quantity be estimated from 50, 100 or 300 paths per arm? (PLAN.md §5 step 2)

For each n_sub, 200 random subsets of episode numbers (without replacement), the SAME subsets
for every arm and every setting, so comparisons between settings stay paired. Each quantity is
recomputed on each subset and compared with its value on all 1,000 episodes.

Usage, from the repository root (after build_states.py):
    .venv/bin/python analysis/task003_llm_free/control_efficiency/downsample.py [--date <date>]
Outputs in results/control_efficiency/<date>/:
  downsample.csv  per setting x time x resolution x n_sub x quantity: full value, mean, sd, bias,
                  RMSE, share of subsets where the value is infinite / undefined
  ranking.csv     neighbouring rates and gate on/off: how often a subset orders the two settings
                  as the full sample does
"""
from __future__ import annotations
import argparse, datetime

import numpy as np
import pandas as pd

from efficiency import RES, arm_arrays, measures, resolutions

N_SUB = (50, 100, 300)
REPS = 200
QUANTITIES = ("I", "C", "eta", "eta_JS", "I3")
TIMES = {"sim1": (30.0,), "sim2": (30.0, 40.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    d = RES / (a.date or f"{datetime.date.today():%Y-%m-%d}")
    states = pd.read_parquet(d / "states.parquet")
    n = states.episode.nunique()
    rng = np.random.default_rng(20261012)
    subsets = {m: np.array([rng.choice(n, m, replace=False) for _ in range(REPS)]) for m in N_SUB}
    rows, reps = [], {}
    for key, arms in arm_arrays(states).items():
        sim, setup, rho, setting = key
        for t in TIMES[sim]:
            F = {x: arms[x][arms[x].t == t].sort_values("episode") for x in arms}
            for name, level, source, fy in resolutions():
                Y = {x: fy(F[x]) for x in F}
                full = measures(*(Y[x].mean(axis=0) for x in ("+", "-", "0")))
                for m, idx in subsets.items():
                    sub = measures(*(Y[x][idx].mean(axis=1) for x in ("+", "-", "0")))
                    for q in QUANTITIES:
                        v, f = sub[q], float(full[q])
                        ok = np.isfinite(v)
                        rows.append(dict(sim=sim, setup=setup, rho=rho, setting=setting, t=t, resolution=name,
                                         level=level, source=source, n_sub=m, quantity=q, full=f,
                                         mean=v[ok].mean() if ok.any() else np.nan,
                                         sd=v[ok].std() if ok.any() else np.nan,
                                         bias=v[ok].mean() - f if ok.any() and np.isfinite(f) else np.nan,
                                         rmse=np.sqrt(((v[ok] - f) ** 2).mean()) if ok.any() and np.isfinite(f) else np.nan,
                                         share_not_finite=1 - ok.mean()))
                        reps[(sim, setup, rho, setting, t, name, m, q)] = (f, v)
    pd.DataFrame(rows).to_csv(d / "downsample.csv", index=False)

    # ranking: neighbouring rates (same gate) and gate on vs off (same rate), Simulation 2
    rank = []
    rates = ["0.5", "1", "2", "4", "8"]
    pairs = []
    for g in ("on", "off"):
        pairs += [(f"lc{r1}_gate{g}", f"lc{r2}_gate{g}", "rate") for r1, r2 in zip(rates, rates[1:])]
    pairs += [(f"lc{r}_gateon", f"lc{r}_gateoff", "gate") for r in rates]
    for (sim, setup, rho, setting, t, name, m, q), (f1, v1) in reps.items():
        for s1, s2, kind in pairs:
            if sim != "sim2" or setting != s1:
                continue
            other = reps.get((sim, setup, rho, s2, t, name, m, q))
            if other is None:
                continue
            f2, v2 = other
            if not (np.isfinite(f1) and np.isfinite(f2)) or f1 == f2:
                continue
            ok = np.isfinite(v1) & np.isfinite(v2)
            agree = np.sign(v2[ok] - v1[ok]) == np.sign(f2 - f1)
            rank.append(dict(setup=setup, rho=rho, t=t, resolution=name, n_sub=m, quantity=q, kind=kind,
                             setting_a=s1, setting_b=s2, full_diff=f2 - f1,
                             share_same_order=agree.mean() if ok.any() else np.nan, share_usable=ok.mean()))
    pd.DataFrame(rank).to_csv(d / "ranking.csv", index=False)
    print(f"{len(rows)} downsample rows, {len(rank)} ranking rows in {d}")


if __name__ == "__main__":
    main()
