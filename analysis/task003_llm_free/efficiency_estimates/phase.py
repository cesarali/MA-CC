"""Efficiency over the whole control plane, for phase diagrams (point estimates at t = 30).

Simulation 2: every qc (3, 6, 12, 24) x rate x gate, q 6, rho 0.75 / 1, stop on.
Simulation 1: every qc x b (1, 2, 3, 6, 9), q 6, rho 0.75 / 1, probability matching (sim1_pm).
Same estimator as efficiency.py (measures(), resolutions()); no bootstrap (precision is
known from downsample.py). Reads only the recorded snapshot / day tables, present in every cell.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/phase.py [--date <date>]
Output: results/efficiency_estimates/<date>/efficiency_phase.csv
"""
from __future__ import annotations
import argparse, datetime, multiprocessing

import numpy as np
import pandas as pd

from build_states import RATES, SETUPS, SIM1, SIM2_GRID, fmt
from efficiency import RES, measures, resolutions

T = 30.0


def load(job) -> pd.DataFrame:
    if job["sim"] == "sim2":
        s = pd.read_parquet(SIM2_GRID / job["cell"] / "snapshots.parquet", filters=[("t", "==", T)],
                            columns=["episode", "mean_p_A0", "mean_p_A1", "mean_p_A2", "share_A0", "share_A1",
                                     "share_A2", "share_voted"])
        xn = 1 - s.share_voted.to_numpy()
    else:
        s = pd.read_parquet(SIM1 / job["cell"] / "days.parquet", filters=[("day", "==", int(T) - 1)],
                            columns=["episode", "mean_p_A0", "mean_p_A1", "mean_p_A2", "share_A0", "share_A1", "share_A2"])
        xn = 0.0
    out = pd.DataFrame({"episode": s.episode.to_numpy(), "m0": s.mean_p_A0.to_numpy(), "m1": s.mean_p_A1.to_numpy(),
                        "m2": s.mean_p_A2.to_numpy(), "x0": s.share_A0.to_numpy(), "x1": s.share_A1.to_numpy(),
                        "x2": s.share_A2.to_numpy(), "xn": xn})
    return out.sort_values("episode").reset_index(drop=True)


def triplets():
    out = []
    for setup in SETUPS:
        for rho in (0.75, 1.0):
            r1 = "0.75" if rho == 0.75 else "1.0"
            for qc in (3, 6, 12, 24):
                for lc in RATES:
                    for gate in ("on", "off"):
                        c = {a: f"{setup}__{a}__q6__qc{qc}__rho{fmt(rho)}__lc{fmt(lc)}__gate{gate}__stopon"
                             for a in ("truth", "false")}
                        out.append(dict(sim="sim2", setup=setup, rho=rho, qc=qc, x=lc, gate=gate,
                                        silent=f"{setup}__silent__q6__rho{fmt(rho)}", **c))
                for b in (1, 2, 3, 6, 9):
                    c = {a: f"{setup}__{a}__q6__qc{qc}__b{b}__rho{r1}" for a in ("truth", "false")}
                    out.append(dict(sim="sim1", setup=setup, rho=rho, qc=qc, x=b, gate="on",
                                    silent=f"{setup}__silent__q6__rho{r1}", **c))
    return out


def facts_posted(sim, cell) -> float:
    """Mean facts the controller posted per episode (Simulation 2: messages, one fact each)."""
    if sim == "sim2":
        return float(pd.read_parquet(SIM2_GRID / cell / "episodes.parquet", columns=["controller_messages"])
                     .controller_messages.mean())
    n = pd.read_parquet(SIM1 / cell / "nights.parquet", columns=["episode", "n_posted"])
    eps = pd.read_parquet(SIM1 / cell / "days.parquet", columns=["episode"]).episode.nunique()
    return float(n.n_posted.sum() / eps)


def one(tr) -> list[dict]:
    F = {"+": load(dict(sim=tr["sim"], cell=tr["truth"])), "-": load(dict(sim=tr["sim"], cell=tr["false"])),
         "0": load(dict(sim=tr["sim"], cell=tr["silent"]))}
    facts = {"plus": facts_posted(tr["sim"], tr["truth"]), "minus": facts_posted(tr["sim"], tr["false"])}
    rows = []
    for name, level, source, fy in resolutions():
        if name not in ("agent_votes", "agent_beliefs", "swarm_beliefs_k3", "swarm_votes_k3"):
            continue
        tg = (0, 2) if level == "agent" else None
        m = {k: float(v) for k, v in measures(*(fy(F[x]).mean(axis=0) for x in ("+", "-", "0")), targets=tg).items()}
        for sign in ("plus", "minus"):
            m[f"facts_{sign}"] = facts[sign]
            if tg is not None:          # price: minimum information injected per fact posted (bits per fact)
                m[f"S_per_fact_{sign}"] = m[f"S_{sign}"] / facts[sign] if facts[sign] > 0 else np.nan
        rows.append({**{k: tr[k] for k in ("sim", "setup", "rho", "qc", "x", "gate")}, "resolution": name, **m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    d = RES / (a.date or f"{datetime.date.today():%Y-%m-%d}")
    with multiprocessing.Pool(12) as pool:
        parts = pool.map(one, triplets())
    out = pd.DataFrame([r for p in parts for r in p])
    out.to_csv(d / "efficiency_phase.csv", index=False)
    print(f"{len(out)} rows -> {d / 'efficiency_phase.csv'}")


if __name__ == "__main__":
    main()
