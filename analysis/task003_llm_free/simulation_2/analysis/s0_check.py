"""S0 check (spec §9): does S0, Simulation 1's rules rebuilt in the new code, reproduce Simulation 1?

The two codes draw different random numbers, so the check is statistical. For every quantity
on the fixed list, z = (S0 mean - Simulation 1 mean) / standard error of the difference, from
per-episode values. S0 passes if no |z| exceeds 4.5 and at most 10% of |z| exceed 2.

Compared with the Simulation 1 run 2026-10-07_sim1_pm, cells q6 / qc12 / b1 / rho0.75 and
silent q6 / rho0.75. Days 1-30 only: S0's measurement at t = d is the end of day d.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/s0_check.py <Simulation 2 run folder>
Output: <run folder>/s0_check.csv, and the verdict printed.
"""
from __future__ import annotations
import pathlib, sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]                 # analysis/task003_llm_free
SIM1_RUN = ROOT / "results" / "simulation_1" / "2026-10-07_sim1_pm"
ABSTAINED = 4
DAILY = ["mean_p_A0", "mean_p_A2", "share_A0", "share_A2", "proof_rate_A0", "abstention_rate"]
Z_MAX, SHARE_OVER_2 = 4.5, 0.10


def sim1_cell(setup: str, arm: str) -> pathlib.Path:
    name = f"{setup}__silent__q6__rho0.75" if arm == "silent" else f"{setup}__{arm}__q6__qc12__b1__rho0.75"
    return SIM1_RUN / "cells" / name


def sim1_values(setup: str, arm: str):
    cell = sim1_cell(setup, arm)
    days = pd.read_parquet(cell / "days.parquet")
    days["day"] = days["day"] + 1                                  # Simulation 1 counts days from 0
    per_ep = None
    if arm != "silent":
        n = pd.read_parquet(cell / "nights.parquet", columns=["episode", "decision", "n_posted"])
        per_ep = n.groupby("episode").agg(facts_posted=("n_posted", "sum"),
                                          nights_gate=("decision", lambda d: (d == "gate").sum()),
                                          nights_proved=("decision", lambda d: (d == "proved").sum()))
    return days[["episode", "day"] + DAILY], per_ep


def sim2_values(cell: pathlib.Path, arm: str):
    snap = pd.read_parquet(cell / "snapshots.parquet")
    snap = snap[(snap.t >= 1) & (snap.t <= 30) & ((snap.t % 1) == 0)].copy()
    snap["day"] = snap.t.astype(int)
    act = pd.read_parquet(cell / "actions.parquet", columns=["episode", "t", "post_reason"])
    act["day"] = act.t.astype(int) + 1
    ab = act.groupby(["episode", "day"]).post_reason.apply(lambda r: (r == ABSTAINED).mean()).rename("abstention_rate")
    days = snap.merge(ab.reset_index(), on=["episode", "day"])
    per_ep = None
    if arm != "silent":
        c = pd.read_parquet(cell / "controller.parquet", columns=["episode", "decision", "posted_fact"])
        eps = pd.read_parquet(cell / "episodes.parquet", columns=["episode"]).episode
        per_ep = c.groupby("episode").agg(facts_posted=("posted_fact", lambda f: (f >= 0).sum()),
                                          nights_gate=("decision", lambda d: (d == "gate").sum()),
                                          nights_proved=("decision", lambda d: (d == "proved").sum()))
        per_ep = per_ep.reindex(eps, fill_value=0)
    return days[["episode", "day"] + DAILY], per_ep


def z_score(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    if se == 0:
        return a.mean(), b.mean(), 0.0 if diff == 0 else np.inf, se
    return a.mean(), b.mean(), diff / se, se


def check(run: pathlib.Path) -> pd.DataFrame:
    rows = []
    for setup in ("task003_symmetric", "task003_nosolution"):
        for arm in ("silent", "truth", "false"):
            cell = run / "cells" / f"S0__{setup}__{arm}"
            d1, e1 = sim1_values(setup, arm)
            d2, e2 = sim2_values(cell, arm)
            for day in range(1, 31):
                x1, x2 = d1[d1.day == day], d2[d2.day == day]
                for q in DAILY:
                    m2, m1, z, se = z_score(x2[q].to_numpy(), x1[q].to_numpy())
                    rows.append((setup, arm, q, day, m2, m1, se, z))
            if e1 is not None:
                for q in e1.columns:
                    m2, m1, z, se = z_score(e2[q].to_numpy(float), e1[q].to_numpy(float))
                    rows.append((setup, arm, q, None, m2, m1, se, z))
    return pd.DataFrame(rows, columns=["setup", "arm", "quantity", "day", "s0_mean", "sim1_mean", "se", "z"])


def main():
    run = pathlib.Path(sys.argv[1]).resolve()
    out = check(run)
    out.to_csv(run / "s0_check.csv", index=False)
    z = out.z.abs()
    over2, worst = (z > 2).mean(), out.loc[z.idxmax()]
    passed = z.max() <= Z_MAX and over2 <= SHARE_OVER_2
    print(f"{len(out)} comparisons; |z| > 2: {over2:.1%} (limit {SHARE_OVER_2:.0%}); max |z| = {z.max():.2f} (limit {Z_MAX})")
    print(f"largest: {worst.setup} {worst.arm} {worst.quantity} day {worst.day}: "
          f"S0 {worst.s0_mean:.4f} vs Simulation 1 {worst.sim1_mean:.4f}")
    print("by quantity, share with |z| > 2:")
    print(out.assign(over2=z > 2).groupby("quantity").over2.mean().round(3).to_string())
    print("S0 CHECK:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
