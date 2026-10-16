"""Per-episode state table for the control-efficiency study (PLAN.md §1, §4).

One row per (simulation, setting, arm, episode, time) with:
  m0, m1, m2      mean beliefs over the 24 agents
  x0, x1, x2, xn  vote shares (last votes; xn = not voted yet, Simulation 2 only)
  Hbar, JSD       mean individual belief entropy and disagreement (bits), where per-agent
                  beliefs are recorded (Simulation 1: every cell; Simulation 2: full-table cells)

Settings: Simulation 2 grid, q 6, qc 12, rho 0.75 / 1, every rate, gate on/off, stop on, t = 0..40;
Simulation 1 sim1_pm, q 6, qc 12, b 1 / 2 / 3 / 6 / 9, rho 0.75 / 1, end of days 1..30.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/efficiency_estimates/build_states.py [--date <date>]
Output: results/efficiency_estimates/<date>/states.parquet
"""
from __future__ import annotations
import argparse, datetime, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                              # analysis/task003_llm_free
sys.path.insert(0, str(ROOT))
from llmfree_core.world import World  # noqa: E402

RES = ROOT / "results"
SIM2_GRID = RES / "simulation_2" / "2026-10-09_sim2_grid" / "cells"
SIM2_RHO1 = RES / "simulation_2" / "2026-10-12_sim2_main_rho1_full" / "cells"
SIM1 = RES / "simulation_1" / "2026-10-07_sim1_pm" / "cells"
WORLD_FILE = ROOT / "llmfree_core" / "world_task003.json"
SETUPS = ("task003_symmetric", "task003_nosolution")
RATES = (0.5, 1.0, 2.0, 4.0, 8.0)


def entropy_bits(P: np.ndarray) -> np.ndarray:
    """Entropy along the last axis, in bits; 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(P > 0, P * np.log2(P), 0.0), axis=-1)


def uncertainty(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P: (..., agents, 3) beliefs. Returns (Hbar, JSD) with H(mean) = Hbar + JSD."""
    hbar = entropy_bits(P).mean(axis=-1)
    jsd = entropy_bits(P.mean(axis=-2)) - hbar
    return hbar, jsd


def fmt(x):
    return f"{x:g}"


def sim2_jobs():
    jobs = []
    for setup in SETUPS:
        for rho in (0.75, 1.0):
            silent = f"{setup}__silent__q6__rho{fmt(rho)}"
            jobs.append(dict(sim="sim2", setup=setup, rho=rho, setting="silent", arm="silent", cell=silent))
            for lc in RATES:
                for gate in ("on", "off"):
                    for arm in ("truth", "false"):
                        cell = f"{setup}__{arm}__q6__qc12__rho{fmt(rho)}__lc{fmt(lc)}__gate{gate}__stopon"
                        jobs.append(dict(sim="sim2", setup=setup, rho=rho, setting=f"lc{fmt(lc)}_gate{gate}",
                                         arm=arm, cell=cell))
    return jobs


def sim1_jobs():
    jobs = []
    for setup in SETUPS:
        for rho in (0.75, 1.0):
            r = "0.75" if rho == 0.75 else "1.0"
            jobs.append(dict(sim="sim1", setup=setup, rho=rho, setting="silent", arm="silent",
                             cell=f"{setup}__silent__q6__rho{r}"))
            for b in (1, 2, 3, 6, 9):
                for arm in ("truth", "false"):
                    jobs.append(dict(sim="sim1", setup=setup, rho=rho, setting=f"b{b}", arm=arm,
                                     cell=f"{setup}__{arm}__q6__qc12__b{b}__rho{r}"))
    return jobs


def run_sim2(job) -> pd.DataFrame:
    grid = SIM2_GRID / job["cell"]
    s = pd.read_parquet(grid / "snapshots.parquet",
                        columns=["episode", "t", "mean_p_A0", "mean_p_A1", "mean_p_A2", "share_A0", "share_A1",
                                 "share_A2", "share_voted"])
    s = s[(s.t % 1) == 0].copy()
    out = pd.DataFrame({"episode": s.episode.to_numpy(), "t": s.t.to_numpy(),
                        "m0": s.mean_p_A0.to_numpy(), "m1": s.mean_p_A1.to_numpy(), "m2": s.mean_p_A2.to_numpy(),
                        "x0": s.share_A0.to_numpy(), "x1": s.share_A1.to_numpy(), "x2": s.share_A2.to_numpy(),
                        "xn": 1 - s.share_voted.to_numpy()})
    # the rho-1 rerun has one value of q, qc and rho, so its cell names leave them out
    rerun = job["cell"].replace("__q6", "").replace("__qc12", "").replace("__rho1", "")
    full = grid if job["rho"] == 0.75 else SIM2_RHO1 / rerun
    if (full / "agent_snapshots.parquet").exists():
        world = World(WORLD_FILE)
        a = pd.read_parquet(full / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active"])
        a = a[a.grid_index % 4 == 0].sort_values(["episode", "grid_index", "agent"])
        P = np.array([world.posterior(int(m)) for m in a.active.to_numpy()]).reshape(-1, 24, 3)
        hbar, jsd = uncertainty(P)
        keys = a[["episode", "grid_index"]].drop_duplicates().to_numpy()
        u = pd.DataFrame({"episode": keys[:, 0], "t": keys[:, 1] / 4.0, "Hbar": hbar, "JSD": jsd})
        out = out.merge(u, on=["episode", "t"], how="left")
    for k in ("sim", "setup", "rho", "setting", "arm"):
        out[k] = job[k]
    return out


def run_sim1(job) -> pd.DataFrame:
    cell = SIM1 / job["cell"]
    d = pd.read_parquet(cell / "days.parquet", columns=["episode", "day", "mean_p_A0", "mean_p_A1", "mean_p_A2",
                                                        "share_A0", "share_A1", "share_A2"])
    out = pd.DataFrame({"episode": d.episode.to_numpy(), "t": d.day.to_numpy() + 1.0,
                        "m0": d.mean_p_A0.to_numpy(), "m1": d.mean_p_A1.to_numpy(), "m2": d.mean_p_A2.to_numpy(),
                        "x0": d.share_A0.to_numpy(), "x1": d.share_A1.to_numpy(), "x2": d.share_A2.to_numpy(),
                        "xn": 0.0})
    a = pd.read_parquet(cell / "agents.parquet", columns=["episode", "day", "agent", "p_A0", "p_A1", "p_A2"])
    a = a.sort_values(["episode", "day", "agent"])
    P = a[["p_A0", "p_A1", "p_A2"]].to_numpy(np.float64).reshape(-1, 24, 3)
    P = P / P.sum(axis=-1, keepdims=True)                           # stored as float32
    hbar, jsd = uncertainty(P)
    keys = a[["episode", "day"]].drop_duplicates().to_numpy()
    u = pd.DataFrame({"episode": keys[:, 0], "t": keys[:, 1] + 1.0, "Hbar": hbar, "JSD": jsd})
    out = out.merge(u, on=["episode", "t"], how="left")
    for k in ("sim", "setup", "rho", "setting", "arm"):
        out[k] = job[k]
    return out


def run(job):
    return run_sim2(job) if job["sim"] == "sim2" else run_sim1(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    d = RES / "efficiency_estimates" / (a.date or f"{datetime.date.today():%Y-%m-%d}")
    d.mkdir(parents=True, exist_ok=True)
    jobs = sim2_jobs() + sim1_jobs()
    with multiprocessing.Pool(12) as pool:
        parts = pool.map(run, jobs)
    states = pd.concat(parts, ignore_index=True)
    states.to_parquet(d / "states.parquet", index=False)
    print(f"{len(jobs)} cells, {len(states):,} rows -> {d / 'states.parquet'}")
    print(states.groupby(["sim", "rho"]).Hbar.apply(lambda s: s.notna().mean()).rename("share with Hbar"))


if __name__ == "__main__":
    main()
