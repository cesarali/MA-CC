"""Shared helpers for the Simulation 2 analysis (ANALYSIS_PLAN.md).

Tables are built from each cell's params.json and the run manifest, never from folder names.
Output goes to results/simulation_2/analysis/<date>/ (not in git).
"""
from __future__ import annotations
import datetime, hashlib, json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent                     # simulation_2/analysis
ROOT = HERE.parents[1]                                             # analysis/task003_llm_free
SIM = HERE.parent / "simulator"
RESULTS = ROOT / "results" / "simulation_2"
SIM1_RESULTS = ROOT / "results" / "simulation_1"
sys.path.insert(0, str(ROOT))

TARGET = {"truth": 0, "false": 2}
SIZE = 1 / 24                                                       # one agent's worth of belief
B = 1000                                                            # bootstrap resamples
SETUP_NAME = {"task003_symmetric": "symmetric", "task003_nosolution": "nosolution"}
GATE = {"votes": "on", "always": "off"}


def run_dir(study: str) -> pathlib.Path:
    """The latest official run of a study, from the runs index."""
    idx = pd.read_csv(SIM / "runs_index.csv")
    rows = idx[idx.study == study]
    if rows.empty:
        raise FileNotFoundError(f"no official run of {study} in runs_index.csv")
    return ROOT / rows.iloc[-1].results_folder


def out_dir(date: str | None = None) -> pathlib.Path:
    d = RESULTS / "analysis" / (date or f"{datetime.date.today():%Y-%m-%d}")
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def cells_table(run: pathlib.Path) -> pd.DataFrame:
    """One row per cell, including reused cells (column `folder` = the cell that was run)."""
    manifest = json.loads((run / "manifest.json").read_text())
    reused = manifest.get("reused_cells", manifest.get("reused_silent_cells", {}))
    rows = []
    for folder in sorted(p for p in (run / "cells").iterdir() if p.is_dir()):
        prm = json.loads((folder / "params.json").read_text())
        prm["cell"] = prm.get("cell") or folder.name
        prm["folder"] = folder.name
        rows.append(prm)
    df = pd.DataFrame(rows)
    by_cell = df.set_index("cell")
    extra = []
    for skipped, kept in reused.items():
        r = by_cell.loc[kept].to_dict()
        r["cell"], r["folder"] = skipped, kept
        if "stopoff" in skipped:
            r["silent_when_target_proved"] = False
        elif skipped.startswith("S"):
            r["step"] = skipped.split("__")[0]
        extra.append(r)
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df
    df["run"] = run.name
    df["gate"] = df.controller_gate.map(GATE)
    df["stop"] = np.where(df.silent_when_target_proved, "on", "off")
    df["setup_name"] = df.setup.map(SETUP_NAME)
    return df.sort_values("cell").reset_index(drop=True)


def _episode_values(args):
    run, folder = args
    f = pathlib.Path(run) / "cells" / folder
    s = pd.read_parquet(f / "snapshots.parquet",
                        columns=["episode", "t", "mean_p_A0", "mean_p_A2", "share_A0", "share_A2", "share_voted"])
    out = s[s.t.isin([30.0, 40.0])].pivot(index="episode", columns="t")
    out.columns = [f"{c}_t{int(t)}" for c, t in out.columns]
    w = s[s.t <= 30.0].sort_values(["episode", "t"])
    for k in ("A0", "A2"):
        out[f"avg_p_{k}_0_30"] = w.groupby("episode").apply(
            lambda g: np.trapezoid(g[f"mean_p_{k}"], g.t) / 30.0, include_groups=False)
    ep = pd.read_parquet(f / "episodes.parquet").set_index("episode")
    out = out.join(ep[["controller_messages", "controller_reads", "budget_exhausted_at"]])
    out["folder"] = folder
    return out.reset_index()


def episode_values(run: pathlib.Path, folders, workers: int = 12) -> pd.DataFrame:
    """One row per (cell folder, episode): beliefs and votes at t = 30, 40, the [0, 30] average, dose."""
    jobs = [(str(run), f) for f in sorted(set(folders))]
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(_episode_values, jobs)
    return pd.concat(parts, ignore_index=True)


def _trajectory(args):
    run, folder = args
    s = pd.read_parquet(pathlib.Path(run) / "cells" / folder / "snapshots.parquet")
    g = s.drop(columns=["episode", "grid_index"]).groupby("t").mean().reset_index()
    g["folder"] = folder
    return g


def trajectories(run: pathlib.Path, folders, workers: int = 12) -> pd.DataFrame:
    """Means over episodes at every measurement time, per cell folder."""
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(_trajectory, [(str(run), f) for f in sorted(set(folders))])
    return pd.concat(parts, ignore_index=True)


def boot_index(n: int, name: str, b: int = B) -> np.ndarray:
    """b resamples of n episode numbers; the same for every cell compared under one name, so
    all cells of an episode stay together (plan §1). Seeded by sha256, reproducible."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def mean_ci(x: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% percentile interval of a per-episode quantity under resampling idx."""
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Ratio of means with a paired bootstrap interval."""
    r = num.mean() / den.mean() if den.mean() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return float(r), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def verdict(mean: float, lo: float, hi: float, size: float = SIZE) -> str:
    """Plan §6. effect: interval excludes 0 and |mean| >= size; no effect: interval within
    ±size; small effect: interval excludes 0 but |mean| < size; otherwise inconclusive.
    Differences below 1e-12 are rounding noise and count as 0."""
    mean, lo, hi = (0.0 if abs(x) < 1e-12 else x for x in (mean, lo, hi))
    if lo > 0 or hi < 0:
        return "effect" if abs(mean) >= size else "small effect"
    if -size <= lo and hi <= size:
        return "no effect"
    return "inconclusive"


def write_index(d: pathlib.Path, rows: list[dict]):
    """Append to the output folder's index of tables and figures."""
    f = d / "index.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~old.file.isin(new.file)]
    pd.concat([old, new], ignore_index=True).sort_values("file").to_csv(f, index=False)
