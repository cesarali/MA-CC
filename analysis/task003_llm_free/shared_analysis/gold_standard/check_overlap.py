"""Do the new light-record runs reproduce the old runs on the cells they share? (same seed, same code)

Compares, cell by cell and on every table both runs recorded, all episodes:
  Simulation 1: new cells of the given runs against 2026-10-07_sim1_pm (days.parquet, nights.parquet)
  Simulation 2: new cells against 2026-10-09_sim2_grid (old names end in __stopon) and
                2026-10-12_sim2_main_rho1_full (q 6, qc 12, rho 1; old names omit q, qc and rho)
                (snapshots.parquet, controller.parquet, episodes.parquet)
Prints one line per (simulation, old run): cells compared, cells identical, and the first differing cell, if any.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/check_overlap.py \
        simulation_1/2026-10-16_sim1_pm_nosolution_fine simulation_2/2026-10-16_sim2_nosolution_fine
"""
from __future__ import annotations
import json, pathlib, sys

import pandas as pd

RES = pathlib.Path(__file__).resolve().parents[2] / "results"
TABLES = {"sim1": ("days.parquet", "nights.parquet"), "sim2": ("snapshots.parquet", "controller.parquet", "episodes.parquet")}


def same(a: pathlib.Path, b: pathlib.Path) -> bool:
    """Equal on every column both tables have (light records leave out some columns, e.g. posts_read)."""
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    cols = [c for c in x.columns if c in y.columns]
    return len(cols) >= len(x.columns) - 1 and x[cols].reset_index(drop=True).equals(y[cols].reset_index(drop=True))


def counterparts(sim: str, cell: pathlib.Path):
    name = cell.name
    if sim == "sim1":
        yield "2026-10-07_sim1_pm", RES / "simulation_1/2026-10-07_sim1_pm/cells" / name
        return
    old = name if "__silent__" in name else name + "__stopon"
    yield "2026-10-09_sim2_grid", RES / "simulation_2/2026-10-09_sim2_grid/cells" / old
    prm = json.loads((cell / "params.json").read_text())
    if prm["q"] == 6 and prm.get("qc") in (12, None) and float(prm["rho"]) == 1.0:
        setup, arm = prm["setup"], prm["arm"]
        if arm == "silent":
            short = f"{setup}__silent"
        else:
            gate = "gateon" if prm["controller_gate"] == "votes" else "gateoff"
            short = f"{setup}__{arm}__lc{prm['lambda_c']:g}__{gate}__stopon"
        yield "2026-10-12_sim2_main_rho1_full", RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells" / short


def main(runs):
    stats = {}
    for run in runs:
        sim = "sim1" if run.startswith("simulation_1") else "sim2"
        for cell in sorted((RES / run / "cells").iterdir()):
            for old_run, old in counterparts(sim, cell):
                if not old.exists():
                    continue
                tabs = [t for t in TABLES[sim] if (cell / t).exists() and (old / t).exists()]
                ok = bool(tabs) and all(same(cell / t, old / t) for t in tabs)
                s = stats.setdefault((sim, old_run), [0, 0, None])
                s[0] += 1
                s[1] += ok
                if not ok and s[2] is None:
                    s[2] = cell.name
    for (sim, old_run), (n, k, bad) in stats.items():
        print(f"{sim} vs {old_run}: {n} shared cells, {k} identical" + (f"; first difference: {bad}" if bad else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
