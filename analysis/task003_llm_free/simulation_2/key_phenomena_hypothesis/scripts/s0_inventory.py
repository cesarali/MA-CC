"""Step 0: inventory of every Simulation 2 run on disk.

For each cell folder under results/simulation_2/<run>/cells/ it reads params.json and lists the tables recorded, and
writes one row per cell to results/key_phenomena/inventory_cells.csv. It also prints and writes a grouped summary
(inventory_summary.csv): per run, setup, arm and recording level, the values of every factor that occurs.

Recording levels: "light" = snapshots, controller, episodes (vote shares, mean beliefs, proof rates every 0.25;
controller actions); "posts" = light + posts (every board post: time, author, fact, author's vote); "full" = posts +
actions, agent_snapshots, forgetting (each agent's remembered facts at every 0.25 and at every action).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s0_inventory.py
"""
from __future__ import annotations
import json, pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]          # analysis/task003_llm_free
RES = ROOT / "results"
OUT = RES / "key_phenomena"


def level(tables: set[str]) -> str:
    if {"actions", "agent_snapshots"} <= tables:
        return "full"
    if "posts" in tables:
        return "posts"
    return "light"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(p for p in (RES / "simulation_2").iterdir() if (p / "cells").is_dir()):
        man = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
        for cell in sorted(c for c in (run / "cells").iterdir() if c.is_dir()):
            prm = json.loads((cell / "params.json").read_text())
            tabs = {f.stem for f in cell.glob("*.parquet")}
            rows.append(dict(run=run.name, cell=cell.name, setup=prm["setup"].replace("task003_", ""), arm=prm["arm"],
                             rho=float(prm["rho"]), q=prm["q"], qc=prm.get("qc"), lambda_c=prm.get("lambda_c"),
                             gate="on" if prm.get("controller_gate") == "votes" else "off",
                             proof_stop=prm.get("silent_when_target_proved"), W=prm.get("W"), budget=prm.get("budget"),
                             t_control=prm.get("t_control"), t_end=prm.get("t_end"), seed=prm.get("seed"),
                             episodes=man.get("episodes_per_cell"), level=level(tabs),
                             tables=",".join(sorted(tabs))))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "inventory_cells.csv", index=False)
    fmt = lambda s: ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in sorted(s.dropna().unique()))  # noqa: E731
    g = (d.groupby(["run", "setup", "arm", "level"])
          .agg(cells=("cell", "size"), rho=("rho", fmt), q=("q", fmt), qc=("qc", fmt), lambda_c=("lambda_c", fmt),
               gate=("gate", fmt), W=("W", fmt), budget=("budget", fmt), t_end=("t_end", fmt), seed=("seed", fmt),
               episodes=("episodes", fmt))
          .reset_index())
    g.to_csv(OUT / "inventory_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 40)
    print(g.to_string(index=False))
    print(f"\n{len(d)} cells -> {OUT / 'inventory_cells.csv'}")


if __name__ == "__main__":
    main()
