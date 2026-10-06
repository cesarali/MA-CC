"""Run the Simulation 1 grid from simulation_1_config.yaml.

Each cell writes three Parquet tables to <output>/<cell id>/:
  agents.parquet  one row per agent per day
  nights.parquet  one row per night (controlled arms only)
  days.parquet    one row per day: population averages
and the run writes manifest.json (settings, input file hashes, git commit).

Examples, from the repository root:
  # smoke test: 5 episodes of every cell
  .venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --episodes 5
  # one cell, to measure speed
  .venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --episodes 50 \\
      --only task003_nosolution__false__q6__qc12__b3__rho0.75
  # the full grid on 12 cores
  .venv/bin/python analysis/task003_llm_free/simulator/run_simulation_1.py --workers 12
"""
from __future__ import annotations
import argparse, datetime, itertools, json, multiprocessing, pathlib, platform, subprocess, sys, time

import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llmfree.setups import load_setup  # noqa: E402
from llmfree.simulation_1 import (AGENT_COLUMNS, DAY_COLUMNS, NIGHT_COLUMNS, Params,  # noqa: E402
                                  params_dict, run_episode)
from llmfree.world import World  # noqa: E402

GRID_KEYS = ("setup", "arm", "q", "qc", "b", "rho")


def build_cells(cfg: dict) -> list[Params]:
    grid, fixed = cfg["grid"], cfg["fixed"]
    unknown = set(grid) - set(GRID_KEYS)
    if unknown:
        raise ValueError(f"unknown grid factors: {sorted(unknown)}")
    ignored = set(cfg["cells"].get("silent_ignores", []))
    if not all(isinstance(x, str) for x in grid["arm"]):
        raise ValueError(f"arm names must be quoted strings in the YAML, got {grid['arm']!r}")
    fixed_params = {k: v for k, v in fixed.items() if k != "population_size"}
    cells, seen = [], set()
    for values in itertools.product(*(grid[k] for k in GRID_KEYS)):
        d = dict(zip(GRID_KEYS, values))
        if d["arm"] == "silent":
            for k in ignored:
                d[k] = None
        p = Params(**d, **fixed_params)
        if p.cell_id() not in seen:
            seen.add(p.cell_id())
            cells.append(p)
    return cells


_WORLD = _SETUPS = None


def _init_worker(world_file: str, setups_dir: str, setup_names: list[str]):
    global _WORLD, _SETUPS
    _WORLD = World(world_file)
    _SETUPS = {s: load_setup(s, setups_dir, _WORLD) for s in setup_names}


def _run_cell(job):
    p, episodes, out_dir = job
    t0 = time.time()
    setup = _SETUPS[p.setup]
    agents, nights, days = [], [], []
    for e in range(episodes):
        a, n, d = run_episode(_WORLD, setup, p, e)
        agents += a
        nights += n
        days += d
    folder = pathlib.Path(out_dir) / p.cell_id()
    folder.mkdir(parents=True, exist_ok=True)
    da = pd.DataFrame(agents, columns=AGENT_COLUMNS).astype(
        {"episode": "int32", "day": "int16", "position": "int16", "agent": "int16", "active_facts": "int64",
         "facts_read": "int64", "n_posts_read": "int16", "p_A0": "float32", "p_A1": "float32",
         "p_A2": "float32", "vote": "int8", "posted_fact": "int8", "proves_A0": "bool"})
    da.to_parquet(folder / "agents.parquet", index=False)
    pd.DataFrame(days, columns=DAY_COLUMNS).to_parquet(folder / "days.parquet", index=False)
    if nights:
        pd.DataFrame(nights, columns=NIGHT_COLUMNS).astype({"posted_facts": "int64"}).to_parquet(
            folder / "nights.parquet", index=False)
    (folder / "params.json").write_text(json.dumps(params_dict(p), indent=2) + "\n")
    last = pd.DataFrame(days, columns=DAY_COLUMNS).query(f"day == {p.M - 1}")
    return {"cell": p.cell_id(), "seconds": round(time.time() - t0, 1), "episodes": episodes,
            "final_share_A0": round(last.share_A0.mean(), 4), "final_share_A2": round(last.share_A2.mean(), 4),
            "final_mean_p_A0": round(last.mean_p_A0.mean(), 4)}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "simulation_1_config.yaml"))
    ap.add_argument("--episodes", type=int, help="override episodes_per_cell (for tests and timing)")
    ap.add_argument("--only", action="append", default=[], help="run only cells whose id contains this; repeatable")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", help="output folder (default: the config's output.directory)")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    base = cfg_path.parent
    world_file = str((base / cfg["paths"]["world_file"]).resolve())
    setups_dir = str((base / cfg["paths"]["setups_dir"]).resolve())

    cells = build_cells(cfg)
    expected = cfg["cells"]["expected_total"]
    if len(cells) != expected:
        sys.exit(f"REFUSING TO RUN: the config gives {len(cells)} cells, expected_total says {expected}.")
    warnings = sorted({w for p in cells for w in p.check()})
    for w in warnings:
        print("WARNING:", w)
    if args.only:
        cells = [p for p in cells if any(s in p.cell_id() for s in args.only)]
        if not cells:
            sys.exit("no cell matches --only")
    episodes = args.episodes or cfg["cells"]["episodes_per_cell"]
    out = pathlib.Path(args.out) if args.out else (base.parent / cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)

    world = World(world_file)
    setup_names = sorted({p.setup for p in cells})
    setups = {s: load_setup(s, setups_dir, world) for s in setup_names}
    manifest = {
        "study": cfg["study"], "config": str(cfg_path), "config_contents": cfg,
        "cells_in_grid": expected, "cells_run": [p.cell_id() for p in cells], "episodes_per_cell": episodes,
        "world_file_sha256": world.sha256,
        "setup_file_sha256": {s: setups[s].file_hashes for s in setup_names},
        "git_commit": _git_commit(), "python": platform.python_version(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"), "warnings": warnings,
    }
    print(f"{len(cells)} cells x {episodes} episodes on {args.workers} worker(s) -> {out}")

    jobs = [(p, episodes, str(out)) for p in cells]
    results = []
    t0 = time.time()
    if args.workers > 1:
        with multiprocessing.Pool(args.workers, _init_worker, (world_file, setups_dir, setup_names)) as pool:
            for r in pool.imap_unordered(_run_cell, jobs):
                results.append(r)
                print(f"[{len(results)}/{len(jobs)}] {r['cell']}: {r['seconds']} s")
    else:
        _init_worker(world_file, setups_dir, setup_names)
        for job in jobs:
            r = _run_cell(job)
            results.append(r)
            print(f"[{len(results)}/{len(jobs)}] {r['cell']}: {r['seconds']} s")
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    manifest["cells"] = sorted(results, key=lambda r: r["cell"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"done in {manifest['wall_seconds']} s")


if __name__ == "__main__":
    main()
