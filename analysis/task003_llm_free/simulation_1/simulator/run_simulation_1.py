"""Run a Simulation 1 study from a config in configs/.

Each run goes to a NEW folder <results_dir>/<date>_<study>/ (results/simulation_1/) and never overwrites one:
  config.yaml      exact copy of the config that ran
  manifest.json    settings, input-file hashes, git commit, times, per-cell timings
  summary.csv      one row per cell (also copied to run_summaries/ and committed)
  cells/<cell id>/ agents.parquet, nights.parquet, days.parquet, params.json
and one line is appended to runs_index.csv.

Examples, from the repository root:
  # tests and timing go to a scratch folder with --out, and are not indexed
  .venv/bin/python analysis/task003_llm_free/simulation_1/simulator/run_simulation_1.py \\
      --config analysis/task003_llm_free/simulation_1/simulator/configs/sim1_base.yaml --episodes 5 --out /tmp/check
  # a real run
  .venv/bin/python analysis/task003_llm_free/simulation_1/simulator/run_simulation_1.py \\
      --config analysis/task003_llm_free/simulation_1/simulator/configs/sim1_base.yaml --workers 12
"""
from __future__ import annotations
import argparse, csv, datetime, itertools, json, multiprocessing, pathlib, platform, shutil, subprocess, sys, time

import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent              # simulation_1/simulator
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                    # analysis/task003_llm_free, for llmfree_core
from llmfree_core.setups import load_setup  # noqa: E402
from llmfree_core.world import World  # noqa: E402
from sim1.simulation_1 import (AGENT_COLUMNS, DAY_COLUMNS, NIGHT_COLUMNS, Params,  # noqa: E402
                               params_dict, run_episode)
from summarize_simulation_1 import summarize  # noqa: E402

GRID_KEYS = ("setup", "arm", "q", "qc", "b", "rho")
RECORD_KEYS = {"agent_days", "posts_read", "memory_after_dawn", "post_reason", "nights", "proof_two_ways"}
INDEX = HERE / "runs_index.csv"
SUMMARIES = HERE / "run_summaries"


def build_cells(cfg: dict) -> list[Params]:
    grid, fixed = cfg["grid"], cfg["fixed"]
    unknown = set(grid) - set(GRID_KEYS)
    if unknown:
        raise ValueError(f"unknown grid factors: {sorted(unknown)}")
    if not all(isinstance(x, str) for x in grid["arm"]):
        raise ValueError(f"arm names must be quoted strings in the YAML, got {grid['arm']!r}")
    ignored = set(cfg["cells"].get("silent_ignores", []))
    cells, seen = [], set()
    for values in itertools.product(*(grid[k] for k in GRID_KEYS)):
        d = dict(zip(GRID_KEYS, values))
        if d["arm"] == "silent":
            for k in ignored:
                d[k] = None
        p = Params(**d, **fixed)
        if p.cell_id() not in seen:
            seen.add(p.cell_id())
            cells.append(p)
    return cells


def check_record(cfg: dict) -> bool:
    """Returns whether to record every post read. The other record items are always on."""
    rec = cfg.get("record", {})
    if set(rec) != RECORD_KEYS:
        raise ValueError(f"record block must list exactly {sorted(RECORD_KEYS)}")
    for k in RECORD_KEYS - {"posts_read"}:
        if rec[k] is not True:
            raise NotImplementedError(f"record.{k}: false is not implemented; it is always recorded")
    return bool(rec["posts_read"])


_WORLD = _SETUPS = None


def _init_worker(world_file: str, setups_dir: str, setup_names: list[str]):
    global _WORLD, _SETUPS
    _WORLD = World(world_file)
    _SETUPS = {s: load_setup(s, setups_dir, _WORLD) for s in setup_names}


def _run_cell(job):
    p, episodes, cells_dir, record_posts_read = job
    t0 = time.time()
    setup = _SETUPS[p.setup]
    agents, nights, days = [], [], []
    for e in range(episodes):
        a, n, d = run_episode(_WORLD, setup, p, e, record_posts_read)
        agents += a
        nights += n
        days += d
    folder = pathlib.Path(cells_dir) / p.cell_id()
    folder.mkdir(parents=True, exist_ok=True)
    da = pd.DataFrame(agents, columns=AGENT_COLUMNS)
    if not record_posts_read:
        da = da.drop(columns="posts_read")
    da.astype({"episode": "int32", "day": "int16", "position": "int16", "agent": "int16",
               "memory_after_dawn": "int64", "known_before_reading": "int64", "active_facts": "int64",
               "facts_read": "int64", "n_posts_read": "int16", "p_A0": "float32", "p_A1": "float32",
               "p_A2": "float32", "vote": "int8", "posted_fact": "int8", "post_reason": "int8",
               "proves_A0": "bool", "proves_A0_own_evidence": "bool"}).to_parquet(folder / "agents.parquet", index=False)
    pd.DataFrame(days, columns=DAY_COLUMNS).to_parquet(folder / "days.parquet", index=False)
    if nights:
        dn = pd.DataFrame(nights, columns=NIGHT_COLUMNS)
        if not record_posts_read:
            dn = dn.drop(columns="posts_read")
        dn.astype({"posted_facts": "int64"}).to_parquet(folder / "nights.parquet", index=False)
    (folder / "params.json").write_text(json.dumps(params_dict(p), indent=2) + "\n")
    return {"cell": p.cell_id(), "seconds": round(time.time() - t0, 1)}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def dirty_check_args() -> list[str]:
    """git status arguments for "has simulator code changed since the commit?". Covers this
    folder and the shared llmfree_core/. The runner's own bookkeeping (runs_index.csv,
    run_summaries/) is excluded: it changes after every run."""
    return ["status", "--porcelain", "--", str(HERE), str(HERE.parents[1] / "llmfree_core"),
            f":(exclude){INDEX.relative_to(HERE).as_posix()}",
            f":(exclude){SUMMARIES.relative_to(HERE).as_posix()}"]


def simulator_is_dirty() -> bool:
    return bool(_git(*dirty_check_args()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, help="override episodes_per_cell (tests and timing only)")
    ap.add_argument("--only", action="append", default=[], help="run only cells whose id contains this; repeatable")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", help="write to this folder instead of a dated run folder; not indexed")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    base = cfg_path.parent
    world_file = str((base / cfg["paths"]["world_file"]).resolve())
    setups_dir = str((base / cfg["paths"]["setups_dir"]).resolve())
    record_posts_read = check_record(cfg)

    cells = build_cells(cfg)
    expected = cfg["cells"]["expected_total"]
    if len(cells) != expected:
        sys.exit(f"REFUSING TO RUN: the config gives {len(cells)} cells, expected_total says {expected}.")
    warnings = sorted({w for p in cells for w in p.check()})
    for w in warnings:
        print("WARNING:", w)
    official = not (args.only or args.episodes or args.out)
    if args.only:
        cells = [p for p in cells if any(s in p.cell_id() for s in args.only)]
        if not cells:
            sys.exit("no cell matches --only")
    episodes = args.episodes or cfg["cells"]["episodes_per_cell"]

    started = datetime.datetime.now()
    run_name = f"{started:%Y-%m-%d}_{cfg['study']}"
    run_dir = pathlib.Path(args.out) if args.out else (base / cfg["paths"]["results_dir"]).resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"REFUSING TO OVERWRITE: {run_dir} already exists and is not empty.")
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, run_dir / "config.yaml")

    world = World(world_file)
    setup_names = sorted({p.setup for p in cells})
    setups = {s: load_setup(s, setups_dir, world) for s in setup_names}
    commit, dirty = _git("rev-parse", "HEAD"), simulator_is_dirty()
    if dirty and official:
        print("WARNING: uncommitted changes in simulation_1/simulator/ or llmfree_core/; the recorded commit does not fully describe this run")
    manifest = {
        "run": run_name if not args.out else str(run_dir), "study": cfg["study"],
        "description": cfg.get("description", ""), "config": str(cfg_path),
        "cells_in_grid": expected, "cells_run": len(cells), "episodes_per_cell": episodes,
        "official_run": official, "world_file_sha256": world.sha256,
        "setup_file_sha256": {s: setups[s].file_hashes for s in setup_names},
        "git_commit": commit, "simulator_has_uncommitted_changes": dirty,
        "python": platform.python_version(), "started": started.isoformat(timespec="seconds"),
        "warnings": warnings,
    }
    print(f"{len(cells)} cells x {episodes} episodes on {args.workers} worker(s) -> {run_dir}", flush=True)

    jobs = [(p, episodes, str(cells_dir), record_posts_read) for p in cells]
    timings = []
    t0 = time.time()
    if args.workers > 1:
        with multiprocessing.Pool(args.workers, _init_worker, (world_file, setups_dir, setup_names)) as pool:
            for r in pool.imap_unordered(_run_cell, jobs):
                timings.append(r)
                print(f"[{len(timings)}/{len(jobs)}] {r['cell']}: {r['seconds']} s", flush=True)
    else:
        _init_worker(world_file, setups_dir, setup_names)
        for job in jobs:
            r = _run_cell(job)
            timings.append(r)
            print(f"[{len(timings)}/{len(jobs)}] {r['cell']}: {r['seconds']} s", flush=True)
    finished = datetime.datetime.now()
    manifest["finished"] = finished.isoformat(timespec="seconds")
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    manifest["cell_seconds"] = sorted(timings, key=lambda r: r["cell"])
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    summary = summarize(run_dir)
    print(f"wrote {run_dir / 'summary.csv'} ({len(summary)} cells)")
    if official:
        SUMMARIES.mkdir(exist_ok=True)
        shutil.copy(run_dir / "summary.csv", SUMMARIES / f"{run_name}_summary.csv")
        new = not INDEX.exists()
        with INDEX.open("a", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            if new:
                w.writerow(["run", "study", "description", "config", "git_commit", "uncommitted_changes",
                            "cells", "episodes_per_cell", "started", "finished", "wall_seconds", "results_folder",
                            "note"])
            w.writerow([run_name, cfg["study"], cfg.get("description", ""),
                        cfg_path.relative_to(HERE).as_posix(), commit, dirty, len(cells), episodes,
                        manifest["started"], manifest["finished"], manifest["wall_seconds"],
                        f"results/simulation_1/{run_name}", ""])
        print(f"indexed in {INDEX.name}; summary copied to {SUMMARIES.name}/")
    print(f"done in {manifest['wall_seconds']} s")


if __name__ == "__main__":
    main()
