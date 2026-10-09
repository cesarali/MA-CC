"""Run a Simulation 2 study from a config in configs/.

Each run goes to a NEW folder <results_dir>/<date>_<study>/ (results/simulation_2/) and never overwrites one:
  config.yaml      exact copy of the config that ran
  manifest.json    settings, input-file hashes, git commit, times, per-cell timings, reused silent cells
  summary.csv      one row per cell (also copied to run_summaries/ and committed)
  cells/<cell id>/ posts, actions, forgetting, controller, snapshots, agent_snapshots, episodes (.parquet),
                   params.json
and one line is appended to runs_index.csv.

Examples, from the repository root:
  # tests and timing go to a scratch folder with --out, and are not indexed
  .venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py \\
      --config analysis/task003_llm_free/simulation_2/simulator/configs/sim2_bridge.yaml --episodes 20 --out /tmp/check
  # a real run
  .venv/bin/python analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py \\
      --config analysis/task003_llm_free/simulation_2/simulator/configs/sim2_bridge.yaml --workers 12
"""
from __future__ import annotations
import argparse, csv, datetime, json, multiprocessing, pathlib, platform, shutil, subprocess, sys, time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HERE = pathlib.Path(__file__).resolve().parent              # simulation_2/simulator
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                    # analysis/task003_llm_free, for llmfree_core
from llmfree_core.setups import load_setup  # noqa: E402
from llmfree_core.world import World  # noqa: E402
from sim2.simulation_2 import (AGENT_SETTINGS, COLUMNS, CONTROLLER_SETTINGS, Params,  # noqa: E402
                               params_dict, run_episode)
from sim2.checks import check_setup  # noqa: E402
from summarize_simulation_2 import summarize  # noqa: E402

STEP_SETTINGS = set(AGENT_SETTINGS) | set(CONTROLLER_SETTINGS) - {"qc", "budget", "theta_vote"}
INDEX = HERE / "runs_index.csv"
SUMMARIES = HERE / "run_summaries"
CHUNK = 100                                                 # episodes held in memory before writing

DTYPES = {
    "posts": {"episode": "int32", "post_id": "int32", "author": "int16", "fact": "int8", "author_vote": "int8"},
    "actions": {"episode": "int32", "agent": "int16", "action_index": "int16", "active_before": "int64",
                "known_before": "int64", "facts_read": "int64", "window_size": "int16", "active_after": "int64",
                "vote": "int8", "posted_fact": "int8", "post_id": "int32", "post_reason": "int8",
                "candidates_in_context": "int64", "candidates_fallback": "int64", "candidates_final": "int64"},
    "forgetting": {"episode": "int32", "agent": "int16", "fact": "int8"},
    "controller": {"episode": "int32", "action_index": "int16", "window_size": "int16", "facts_read": "int64",
                   "posted_fact": "int8", "post_id": "int32", "budget_before": "int16", "budget_after": "int16",
                   "cumulative_reads": "int32"},
    "snapshots": {"episode": "int32", "grid_index": "int16", "board_size": "int32", "agent_posts": "int32",
                  "controller_messages": "int16", "controller_reads": "int32"},
    "agent_snapshots": {"episode": "int32", "grid_index": "int16", "agent": "int16", "active": "int64",
                        "last_vote": "int8", "vote_age": "float32"},
    "episodes": {"episode": "int32", "controller_messages": "int16", "controller_reads": "int32",
                 "agent_actions": "int32", "agent_posts": "int32", "facts_forgotten": "int32"},
}


def resolve_steps(cfg: dict) -> dict[str, dict]:
    """Each step's full settings. In cumulative mode a step inherits the previous step's settings
    and may change only one of them (lambda_c counts with controller_schedule)."""
    steps, mode = cfg["steps"], cfg.get("steps_mode")
    if mode not in ("cumulative", "independent"):
        raise ValueError("steps_mode must be cumulative or independent")
    out, prev = {}, None
    for name, change in steps.items():
        unknown = set(change) - STEP_SETTINGS
        if unknown:
            raise ValueError(f"step {name}: unknown settings {sorted(unknown)}")
        if mode == "cumulative" and prev is not None:
            real = {k for k, v in change.items() if prev.get(k) != v} - {"lambda_c"}
            if len(real) != 1:
                raise ValueError(f"step {name} changes {sorted(real) or 'nothing'}; a cumulative step changes exactly one setting")
            full = {**prev, **change}
        else:
            full = dict(change)
        if full.get("controller_schedule") == "nights":
            full["lambda_c"] = None
        out[name] = full
        prev = full
    return out


def build_cells(cfg: dict) -> tuple[list[Params], dict[str, str]]:
    """Returns (cells to run, reused silent cells {skipped cell id: cell id it reuses})."""
    grid, fixed = cfg["grid"], cfg["fixed"]
    if set(grid) != {"setup", "arm"}:
        raise ValueError("the grid has exactly two factors: setup and arm")
    if not all(isinstance(x, str) for x in grid["arm"]):
        raise ValueError(f"arm names must be quoted strings in the YAML, got {grid['arm']!r}")
    cells, reused, silent_seen = [], {}, {}
    for step, settings in resolve_steps(cfg).items():
        for setup in grid["setup"]:
            for arm in grid["arm"]:
                p = Params(setup=setup, arm=arm, step=step, **fixed, **settings)
                if arm == "silent":
                    key = p.silent_key()
                    if key in silent_seen:
                        reused[p.cell_id()] = silent_seen[key]
                        continue
                    silent_seen[key] = p.cell_id()
                cells.append(p)
    return cells, reused


_WORLD = _SETUPS = None


def _init_worker(world_file: str, setups_dir: str, setup_names: list[str]):
    global _WORLD, _SETUPS
    _WORLD = World(world_file)
    _SETUPS = {s: load_setup(s, setups_dir, _WORLD) for s in setup_names}


def _frame(table: str, rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS[table])
    return df.astype(DTYPES.get(table, {}))


def _run_cell(job):
    p, episodes, cells_dir, agent_snapshots = job
    t0 = time.time()
    setup = _SETUPS[p.setup]
    folder = pathlib.Path(cells_dir) / p.cell_id()
    folder.mkdir(parents=True, exist_ok=True)
    tables = [t for t in COLUMNS if agent_snapshots or t != "agent_snapshots"]
    writers: dict[str, pq.ParquetWriter] = {}
    buffer = {t: [] for t in tables}
    for e in range(episodes):
        rows = run_episode(_WORLD, setup, p, e, agent_snapshots)
        for t in tables:
            buffer[t] += rows[t]
        if (e + 1) % CHUNK == 0 or e == episodes - 1:
            for t in tables:
                df = _frame(t, buffer[t])
                tab = pa.Table.from_pandas(df, preserve_index=False)
                if t not in writers:
                    writers[t] = pq.ParquetWriter(folder / f"{t}.parquet", tab.schema)
                writers[t].write_table(tab.cast(writers[t].schema))
                buffer[t] = []
    for w in writers.values():
        w.close()
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
    if not cfg["study"].startswith("sim2_"):
        sys.exit("study names start with sim2_")
    base = cfg_path.parent
    world_file = str((base / cfg["paths"]["world_file"]).resolve())
    setups_dir = str((base / cfg["paths"]["setups_dir"]).resolve())
    agent_snapshots = bool(cfg.get("record", {}).get("agent_snapshots", True))

    cells, reused = build_cells(cfg)
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

    world = World(world_file)
    setup_names = sorted({p.setup for p in cells})
    setups = {s: load_setup(s, setups_dir, world) for s in setup_names}
    for s in setup_names:
        check_setup(world, setups[s])                       # spec §1: stops on any failure

    started = datetime.datetime.now()
    run_name = f"{started:%Y-%m-%d}_{cfg['study']}"
    run_dir = pathlib.Path(args.out) if args.out else (base / cfg["paths"]["results_dir"]).resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"REFUSING TO OVERWRITE: {run_dir} already exists and is not empty.")
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, run_dir / "config.yaml")

    commit, dirty = _git("rev-parse", "HEAD"), simulator_is_dirty()
    if dirty and official:
        print("WARNING: uncommitted changes in simulation_2/simulator/ or llmfree_core/; the recorded commit does not fully describe this run")
    manifest = {
        "run": run_name if not args.out else str(run_dir), "study": cfg["study"],
        "description": cfg.get("description", ""), "config": str(cfg_path),
        "cells_in_grid": expected, "cells_run": len(cells), "episodes_per_cell": episodes,
        "reused_silent_cells": reused, "steps": resolve_steps(cfg),
        "official_run": official, "world_file_sha256": world.sha256,
        "setup_file_sha256": {s: setups[s].file_hashes for s in setup_names},
        "git_commit": commit, "simulator_has_uncommitted_changes": dirty,
        "python": platform.python_version(), "pyarrow": pa.__version__, "pandas": pd.__version__,
        "started": started.isoformat(timespec="seconds"), "warnings": warnings,
    }
    print(f"{len(cells)} cells x {episodes} episodes on {args.workers} worker(s) -> {run_dir}", flush=True)

    jobs = [(p, episodes, str(cells_dir), agent_snapshots) for p in cells]
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
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
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
                        f"results/simulation_2/{run_name}", ""])
        print(f"indexed in {INDEX.name}; summary copied to {SUMMARIES.name}/")
    print(f"done in {manifest['wall_seconds']} s")


if __name__ == "__main__":
    main()
