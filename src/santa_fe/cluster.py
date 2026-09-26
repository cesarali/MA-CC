"""Resumable cell-array execution for synthetic Santa Fe studies.

The commands here can run locally for checks or as Slurm array tasks. This
module does not submit jobs; ``plan`` writes a reviewable Cygnus submit script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shlex

import pandas as pd
from tqdm.auto import tqdm

from .config import Config, load_config
from .beta_cmi import produce_beta_outputs
from .llm_parallel import _analyze_group, _report, adapt_trajectories
from .measurements import final_summary, information_summary, susceptibility_summary
from .plotting import save_phase_diagrams
from .runner import run, save_trajectories


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parquet(path: Path, data: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    data.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _cell_dir(config: Config, cell_id: int) -> Path:
    return config.results_dir / "cells" / f"cell-{cell_id:04d}"


def _load_manifest(config: Config) -> dict:
    path = config.results_dir / "execution_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"run `santa_fe.cluster prepare` first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha(config.path):
        raise ValueError("config changed after cluster preparation")
    if manifest["parameter_cells"] != len(config.cells):
        raise ValueError("cell count changed after cluster preparation")
    return manifest


def prepare(config: Config) -> dict:
    root = config.results_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "cells").mkdir(exist_ok=True)
    copy = root / "config.yaml"
    if copy.is_file() and copy.read_bytes() != config.path.read_bytes():
        raise ValueError(f"result root already belongs to a different config: {root}")
    copy.write_bytes(config.path.read_bytes())
    policy = config.raw.get("slurm", {})
    if not isinstance(policy, dict):
        raise ValueError("slurm policy must be a mapping")
    for stage in ("simulation", "information", "finalize"):
        value = policy.get(stage, {})
        if not isinstance(value, dict):
            raise ValueError(f"slurm.{stage} must be a mapping")
        if int(value.get("cpus_per_task", 1)) < 1:
            raise ValueError(f"slurm.{stage}.cpus_per_task must be positive")
        if stage != "finalize" and int(value.get("throttle", 1)) < 1:
            raise ValueError(f"slurm.{stage}.throttle must be positive")
    simulation_cpus = int(policy.get("simulation", {}).get("cpus_per_task", 4))
    information_cpus = int(policy.get("information", {}).get("cpus_per_task", 4))
    if config.processes > simulation_cpus:
        raise ValueError("experiment.processes exceeds Slurm simulation CPUs per task")
    if int(config.raw.get("analysis", {}).get("llm_parallel", {}).get("processes", 1)) > information_cpus:
        raise ValueError("llm_parallel.processes exceeds Slurm information CPUs per task")
    manifest = {"schema_version": 1, "config_path": str(config.path),
                "config_sha256": _sha(config.path), "results_dir": str(root),
                "parameter_cells": len(config.cells), "episodes_per_cell": config.episodes,
                "total_episodes": len(config.cells) * config.episodes,
                "round_rows_expected": len(config.cells) * config.episodes * (config.params.rounds + 1),
                "slurm": policy}
    _json(root / "execution_plan.json", manifest)
    return manifest


def _valid_seal(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if seal.get(key) != value:
                return False
        for name, digest in seal["files"].items():
            if not (path.parent / name).is_file() or _sha(path.parent / name) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_cell(config: Config, cell_id: int) -> dict:
    manifest = _load_manifest(config)
    if not 0 <= cell_id < len(config.cells):
        raise ValueError(f"cell ID outside 0..{len(config.cells)-1}")
    destination = _cell_dir(config, cell_id)
    destination.mkdir(parents=True, exist_ok=True)
    seal_path = destination / "cell_complete.json"
    expected = {"config_sha256": manifest["config_sha256"], "cell_id": cell_id,
                "episodes": config.episodes, "round_rows": config.episodes * (config.params.rounds + 1)}
    if _valid_seal(seal_path, expected):
        return {"cell_id": cell_id, "status": "already_complete", "path": str(destination)}
    cell_config = replace(config, cells=(config.cells[cell_id],), processes=min(config.processes, int(os.environ.get("SLURM_CPUS_PER_TASK", config.processes))))
    rounds, micro = run(cell_config)
    if rounds.seed.nunique() != config.episodes or len(rounds) != expected["round_rows"]:
        raise ValueError("generated cell has unexpected episode or round count")
    round_path = destination / "rounds.parquet"
    _parquet(round_path, rounds)
    files = {round_path.name: _sha(round_path)}
    if config.save_micro:
        micro_path = destination / "micro.parquet"
        _parquet(micro_path, micro)
        files[micro_path.name] = _sha(micro_path)
    _json(seal_path, {**expected, "files": files,
                      "beta_evidence": config.cells[cell_id].params.beta_evidence,
                      "beta_social": config.cells[cell_id].params.beta_social,
                      "beta_regime": config.cells[cell_id].beta_regime,
                      "rho": config.cells[cell_id].params.rho,
                      "budget_fraction": config.cells[cell_id].params.budget_fraction,
                      "budget": config.cells[cell_id].params.budget})
    return {"cell_id": cell_id, "status": "complete", "path": str(destination)}


def _sealed_rounds(config: Config, cell_id: int) -> pd.DataFrame:
    manifest = _load_manifest(config)
    destination = _cell_dir(config, cell_id)
    expected = {"config_sha256": manifest["config_sha256"], "cell_id": cell_id,
                "episodes": config.episodes, "round_rows": config.episodes * (config.params.rounds + 1)}
    if not _valid_seal(destination / "cell_complete.json", expected):
        raise ValueError(f"cell {cell_id} is incomplete or its files changed")
    rounds = pd.read_parquet(destination / "rounds.parquet")
    if len(rounds) != expected["round_rows"] or rounds.seed.nunique() != config.episodes:
        raise ValueError(f"cell {cell_id} contents do not match its seal")
    return rounds


def aggregate_cells(config: Config) -> dict:
    _load_manifest(config)
    frames = [_sealed_rounds(config, cell.cell_id) for cell in tqdm(config.cells, desc="sealed cells")]
    rounds = pd.concat(frames, ignore_index=True)
    if rounds.duplicated(["cell_id", "seed", "round"]).any():
        raise ValueError("duplicate scientific round coordinates")
    micro_frames = []
    if config.save_micro:
        micro_frames = [pd.read_parquet(_cell_dir(config, cell.cell_id) / "micro.parquet") for cell in config.cells]
    micro = pd.concat(micro_frames, ignore_index=True) if micro_frames else pd.DataFrame()
    save_trajectories(config, rounds, micro)
    root = config.results_dir
    summary = root / "summaries"
    summary.mkdir(exist_ok=True)
    final = final_summary(rounds)
    final.to_csv(summary / "final_summary.csv", index=False)
    if config.information:
        information_summary(rounds, config.bins, config.coordinate).to_csv(summary / "information_summary.csv", index=False)
        susceptibility_summary(rounds, config.bins, config.coordinate).to_csv(summary / "susceptibility.csv", index=False)
    save_phase_diagrams(final, None, root / "plots" / "phase_diagrams")
    lines = ["# Santa Fe full-grid summary", "",
             f"Cells: {len(config.cells)}; episodes per cell: {config.episodes}; round rows: {len(rounds)}.",
             "", "Phase diagrams are in `plots/phase_diagrams/`; information diagrams are added after information aggregation.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _json(root / "cells_aggregation.json", {"config_sha256": _sha(config.path),
                                            "cells": len(config.cells), "episodes": len(config.cells) * config.episodes,
                                            "round_rows": len(rounds), "round_trajectory_sha256": _sha(root / "trajectories" / "round_trajectories.parquet")})
    return {"cells": len(config.cells), "episodes": len(config.cells) * config.episodes,
            "round_rows": len(rounds), "results_dir": str(root)}


def run_information_cell(config: Config, cell_id: int) -> dict:
    _load_manifest(config)
    rounds = _sealed_rounds(config, cell_id)
    source = _cell_dir(config, cell_id) / "rounds.parquet"
    settings = config.raw.get("analysis", {}).get("llm_parallel", {})
    statistics = tuple(settings.get("statistics", ()))
    if not statistics:
        raise ValueError("analysis.llm_parallel.statistics is required")
    bootstrap = int(settings.get("bootstrap_resamples", 1000))
    permutations = int(settings.get("null_permutations", 1000))
    confidence = float(settings.get("confidence", .95))
    seed = int(settings.get("seed", config.seed))
    processes = min(int(settings.get("processes", 1)), int(os.environ.get("SLURM_CPUS_PER_TASK", settings.get("processes", 1))))
    destination = _cell_dir(config, cell_id) / "information"
    destination.mkdir(exist_ok=True)
    seal_path = destination / "information_complete.json"
    expected = {"config_sha256": _sha(config.path), "source_sha256": _sha(source), "cell_id": cell_id}
    if _valid_seal(seal_path, expected):
        return {"cell_id": cell_id, "status": "already_complete", "path": str(destination)}
    events = adapt_trajectories(rounds, bins=config.bins)
    groups: dict[int, list] = {}
    for event in events:
        groups.setdefault(event.round_index, []).append(event)
    payloads = [(cell_id, "pooled", None, events, statistics, bootstrap, permutations, confidence, seed)]
    payloads += [(cell_id, "per_round", time, rows, statistics, bootstrap, permutations, confidence, seed)
                 for time, rows in sorted(groups.items())]
    estimates, nulls = [], []
    if processes == 1:
        iterator = map(_analyze_group, payloads)
        for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell_id}"):
            estimates.extend(estimate_rows)
            nulls.extend(null_rows)
    else:
        with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
            iterator = pool.map(_analyze_group, payloads, chunksize=1)
            for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell_id}"):
                estimates.extend(estimate_rows)
                nulls.extend(null_rows)
    estimate_path = destination / "estimates.parquet"
    null_path = destination / "nulls.parquet"
    _parquet(estimate_path, pd.DataFrame(estimates))
    _parquet(null_path, pd.DataFrame(nulls))
    _json(seal_path, {**expected, "estimates": len(estimates), "null_draws": len(nulls),
                      "files": {estimate_path.name: _sha(estimate_path), null_path.name: _sha(null_path)}})
    return {"cell_id": cell_id, "status": "complete", "estimates": len(estimates),
            "null_draws": len(nulls), "path": str(destination)}


def aggregate_information(config: Config) -> dict:
    _load_manifest(config)
    estimates = []
    total_null_draws = 0
    for cell in tqdm(config.cells, desc="sealed information cells"):
        cell_id = cell.cell_id
        source = _cell_dir(config, cell_id) / "rounds.parquet"
        destination = _cell_dir(config, cell_id) / "information"
        expected = {"config_sha256": _sha(config.path), "source_sha256": _sha(source), "cell_id": cell_id}
        seal = destination / "information_complete.json"
        if not _valid_seal(seal, expected):
            raise ValueError(f"information cell {cell_id} is incomplete or changed")
        payload = json.loads(seal.read_text(encoding="utf-8"))
        total_null_draws += int(payload["null_draws"])
        estimates.append(pd.read_parquet(destination / "estimates.parquet"))
    table = pd.concat(estimates, ignore_index=True)
    output = config.results_dir / "information" / "llm_parallel"
    output.mkdir(parents=True, exist_ok=True)
    _parquet(output / "round_information_estimates.parquet", table)
    table.to_csv(output / "round_information_estimates.csv", index=False)
    _report(table, config, output)
    final = pd.read_csv(config.results_dir / "summaries" / "final_summary.csv")
    save_phase_diagrams(final, table.loc[table.scope == "pooled"], config.results_dir / "plots" / "phase_diagrams")
    if config.raw.get("analysis", {}).get("beta_cmi", {}).get("enabled", False):
        rounds = pd.read_parquet(config.results_dir / "trajectories" / "round_trajectories.parquet")
        susceptibility = pd.read_csv(config.results_dir / "summaries" / "susceptibility.csv")
        produce_beta_outputs(config, rounds, final, susceptibility, table)
    _json(output / "analysis_config.json", {"config_sha256": _sha(config.path),
                                            "source": "sealed per-cell information outputs",
                                            "cells": len(config.cells), "estimates": len(table),
                                            "null_draws": total_null_draws})
    (output / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return {"cells": len(config.cells), "estimates": len(table), "null_draws": total_null_draws,
            "results_dir": str(output)}


def _resources(policy: dict, stage: str) -> tuple[int, str, str, int]:
    defaults = {"simulation": (4, "8G", "01:00:00", 12),
                "information": (4, "12G", "04:00:00", 12),
                "finalize": (8, "32G", "02:00:00", 1)}
    section = policy.get(stage, {})
    cpus, memory, limit, throttle = defaults[stage]
    return (int(section.get("cpus_per_task", cpus)), str(section.get("memory", memory)),
            str(section.get("time_limit", limit)), int(section.get("throttle", throttle)))


def write_cygnus_commands(config: Config, *, python: str, repository_root: str) -> Path:
    manifest = prepare(config)
    policy = manifest["slurm"]
    root = config.results_dir
    config_arg = shlex.quote(str(config.path))
    py = shlex.quote(python)
    repo = shlex.quote(repository_root)
    logs = shlex.quote(str(root / "logs"))
    cells = len(config.cells)
    def sbatch(stage: str, job_name: str, command: str, *, array: bool, dependency: str | None = None) -> str:
        cpus, memory, limit, throttle = _resources(policy, stage)
        array_arg = f" --array=0-{cells-1}%{throttle}" if array else ""
        index = "%A_%a" if array else "%j"
        dependent = f" --dependency=afterok:${dependency}" if dependency else ""
        return (f"sbatch --parsable --job-name={job_name}{array_arg}{dependent} --cpus-per-task={cpus} "
                f"--mem={shlex.quote(memory)} --time={shlex.quote(limit)} "
                f"--output={logs}/{job_name}-{index}.out --error={logs}/{job_name}-{index}.err "
                f"--wrap={shlex.quote(command)}")
    base = f"cd {repo} && export PYTHONPATH={repo}/src && export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
    simulation = base + f"{py} -m santa_fe.cluster run-cell --config {config_arg} --cell-id $SLURM_ARRAY_TASK_ID"
    cell_finalize = base + f"{py} -m santa_fe.cluster aggregate-cells --config {config_arg}"
    information = base + f"{py} -m santa_fe.cluster run-information-cell --config {config_arg} --cell-id $SLURM_ARRAY_TASK_ID"
    info_finalize = base + f"{py} -m santa_fe.cluster aggregate-information --config {config_arg}"
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "# Review paths/resources before running; no provider calls are made.",
                f"sim_job=$({sbatch('simulation', 'santa-fe-sim', simulation, array=True)})",
                f"cells_job=$({sbatch('finalize', 'santa-fe-cells', cell_finalize, array=False, dependency='sim_job')})",
                f"info_job=$({sbatch('information', 'santa-fe-info', information, array=True, dependency='cells_job')})",
                f"final_job=$({sbatch('finalize', 'santa-fe-final', info_finalize, array=False, dependency='info_job')})",
                'printf "simulation=%s cell_aggregation=%s information=%s final=%s\\n" "$sim_job" "$cells_job" "$info_job" "$final_job"', ""]
    path = root / "submit_cygnus.sh"
    path.write_text("\n".join(commands), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run-cell", "aggregate-cells", "run-information-cell", "aggregate-information", "plan-cygnus"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell-id", type=int)
    parser.add_argument("--python", help="absolute Cygnus Python path for the plan")
    parser.add_argument("--repository-root", help="absolute Cygnus checkout path for the plan")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.operation in {"run-cell", "run-information-cell"}:
        cell_id = args.cell_id if args.cell_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_cell(config, cell_id) if args.operation == "run-cell" else run_information_cell(config, cell_id)
    elif args.operation == "prepare":
        result = prepare(config)
    elif args.operation == "aggregate-cells":
        result = aggregate_cells(config)
    elif args.operation == "aggregate-information":
        result = aggregate_information(config)
    else:
        if not args.python or not args.repository_root:
            parser.error("plan-cygnus needs --python and --repository-root")
        script = write_cygnus_commands(config, python=args.python, repository_root=args.repository_root)
        result = {"script": str(script), "results_dir": str(config.results_dir),
                  "cells": len(config.cells), "episodes": len(config.cells) * config.episodes}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
