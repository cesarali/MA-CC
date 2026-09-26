"""Write, but never submit, a Cygnus array plan for saved-state theory comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from .config import load_config


def plan(config_path: str, *, python: str, repository_root: str, afterok: str,
         throttle: int = 8, cpus: int = 4, memory: str = "16G",
         time_limit: str = "08:00:00", initial_blocks: int = 2,
         theory_replicas: int = 16, substeps: int = 24,
         branch_pairs: int = 16) -> Path:
    config = load_config(config_path)
    if not config.cells or config.params.model_version != "santa_fe_epistemic_feedback_v3":
        raise ValueError("theory array needs Santa Fe v3 cells")
    if any(value < 1 for value in (throttle, cpus, initial_blocks, theory_replicas, substeps, branch_pairs)):
        raise ValueError("array resources and theory sampling must be positive")
    if not afterok.isdecimal():
        raise ValueError("afterok must be the numeric cell-aggregation Slurm job ID")
    root = Path(repository_root).resolve()
    py = Path(python)
    if not py.is_absolute():
        raise ValueError("--python must be absolute")
    logs = config.results_dir / "theory" / "logs"
    output = config.results_dir / "theory"
    # The variables below expand in the Slurm worker's shell, after sbatch.
    command = (f"cd {shlex.quote(str(root))} && export PYTHONPATH={shlex.quote(str(root / 'src'))} "
               "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
               f"{shlex.quote(str(py))} -m santa_fe.theory_integration "
               f"--config {shlex.quote(str(config.path))} "
               f"--out {shlex.quote(str(output))}/$(printf 'cell-%04d' \"$SLURM_ARRAY_TASK_ID\") "
               ' --cell-ids "$SLURM_ARRAY_TASK_ID" '
               f"--initial-blocks {initial_blocks} --theory-replicas {theory_replicas} "
               f"--substeps {substeps} --branch-pairs {branch_pairs} --branch-snapshots 1")
    script = output / "submit_theory_cygnus.sh"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    script.write_text("\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        "# Review paths, resource estimates and the completed cell aggregation before submitting.",
        f"job=$(sbatch --parsable --job-name=sf-theory --dependency=afterok:{afterok} "
        f"--array=0-{len(config.cells)-1}%{throttle} --cpus-per-task={cpus} "
        f"--mem={shlex.quote(memory)} --time={shlex.quote(time_limit)} "
        f"--output={shlex.quote(str(logs))}/theory-%A_%a.out "
        f"--error={shlex.quote(str(logs))}/theory-%A_%a.err "
        f"--wrap={shlex.quote(command)})",
        'printf "theory=%s\\n" "$job"', ""]), encoding="utf-8")
    return script


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--afterok", required=True)
    parser.add_argument("--throttle", type=int, default=8)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--time-limit", default="08:00:00")
    parser.add_argument("--initial-blocks", type=int, default=2)
    parser.add_argument("--theory-replicas", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=24)
    parser.add_argument("--branch-pairs", type=int, default=16)
    args = parser.parse_args(argv)
    print(plan(args.config, python=args.python, repository_root=args.repository_root,
               afterok=args.afterok, throttle=args.throttle, cpus=args.cpus,
               memory=args.memory, time_limit=args.time_limit,
               initial_blocks=args.initial_blocks, theory_replicas=args.theory_replicas,
               substeps=args.substeps, branch_pairs=args.branch_pairs))


if __name__ == "__main__":
    main()
