"""Fresh, seeded task datasets - produced by the generator's own CLI.

The generator folder is deliberately not a Python package (its README says so),
so it is driven the documented way: as a subprocess, through
``generate_dataset.py`` and ``validate_dataset.py``.  That costs one process per
grid cell and buys the two things this benchmark needs from it:

* ``manifest.json`` with per-task and whole-dataset SHA-256 fingerprints, which
  become the provenance record of the benchmark run;
* ``validate_dataset.py``'s default mode, which **regenerates every task from
  the seed stored inside it and compares canonical JSON exactly**.  That is the
  reproducibility check, executed rather than asserted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import ParameterCondition

GENERATOR_DIR = (
    Path(__file__).resolve().parents[2]
    / "relational_task_generator"
    / "relational_task_generator"
)
"""``src/mas_cc/relational_task_generator/relational_task_generator``."""


class DatasetGenerationError(RuntimeError):
    """The generator or its validator refused a configuration."""


def _run(script: str, arguments: list[str]) -> str:
    command = [sys.executable, script, *arguments]
    completed = subprocess.run(
        command,
        cwd=GENERATOR_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DatasetGenerationError(
            f"{script} failed ({completed.returncode}):\n"
            f"  command: {' '.join(command)}\n"
            f"  stderr: {completed.stderr.strip()}\n"
            f"  stdout: {completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def generate_condition_dataset(
    condition: ParameterCondition, destination: Path, *, overwrite: bool = True
) -> dict:
    """Generate one grid cell's dataset and return its manifest."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arguments = [*condition.generator_arguments(), "--output", str(destination)]
    if overwrite:
        arguments.append("--overwrite")
    _run("generate_dataset.py", arguments)
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetGenerationError(f"generator wrote no manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_dataset_reproducible(destination: Path) -> str:
    """Regenerate every task from its stored seed and compare canonically."""

    return _run("validate_dataset.py", [str(Path(destination).resolve())])
