"""Reading a frozen task for single-model use.

Schema validation is not re-implemented here: ``load_relational_task`` already
accepts exactly ``spatial_relational_task_v1`` and refuses everything else, and
a second copy of those rules would be a second thing to drift.  It is a pure
data module - it imports no game, no runner and no controller - so reusing it
does not couple the benchmark to the multi-agent code.

What it does not keep is ``query.subject`` and ``query.reference``: the game
never needs the endpoints because it only ever shows the rendered question.  The
benchmark needs them, because its central check re-solves the query from the
shown constraints.  So this wrapper reads those two names straight from the
frozen JSON and delegates everything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_cc.games.relational_reasoning.data import (
    RelationalTask,
    RelationalTaskError,
    list_relational_task_ids,
    load_relational_task,
)

__all__ = ["BenchmarkTask", "load_benchmark_task", "load_benchmark_tasks"]


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A validated task plus the query endpoints the solver check needs."""

    task: RelationalTask
    question_subject: str
    question_reference: str
    generation: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "task"), name)


def load_benchmark_task(dataset_dir: str | Path, task_id: str) -> BenchmarkTask:
    task = load_relational_task(dataset_dir, task_id)
    payload = json.loads(Path(task.source_path).read_text(encoding="utf-8"))
    query = payload.get("query")
    if not isinstance(query, dict) or not query.get("subject") or not query.get("reference"):
        raise RelationalTaskError(
            f"task file {task.source_path}: query.subject and query.reference are required"
        )
    generation = payload.get("generation")
    return BenchmarkTask(
        task=task,
        question_subject=str(query["subject"]),
        question_reference=str(query["reference"]),
        generation=dict(generation) if isinstance(generation, dict) else {},
    )


def load_benchmark_tasks(dataset_dir: str | Path) -> tuple[BenchmarkTask, ...]:
    """Every task in a generated dataset directory, in file order."""

    return tuple(
        load_benchmark_task(dataset_dir, task_id)
        for task_id in list_relational_task_ids(dataset_dir)
    )
