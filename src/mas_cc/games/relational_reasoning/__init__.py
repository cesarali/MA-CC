"""Games played on frozen synthetic relational-reasoning tasks.

The tasks themselves are produced by the standalone generator under
``src/mas_cc/relational_task_generator/`` and are **consumed here as frozen
JSON**: nothing in this package generates, redistributes, or repairs a task.
See ``data.py`` for the loader and the exact schema it accepts.
"""

from .data import (
    DEFAULT_TASK_DATASET_DIR,
    RelationalFact,
    RelationalTask,
    RelationalTaskError,
    list_relational_task_ids,
    load_relational_task,
)

__all__ = [
    "DEFAULT_TASK_DATASET_DIR",
    "RelationalFact",
    "RelationalTask",
    "RelationalTaskError",
    "list_relational_task_ids",
    "load_relational_task",
]
