"""The benchmark configuration: a grid over *task-difficulty* parameters only.

The generator exposes seven knobs.  Three of them change the reasoning item a
single model actually sees:

* ``reasoning_depth`` (L) - how many facts the chain has;
* ``distractors`` - how many irrelevant facts share the page;
* ``num_options`` - how many answers to choose between, i.e. the chance floor.

The other four - ``population_size``, ``support_redundancy``,
``distractor_redundancy``, ``no_single_agent_solution`` - only decide *who gets
told what*, and this benchmark tells one model everything its condition allows.
They are held fixed, declared under ``fixed:``, and recorded in every output row
so a later run cannot quietly disagree about them.  Sweeping them here would
multiply cost by a factor that provably cannot move the measurement.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mas_cc.config import LLMProviderConfig

from .presentation import ALL_POSITIONS, PRESENTATION_MODES

_PROVIDER_FIELDS = {
    "type",
    "model",
    "schema_version",
    "credentials_env",
    "base_url_env",
    "timeout_seconds",
    "max_retries",
    "request_concurrency",
    "temperature",
    "max_output_tokens",
    "options",
}


@dataclass(frozen=True, slots=True)
class ParameterCondition:
    """One cell of the difficulty grid: a dataset to generate and label."""

    reasoning_depth: int
    distractors: int
    num_options: int
    population_size: int
    support_redundancy: int
    distractor_redundancy: int
    no_single_agent_solution: bool
    dataset_seed: int
    num_tasks: int

    @property
    def label(self) -> str:
        return f"L{self.reasoning_depth}_D{self.distractors}_O{self.num_options}"

    def generator_arguments(self) -> list[str]:
        """The exact ``generate_dataset.py`` flags, using its real option names."""

        flags = [
            "--num-tasks", str(self.num_tasks),
            "--population-size", str(self.population_size),
            "--reasoning-depth", str(self.reasoning_depth),
            "--support-redundancy", str(self.support_redundancy),
            "--distractors", str(self.distractors),
            "--distractor-redundancy", str(self.distractor_redundancy),
            "--num-options", str(self.num_options),
            "--seed", str(self.dataset_seed),
        ]
        flags.append(
            "--no-single-agent-solution"
            if self.no_single_agent_solution
            else "--allow-single-agent-solution"
        )
        return flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_condition": self.label,
            "reasoning_depth": self.reasoning_depth,
            "distractors": self.distractors,
            "num_options": self.num_options,
            "population_size": self.population_size,
            "support_redundancy": self.support_redundancy,
            "distractor_redundancy": self.distractor_redundancy,
            "no_single_agent_solution": self.no_single_agent_solution,
            "dataset_seed": self.dataset_seed,
            "num_tasks": self.num_tasks,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    seed: int
    tasks_per_condition: int
    grid: Mapping[str, tuple[int, ...]]
    fixed: Mapping[str, Any]
    include_zero_condition: bool
    max_subsets_per_k: int
    presentation_mode: str
    provider: LLMProviderConfig
    output_dir: Path
    max_requests: int
    prompt_example_tasks: int
    description: str = ""
    source_path: Path | None = field(default=None)

    def conditions(self) -> tuple[ParameterCondition, ...]:
        """Every grid cell, with a per-cell dataset seed derived from ``seed``."""

        depths = self.grid["reasoning_depth"]
        distractors = self.grid["distractors"]
        options = self.grid["num_options"]
        cells: list[ParameterCondition] = []
        for index, (depth, distractor, option_count) in enumerate(
            itertools.product(depths, distractors, options)
        ):
            # L = 1 makes ``no_single_agent_solution`` unsatisfiable by
            # definition - one fact handed to one agent *is* the whole proof -
            # and the generator rejects that combination outright.
            hidden_profile = bool(self.fixed["no_single_agent_solution"]) and depth > 1
            cells.append(
                ParameterCondition(
                    reasoning_depth=depth,
                    distractors=distractor,
                    num_options=option_count,
                    population_size=int(self.fixed["population_size"]),
                    support_redundancy=int(self.fixed["support_redundancy"]),
                    distractor_redundancy=int(self.fixed["distractor_redundancy"]),
                    no_single_agent_solution=hidden_profile,
                    dataset_seed=self.seed + 1000 * (index + 1),
                    num_tasks=self.tasks_per_condition,
                )
            )
        return tuple(cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "tasks_per_condition": self.tasks_per_condition,
            "grid": {key: list(value) for key, value in self.grid.items()},
            "fixed": dict(self.fixed),
            "evidence": {
                "include_zero_condition": self.include_zero_condition,
                "max_subsets_per_k": self.max_subsets_per_k,
            },
            "presentation": {"mode": self.presentation_mode},
            "llm_provider": self.provider.to_dict(),
            "output_dir": str(self.output_dir),
            "max_requests": self.max_requests,
            "prompt_example_tasks": self.prompt_example_tasks,
            "source_path": str(self.source_path) if self.source_path else None,
        }


def _int_list(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"benchmark grid.{name} must be a non-empty list of integers")
    return tuple(int(item) for item in value)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Read one benchmark YAML.  Unknown keys are rejected, not ignored."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark config {source} is not a mapping")

    unknown = set(payload) - {
        "benchmark", "grid", "fixed", "evidence", "presentation", "llm_provider",
        "output", "limits",
    }
    if unknown:
        raise ValueError(f"benchmark config {source} has unknown top-level keys: {sorted(unknown)}")

    meta = dict(payload.get("benchmark") or {})
    grid_raw = dict(payload.get("grid") or {})
    fixed = {
        "population_size": 12,
        "support_redundancy": 4,
        "distractor_redundancy": 1,
        "no_single_agent_solution": True,
        **dict(payload.get("fixed") or {}),
    }
    evidence = dict(payload.get("evidence") or {})
    presentation = dict(payload.get("presentation") or {})
    output = dict(payload.get("output") or {})
    limits = dict(payload.get("limits") or {})

    provider_raw = dict(payload.get("llm_provider") or {})
    provider_unknown = set(provider_raw) - _PROVIDER_FIELDS
    if provider_unknown:
        raise ValueError(f"llm_provider has unknown keys: {sorted(provider_unknown)}")
    if not provider_raw.get("type") or not provider_raw.get("model"):
        raise ValueError("llm_provider.type and llm_provider.model are required")
    provider = LLMProviderConfig(**provider_raw)

    grid = {
        "reasoning_depth": _int_list(grid_raw.get("reasoning_depth", 2), "reasoning_depth"),
        "distractors": _int_list(grid_raw.get("distractors", 4), "distractors"),
        "num_options": _int_list(grid_raw.get("num_options", 3), "num_options"),
    }
    for depth in grid["reasoning_depth"]:
        if depth not in (1, 2, 3, 4):
            raise ValueError(f"reasoning_depth {depth} is outside the generator's supported 1..4")
    for option_count in grid["num_options"]:
        if not 2 <= option_count <= 8:
            raise ValueError(f"num_options {option_count} is outside the generator's supported 2..8")

    presentation_mode = str(presentation.get("mode", ALL_POSITIONS))
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError(
            f"presentation.mode {presentation_mode!r} is not one of {PRESENTATION_MODES}"
        )

    return BenchmarkConfig(
        name=str(meta.get("name") or source.stem),
        description=str(meta.get("description") or ""),
        seed=int(meta.get("seed", 20260818)),
        tasks_per_condition=int(meta.get("tasks_per_condition", 20)),
        grid=grid,
        fixed=fixed,
        include_zero_condition=bool(evidence.get("include_zero_condition", True)),
        max_subsets_per_k=int(evidence.get("max_subsets_per_k", 4)),
        presentation_mode=presentation_mode,
        provider=provider,
        output_dir=Path(output.get("dir", "results/benchmarks/relational_support")),
        max_requests=int(limits.get("max_requests", 2000)),
        prompt_example_tasks=int(limits.get("prompt_example_tasks", 2)),
        source_path=source,
    )
