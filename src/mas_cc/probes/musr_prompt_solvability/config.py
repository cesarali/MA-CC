"""Configuration for MuSR prompt/full-profile solvability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_prompt_solvability"


class SolvabilityConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SolvabilityConfig:
    source_path: str
    provider: LLMProviderConfig
    task_dir: Path
    development_tasks: tuple[str, ...]
    heldout_tasks: tuple[str, ...]
    population_size: int
    prompt_repetitions: int
    packet_repetitions: int
    heldout_repetitions: int
    seed: int
    workers: int
    output_dir: Path
    max_requests: int
    max_input_tokens: int
    max_output_tokens_total: int
    max_cost: float
    accounting_unit: str

    @property
    def nominal_calls(self) -> int:
        phase_a = len(self.development_tasks) * 4 * self.prompt_repetitions
        phase_b = len(self.development_tasks) * 3 * self.packet_repetitions
        phase_c = len(self.heldout_tasks) * (
            2 * self.heldout_repetitions + self.population_size * self.heldout_repetitions
        )
        return phase_a + phase_b + phase_c

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "source_path": self.source_path,
            "provider": self.provider.to_dict(),
            "tasks": {
                "dataset_dir": str(self.task_dir),
                "development": list(self.development_tasks),
                "heldout": list(self.heldout_tasks),
                "population_size": self.population_size,
            },
            "design": {
                "prompt_repetitions": self.prompt_repetitions,
                "packet_repetitions": self.packet_repetitions,
                "heldout_repetitions": self.heldout_repetitions,
                "seed": self.seed,
            },
            "execution": {"workers": self.workers},
            "storage": {"output_dir": str(self.output_dir)},
            "budget": {
                "max_requests": self.max_requests,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens_total,
                "max_cost": self.max_cost,
                "accounting_unit": self.accounting_unit,
            },
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SolvabilityConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> SolvabilityConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("probe") != PROBE_NAME:
        raise SolvabilityConfigError(f"probe must be {PROBE_NAME!r}")
    provider_raw = _mapping(payload.get("provider"), "provider")
    provider = LLMProviderConfig(
        type=str(provider_raw.get("type")), model=str(provider_raw.get("model")),
        credentials_env=str(provider_raw.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(provider_raw.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(provider_raw.get("timeout_seconds", 180)),
        max_retries=int(provider_raw.get("max_retries", 0)),
        request_concurrency=int(provider_raw.get("request_concurrency", 4)),
        temperature=float(provider_raw.get("temperature", 1.0)),
        max_output_tokens=int(provider_raw.get("max_output_tokens", 4096)),
        options=dict(provider_raw.get("options") or {}),
    )
    if (provider.type, provider.model) != ("university", "gwdg/openai-gpt-oss-120b"):
        raise SolvabilityConfigError("calibration requires university/gwdg/openai-gpt-oss-120b")
    if provider.max_retries != 0:
        raise SolvabilityConfigError("provider.max_retries must be 0")
    tasks = _mapping(payload.get("tasks"), "tasks")
    development = tuple(str(item) for item in tasks.get("development", ("task_001", "task_002")))
    heldout = tuple(str(item) for item in tasks.get("heldout", ("task_003",)))
    if len(development) < 2 or not heldout or set(development) & set(heldout):
        raise SolvabilityConfigError("use at least two development tasks and disjoint held-out tasks")
    design = _mapping(payload.get("design", {}), "design")
    execution = _mapping(payload.get("execution", {}), "execution")
    storage = _mapping(payload.get("storage", {}), "storage")
    budget = _mapping(payload.get("budget", {}), "budget")
    config = SolvabilityConfig(
        source_path=str(source), provider=provider,
        task_dir=Path(str(tasks.get("dataset_dir", "results/studies/musr_team_allocation_validation_01/tasks"))),
        development_tasks=development, heldout_tasks=heldout,
        population_size=int(tasks.get("population_size", 12)),
        prompt_repetitions=int(design.get("prompt_repetitions", 20)),
        packet_repetitions=int(design.get("packet_repetitions", 20)),
        heldout_repetitions=int(design.get("heldout_repetitions", 10)),
        seed=int(design.get("seed", 20260901)), workers=int(execution.get("workers", 4)),
        output_dir=Path(str(storage.get("output_dir", "results/studies/musr_prompt_solvability_calibration_01"))),
        max_requests=int(budget.get("max_requests", 400)),
        max_input_tokens=int(budget.get("max_input_tokens", 5_000_000)),
        max_output_tokens_total=int(budget.get("max_output_tokens", 1_500_000)),
        max_cost=float(budget.get("max_cost", 5.0)),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    if min(config.prompt_repetitions, config.packet_repetitions, config.heldout_repetitions, config.workers) < 1:
        raise SolvabilityConfigError("repetitions and workers must be positive")
    if config.workers > provider.request_concurrency:
        raise SolvabilityConfigError("workers cannot exceed provider request_concurrency")
    return config


__all__ = ["PROBE_NAME", "SolvabilityConfig", "SolvabilityConfigError", "load_config"]
