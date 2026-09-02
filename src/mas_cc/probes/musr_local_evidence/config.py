"""Strict configuration for the self-contained MuSR local-evidence probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_local_evidence"
REQUIRED_PROVIDER = "university"
REQUIRED_MODEL = "gwdg/openai-gpt-oss-120b"


class LocalEvidenceConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalEvidenceProbeConfig:
    source_path: str
    provider: LLMProviderConfig
    task_dir: Path
    task_id: str
    population_size: int
    agents: tuple[int, ...]
    pair_repetitions: int
    doses: tuple[int, ...]
    dose_repetitions: int
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
        return len(self.agents) * self.pair_repetitions * 2 + len(self.agents) * len(self.doses) * self.dose_repetitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "source_path": self.source_path,
            "provider": self.provider.to_dict(),
            "task": {
                "dataset_dir": str(self.task_dir),
                "task_id": self.task_id,
                "population_size": self.population_size,
            },
            "design": {
                "agents": list(self.agents),
                "pair_repetitions": self.pair_repetitions,
                "doses": list(self.doses),
                "dose_repetitions": self.dose_repetitions,
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
        raise LocalEvidenceConfigError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LocalEvidenceConfigError(f"{name} must be a positive integer")
    return value


def load_probe_config(path: str | Path) -> LocalEvidenceProbeConfig:
    source = Path(path)
    if not source.is_file():
        raise LocalEvidenceConfigError(f"probe config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("probe") != PROBE_NAME:
        raise LocalEvidenceConfigError(f"probe must be {PROBE_NAME!r}")
    provider_raw = _mapping(payload.get("provider"), "provider")
    provider = LLMProviderConfig(
        type=str(provider_raw.get("type", "")),
        model=str(provider_raw.get("model", "")),
        credentials_env=str(provider_raw.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(provider_raw.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(provider_raw.get("timeout_seconds", 180)),
        max_retries=int(provider_raw.get("max_retries", 0)),
        request_concurrency=int(provider_raw.get("request_concurrency", 4)),
        temperature=float(provider_raw.get("temperature", 1.0)),
        max_output_tokens=int(provider_raw.get("max_output_tokens", 4096)),
        options=dict(provider_raw.get("options") or {}),
    )
    if (provider.type, provider.model) != (REQUIRED_PROVIDER, REQUIRED_MODEL):
        raise LocalEvidenceConfigError(
            f"this probe requires {REQUIRED_PROVIDER}/{REQUIRED_MODEL}"
        )
    if provider.max_retries != 0:
        raise LocalEvidenceConfigError("provider.max_retries must be 0 for exact attempt accounting")
    task = _mapping(payload.get("task"), "task")
    design = _mapping(payload.get("design"), "design")
    agents = tuple(int(item) for item in design.get("agents", (1, 4, 3)))
    if len(agents) < 3 or len(set(agents)) != len(agents) or any(item < 1 for item in agents):
        raise LocalEvidenceConfigError("design.agents must contain at least three unique positive IDs")
    doses = tuple(int(item) for item in design.get("doses", (0, 3, 6, 9, 12, 18, 27)))
    if doses != (0, 3, 6, 9, 12, 18, 27):
        raise LocalEvidenceConfigError("design.doses must be exactly [0, 3, 6, 9, 12, 18, 27]")
    execution = _mapping(payload.get("execution", {}), "execution")
    storage = _mapping(payload.get("storage", {}), "storage")
    budget = _mapping(payload.get("budget", {}), "budget")
    config = LocalEvidenceProbeConfig(
        source_path=str(source),
        provider=provider,
        task_dir=Path(str(task.get("dataset_dir", "results/studies/musr_team_allocation_validation_01/tasks"))),
        task_id=str(task.get("task_id", "task_001")),
        population_size=_positive(int(task.get("population_size", 12)), "task.population_size"),
        agents=agents,
        pair_repetitions=_positive(int(design.get("pair_repetitions", 10)), "design.pair_repetitions"),
        doses=doses,
        dose_repetitions=_positive(int(design.get("dose_repetitions", 3)), "design.dose_repetitions"),
        seed=int(design.get("seed", 20260901)),
        workers=_positive(int(execution.get("workers", 4)), "execution.workers"),
        output_dir=Path(str(storage.get("output_dir", "results/studies/musr_local_evidence_probe_01"))),
        max_requests=_positive(int(budget.get("max_requests", 150)), "budget.max_requests"),
        max_input_tokens=_positive(int(budget.get("max_input_tokens", 2_000_000)), "budget.max_input_tokens"),
        max_output_tokens_total=_positive(int(budget.get("max_output_tokens", 600_000)), "budget.max_output_tokens"),
        max_cost=float(budget.get("max_cost", 5.0)),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    if config.nominal_calls != 123:
        raise LocalEvidenceConfigError(f"the paper probe must schedule 123 calls, got {config.nominal_calls}")
    if config.workers > provider.request_concurrency:
        raise LocalEvidenceConfigError("execution.workers cannot exceed provider.request_concurrency")
    return config


__all__ = ["LocalEvidenceConfigError", "LocalEvidenceProbeConfig", "PROBE_NAME", "load_probe_config"]
