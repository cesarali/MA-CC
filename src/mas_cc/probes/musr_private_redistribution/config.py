"""Strict config for private-evidence redistribution calibration."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml
from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_private_redistribution"


@dataclass(frozen=True, slots=True)
class RedistributionConfig:
    source_path: str
    provider: LLMProviderConfig
    task_dir: Path
    tasks: tuple[str, ...]
    development: tuple[str, ...]
    heldout: tuple[str, ...]
    population_size: int
    private_repetitions: int
    endpoint_repetitions: int
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
        return len(self.tasks) * (
            4 * self.population_size * self.private_repetitions
            + 2 * self.endpoint_repetitions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "source_path": self.source_path,
            "provider": self.provider.to_dict(),
            "tasks": {
                "dataset_dir": str(self.task_dir),
                "ids": list(self.tasks),
                "development": list(self.development),
                "heldout": list(self.heldout),
                "population_size": self.population_size,
            },
            "design": {
                "private_repetitions": self.private_repetitions,
                "endpoint_repetitions": self.endpoint_repetitions,
                "seed": self.seed,
                "prompt_variant": "P2",
                "full_profile": "F9",
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


def _map(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> RedistributionConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    p = _map(raw.get("provider"), "provider")
    provider = LLMProviderConfig(
        type=str(p.get("type")),
        model=str(p.get("model")),
        credentials_env=str(p.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(p.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(p.get("timeout_seconds", 180)),
        max_retries=int(p.get("max_retries", 0)),
        request_concurrency=int(p.get("request_concurrency", 4)),
        temperature=float(p.get("temperature", 1)),
        max_output_tokens=int(p.get("max_output_tokens", 4096)),
        options=dict(p.get("options") or {}),
    )
    if (provider.type, provider.model) != (
        "university",
        "gwdg/openai-gpt-oss-120b",
    ) or provider.max_retries != 0:
        raise ValueError(
            "requires university/gwdg/openai-gpt-oss-120b with max_retries 0"
        )
    t = _map(raw.get("tasks"), "tasks")
    d = _map(raw.get("design", {}), "design")
    e = _map(raw.get("execution", {}), "execution")
    s = _map(raw.get("storage", {}), "storage")
    b = _map(raw.get("budget", {}), "budget")
    tasks = tuple(str(x) for x in t.get("ids", ("task_001", "task_002", "task_003")))
    development = tuple(str(x) for x in t.get("development", ("task_001", "task_002")))
    heldout = tuple(str(x) for x in t.get("heldout", ("task_003",)))
    if set(development) | set(heldout) != set(tasks) or set(development) & set(heldout):
        raise ValueError("development and heldout must partition tasks")
    cfg = RedistributionConfig(
        str(source),
        provider,
        Path(
            str(
                t.get(
                    "dataset_dir",
                    "results/studies/musr_team_allocation_validation_01/tasks",
                )
            )
        ),
        tasks,
        development,
        heldout,
        int(t.get("population_size", 12)),
        int(d.get("private_repetitions", 3)),
        int(d.get("endpoint_repetitions", 10)),
        int(d.get("seed", 20260901)),
        int(e.get("workers", 4)),
        Path(
            str(
                s.get(
                    "output_dir",
                    "results/studies/musr_private_redistribution_calibration_01",
                )
            )
        ),
        int(b.get("max_requests", 550)),
        int(b.get("max_input_tokens", 6000000)),
        int(b.get("max_output_tokens", 2200000)),
        float(b.get("max_cost", 5)),
        str(b.get("accounting_unit", "proxy_accounting_unit")),
    )
    if cfg.nominal_calls != 492:
        raise ValueError(f"expected 492 calls, got {cfg.nominal_calls}")
    if cfg.workers < 1 or cfg.workers > provider.request_concurrency:
        raise ValueError("workers must fit provider concurrency")
    return cfg


__all__ = ["PROBE_NAME", "RedistributionConfig", "load_config"]
