"""Frozen configuration for the isolated OSS evidence-to-answer evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_truthful_selective_isolated"


def _map(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    concurrency: int
    requests_per_minute: int


@dataclass(frozen=True, slots=True)
class IsolatedOSSConfig:
    source_path: str
    calibration_root: Path
    revision_manifest: Path
    expected_tasks: Mapping[str, int]
    budgets: tuple[int, ...]
    random_replicates: int
    answer_orders: int
    assignment_population: int
    prompt_variant: str
    seed: int
    provider: LLMProviderConfig
    smoke: ExecutionProfile
    cluster: ExecutionProfile
    invalid_response_retries: int
    output_dir: Path
    max_logical_calls: int
    max_provider_attempts: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost: float
    accounting_unit: str
    assumed_latency_seconds: float

    @property
    def logical_calls(self) -> int:
        tasks = len(self.expected_tasks)
        agents = self.assignment_population
        orders = self.answer_orders
        budgets = len(self.budgets)
        return tasks * orders + tasks * agents * orders * (
            1 + budgets + budgets * self.random_replicates
        )

    def profile(self, name: str) -> ExecutionProfile:
        if name == "smoke":
            return self.smoke
        if name == "cluster":
            return self.cluster
        raise ValueError("execution profile must be smoke or cluster")

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "source": {
                "calibration_root": str(self.calibration_root),
                "revision_manifest": str(self.revision_manifest),
                "tasks": dict(self.expected_tasks),
                "assignment_population": self.assignment_population,
            },
            "evaluation": {
                "budgets": list(self.budgets),
                "answer_orders": "all_six",
                "random_replicates": self.random_replicates,
                "prompt_variant": self.prompt_variant,
                "seed": self.seed,
                "equal_task_weight_primary": True,
            },
            "provider": self.provider.to_dict(),
            "execution_profiles": {
                "smoke": asdict(self.smoke),
                "cluster": asdict(self.cluster),
                "invalid_response_retries": self.invalid_response_retries,
            },
            "storage": {"output_dir": str(self.output_dir)},
            "budget": {
                "accounting_unit": self.accounting_unit,
                "max_logical_calls": self.max_logical_calls,
                "max_provider_attempts": self.max_provider_attempts,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens,
                "max_cost": self.max_cost,
            },
            "planning": {"assumed_latency_seconds": self.assumed_latency_seconds},
        }


def load_isolated_config(path: str | Path) -> IsolatedOSSConfig:
    source_path = Path(path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    source = _map(raw.get("source"), "source")
    evaluation = _map(raw.get("evaluation"), "evaluation")
    provider_raw = _map(raw.get("provider"), "provider")
    profiles = _map(raw.get("execution_profiles"), "execution_profiles")
    smoke = _map(profiles.get("smoke"), "execution_profiles.smoke")
    cluster = _map(profiles.get("cluster"), "execution_profiles.cluster")
    storage = _map(raw.get("storage"), "storage")
    budget = _map(raw.get("budget"), "budget")
    planning = _map(raw.get("planning", {}), "planning")
    calibration_root = Path(str(source["calibration_root"]))
    revision = Path(str(source.get("revision_manifest", "analysis/task_revision_manifest.json")))
    if not revision.is_absolute():
        revision = calibration_root / revision
    provider = LLMProviderConfig(
        type=str(provider_raw.get("type", "university")),
        model=str(provider_raw["model"]),
        credentials_env=str(provider_raw.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(provider_raw.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(provider_raw.get("timeout_seconds", 180)),
        max_retries=int(provider_raw.get("max_retries", 2)),
        request_concurrency=int(provider_raw.get("request_concurrency", 30)),
        temperature=float(provider_raw.get("temperature", 1.0)),
        max_output_tokens=int(provider_raw.get("max_output_tokens", 4096)),
        options=dict(provider_raw.get("options") or {}),
    )
    config = IsolatedOSSConfig(
        source_path=str(source_path),
        calibration_root=calibration_root,
        revision_manifest=revision,
        expected_tasks={str(k): int(v) for k, v in _map(source.get("tasks"), "source.tasks").items()},
        budgets=tuple(int(value) for value in evaluation.get("budgets", (3, 6, 9, 12))),
        random_replicates=int(evaluation.get("random_replicates", 5)),
        answer_orders=6,
        assignment_population=int(source.get("assignment_population", 24)),
        prompt_variant=str(evaluation.get("prompt_variant", "P2")),
        seed=int(evaluation.get("seed", 20260907)),
        provider=provider,
        smoke=ExecutionProfile(int(smoke.get("concurrency", 4)), int(smoke.get("requests_per_minute", 60))),
        cluster=ExecutionProfile(int(cluster.get("concurrency", 30)), int(cluster.get("requests_per_minute", 300))),
        invalid_response_retries=int(profiles.get("invalid_response_retries", 1)),
        output_dir=Path(str(storage["output_dir"])),
        max_logical_calls=int(budget["max_logical_calls"]),
        max_provider_attempts=int(budget["max_provider_attempts"]),
        max_input_tokens=int(budget["max_input_tokens"]),
        max_output_tokens=int(budget["max_output_tokens"]),
        max_cost=float(budget["max_cost"]),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
        assumed_latency_seconds=float(planning.get("assumed_latency_seconds", 10.0)),
    )
    if config.expected_tasks != {"task_001": 42, "task_002": 237, "task_003": 130}:
        raise ValueError("isolated evaluation requires approved candidates 42/237/130")
    if config.budgets != (3, 6, 9, 12) or config.random_replicates != 5:
        raise ValueError("isolated design requires budgets 3/6/9/12 and five random packets")
    if config.assignment_population != 24 or config.answer_orders != 6:
        raise ValueError("isolated design requires N=24 and all six answer orders")
    if (provider.type, provider.model, provider.temperature) != (
        "university", "gwdg/openai-gpt-oss-120b", 1.0
    ):
        raise ValueError("provider must be frozen university gwdg/openai-gpt-oss-120b at temperature 1.0")
    if config.logical_calls != 10_818 or config.max_logical_calls != 10_818:
        raise ValueError("isolated full manifest must contain exactly 10,818 logical calls")
    if config.max_provider_attempts < config.logical_calls * (config.invalid_response_retries + 1):
        raise ValueError("provider-attempt cap is below schema retry ceiling")
    return config


__all__ = ["ExecutionProfile", "IsolatedOSSConfig", "PROBE_NAME", "load_isolated_config"]
