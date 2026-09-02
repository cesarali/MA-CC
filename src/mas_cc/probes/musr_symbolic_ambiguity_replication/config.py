"""Strict configuration for the frozen symbolic-ambiguity replication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_symbolic_ambiguity_replication"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _provider(value: Any) -> LLMProviderConfig:
    raw = _mapping(value, "behavioral_provider")
    return LLMProviderConfig(
        type=str(raw.get("type", "university")),
        model=str(raw["model"]),
        credentials_env=str(raw.get("credentials_env", "POTSDAM_API_KEY")),
        base_url_env=str(raw.get("base_url_env", "BASE_POTSDAM_LLM_URL")),
        timeout_seconds=float(raw.get("timeout_seconds", 180)),
        max_retries=int(raw.get("max_retries", 0)),
        request_concurrency=int(raw.get("request_concurrency", 4)),
        temperature=float(raw.get("temperature", 1.0)),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        options=dict(raw.get("options") or {}),
    )


@dataclass(frozen=True, slots=True)
class ReplicationConfig:
    source_path: str
    calibration_root: Path
    expected_manifest_file_sha256: str
    expected_manifest_content_sha256: str
    expected_symbolic_selection_sha256: str
    behavioral_provider: LLMProviderConfig
    population_size: int
    seed: int
    prompt_variant: str
    full_profile: str
    private_breadth: int
    max_predictability: float
    min_normalized_entropy: float
    min_score_margin: int
    private_existing_repetitions: int
    private_additional_repetitions: int
    endpoint_existing_repetitions: int
    endpoint_additional_repetitions: int
    zero_max_truth_rate: float
    private_max_truth_rate: float
    full_min_truth_rate: float
    borderline_private_max: float
    minimum_full_private_separation: float
    task_pathology_full_below: float
    behavioral_workers: int
    output_dir: Path
    max_behavioral_requests: int
    max_behavioral_input_tokens: int
    max_behavioral_output_tokens: int
    max_behavioral_cost: float
    accounting_unit: str

    @property
    def provider(self) -> LLMProviderConfig:
        return self.behavioral_provider

    @property
    def workers(self) -> int:
        return self.behavioral_workers

    @property
    def max_requests(self) -> int:
        return self.max_behavioral_requests

    @property
    def max_input_tokens(self) -> int:
        return self.max_behavioral_input_tokens

    @property
    def max_output_tokens_total(self) -> int:
        return self.max_behavioral_output_tokens

    @property
    def max_cost(self) -> float:
        return self.max_behavioral_cost

    @property
    def additional_calls(self) -> int:
        return 6 * (
            2 * self.endpoint_additional_repetitions
            + self.population_size * self.private_additional_repetitions
        )

    @property
    def final_calls(self) -> int:
        return 6 * (
            2
            * (
                self.endpoint_existing_repetitions
                + self.endpoint_additional_repetitions
            )
            + self.population_size
            * (self.private_existing_repetitions + self.private_additional_repetitions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "source": {
                "calibration_root": str(self.calibration_root),
                "expected_manifest_file_sha256": self.expected_manifest_file_sha256,
                "expected_manifest_content_sha256": self.expected_manifest_content_sha256,
                "expected_symbolic_selection_sha256": self.expected_symbolic_selection_sha256,
            },
            "behavioral_provider": self.behavioral_provider.to_dict(),
            "frozen_design": {
                "population_size": self.population_size,
                "seed": self.seed,
                "prompt_variant": self.prompt_variant,
                "full_profile": self.full_profile,
                "private_breadth": self.private_breadth,
                "max_predictability": self.max_predictability,
                "min_normalized_entropy": self.min_normalized_entropy,
                "min_score_margin": self.min_score_margin,
            },
            "behavioral_validation": {
                "private_existing_repetitions": self.private_existing_repetitions,
                "private_additional_repetitions": self.private_additional_repetitions,
                "endpoint_existing_repetitions": self.endpoint_existing_repetitions,
                "endpoint_additional_repetitions": self.endpoint_additional_repetitions,
                "zero_max_truth_rate": self.zero_max_truth_rate,
                "private_max_truth_rate": self.private_max_truth_rate,
                "full_min_truth_rate": self.full_min_truth_rate,
            },
            "recommendation_rule": {
                "borderline_private_max": self.borderline_private_max,
                "minimum_full_private_separation": self.minimum_full_private_separation,
                "task_pathology_full_below": self.task_pathology_full_below,
            },
            "execution": {"behavioral_workers": self.behavioral_workers},
            "storage": {"output_dir": str(self.output_dir)},
            "budget": {
                "accounting_unit": self.accounting_unit,
                "max_behavioral_requests": self.max_behavioral_requests,
                "max_behavioral_input_tokens": self.max_behavioral_input_tokens,
                "max_behavioral_output_tokens": self.max_behavioral_output_tokens,
                "max_behavioral_cost": self.max_behavioral_cost,
            },
        }


def load_config(path: str | Path) -> ReplicationConfig:
    source_path = Path(path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    source = _mapping(raw.get("source"), "source")
    frozen = _mapping(raw.get("frozen_design"), "frozen_design")
    behavioral = _mapping(raw.get("behavioral_validation"), "behavioral_validation")
    recommendation = _mapping(raw.get("recommendation_rule"), "recommendation_rule")
    execution = _mapping(raw.get("execution"), "execution")
    storage = _mapping(raw.get("storage"), "storage")
    budget = _mapping(raw.get("budget"), "budget")
    provider = _provider(raw.get("behavioral_provider"))
    config = ReplicationConfig(
        source_path=str(source_path),
        calibration_root=Path(str(source["calibration_root"])),
        expected_manifest_file_sha256=str(source["expected_manifest_file_sha256"]),
        expected_manifest_content_sha256=str(
            source["expected_manifest_content_sha256"]
        ),
        expected_symbolic_selection_sha256=str(
            source["expected_symbolic_selection_sha256"]
        ),
        behavioral_provider=provider,
        population_size=int(frozen.get("population_size", 12)),
        seed=int(frozen.get("seed", 20260901)),
        prompt_variant=str(frozen.get("prompt_variant", "P2")),
        full_profile=str(frozen.get("full_profile", "F9")),
        private_breadth=int(frozen.get("private_breadth", 4)),
        max_predictability=float(frozen.get("max_predictability", 0.45)),
        min_normalized_entropy=float(frozen.get("min_normalized_entropy", 0.90)),
        min_score_margin=int(frozen.get("min_score_margin", 2)),
        private_existing_repetitions=int(
            behavioral.get("private_existing_repetitions", 3)
        ),
        private_additional_repetitions=int(
            behavioral.get("private_additional_repetitions", 3)
        ),
        endpoint_existing_repetitions=int(
            behavioral.get("endpoint_existing_repetitions", 10)
        ),
        endpoint_additional_repetitions=int(
            behavioral.get("endpoint_additional_repetitions", 10)
        ),
        zero_max_truth_rate=float(behavioral.get("zero_max_truth_rate", 0.45)),
        private_max_truth_rate=float(behavioral.get("private_max_truth_rate", 0.45)),
        full_min_truth_rate=float(behavioral.get("full_min_truth_rate", 0.80)),
        borderline_private_max=float(
            recommendation.get("borderline_private_max", 0.50)
        ),
        minimum_full_private_separation=float(
            recommendation.get("minimum_full_private_separation", 0.25)
        ),
        task_pathology_full_below=float(
            recommendation.get("task_pathology_full_below", 0.50)
        ),
        behavioral_workers=int(execution.get("behavioral_workers", 4)),
        output_dir=Path(
            str(
                storage.get(
                    "output_dir",
                    "results/studies/musr_symbolic_ambiguity_replication_01",
                )
            )
        ),
        max_behavioral_requests=int(budget.get("max_behavioral_requests", 360)),
        max_behavioral_input_tokens=int(
            budget.get("max_behavioral_input_tokens", 4_000_000)
        ),
        max_behavioral_output_tokens=int(
            budget.get("max_behavioral_output_tokens", 1_500_000)
        ),
        max_behavioral_cost=float(budget.get("max_behavioral_cost", 5.0)),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    frozen_values = (
        config.population_size,
        config.seed,
        config.prompt_variant,
        config.full_profile,
        config.private_breadth,
        config.max_predictability,
        config.min_normalized_entropy,
        config.min_score_margin,
        config.private_existing_repetitions,
        config.private_additional_repetitions,
        config.endpoint_existing_repetitions,
        config.endpoint_additional_repetitions,
    )
    if frozen_values != (12, 20260901, "P2", "F9", 4, 0.45, 0.90, 2, 3, 3, 10, 10):
        raise ValueError("replication design does not match the frozen handoff")
    if (
        provider.type,
        provider.model,
        provider.temperature,
        provider.max_output_tokens,
    ) != (
        "university",
        "gwdg/openai-gpt-oss-120b",
        1.0,
        4096,
    ):
        raise ValueError("replication must use the frozen gpt-oss provider settings")
    if config.additional_calls != 336 or config.final_calls != 672:
        raise ValueError(
            "replication sample-size contract is not 336 new / 672 final calls"
        )
    if config.additional_calls > config.max_behavioral_requests:
        raise ValueError("behavioral request budget is below the additional call plan")
    if config.behavioral_workers > provider.request_concurrency:
        raise ValueError("behavioral workers exceed provider request concurrency")
    return config


__all__ = ["PROBE_NAME", "ReplicationConfig", "load_config"]
