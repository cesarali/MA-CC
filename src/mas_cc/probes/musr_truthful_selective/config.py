"""Configuration for truthful selective-disclosure task calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.musr_team_allocation_generator.selective_design import SelectiveThresholds

PROBE_NAME = "musr_truthful_selective"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _provider(value: Any, name: str) -> LLMProviderConfig:
    raw = _mapping(value, name)
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
class BehavioralThresholds:
    zero_min_truth: float = 0.20
    zero_max_truth: float = 0.50
    zero_min_false: float = 0.20
    zero_max_false: float = 0.50
    private_max_truth: float = 0.50
    private_max_false: float = 0.50
    controller_max_false: float = 0.85
    decisive_min_truth: float = 0.70
    mixed_min_truth: float = 0.80
    full_min_truth: float = 0.90
    development_full_min_truth: float = 0.80


@dataclass(frozen=True, slots=True)
class TruthfulSelectiveConfig:
    source_path: str
    generation_provider: LLMProviderConfig
    behavioral_provider: LLMProviderConfig
    candidate_worlds: int
    development_tasks: int
    seed: int
    symbolic: SelectiveThresholds
    behavioral: BehavioralThresholds
    zero_repetitions: int
    private_repetitions: int
    controller_repetitions: int
    decisive_repetitions: int
    mixed_repetitions: int
    full_repetitions: int
    alternative_subset_repetitions: int
    alternative_subsets_per_budget: int
    generation_workers: int
    behavioral_workers: int
    semantic_retries: int
    output_dir: Path
    max_generation_requests: int
    max_behavioral_requests: int
    max_generation_input_tokens: int
    max_behavioral_input_tokens: int
    max_generation_output_tokens: int
    max_behavioral_output_tokens: int
    max_generation_cost: float
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
    def nominal_generation_calls(self) -> int:
        # At most 99 generated propositions plus 99 semantic audits per task.
        return self.development_tasks * 99 * 2

    @property
    def maximum_generation_calls(self) -> int:
        # Evidence generation can use semantic retries; its audit is one call.
        facts = self.development_tasks * 99
        return facts * self.semantic_retries + facts

    @property
    def base_behavioral_calls_per_task(self) -> int:
        budgets = len(self.symbolic.controller_budgets)
        return (
            self.zero_repetitions
            + self.symbolic.population_size * self.private_repetitions
            + budgets * self.controller_repetitions
            + self.decisive_repetitions
            + budgets * self.mixed_repetitions
            + self.full_repetitions
        )

    @property
    def behavioral_calls(self) -> int:
        """Conservative ceiling; exact count is frozen from task pool sizes."""

        alternatives = max(0, self.alternative_subsets_per_budget) * 3
        return self.development_tasks * (
            self.base_behavioral_calls_per_task
            + alternatives * self.alternative_subset_repetitions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "generation_provider": self.generation_provider.to_dict(),
            "behavioral_provider": self.behavioral_provider.to_dict(),
            "scan": {
                "candidate_worlds": self.candidate_worlds,
                "development_tasks": self.development_tasks,
                "seed": self.seed,
            },
            "symbolic_thresholds": asdict(self.symbolic),
            "behavioral_thresholds": asdict(self.behavioral),
            "replications": {
                "zero": self.zero_repetitions,
                "private": self.private_repetitions,
                "controller": self.controller_repetitions,
                "decisive": self.decisive_repetitions,
                "mixed": self.mixed_repetitions,
                "full": self.full_repetitions,
                "alternative_subset": self.alternative_subset_repetitions,
                "alternative_subsets_per_budget": self.alternative_subsets_per_budget,
            },
            "generation": {"semantic_retries": self.semantic_retries},
            "execution": {
                "generation_workers": self.generation_workers,
                "behavioral_workers": self.behavioral_workers,
            },
            "storage": {"output_dir": str(self.output_dir)},
            "budget": {
                "accounting_unit": self.accounting_unit,
                "max_generation_requests": self.max_generation_requests,
                "max_behavioral_requests": self.max_behavioral_requests,
                "max_generation_input_tokens": self.max_generation_input_tokens,
                "max_behavioral_input_tokens": self.max_behavioral_input_tokens,
                "max_generation_output_tokens": self.max_generation_output_tokens,
                "max_behavioral_output_tokens": self.max_behavioral_output_tokens,
                "max_generation_cost": self.max_generation_cost,
                "max_behavioral_cost": self.max_behavioral_cost,
            },
        }


def load_config(path: str | Path) -> TruthfulSelectiveConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    generation_provider = _provider(
        raw.get("generation_provider"), "generation_provider"
    )
    behavioral_provider = _provider(
        raw.get("behavioral_provider"), "behavioral_provider"
    )
    if generation_provider.model != "microsoft/gpt-5.6-terra":
        raise ValueError("generation requires microsoft/gpt-5.6-terra")
    if behavioral_provider.model != "gwdg/openai-gpt-oss-120b":
        raise ValueError("behavioral validation requires gwdg/openai-gpt-oss-120b")
    scan = _mapping(raw.get("scan", {}), "scan")
    symbolic_raw = _mapping(raw.get("symbolic_thresholds", {}), "symbolic_thresholds")
    behavioral_raw = _mapping(
        raw.get("behavioral_thresholds", {}), "behavioral_thresholds"
    )
    reps = _mapping(raw.get("replications", {}), "replications")
    generation = _mapping(raw.get("generation", {}), "generation")
    execution = _mapping(raw.get("execution", {}), "execution")
    storage = _mapping(raw.get("storage", {}), "storage")
    budget = _mapping(raw.get("budget", {}), "budget")
    symbolic = SelectiveThresholds(
        zero_max_probability=float(symbolic_raw.get("zero_max_probability", 0.45)),
        zero_min_entropy=float(symbolic_raw.get("zero_min_entropy", 0.90)),
        private_max_probability=float(
            symbolic_raw.get("private_max_probability", 0.45)
        ),
        private_min_entropy=float(symbolic_raw.get("private_min_entropy", 0.90)),
        minimum_controller_facts=int(symbolic_raw.get("minimum_controller_facts", 24)),
        controller_budgets=tuple(
            int(x) for x in symbolic_raw.get("controller_budgets", (3, 6, 12, 24))
        ),
        controller_min_lift=float(symbolic_raw.get("controller_min_lift", 1e-12)),
        controller_max_false_probability=float(
            symbolic_raw.get("controller_max_false_probability", 0.70)
        ),
        decisive_min_truth_probability=float(
            symbolic_raw.get("decisive_min_truth_probability", 0.80)
        ),
        subset_positive_lift_fraction=float(
            symbolic_raw.get("subset_positive_lift_fraction", 0.70)
        ),
        subset_samples=int(symbolic_raw.get("subset_samples", 64)),
        population_size=int(symbolic_raw.get("population_size", 24)),
        private_assignments_tested=int(
            symbolic_raw.get("private_assignments_tested", 32)
        ),
        private_facts_per_agent=int(symbolic_raw.get("private_facts_per_agent", 1)),
        minimum_private_holders=int(symbolic_raw.get("minimum_private_holders", 1)),
    )
    behavioral = BehavioralThresholds(
        **{
            field: float(behavioral_raw.get(field, default))
            for field, default in asdict(BehavioralThresholds()).items()
        }
    )
    config = TruthfulSelectiveConfig(
        source_path=str(source),
        generation_provider=generation_provider,
        behavioral_provider=behavioral_provider,
        candidate_worlds=int(scan.get("candidate_worlds", 10_000)),
        development_tasks=int(scan.get("development_tasks", 3)),
        seed=int(scan.get("seed", 20260904)),
        symbolic=symbolic,
        behavioral=behavioral,
        zero_repetitions=int(reps.get("zero", 20)),
        private_repetitions=int(reps.get("private", 5)),
        controller_repetitions=int(reps.get("controller", 20)),
        decisive_repetitions=int(reps.get("decisive", 20)),
        mixed_repetitions=int(reps.get("mixed", 20)),
        full_repetitions=int(reps.get("full", 20)),
        alternative_subset_repetitions=int(reps.get("alternative_subset", 3)),
        alternative_subsets_per_budget=int(
            reps.get("alternative_subsets_per_budget", 5)
        ),
        generation_workers=int(execution.get("generation_workers", 4)),
        behavioral_workers=int(execution.get("behavioral_workers", 4)),
        semantic_retries=int(generation.get("semantic_retries", 3)),
        output_dir=Path(
            str(
                storage.get(
                    "output_dir",
                    "results/studies/musr_truthful_selective_task_calibration_01",
                )
            )
        ),
        max_generation_requests=int(budget.get("max_generation_requests", 200)),
        max_behavioral_requests=int(budget.get("max_behavioral_requests", 2_000)),
        max_generation_input_tokens=int(
            budget.get("max_generation_input_tokens", 800_000)
        ),
        max_behavioral_input_tokens=int(
            budget.get("max_behavioral_input_tokens", 10_000_000)
        ),
        max_generation_output_tokens=int(
            budget.get("max_generation_output_tokens", 500_000)
        ),
        max_behavioral_output_tokens=int(
            budget.get("max_behavioral_output_tokens", 5_000_000)
        ),
        max_generation_cost=float(budget.get("max_generation_cost", 10)),
        max_behavioral_cost=float(budget.get("max_behavioral_cost", 20)),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    if config.candidate_worlds < 10_000:
        raise ValueError("the symbolic scan requires at least 10,000 candidate worlds")
    if not 3 <= config.development_tasks <= 6:
        raise ValueError("development_tasks must be between three and six")
    if config.symbolic.population_size != 24:
        raise ValueError("truthful-selective calibration requires N=24")
    if config.symbolic.controller_budgets != (3, 6, 12, 24):
        raise ValueError("controller budgets must be [3, 6, 12, 24]")
    if config.maximum_generation_calls > config.max_generation_requests:
        raise ValueError("generation request budget is below the retry ceiling")
    if config.behavioral_calls > config.max_behavioral_requests:
        raise ValueError("behavioral request budget is below the call plan")
    return config


__all__ = [
    "BehavioralThresholds",
    "PROBE_NAME",
    "TruthfulSelectiveConfig",
    "load_config",
]
