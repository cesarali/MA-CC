"""Strict configuration for symbolic MuSR construction and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.config import LLMProviderConfig

PROBE_NAME = "musr_symbolic_ambiguity"


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
class SymbolicAmbiguityConfig:
    source_path: str
    generation_provider: LLMProviderConfig
    behavioral_provider: LLMProviderConfig
    candidate_worlds: int
    private_breadth_candidates: tuple[int, ...]
    preferred_max_predictability: float
    fallback_max_predictability: float
    min_normalized_entropy: float
    margin_candidates: tuple[int, ...]
    final_tasks: int
    population_size: int
    minimum_holders: int
    seed: int
    branches_per_latent_fact: int
    statements_per_branch: int
    tree_depth: int
    semantic_retries: int
    private_repetitions: int
    endpoint_repetitions: int
    generation_workers: int
    behavioral_workers: int
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

    # Compatibility surface for the shared resumable behavioral executor.
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
        return self.final_tasks * 9

    @property
    def maximum_generation_calls(self) -> int:
        return self.nominal_generation_calls * self.semantic_retries

    @property
    def behavioral_calls(self) -> int:
        return self.final_tasks * (
            2 * self.endpoint_repetitions
            + self.population_size * self.private_repetitions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_NAME,
            "generation_provider": self.generation_provider.to_dict(),
            "behavioral_provider": self.behavioral_provider.to_dict(),
            "symbolic_ambiguity": {
                "candidate_worlds": self.candidate_worlds,
                "private_breadth_candidates": list(self.private_breadth_candidates),
                "preferred_max_predictability": self.preferred_max_predictability,
                "fallback_max_predictability": self.fallback_max_predictability,
                "min_normalized_entropy": self.min_normalized_entropy,
                "evaluate_all_subsets": True,
            },
            "world_filter": {
                "require_unique_optimum": True,
                "min_score_margin_candidates": list(self.margin_candidates),
                "balance_gold_allocations": True,
                "final_tasks": self.final_tasks,
            },
            "generation": {
                "population_size": self.population_size,
                "minimum_holders": self.minimum_holders,
                "seed": self.seed,
                "branches_per_latent_fact": self.branches_per_latent_fact,
                "statements_per_branch": self.statements_per_branch,
                "tree_depth": self.tree_depth,
                "semantic_retries": self.semantic_retries,
            },
            "behavioral_validation": {
                "model": self.behavioral_provider.model,
                "prompt_variant": "P2",
                "full_profile": "F9",
                "private_repetitions": self.private_repetitions,
                "endpoint_repetitions": self.endpoint_repetitions,
                "zero_max_truth_rate": 0.45,
                "private_max_truth_rate": 0.45,
                "full_min_truth_rate": 0.80,
                "preferred_full_truth_rate": 0.90,
            },
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


def load_config(path: str | Path) -> SymbolicAmbiguityConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("probe") != PROBE_NAME:
        raise ValueError(f"probe must be {PROBE_NAME}")
    generation_provider = _provider(raw.get("generation_provider"), "generation_provider")
    behavioral_provider = _provider(raw.get("behavioral_provider"), "behavioral_provider")
    if (behavioral_provider.type, behavioral_provider.model) != (
        "university",
        "gwdg/openai-gpt-oss-120b",
    ):
        raise ValueError("behavioral validation requires gwdg/openai-gpt-oss-120b")
    if generation_provider.model == behavioral_provider.model:
        raise ValueError("the game-playing model must not be used as the evidence generator")
    symbolic = _mapping(raw.get("symbolic_ambiguity", {}), "symbolic_ambiguity")
    world = _mapping(raw.get("world_filter", {}), "world_filter")
    generation = _mapping(raw.get("generation", {}), "generation")
    behavioral = _mapping(raw.get("behavioral_validation", {}), "behavioral_validation")
    execution = _mapping(raw.get("execution", {}), "execution")
    storage = _mapping(raw.get("storage", {}), "storage")
    budget = _mapping(raw.get("budget", {}), "budget")
    config = SymbolicAmbiguityConfig(
        source_path=str(source),
        generation_provider=generation_provider,
        behavioral_provider=behavioral_provider,
        candidate_worlds=int(symbolic.get("candidate_worlds", 10_000)),
        private_breadth_candidates=tuple(int(x) for x in symbolic.get("private_breadth_candidates", (2, 3, 4))),
        preferred_max_predictability=float(symbolic.get("preferred_max_predictability", 0.45)),
        fallback_max_predictability=float(symbolic.get("fallback_max_predictability", 0.50)),
        min_normalized_entropy=float(symbolic.get("min_normalized_entropy", 0.90)),
        margin_candidates=tuple(int(x) for x in world.get("min_score_margin_candidates", (1, 2))),
        final_tasks=int(world.get("final_tasks", 6)),
        population_size=int(generation.get("population_size", 12)),
        minimum_holders=int(generation.get("minimum_holders", 2)),
        seed=int(generation.get("seed", 20260901)),
        branches_per_latent_fact=int(generation.get("branches_per_latent_fact", 3)),
        statements_per_branch=int(generation.get("statements_per_branch", 2)),
        tree_depth=int(generation.get("tree_depth", 2)),
        semantic_retries=int(generation.get("semantic_retries", 3)),
        private_repetitions=int(behavioral.get("private_repetitions", 3)),
        endpoint_repetitions=int(behavioral.get("endpoint_repetitions", 10)),
        generation_workers=int(execution.get("generation_workers", 4)),
        behavioral_workers=int(execution.get("behavioral_workers", 4)),
        output_dir=Path(str(storage.get("output_dir", "results/studies/musr_symbolic_ambiguity_calibration_01"))),
        max_generation_requests=int(budget.get("max_generation_requests", 180)),
        max_behavioral_requests=int(budget.get("max_behavioral_requests", 360)),
        max_generation_input_tokens=int(budget.get("max_generation_input_tokens", 500_000)),
        max_behavioral_input_tokens=int(budget.get("max_behavioral_input_tokens", 4_000_000)),
        max_generation_output_tokens=int(budget.get("max_generation_output_tokens", 360_000)),
        max_behavioral_output_tokens=int(budget.get("max_behavioral_output_tokens", 1_500_000)),
        max_generation_cost=float(budget.get("max_generation_cost", 5.0)),
        max_behavioral_cost=float(budget.get("max_behavioral_cost", 5.0)),
        accounting_unit=str(budget.get("accounting_unit", "proxy_accounting_unit")),
    )
    if config.candidate_worlds < 10_000:
        raise ValueError("the symbolic scan requires at least 10,000 candidate worlds")
    if config.private_breadth_candidates != (2, 3, 4):
        raise ValueError("private breadth candidates must be exactly [2, 3, 4]")
    if config.margin_candidates != (1, 2):
        raise ValueError("score-margin candidates must be exactly [1, 2]")
    if config.final_tasks < 6 or config.final_tasks % 3:
        raise ValueError("final_tasks must be a multiple of three and at least six")
    if config.population_size != 12:
        raise ValueError("calibration_01 freezes population_size at 12")
    if config.private_repetitions < 3 or config.endpoint_repetitions < 10:
        raise ValueError("behavioral sample sizes are below the handoff minimum")
    if config.maximum_generation_calls > config.max_generation_requests:
        raise ValueError("generation request budget is below the retry ceiling")
    if config.behavioral_calls > config.max_behavioral_requests:
        raise ValueError("behavioral request budget is below the call plan")
    if config.generation_workers > generation_provider.request_concurrency:
        raise ValueError("generation workers exceed provider concurrency")
    if config.behavioral_workers > behavioral_provider.request_concurrency:
        raise ValueError("behavioral workers exceed provider concurrency")
    return config


__all__ = ["PROBE_NAME", "SymbolicAmbiguityConfig", "load_config"]
