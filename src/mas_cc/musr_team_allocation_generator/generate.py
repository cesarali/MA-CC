"""End-to-end native Team Allocation world and dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_cc.core.random import Seed

from .distribute import distribute_evidence
from .evidence_generation import generate_all_evidence
from .io_utils import sha256_file, sha256_object, write_json_atomic
from .latent_problem import generate_latent_problem, latent_facts, scenario_for
from .provider_adapter import MuSRGenerationModel
from .schemas import (
    MUSR_COMMIT,
    MUSR_REPOSITORY,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TASK_FAMILY,
    FrozenTask,
)
from .validate import (
    validate_distribution,
    validate_exact_problem,
    validate_frozen_task,
    validate_full_information,
    validate_leakage,
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    num_tasks: int = 20
    population_size: int = 24
    branches_per_latent_fact: int = 3
    statements_per_branch: int = 2
    tree_depth: int = 2
    evidence_redundancy: int = 3
    min_margin: int = 1
    seed: int = 0
    semantic_retries: int = 3
    world_retries: int = 3
    full_validation_attempts: int = 3
    full_validation_required: int = 2
    run_full_information_validation: bool = True
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        positive = (
            "num_tasks",
            "population_size",
            "branches_per_latent_fact",
            "statements_per_branch",
            "tree_depth",
            "evidence_redundancy",
            "min_margin",
            "semantic_retries",
            "world_retries",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.evidence_redundancy >= self.population_size:
            raise ValueError("evidence_redundancy must be smaller than population_size")
        if self.run_full_information_validation:
            if not 1 <= self.full_validation_required <= self.full_validation_attempts:
                raise ValueError(
                    "full_validation_required must be between 1 and full_validation_attempts"
                )


def option_rows(problem: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, allocation in enumerate(problem.candidate_allocations):
        first_task, second_task = problem.tasks
        display = (
            f"{first_task.name}: {allocation.singleton}; "
            f"{second_task.name}: {allocation.pair[0]} and {allocation.pair[1]}"
        )
        rows.append(
            {
                "id": f"ALLOCATION_{index}",
                "index": index,
                "display_text": display,
                **allocation.to_dict(problem.tasks),
            }
        )
    return rows


async def generate_world(
    model: MuSRGenerationModel,
    config: GenerationConfig,
    *,
    task_index: int,
) -> FrozenTask:
    if config.prompt_version != model.prompt_version:
        raise ValueError(
            "GenerationConfig.prompt_version must match the provider adapter"
        )
    task_seed = Seed(config.seed).derive(task_index)
    failure_reasons: list[str] = []
    for world_attempt in range(config.world_retries):
        attempt_seed = task_seed.derive(world_attempt)
        problem = generate_latent_problem(
            attempt_seed.derive("latent").create_random(),
            min_margin=config.min_margin,
        )
        facts = latent_facts(problem)
        generated = await generate_all_evidence(
            model,
            problem,
            facts,
            branches_per_latent_fact=config.branches_per_latent_fact,
            statements_per_branch=config.statements_per_branch,
            tree_depth=config.tree_depth,
            seed=attempt_seed.derive("evidence"),
            max_attempts=config.semantic_retries,
        )
        assignments = distribute_evidence(
            generated.cards,
            problem,
            population_size=config.population_size,
            redundancy=config.evidence_redundancy,
            rng=attempt_seed.derive("distribution").create_random(),
        )
        exact_errors = validate_exact_problem(problem)
        fact_map = {fact.fact_id: fact for fact in facts}
        structural_errors = validate_distribution(
            generated.cards,
            assignments,
            population_size=config.population_size,
            problem=problem,
        )
        leakage_errors = validate_leakage(problem, fact_map, generated.cards)
        errors = exact_errors + structural_errors + leakage_errors
        if errors:
            failure_reasons.extend(errors)
            continue

        task: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "task_id": f"musr_team_allocation_{task_index:04d}",
            "task_family": TASK_FAMILY,
            "scenario": scenario_for(problem),
            "question": "Which allocation is expected to be most effective?",
            "people": list(problem.people),
            "tasks": [task.to_dict() for task in problem.tasks],
            "skills": [task.skill for task in problem.tasks],
            "options": option_rows(problem),
            "gold_index": problem.gold_index,
            "gold_answer": f"ALLOCATION_{problem.gold_index}",
            "evidence": [card.to_dict() for card in generated.cards],
            "agent_evidence_ids": assignments,
            "latent": problem.to_dict(),
            "reasoning_provenance": {
                "latent_facts": [fact.to_dict() for fact in facts],
                "trees": [tree.to_dict() for tree in generated.trees],
                "generation_attempts": generated.attempts,
                "failed_generation_attempt_count": len(generated.failures),
            },
            "generation": {
                "seed": config.seed,
                "task_seed": int(task_seed),
                "world_attempt": world_attempt + 1,
                "provider": model.provider.name,
                "model": model.provider.model,
                "temperature_requested": model.temperature,
                "max_output_tokens": model.max_output_tokens,
                "tree_depth": config.tree_depth,
                "branches_per_latent_fact": config.branches_per_latent_fact,
                "statements_per_branch": config.statements_per_branch,
                "evidence_redundancy": config.evidence_redundancy,
                "prompt_version": config.prompt_version,
                "musr_repo": MUSR_REPOSITORY,
                "musr_commit": MUSR_COMMIT,
                "musr_license": "MIT",
            },
            "validation": {
                "unique_exact_winner": True,
                "score_margin": problem.margin_to_second_best,
                "population_complete": True,
                "no_single_agent_structural_solution": True,
                "forbidden_leakage_free": True,
            },
        }
        if config.run_full_information_validation:
            result = await validate_full_information(
                model,
                task,
                attempts=config.full_validation_attempts,
                required_successes=config.full_validation_required,
                seed=attempt_seed.derive("full_information_validation"),
            )
            task["validation"]["full_information"] = result.to_dict()
            if not result.accepted:
                failure_reasons.append(
                    f"full-information validation failed: {result.successes}/{result.attempts}"
                )
                continue
        else:
            task["validation"]["full_information"] = {
                "accepted": True,
                "skipped": True,
                "reason": "disabled explicitly; run dataset QA before scientific use",
            }
        task["fingerprint_sha256"] = sha256_object(task)
        errors = validate_frozen_task(task)
        if not errors:
            return FrozenTask(task)
        failure_reasons.extend(errors)
    raise RuntimeError(
        f"could not generate task {task_index} after {config.world_retries} world attempts: "
        + "; ".join(failure_reasons)
    )


async def generate_dataset(
    model: MuSRGenerationModel,
    config: GenerationConfig,
    *,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty frozen dataset directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    task_hashes: dict[str, str] = {}
    task_fingerprints: dict[str, str] = {}
    for task_index in range(1, config.num_tasks + 1):
        task = await generate_world(model, config, task_index=task_index)
        filename = f"{task.task_id}.json"
        path = output / filename
        write_json_atomic(path, task.to_dict())
        task_hashes[filename] = sha256_file(path)
        task_fingerprints[filename] = str(task.payload["fingerprint_sha256"])
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_family": TASK_FAMILY,
        "dataset_seed": config.seed,
        "num_tasks": config.num_tasks,
        "population_size": config.population_size,
        "generation_config": {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
        "provider": model.provider.name,
        "model": model.provider.model,
        "musr_repo": MUSR_REPOSITORY,
        "musr_commit": MUSR_COMMIT,
        "task_hashes": task_hashes,
        "task_fingerprints": task_fingerprints,
        "provider_calls": list(model.calls),
    }
    manifest["dataset_fingerprint_sha256"] = sha256_object(
        {"settings": manifest["generation_config"], "task_hashes": task_hashes}
    )
    write_json_atomic(output / "manifest.json", manifest)
    return manifest
