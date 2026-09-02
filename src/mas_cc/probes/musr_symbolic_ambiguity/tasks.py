"""Post-filter evidence generation and frozen task-pack construction."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.core import Seed
from mas_cc.llm_runtime.providers import (
    BudgetExpectation,
    BudgetGuardedProvider,
    BudgetLimits,
    MonetaryAmount,
    RuntimeBudgetGuard,
    UniversityPricingSource,
    create_llm_provider,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.musr_team_allocation_generator.evidence_generation import generate_all_evidence
from mas_cc.musr_team_allocation_generator.evidence_generation import forbidden_phrases
from mas_cc.musr_team_allocation_generator.generate import option_rows
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    generate_latent_problem,
    latent_facts,
    problem_from_latent_values,
    scenario_for,
)
from mas_cc.musr_team_allocation_generator.prompts import evidence_prompt
from mas_cc.musr_team_allocation_generator.provider_adapter import MuSRGenerationModel
from mas_cc.musr_team_allocation_generator.schemas import (
    MUSR_COMMIT,
    MUSR_REPOSITORY,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TASK_FAMILY,
    EvidenceCard,
    LatentProblem,
)
from mas_cc.musr_team_allocation_generator.validate import (
    agent_can_certify_unique_allocation,
    validate_distribution,
    validate_exact_problem,
    validate_leakage,
)
from mas_cc.probes.musr_prompt_solvability.execution import append

from .config import SymbolicAmbiguityConfig


def _problem(selection: Mapping[str, Any], config: SymbolicAmbiguityConfig) -> LatentProblem:
    task_id = str(selection["task_id"])
    context = generate_latent_problem(
        Seed(config.seed).derive(f"{task_id}:context").create_random(), min_margin=1
    )
    return problem_from_latent_values(
        tuple(int(value) for value in selection["latent_values"]),
        people=context.people,
        tasks=context.tasks,
    )


def generation_input_estimate(
    selections: Sequence[Mapping[str, Any]], config: SymbolicAmbiguityConfig
) -> int:
    counter = RegexTokenCounter()
    total = 0
    for selection in selections:
        problem = _problem(selection, config)
        for fact in latent_facts(problem):
            prompt = evidence_prompt(
                problem,
                fact,
                branches=config.branches_per_latent_fact,
                statements_per_branch=config.statements_per_branch,
                tree_depth=config.tree_depth,
                forbidden_phrases=forbidden_phrases(problem, fact),
            )
            total += counter.count_tokens(prompt)
    return total


def _assign_cards(
    cards: Sequence[EvidenceCard], selection: Mapping[str, Any]
) -> dict[str, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    fact_ids = [
        "skill_p0_t0",
        "skill_p0_t1",
        "skill_p1_t0",
        "skill_p1_t1",
        "skill_p2_t0",
        "skill_p2_t1",
        "coop_p0_p1",
        "coop_p0_p2",
        "coop_p1_p2",
    ]
    fact_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    for card in cards:
        groups[fact_index[card.latent_fact_id]].append(card.evidence_id)
    if any(len(groups[index]) != 3 for index in range(9)):
        raise RuntimeError("expected exactly three generated cards per latent value")

    views = [tuple(int(x) for x in row["visible_indices"]) for row in selection["private_views"]]
    holdings: dict[str, list[str]] = {str(index): [] for index in range(len(views))}
    cursors = Counter()
    for agent, view in enumerate(views):
        for latent in view:
            choices = groups[latent]
            holdings[str(agent)].append(choices[cursors[latent] % len(choices)])
            cursors[latent] += 1

    # Cover every generated branch before adding arbitrary within-view redundancy.
    missing = [card.evidence_id for card in cards if not any(card.evidence_id in held for held in holdings.values())]
    card_latent = {card.evidence_id: fact_index[card.latent_fact_id] for card in cards}
    for evidence_id in missing:
        latent = card_latent[evidence_id]
        eligible = [
            str(agent)
            for agent, view in enumerate(views)
            if latent in view and len(holdings[str(agent)]) < 6
        ]
        if not eligible:
            raise RuntimeError("could not cover every generated evidence branch")
        target = min(eligible, key=lambda agent: (len(holdings[agent]), int(agent)))
        holdings[target].append(evidence_id)

    for agent, view in enumerate(views):
        key = str(agent)
        cursor = 0
        while len(holdings[key]) < 6:
            latent = view[cursor % len(view)]
            candidates = [item for item in groups[latent] if item not in holdings[key]]
            if candidates:
                holdings[key].append(candidates[0])
            cursor += 1
            if cursor > 100:
                raise RuntimeError("could not fill six cards within assigned latent values")
    order = [card.evidence_id for card in cards]
    return {
        agent: [evidence_id for evidence_id in order if evidence_id in set(held)]
        for agent, held in holdings.items()
    }


def _distribution(
    base: Mapping[str, Any],
    problem: LatentProblem,
    cards: Sequence[EvidenceCard],
    assignments: Mapping[str, Sequence[str]],
    selection: Mapping[str, Any],
    config: SymbolicAmbiguityConfig,
) -> dict[str, Any]:
    errors = validate_distribution(
        cards,
        assignments,
        population_size=config.population_size,
        problem=problem,
    )
    if errors:
        raise RuntimeError("invalid symbolic private distribution: " + "; ".join(errors))
    card_to_fact = {card.evidence_id: card.latent_fact_id for card in cards}
    diagnostics = []
    violations = 0
    for agent, (evidence_ids, symbolic) in enumerate(
        zip(assignments.values(), selection["private_views"], strict=True)
    ):
        held_facts = {card_to_fact[item] for item in evidence_ids}
        certified, winner = agent_can_certify_unique_allocation(problem, held_facts)
        violations += int(certified)
        diagnostics.append(
            {
                "agent_id": str(agent),
                "evidence_cards": len(evidence_ids),
                "distinct_latent_facts": len(held_facts),
                "structurally_certifies_solution": certified,
                "certified_index": winner,
                **dict(symbolic),
            }
        )
    payload = {
        "schema_version": "musr_team_allocation_distribution_v1",
        "task_id": base["task_id"],
        "semantic_world_sha256": base["semantic_world_sha256"],
        "population_size": config.population_size,
        "evidence_redundancy": None,
        "distribution_seed": int(Seed(config.seed).derive(f"{base['task_id']}:symbolic-distribution")),
        "algorithm_version": "symbolic_ambiguity_private_views_v1",
        "agent_evidence_ids": {key: list(value) for key, value in assignments.items()},
        "agent_diagnostics": diagnostics,
        "latent_holder_counts": dict(
            sorted(
                Counter(
                    latent
                    for row in selection["private_views"]
                    for latent in row["visible_indices"]
                ).items()
            )
        ),
        "no_single_agent_violations": violations,
    }
    payload["fingerprint_sha256"] = sha256_object(payload)
    return payload


async def _generate_one(
    model: MuSRGenerationModel,
    selection: Mapping[str, Any],
    config: SymbolicAmbiguityConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = str(selection["task_id"])
    problem = _problem(selection, config)
    if validate_exact_problem(problem):
        raise RuntimeError("selected symbolic world failed exact validation")
    facts = latent_facts(problem)
    task_seed = Seed(config.seed).derive(f"{task_id}:evidence")
    generated = await generate_all_evidence(
        model,
        problem,
        facts,
        branches_per_latent_fact=config.branches_per_latent_fact,
        statements_per_branch=config.statements_per_branch,
        tree_depth=config.tree_depth,
        seed=task_seed,
        max_attempts=config.semantic_retries,
    )
    leakage = validate_leakage(
        problem, {fact.fact_id: fact for fact in facts}, generated.cards
    )
    if leakage:
        raise RuntimeError("generated evidence failed leakage validation: " + "; ".join(leakage))
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
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
            "provider": config.generation_provider.type,
            "model": config.generation_provider.model,
            "temperature_requested": config.generation_provider.temperature,
            "max_output_tokens": config.generation_provider.max_output_tokens,
            "tree_depth": config.tree_depth,
            "branches_per_latent_fact": config.branches_per_latent_fact,
            "statements_per_branch": config.statements_per_branch,
            "prompt_version": PROMPT_VERSION,
            "musr_repo": MUSR_REPOSITORY,
            "musr_commit": MUSR_COMMIT,
            "musr_license": "MIT",
            "symbolic_candidate_id": selection["candidate_id"],
        },
        "validation": {
            "unique_exact_winner": True,
            "score_margin": problem.margin_to_second_best,
            "symbolic_ambiguity_gate": True,
            "language_generated_after_symbolic_acceptance": True,
            "forbidden_leakage_free": True,
        },
    }
    base["semantic_world_sha256"] = sha256_object(base)
    assignments = _assign_cards(generated.cards, selection)
    distribution = _distribution(
        base, problem, generated.cards, assignments, selection, config
    )
    return base, distribution


def _complete_manifest(root: Path, selections: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    path = root / "generation_manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(item["task_id"]) for item in selections}
    if set(manifest.get("tasks", {})) != expected:
        return None
    for task_id, hashes in manifest["tasks"].items():
        for name, expected_hash in hashes.items():
            target = root / task_id / name
            if not target.is_file() or sha256_file(target) != expected_hash:
                return None
    return manifest


async def generate_task_pack(
    config: SymbolicAmbiguityConfig,
    selections: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    existing = _complete_manifest(output, selections)
    if existing is not None:
        return existing

    quote = UniversityPricingSource(config.generation_provider).fetch(
        config.generation_provider.type, config.generation_provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"generation pricing does not permit launch: {quote.status}")
    input_estimate = generation_input_estimate(selections, config)
    limit = MonetaryAmount(
        config.max_generation_cost,
        config.accounting_unit,
        "symbolic ambiguity config",
        config.generation_provider.type,
        config.generation_provider.model,
        "MuSR evidence generation after symbolic acceptance",
        quote.retrieved_at,
        "symbolic-ambiguity-v1",
    )
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=limit,
            max_requests=config.max_generation_requests,
            max_input_tokens=config.max_generation_input_tokens,
            max_output_tokens=config.max_generation_output_tokens,
        ),
        expectation=BudgetExpectation(
            config.nominal_generation_calls,
            input_estimate,
            config.nominal_generation_calls * config.generation_provider.max_output_tokens,
        ),
    )
    raw = create_llm_provider(config.generation_provider)
    counter = RegexTokenCounter()
    provider = BudgetGuardedProvider(
        raw,
        guard,
        quote.pricing,
        input_token_estimator=lambda request: sum(
            counter.count_tokens(message.content) for message in request.messages
        ),
    )
    model = MuSRGenerationModel(
        provider,
        temperature=config.generation_provider.temperature,
        max_output_tokens=config.generation_provider.max_output_tokens,
        prompt_version=PROMPT_VERSION,
        audit_sink=lambda row: append(output.parent / "generation" / "raw_calls.jsonl", row),
    )
    semaphore = asyncio.Semaphore(config.generation_workers)

    async def one(selection: Mapping[str, Any]):
        async with semaphore:
            return await _generate_one(model, selection, config)

    try:
        generated = await asyncio.gather(*(one(selection) for selection in selections))
    finally:
        provider.close()

    task_hashes: dict[str, dict[str, str]] = {}
    for selection, (base, distribution) in zip(selections, generated, strict=True):
        task_id = str(selection["task_id"])
        destination = output / task_id
        destination.mkdir(parents=True, exist_ok=True)
        base_path = destination / "base_task.json"
        distribution_path = destination / f"distribution_N{config.population_size}.json"
        write_json_atomic(base_path, base)
        write_json_atomic(distribution_path, distribution)
        task_hashes[task_id] = {
            base_path.name: sha256_file(base_path),
            distribution_path.name: sha256_file(distribution_path),
        }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "provider": config.generation_provider.type,
        "model": config.generation_provider.model,
        "prompt_version": PROMPT_VERSION,
        "tasks": task_hashes,
        "calls": len(model.calls),
        "usage": {
            "input_tokens": sum(int(row["usage"].get("input_tokens") or 0) for row in model.calls),
            "output_tokens": sum(int(row["usage"].get("output_tokens") or 0) for row in model.calls),
        },
        "budget_status": guard.status(),
    }
    manifest["fingerprint_sha256"] = sha256_object(manifest)
    write_json_atomic(output / "generation_manifest.json", manifest)
    return manifest


__all__ = ["generate_task_pack", "generation_input_estimate"]
