"""Exact structural, leakage, and full-information validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mas_cc.core.random import Seed

from .evidence_generation import extract_json_object, forbidden_phrases
from .io_utils import sha256_object
from .latent_problem import latent_facts, score_allocation
from .prompts import full_information_prompt
from .provider_adapter import MuSRGenerationModel
from .schemas import EvidenceCard, LatentFact, LatentProblem


@dataclass(frozen=True, slots=True)
class FullInformationResult:
    accepted: bool
    successes: int
    attempts: int
    required_successes: int
    records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "successes": self.successes,
            "attempts": self.attempts,
            "required_successes": self.required_successes,
            "records": list(self.records),
        }


def _known_values(
    problem: LatentProblem, known_fact_ids: Iterable[str]
) -> dict[str, int]:
    known_ids = set(known_fact_ids)
    return {
        fact.fact_id: fact.value
        for fact in latent_facts(problem)
        if fact.fact_id in known_ids
    }


def _candidate_coefficients(
    problem: LatentProblem, candidate_index: int
) -> dict[str, int]:
    allocation = problem.candidate_allocations[candidate_index]
    person_indices = {person: index for index, person in enumerate(problem.people)}
    pair_indices = sorted(person_indices[person] for person in allocation.pair)
    keys = [
        f"skill_p{person_indices[allocation.singleton]}_t0",
        *(f"skill_p{person_indices[person]}_t1" for person in allocation.pair),
        f"coop_p{pair_indices[0]}_p{pair_indices[1]}",
    ]
    return {key: keys.count(key) for key in set(keys)}


def _minimum_score_difference(
    problem: LatentProblem,
    candidate_index: int,
    other_index: int,
    known: Mapping[str, int],
) -> int:
    left = _candidate_coefficients(problem, candidate_index)
    right = _candidate_coefficients(problem, other_index)
    minimum = 0
    for fact_id in set(left) | set(right):
        coefficient = left.get(fact_id, 0) - right.get(fact_id, 0)
        if not coefficient:
            continue
        if fact_id in known:
            value = known[fact_id]
        else:
            value = 1 if coefficient > 0 else 3
        minimum += coefficient * value
    return minimum


def agent_can_certify_unique_allocation(
    problem: LatentProblem,
    known_fact_ids: Iterable[str],
) -> tuple[bool, int | None]:
    known = _known_values(problem, known_fact_ids)
    candidate_count = len(problem.candidate_allocations)
    for index in range(candidate_count):
        if all(
            index == other
            or _minimum_score_difference(problem, index, other, known) > 0
            for other in range(candidate_count)
        ):
            return True, index
    return False, None


def validate_exact_problem(problem: LatentProblem) -> list[str]:
    errors: list[str] = []
    recomputed = tuple(
        score_allocation(problem, item) for item in problem.candidate_allocations
    )
    if recomputed != problem.candidate_scores:
        errors.append("candidate_scores do not match exact recomputation")
    winners = [
        index for index, score in enumerate(recomputed) if score == max(recomputed)
    ]
    if len(winners) != 1:
        errors.append("latent problem does not have exactly one optimum")
    elif winners[0] != problem.gold_index:
        errors.append("gold_index does not match the exact optimum")
    ranked = sorted(recomputed, reverse=True)
    if ranked[0] - ranked[1] != problem.margin_to_second_best:
        errors.append("margin_to_second_best is inconsistent")
    return errors


def validate_leakage(
    problem: LatentProblem,
    facts_by_id: Mapping[str, Any],
    cards: Sequence[EvidenceCard],
) -> list[str]:
    errors: list[str] = []
    observed_statements: set[str] = set()
    option_texts = [
        " ".join((allocation.singleton, *allocation.pair)).casefold()
        for allocation in problem.candidate_allocations
    ]
    for card in cards:
        fact = facts_by_id.get(card.latent_fact_id)
        if fact is None:
            errors.append(f"{card.evidence_id}: unknown latent fact")
            continue
        forbidden = tuple(
            phrase.casefold() for phrase in forbidden_phrases(problem, fact)
        )
        for statement in card.statements:
            lowered = statement.casefold()
            normalized = " ".join(lowered.split())
            if normalized in observed_statements:
                errors.append(
                    f"{card.evidence_id}: duplicates another evidence statement"
                )
            observed_statements.add(normalized)
            if any(phrase and phrase in lowered for phrase in forbidden):
                errors.append(f"{card.evidence_id}: forbidden leakage")
            if any(option and option in lowered for option in option_texts):
                errors.append(f"{card.evidence_id}: copies an answer option")
    return errors


def validate_distribution(
    cards: Sequence[EvidenceCard],
    assignments: Mapping[str, Sequence[str]],
    *,
    population_size: int,
    problem: LatentProblem,
) -> list[str]:
    errors: list[str] = []
    evidence_ids = {card.evidence_id for card in cards}
    card_to_fact = {card.evidence_id: card.latent_fact_id for card in cards}
    if len(assignments) != population_size:
        errors.append("agent assignment count does not equal population_size")
    expected_agents = {str(index) for index in range(population_size)}
    if set(assignments) != expected_agents:
        errors.append("agent IDs must be exactly the integers 0 through N-1")
    population_union: set[str] = set()
    for agent, held in assignments.items():
        if not held:
            errors.append(f"agent {agent} has no useful evidence")
        if len(held) != len(set(held)):
            errors.append(f"agent {agent} contains duplicate evidence IDs")
        unknown = set(held) - evidence_ids
        if unknown:
            errors.append(f"agent {agent} references unknown evidence IDs")
        population_union.update(held)
        fact_ids = {card_to_fact[item] for item in held if item in card_to_fact}
        if agent_can_certify_unique_allocation(problem, fact_ids)[0]:
            errors.append(f"agent {agent} can structurally certify a unique allocation")
    if population_union != evidence_ids:
        errors.append("population evidence union is incomplete")
    if {card.latent_fact_id for card in cards} != {
        fact.fact_id for fact in latent_facts(problem)
    }:
        errors.append("evidence does not cover every latent fact")
    return errors


async def validate_full_information(
    model: MuSRGenerationModel,
    task: dict[str, Any],
    *,
    attempts: int,
    required_successes: int,
    seed: Seed,
) -> FullInformationResult:
    if not 1 <= required_successes <= attempts:
        raise ValueError(
            "required_successes must satisfy 1 <= required_successes <= attempts"
        )
    records: list[dict[str, Any]] = []
    successes = 0
    for attempt in range(attempts):
        response = await model.inference(
            full_information_prompt(task),
            seed=int(seed.derive(attempt)),
            purpose="full_information_validation",
            metadata={"task_id": task["task_id"], "attempt": attempt + 1},
        )
        parsed_index: int | None = None
        rationale = ""
        parse_error: str | None = None
        try:
            payload = extract_json_object(response.content)
            raw_index = payload.get("option_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError("option_index must be an integer")
            if not 0 <= raw_index < len(task["options"]):
                raise ValueError("option_index is outside the option range")
            parsed_index = raw_index
            rationale = str(payload.get("rationale", ""))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        correct = parsed_index == task["gold_index"]
        successes += int(correct)
        records.append(
            {
                "attempt": attempt + 1,
                "parsed_index": parsed_index,
                "correct": correct,
                "rationale": rationale,
                "parse_error": parse_error,
                "provider": response.provider,
                "model": response.model,
                "request_id": response.request_id,
            }
        )
    return FullInformationResult(
        accepted=successes >= required_successes,
        successes=successes,
        attempts=attempts,
        required_successes=required_successes,
        records=tuple(records),
    )


def validate_frozen_task(task: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "task_id",
        "task_family",
        "scenario",
        "people",
        "tasks",
        "options",
        "gold_index",
        "evidence",
        "agent_evidence_ids",
        "latent",
        "reasoning_provenance",
        "generation",
    }
    missing = required - set(task)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    if task.get("task_family") != "musr_team_allocation":
        errors.append("task_family is not musr_team_allocation")
    fingerprint = task.get("fingerprint_sha256")
    if not isinstance(fingerprint, str) or fingerprint != sha256_object(
        {key: value for key, value in task.items() if key != "fingerprint_sha256"}
    ):
        errors.append("fingerprint_sha256 does not match task content")
    evidence = task.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must be a non-empty list")
        return errors
    evidence_ids = {
        item.get("evidence_id") for item in evidence if isinstance(item, Mapping)
    }
    if None in evidence_ids or len(evidence_ids) != len(evidence):
        errors.append("evidence IDs must be present and unique")
    assignments = task.get("agent_evidence_ids")
    if not isinstance(assignments, Mapping):
        errors.append("agent_evidence_ids must be an object")
        return errors
    referenced = {
        item
        for values in assignments.values()
        if isinstance(values, list)
        for item in values
    }
    if referenced - evidence_ids:
        errors.append("agent assignments reference unknown evidence IDs")
    if referenced != evidence_ids:
        errors.append("population evidence union is incomplete")
    options = task.get("options")
    gold_index = task.get("gold_index")
    if not isinstance(options, list) or not options:
        errors.append("options must be a non-empty list")
    elif (
        isinstance(gold_index, bool)
        or not isinstance(gold_index, int)
        or not 0 <= gold_index < len(options)
    ):
        errors.append("gold_index is outside the options range")
    latent = task.get("latent")
    if isinstance(latent, Mapping):
        try:
            problem = LatentProblem.from_dict(latent)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"latent problem is invalid: {exc}")
            problem = None
        if problem is not None:
            errors.extend(validate_exact_problem(problem))
            raw_cards: list[EvidenceCard] = []
            for item in evidence:
                if not isinstance(item, Mapping):
                    errors.append("evidence entries must be objects")
                    continue
                text = item.get("text")
                if not isinstance(text, list) or any(
                    not isinstance(line, str) for line in text
                ):
                    errors.append("evidence text must be a list of strings")
                    continue
                raw_cards.append(
                    EvidenceCard(
                        evidence_id=str(item["evidence_id"]),
                        latent_fact_id=str(item["latent_fact_id"]),
                        branch_id=str(item["branch_id"]),
                        statements=tuple(text),
                    )
                )
            if len(raw_cards) == len(evidence):
                saved_facts: dict[str, LatentFact] = {}
                provenance_value = task.get("reasoning_provenance")
                if isinstance(provenance_value, Mapping):
                    for raw_fact in provenance_value.get("latent_facts", []):
                        if not isinstance(raw_fact, Mapping):
                            continue
                        fact = LatentFact(
                            fact_id=str(raw_fact["fact_id"]),
                            kind=str(raw_fact["kind"]),
                            value=int(raw_fact["value"]),
                            people=tuple(str(person) for person in raw_fact["people"]),
                            task_index=(
                                None
                                if raw_fact.get("task_index") is None
                                else int(raw_fact["task_index"])
                            ),
                            hidden_claim=str(raw_fact["hidden_claim"]),
                        )
                        saved_facts[fact.fact_id] = fact
                errors.extend(validate_leakage(problem, saved_facts, raw_cards))
                errors.extend(
                    validate_distribution(
                        raw_cards,
                        assignments,
                        population_size=len(assignments),
                        problem=problem,
                    )
                )
    else:
        errors.append("latent must be an object")
    provenance = task.get("reasoning_provenance")
    if isinstance(provenance, Mapping):
        raw_facts = provenance.get("latent_facts")
        raw_trees = provenance.get("trees")
        fact_ids = {
            item.get("fact_id") for item in raw_facts or [] if isinstance(item, Mapping)
        }
        card_fact_ids = {
            item.get("latent_fact_id") for item in evidence if isinstance(item, Mapping)
        }
        if fact_ids != card_fact_ids:
            errors.append("evidence does not cover every saved latent fact")
        tree_branches = {
            item.get("branch_id")
            for item in raw_trees or []
            if isinstance(item, Mapping)
        }
        card_branches = {
            item.get("branch_id") for item in evidence if isinstance(item, Mapping)
        }
        if tree_branches != card_branches:
            errors.append(
                "reasoning-tree provenance does not cover every evidence branch"
            )
    else:
        errors.append("reasoning_provenance must be an object")
    validation = task.get("validation")
    if isinstance(validation, Mapping):
        full = validation.get("full_information")
        if not isinstance(full, Mapping):
            errors.append("full-information validation record is missing")
        elif not full.get("accepted", False):
            errors.append("full-information validation was not accepted")
        elif not full.get("skipped", False):
            successes = full.get("successes")
            required_successes = full.get("required_successes")
            records = full.get("records")
            if (
                not isinstance(successes, int)
                or not isinstance(required_successes, int)
                or successes < required_successes
                or not isinstance(records, list)
                or sum(
                    1
                    for record in records
                    if isinstance(record, Mapping) and record.get("correct") is True
                )
                != successes
            ):
                errors.append("full-information validation evidence is inconsistent")
    else:
        errors.append("validation must be an object")
    return errors
