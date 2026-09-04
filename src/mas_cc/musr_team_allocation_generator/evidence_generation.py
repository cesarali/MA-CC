"""Structured MuSR-style evidence generation through MAS-CC providers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mas_cc.core.random import Seed

from .prompts import evidence_prompt
from .provider_adapter import MuSRGenerationModel
from .reasoning_tree import ReasoningTree, build_reasoning_tree
from .schemas import EvidenceCard, LatentFact, LatentProblem
from .symbolic_facts import CanonicalFact

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class EvidenceGenerationError(RuntimeError):
    """Raised when bounded semantic retries cannot produce valid evidence."""


@dataclass(frozen=True, slots=True)
class GeneratedEvidence:
    cards: tuple[EvidenceCard, ...]
    trees: tuple[ReasoningTree, ...]
    attempts: int
    failures: tuple[str, ...]


def extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = _JSON_FENCE.sub("", text.strip())
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, Mapping):
        raise ValueError("response JSON must be an object")
    return value


def forbidden_phrases(
    problem: LatentProblem, fact: LatentFact | CanonicalFact
) -> tuple[str, ...]:
    hidden_claim = (
        fact.hidden_claim if isinstance(fact, LatentFact) else fact.canonical_text
    )
    phrases = [
        hidden_claim,
        "best allocation",
        "correct allocation",
        "correct assignment",
        "best choice",
        "should be assigned",
        "option 0",
        "option 1",
        "option 2",
        "allocation 0",
        "allocation 1",
        "allocation 2",
    ]
    for allocation in problem.candidate_allocations:
        first_task, second_task = problem.tasks
        phrases.append(f"{allocation.singleton}; {' and '.join(allocation.pair)}")
        phrases.append(
            f"{first_task.name}: {allocation.singleton}; "
            f"{second_task.name}: {allocation.pair[0]} and {allocation.pair[1]}"
        )
        phrases.append(f"assign {allocation.singleton} to {first_task.name}")
    return tuple(phrases)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _validate_branch_payload(
    value: Mapping[str, Any],
    *,
    branches: int,
    statements_per_branch: int,
    tree_depth: int,
    forbidden: Sequence[str],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    raw_branches = value.get("branches")
    if not isinstance(raw_branches, list) or len(raw_branches) != branches:
        raise ValueError(f"expected exactly {branches} branches")
    parsed = []
    all_statements: set[str] = set()
    forbidden_norm = tuple(
        _normalized(phrase) for phrase in forbidden if phrase.strip()
    )
    for branch_index, raw in enumerate(raw_branches):
        if not isinstance(raw, Mapping):
            raise ValueError(f"branch {branch_index} must be an object")
        statements = raw.get("statements")
        bridges = raw.get("commonsense_bridges")
        intermediate = raw.get("intermediate_claims", [])
        if not isinstance(statements, list) or len(statements) != statements_per_branch:
            raise ValueError(
                f"branch {branch_index} must have exactly {statements_per_branch} statements"
            )
        if not isinstance(bridges, list) or not bridges:
            raise ValueError(f"branch {branch_index} needs a commonsense bridge")
        if not isinstance(intermediate, list) or len(intermediate) != max(
            0, tree_depth - 1
        ):
            raise ValueError(
                f"branch {branch_index} needs exactly {max(0, tree_depth - 1)} intermediate claims"
            )
        groups: list[tuple[str, ...]] = []
        for field_name, raw_values in (
            ("statements", statements),
            ("commonsense_bridges", bridges),
            ("intermediate_claims", intermediate),
        ):
            if any(
                not isinstance(item, str) or not item.strip() for item in raw_values
            ):
                raise ValueError(
                    f"branch {branch_index} {field_name} must contain non-empty strings"
                )
            groups.append(tuple(item.strip() for item in raw_values))
        visible, hidden_bridges, intermediate_claims = groups
        for statement in visible:
            normalized = _normalized(statement)
            if normalized in all_statements:
                raise ValueError("duplicate explicit statements are not allowed")
            if any(phrase and phrase in normalized for phrase in forbidden_norm):
                raise ValueError(
                    f"forbidden answer leakage in statement: {statement!r}"
                )
            if re.search(r"\b(score|points?)\s*(?:is|of|=|:)\s*\d+\b", statement, re.I):
                raise ValueError(f"numeric score leakage in statement: {statement!r}")
            all_statements.add(normalized)
        parsed.append((visible, hidden_bridges, intermediate_claims))
    return tuple(parsed)


async def generate_evidence_for_fact(
    model: MuSRGenerationModel,
    problem: LatentProblem,
    fact: LatentFact | CanonicalFact,
    *,
    branches: int,
    statements_per_branch: int,
    tree_depth: int,
    seed: Seed,
    max_attempts: int,
) -> GeneratedEvidence:
    forbidden = forbidden_phrases(problem, fact)
    failures: list[str] = []
    for attempt in range(max_attempts):
        response = await model.inference(
            evidence_prompt(
                problem,
                fact,
                branches=branches,
                statements_per_branch=statements_per_branch,
                tree_depth=tree_depth,
                forbidden_phrases=forbidden,
            ),
            seed=int(seed.derive(attempt)),
            purpose="evidence_generation",
            metadata={"latent_fact_id": fact.fact_id, "attempt": attempt + 1},
        )
        try:
            parsed = _validate_branch_payload(
                extract_json_object(response.content),
                branches=branches,
                statements_per_branch=statements_per_branch,
                tree_depth=tree_depth,
                forbidden=forbidden,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
            continue
        cards: list[EvidenceCard] = []
        trees: list[ReasoningTree] = []
        for index, (statements, bridges, intermediate) in enumerate(parsed):
            branch_id = f"{fact.fact_id}_b{index:02d}"
            evidence_id = f"e_{branch_id}"
            cards.append(EvidenceCard(evidence_id, fact.fact_id, branch_id, statements))
            trees.append(
                build_reasoning_tree(
                    latent_fact_id=fact.fact_id,
                    branch_id=branch_id,
                    hidden_claim=(
                        fact.hidden_claim
                        if isinstance(fact, LatentFact)
                        else fact.canonical_text
                    ),
                    intermediate_claims=intermediate,
                    statements=statements,
                    commonsense_bridges=bridges,
                )
            )
        return GeneratedEvidence(
            tuple(cards), tuple(trees), attempt + 1, tuple(failures)
        )
    raise EvidenceGenerationError(
        f"could not generate valid evidence for {fact.fact_id} after {max_attempts} attempts: "
        + "; ".join(failures)
    )


async def generate_all_evidence(
    model: MuSRGenerationModel,
    problem: LatentProblem,
    facts: Sequence[LatentFact | CanonicalFact],
    *,
    branches_per_latent_fact: int,
    statements_per_branch: int,
    tree_depth: int,
    seed: Seed,
    max_attempts: int,
) -> GeneratedEvidence:
    cards: list[EvidenceCard] = []
    trees: list[ReasoningTree] = []
    failures: list[str] = []
    attempts = 0
    for fact in facts:
        generated = await generate_evidence_for_fact(
            model,
            problem,
            fact,
            branches=branches_per_latent_fact,
            statements_per_branch=statements_per_branch,
            tree_depth=tree_depth,
            seed=seed.derive(fact.fact_id),
            max_attempts=max_attempts,
        )
        cards.extend(generated.cards)
        trees.extend(generated.trees)
        attempts += generated.attempts
        failures.extend(generated.failures)
    return GeneratedEvidence(tuple(cards), tuple(trees), attempts, tuple(failures))
