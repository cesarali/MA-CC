"""Terra language generation after the symbolic gate has frozen tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from mas_cc.musr_team_allocation_generator.evidence_generation import (
    generate_all_evidence,
    generate_evidence_for_fact,
)
from mas_cc.musr_team_allocation_generator.evidence_generation import forbidden_phrases
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)
from mas_cc.musr_team_allocation_generator.provider_adapter import MuSRGenerationModel
from mas_cc.musr_team_allocation_generator.prompts import evidence_prompt
from mas_cc.musr_team_allocation_generator.schemas import (
    EvidenceCard,
    LatentProblem,
    PROMPT_VERSION,
)
from mas_cc.musr_team_allocation_generator.reasoning_tree import build_reasoning_tree
from mas_cc.musr_team_allocation_generator.symbolic_facts import (
    CanonicalFact,
    render_canonical_equality_evidence,
)
from mas_cc.musr_team_allocation_generator.validate import validate_leakage
from mas_cc.probes.musr_prompt_solvability.execution import append

from .config import TruthfulSelectiveConfig
from .generation_validation import audit_cards


def _load_task(root: Path) -> tuple[LatentProblem, tuple[CanonicalFact, ...]]:
    problem = LatentProblem.from_dict(
        json.loads((root / "hidden_world.json").read_text(encoding="utf-8"))
    )
    facts = tuple(
        CanonicalFact.from_dict(row)
        for row in json.loads(
            (root / "facts/all_true_facts.json").read_text(encoding="utf-8")
        )
    )
    return problem, facts


def input_token_estimate(
    config: TruthfulSelectiveConfig, task_roots: Sequence[Path]
) -> int:
    from mas_cc.musr_team_allocation_generator.evidence_generation import (
        forbidden_phrases,
    )
    from mas_cc.musr_team_allocation_generator.prompts import evidence_prompt

    counter = RegexTokenCounter()
    return sum(
        counter.count_tokens(
            evidence_prompt(
                problem,
                fact,
                branches=1,
                statements_per_branch=2,
                tree_depth=2,
                forbidden_phrases=forbidden_phrases(problem, fact),
            )
        )
        for task_root in task_roots
        for problem, facts in [_load_task(task_root)]
        for fact in facts
    )


async def generate(
    config: TruthfulSelectiveConfig, task_roots: Sequence[Path], root: Path
) -> dict[str, Any]:
    manifest_path = root / "generation/terra_generation_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            (task_root / "generation/generated_cards.json").is_file()
            and (task_root / "generation/semantic_validation_summary.json").is_file()
            and read_summary(task_root).get("all_passed") is True
            for task_root in task_roots
        ):
            return existing
    quote = UniversityPricingSource(config.generation_provider).fetch(
        config.generation_provider.type, config.generation_provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"generation pricing does not permit launch: {quote.status}")
    estimate = input_token_estimate(config, task_roots)
    exact_fact_calls = sum(len(_load_task(task_root)[1]) for task_root in task_roots)
    expected_calls = exact_fact_calls * 2
    guard = RuntimeBudgetGuard(
        BudgetLimits(
            max_cost=MonetaryAmount(
                config.max_generation_cost,
                config.accounting_unit,
                "truthful-selective config",
                config.generation_provider.type,
                config.generation_provider.model,
                "MuSR truthful selective evidence generation",
                quote.retrieved_at,
                "truthful-selective-v1",
            ),
            max_requests=config.max_generation_requests,
            max_input_tokens=config.max_generation_input_tokens,
            max_output_tokens=config.max_generation_output_tokens,
        ),
        expectation=BudgetExpectation(
            expected_calls,
            estimate * 2,
            expected_calls * config.generation_provider.max_output_tokens,
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
        audit_sink=lambda row: append(root / "generation/raw_calls.jsonl", row),
    )
    semaphore = asyncio.Semaphore(config.generation_workers)

    async def one(task_root: Path) -> Mapping[str, Any]:
        problem, facts = _load_task(task_root)
        cards_path = task_root / "generation/generated_cards.json"
        provenance_path = task_root / "generation/branch_leaf_provenance.json"
        roles = {
            str(row["fact_id"]): str(row["role"])
            for row in json.loads(
                (task_root / "facts/all_true_facts.json").read_text(encoding="utf-8")
            )
        }
        if cards_path.is_file() and provenance_path.is_file():
            cards = json.loads(cards_path.read_text(encoding="utf-8"))
            cards_by_id = {str(row["fact_id"]): dict(row) for row in cards}
            provenance_by_id = {
                str(row["latent_fact_id"]): dict(row)
                for row in json.loads(provenance_path.read_text(encoding="utf-8"))
            }
            for fact in facts:
                if fact.operator != "eq":
                    continue
                statements = render_canonical_equality_evidence(problem, fact)
                cards_by_id[fact.fact_id] = {
                    "fact_id": fact.fact_id,
                    "canonical_exact_fact": fact.to_dict(),
                    "classification": roles[fact.fact_id],
                    "generated_card_text": list(statements),
                    "branch_id": f"{fact.fact_id}_canonical_equality",
                    "generation_prompt_hash": None,
                    "model_id": None,
                    "generation_seed": None,
                    "rendering_method": "deterministic_canonical_equality_v1",
                    "renderer_input_sha256": sha256_object(fact.to_dict()),
                    "entailment": "direct_by_template",
                    "validation": {"structural": True},
                }
                provenance_by_id[fact.fact_id] = build_reasoning_tree(
                    latent_fact_id=fact.fact_id,
                    branch_id=f"{fact.fact_id}_canonical_equality",
                    hidden_claim=fact.canonical_text,
                    intermediate_claims=(),
                    statements=statements,
                    commonsense_bridges=(),
                ).to_dict()
            cards = [cards_by_id[fact.fact_id] for fact in facts]
            write_json_atomic(cards_path, cards)
            write_json_atomic(
                provenance_path,
                [provenance_by_id[fact.fact_id] for fact in facts],
            )
            generation_attempts = len(cards)
            generation_failures: list[str] = []
            leakage: list[str] = []
        else:
            generated_facts = tuple(fact for fact in facts if fact.operator != "eq")
            async with semaphore:
                generated = await generate_all_evidence(
                    model,
                    problem,
                    generated_facts,
                    branches_per_latent_fact=1,
                    statements_per_branch=2,
                    tree_depth=2,
                    seed=Seed(config.seed).derive(f"terra:{task_root.name}"),
                    max_attempts=config.semantic_retries,
                )
            trees = {tree.latent_fact_id: tree for tree in generated.trees}
            cards = []
            provenance = []
            for fact, card in zip(generated_facts, generated.cards, strict=True):
                tree = trees[fact.fact_id]
                prompt_text = evidence_prompt(
                    problem,
                    fact,
                    branches=1,
                    statements_per_branch=2,
                    tree_depth=2,
                    forbidden_phrases=forbidden_phrases(problem, fact),
                )
                cards.append(
                    {
                        "fact_id": fact.fact_id,
                        "canonical_exact_fact": fact.to_dict(),
                        "classification": roles[fact.fact_id],
                        "generated_card_text": list(card.statements),
                        "branch_id": card.branch_id,
                        "generation_prompt_hash": sha256_object(prompt_text),
                        "model_id": config.generation_provider.model,
                        "generation_seed": int(
                            Seed(config.seed).derive(
                                f"terra:{task_root.name}:{fact.fact_id}"
                            )
                        ),
                        "validation": {"structural": True},
                    }
                )
                provenance.append(tree.to_dict())
            leakage = validate_leakage(
                problem,
                {fact.fact_id: fact for fact in facts},
                generated.cards,
            )
            if leakage:
                raise RuntimeError(
                    f"{task_root.name} generated evidence failed leakage validation: "
                    + "; ".join(leakage)
                )
            write_json_atomic(cards_path, cards)
            write_json_atomic(provenance_path, provenance)
            write_json_atomic(
                task_root / "generation/prompt_hashes.json",
                {row["fact_id"]: row["generation_prompt_hash"] for row in cards},
            )
            generation_attempts = generated.attempts
            generation_failures = list(generated.failures)
        # Equality is rendered canonically for both resumed and fresh tasks.
        cards_by_id = {str(row["fact_id"]): dict(row) for row in cards}
        provenance_by_id = {
            str(row["latent_fact_id"]): dict(row)
            for row in json.loads(provenance_path.read_text(encoding="utf-8"))
        }
        for fact in facts:
            if fact.operator != "eq":
                continue
            statements = render_canonical_equality_evidence(problem, fact)
            cards_by_id[fact.fact_id] = {
                "fact_id": fact.fact_id,
                "canonical_exact_fact": fact.to_dict(),
                "classification": roles[fact.fact_id],
                "generated_card_text": list(statements),
                "branch_id": f"{fact.fact_id}_canonical_equality",
                "generation_prompt_hash": None,
                "model_id": None,
                "generation_seed": None,
                "rendering_method": "deterministic_canonical_equality_v1",
                "renderer_input_sha256": sha256_object(fact.to_dict()),
                "entailment": "direct_by_template",
                "validation": {"structural": True},
            }
            provenance_by_id[fact.fact_id] = build_reasoning_tree(
                latent_fact_id=fact.fact_id,
                branch_id=f"{fact.fact_id}_canonical_equality",
                hidden_claim=fact.canonical_text,
                intermediate_claims=(),
                statements=statements,
                commonsense_bridges=(),
            ).to_dict()
        cards = [cards_by_id[fact.fact_id] for fact in facts]
        write_json_atomic(cards_path, cards)
        write_json_atomic(
            provenance_path, [provenance_by_id[fact.fact_id] for fact in facts]
        )
        semantic = await audit_cards(
            model,
            task_root=task_root,
            problem=problem,
            facts=facts,
            cards=cards,
            seed=Seed(config.seed).derive(f"semantic-audit:{task_root.name}"),
        )
        semantic_repair_rounds = 0
        while (
            not semantic["all_passed"]
            and semantic_repair_rounds < config.semantic_retries - 1
        ):
            semantic_repair_rounds += 1
            validation_rows = json.loads(
                (task_root / "generation/semantic_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            failed_ids = {
                str(row["fact_id"])
                for row in validation_rows
                if row.get("passed") is not True
            }
            cards_by_id = {str(row["fact_id"]): dict(row) for row in cards}
            provenance_by_id = {
                str(row["latent_fact_id"]): dict(row)
                for row in json.loads(provenance_path.read_text(encoding="utf-8"))
            }
            for fact in facts:
                if fact.fact_id not in failed_ids:
                    continue
                if fact.operator == "eq":
                    continue
                regenerated = await generate_evidence_for_fact(
                    model,
                    problem,
                    fact,
                    branches=1,
                    statements_per_branch=2,
                    tree_depth=2,
                    seed=Seed(config.seed).derive(
                        f"semantic-regeneration:{task_root.name}:{semantic_repair_rounds}:{fact.fact_id}"
                    ),
                    max_attempts=config.semantic_retries,
                )
                card = regenerated.cards[0]
                prompt_text = evidence_prompt(
                    problem,
                    fact,
                    branches=1,
                    statements_per_branch=2,
                    tree_depth=2,
                    forbidden_phrases=forbidden_phrases(problem, fact),
                )
                cards_by_id[fact.fact_id] = {
                    "fact_id": fact.fact_id,
                    "canonical_exact_fact": fact.to_dict(),
                    "classification": roles[fact.fact_id],
                    "generated_card_text": list(card.statements),
                    "branch_id": card.branch_id,
                    "generation_prompt_hash": sha256_object(prompt_text),
                    "model_id": config.generation_provider.model,
                    "generation_seed": int(
                        Seed(config.seed).derive(
                            f"semantic-regeneration:{task_root.name}:{semantic_repair_rounds}:{fact.fact_id}"
                        )
                    ),
                    "semantic_regeneration_round": semantic_repair_rounds,
                    "validation": {"structural": True},
                }
                provenance_by_id[fact.fact_id] = regenerated.trees[0].to_dict()
                generation_attempts += regenerated.attempts
                generation_failures.extend(regenerated.failures)
            cards = [cards_by_id[fact.fact_id] for fact in facts]
            provenance = [provenance_by_id[fact.fact_id] for fact in facts]
            evidence_cards = tuple(
                EvidenceCard(
                    evidence_id=f"e_{fact.fact_id}_b00",
                    latent_fact_id=fact.fact_id,
                    branch_id=str(cards_by_id[fact.fact_id]["branch_id"]),
                    statements=tuple(cards_by_id[fact.fact_id]["generated_card_text"]),
                )
                for fact in facts
            )
            leakage = validate_leakage(
                problem, {fact.fact_id: fact for fact in facts}, evidence_cards
            )
            if leakage:
                raise RuntimeError(
                    f"{task_root.name} regenerated evidence failed leakage validation: "
                    + "; ".join(leakage)
                )
            write_json_atomic(cards_path, cards)
            write_json_atomic(provenance_path, provenance)
            semantic = await audit_cards(
                model,
                task_root=task_root,
                problem=problem,
                facts=facts,
                cards=cards,
                seed=Seed(config.seed).derive(
                    f"semantic-audit:{task_root.name}:repair:{semantic_repair_rounds}"
                ),
            )
        return {
            "task_id": task_root.name,
            "facts": len(facts),
            "attempts": generation_attempts,
            "validation_failures": generation_failures,
            "leakage_validation_errors": leakage,
            "semantic_validation": semantic,
            "semantic_repair_rounds": semantic_repair_rounds,
        }

    try:
        rows = await asyncio.gather(*(one(task_root) for task_root in task_roots))
    finally:
        provider.close()
    raw_attempts = 0
    raw_path = root / "generation/raw_calls.jsonl"
    if raw_path.is_file():
        raw_attempts = sum(
            1
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    audit_rows = (
        [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if raw_path.is_file()
        else []
    )
    evidence_calls = sum(
        row.get("purpose") == "evidence_generation" for row in audit_rows
    )
    semantic_calls = sum(
        row.get("purpose") == "evidence_semantic_validation" for row in audit_rows
    )
    all_valid = all(row["semantic_validation"]["all_passed"] for row in rows)
    manifest = {
        "schema_version": 1,
        "status": "complete" if all_valid else "validation_failed",
        "provider": config.generation_provider.type,
        "model": config.generation_provider.model,
        "prompt_version": PROMPT_VERSION,
        "logical_calls": evidence_calls + semantic_calls,
        "evidence_generation_logical_calls": evidence_calls,
        "semantic_validation_logical_calls": semantic_calls,
        "provider_attempts": raw_attempts,
        "retry_count": max(0, raw_attempts - sum(row["facts"] for row in rows)),
        "tasks": rows,
        "generation_validation_status": "PASS" if all_valid else "FAIL",
        "usage": {
            "input_tokens": sum(
                int((row.get("usage") or {}).get("input_tokens") or 0)
                for row in audit_rows
            ),
            "output_tokens": sum(
                int((row.get("usage") or {}).get("output_tokens") or 0)
                for row in audit_rows
            ),
        },
        "budget_status": guard.status(),
    }
    manifest["fingerprint_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "generation/terra_generation_manifest.json", manifest)
    for task_root in task_roots:
        write_json_atomic(
            task_root / "generation/terra_generation_manifest.json", manifest
        )
    return manifest


def read_summary(task_root: Path) -> Mapping[str, Any]:
    return json.loads(
        (task_root / "generation/semantic_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )


__all__ = ["generate", "input_token_estimate"]
