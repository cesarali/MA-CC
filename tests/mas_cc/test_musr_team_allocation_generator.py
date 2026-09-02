from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

import pytest

from mas_cc.llm_runtime.providers import CompletionResponse, ProviderCapabilities
from mas_cc.musr_team_allocation_generator.cli import validate_dataset_dir
from mas_cc.musr_team_allocation_generator.distribute import distribute_evidence
from mas_cc.musr_team_allocation_generator.evidence_generation import (
    _validate_branch_payload,
)
from mas_cc.musr_team_allocation_generator.generate import (
    GenerationConfig,
    generate_dataset,
    generate_world,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    generate_latent_problem,
    latent_facts,
)
from mas_cc.musr_team_allocation_generator.provider_adapter import MuSRGenerationModel
from mas_cc.musr_team_allocation_generator.reasoning_tree import (
    ReasoningTree,
    build_reasoning_tree,
)
from mas_cc.musr_team_allocation_generator.schemas import EvidenceCard, MUSR_COMMIT
from mas_cc.musr_team_allocation_generator.validate import (
    agent_can_certify_unique_allocation,
    validate_distribution,
    validate_exact_problem,
    validate_frozen_task,
)


class ScriptedProvider:
    name = "scripted"
    model = "unit-test-model"
    capabilities = ProviderCapabilities(supports_seed=True)

    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    async def complete(self, request):
        self.requests.append(request)
        prompt = request.messages[-1].content
        if "Solve this Team Allocation problem" in prompt:
            match = re.search(r"GOLD_FOR_TEST=(\d+)", prompt)
            index = int(match.group(1)) if match else 0
            content = json.dumps(
                {"option_index": index, "rationale": "All evidence supports it."}
            )
        elif "Hidden target (never copy into explicit statements)" not in prompt:
            content = "ok"
        else:
            target = json.loads(
                re.search(
                    r"Hidden target \(never copy into explicit statements\): (\{.*\})",
                    prompt,
                ).group(1)
            )
            branches = int(re.search(r"exactly (\d+) branches", prompt).group(1))
            statements = int(
                re.search(r"exactly (\d+) non-empty statements", prompt).group(1)
            )
            intermediates = int(
                re.search(r"exactly (\d+) intermediate_claims", prompt).group(1)
            )
            person = target["people"][0]
            payload = []
            for branch in range(branches):
                payload.append(
                    {
                        "intermediate_claims": [
                            f"Past event set {branch} provides relevant indirect evidence."
                            for _ in range(intermediates)
                        ],
                        "statements": [
                            f"For {target['hidden_fact_id']} in archived project {branch}-{item}, {person} produced a documented outcome under pressure."
                            for item in range(statements)
                        ],
                        "commonsense_bridges": [
                            "Repeated observable outcomes can indicate the relevant hidden capability."
                        ],
                    }
                )
            content = json.dumps({"branches": payload})
        return CompletionResponse(
            content=content,
            provider=self.name,
            model=self.model,
            request_id=f"request-{len(self.requests)}",
        )

    def close(self):
        self.closed = True


def test_latent_generator_is_exact_unique_and_reproducible():
    first = generate_latent_problem(random.Random(17), min_margin=1)
    second = generate_latent_problem(random.Random(17), min_margin=1)
    assert first == second
    assert not validate_exact_problem(first)
    assert len(set(first.candidate_scores)) >= 2
    assert first.candidate_scores.count(max(first.candidate_scores)) == 1
    assert first.margin_to_second_best >= 1
    assert len(latent_facts(first)) == 9


def test_provider_adapter_passes_seed_prompt_and_metadata():
    provider = ScriptedProvider()
    model = MuSRGenerationModel(provider, temperature=0.25, max_output_tokens=512)
    response = asyncio.run(model.inference("hello", seed=42, purpose="adapter_test"))
    request = provider.requests[0]
    assert response.provider == "scripted"
    assert request.seed == 42
    assert request.temperature == 0.25
    assert request.max_output_tokens == 512
    assert request.metadata["purpose"] == "adapter_test"
    assert request.messages[0].content == "hello"


def test_reasoning_tree_round_trip_preserves_explicit_leaves():
    tree = build_reasoning_tree(
        latent_fact_id="skill_p0_t0",
        branch_id="skill_p0_t0_b00",
        hidden_claim="Alice has strong skill.",
        intermediate_claims=("Alice repeatedly succeeded in comparable work.",),
        statements=(
            "Alice repaired three failed systems.",
            "Each repair passed review.",
        ),
        commonsense_bridges=("Repeated verified success indicates ability.",),
    )
    restored = ReasoningTree.from_dict(tree.to_dict())
    assert restored == tree
    assert [node.text for node in restored.root.explicit_leaves()] == [
        "Alice repaired three failed systems.",
        "Each repair passed review.",
    ]


def test_forbidden_answer_leakage_is_rejected():
    with pytest.raises(ValueError, match="forbidden answer leakage"):
        _validate_branch_payload(
            {
                "branches": [
                    {
                        "intermediate_claims": ["This supports a conclusion."],
                        "statements": ["Alice should be assigned to programming."],
                        "commonsense_bridges": ["Past success predicts performance."],
                    }
                ]
            },
            branches=1,
            statements_per_branch=1,
            tree_depth=2,
            forbidden=("should be assigned",),
        )


def _cards_for(problem, branches=3):
    return tuple(
        EvidenceCard(
            evidence_id=f"e_{fact.fact_id}_b{branch:02d}",
            latent_fact_id=fact.fact_id,
            branch_id=f"{fact.fact_id}_b{branch:02d}",
            statements=(f"Observation {fact.fact_id} {branch}.",),
        )
        for fact in latent_facts(problem)
        for branch in range(branches)
    )


def test_distribution_covers_exactly_n_agents_and_all_ids():
    problem = generate_latent_problem(random.Random(21))
    cards = _cards_for(problem)
    assignments = distribute_evidence(
        cards,
        problem,
        population_size=24,
        redundancy=3,
        rng=random.Random(4),
    )
    assert len(assignments) == 24
    assert set().union(*(set(ids) for ids in assignments.values())) == {
        card.evidence_id for card in cards
    }
    assert not validate_distribution(
        cards, assignments, population_size=24, problem=problem
    )


def test_no_single_agent_structural_validator_detects_complete_information():
    problem = generate_latent_problem(random.Random(33), min_margin=3)
    all_facts = {fact.fact_id for fact in latent_facts(problem)}
    certified, winner = agent_can_certify_unique_allocation(problem, all_facts)
    assert certified
    assert winner == problem.gold_index


@pytest.mark.parametrize("seed", range(20))
def test_many_latent_worlds_have_unique_optimum(seed):
    problem = generate_latent_problem(random.Random(seed), min_margin=2)
    assert not validate_exact_problem(problem)
    assert problem.margin_to_second_best >= 2


def test_mocked_end_to_end_generation_without_real_provider_calls(tmp_path: Path):
    provider = ScriptedProvider()
    model = MuSRGenerationModel(provider)
    config = GenerationConfig(
        num_tasks=1,
        population_size=24,
        branches_per_latent_fact=3,
        statements_per_branch=2,
        evidence_redundancy=3,
        seed=8,
        run_full_information_validation=False,
    )
    manifest = asyncio.run(generate_dataset(model, config, output=tmp_path))
    task_path = tmp_path / "musr_team_allocation_0001.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))

    assert manifest["num_tasks"] == 1
    assert task["task_family"] == "musr_team_allocation"
    assert task["generation"]["musr_commit"] == MUSR_COMMIT
    assert len(task["evidence"]) == 27
    assert len(task["agent_evidence_ids"]) == 24
    assert not validate_frozen_task(task)
    validate_dataset_dir(tmp_path)
    assert len(provider.requests) == 9


def test_full_information_majority_is_required_and_recorded(monkeypatch):
    provider = ScriptedProvider()
    model = MuSRGenerationModel(provider)
    config = GenerationConfig(
        num_tasks=1,
        population_size=6,
        branches_per_latent_fact=2,
        statements_per_branch=2,
        evidence_redundancy=2,
        seed=12,
        full_validation_attempts=3,
        full_validation_required=2,
    )

    async def always_accept(_model, _task, *, attempts, required_successes, seed):
        from mas_cc.musr_team_allocation_generator.validate import FullInformationResult

        assert _model is model
        assert _task["task_family"] == "musr_team_allocation"
        assert attempts == 3
        assert required_successes == 2
        assert seed.value >= 0
        return FullInformationResult(
            accepted=True,
            successes=2,
            attempts=3,
            required_successes=2,
            records=({"correct": True}, {"correct": True}, {"correct": False}),
        )

    monkeypatch.setattr(
        "mas_cc.musr_team_allocation_generator.generate.validate_full_information",
        always_accept,
    )
    task = asyncio.run(generate_world(model, config, task_index=1)).to_dict()
    assert task["validation"]["full_information"]["successes"] == 2
    assert task["validation"]["full_information"]["accepted"] is True
