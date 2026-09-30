"""Portable tests for exact agents; no frozen result directories or LLM calls."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.experiments import run_experiment_sync
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback.bayesian import BayesianAgentPolicy
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import RelationalRoundBudgetedControl
from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
    initialization_compatibility_key,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    RecoveryCheckpointError,
    _RecoveryLedger,
    _execute_decision,
    apply_epistemic_persistence,
    run_relational_imitation_round_feedback_game,
)
from mas_cc.core import Seed
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider
from mas_cc.planning import call_plan_for_run
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.latent_problem import problem_from_latent_values
from mas_cc.musr_team_allocation_generator.selective_design import (
    SelectiveThresholds,
    build_selective_design,
)
from mas_cc.probes.musr_truthful_selective.symbolic import write_design_artifacts


@pytest.fixture(scope="module")
def symbolic_task(tmp_path_factory):
    root = tmp_path_factory.mktemp("bayesian-task")
    design = build_selective_design(
        problem_from_latent_values((2, 2, 2, 2, 2, 2, 1, 3, 1)),
        TeamAllocationCompletionIndex(),
        SelectiveThresholds(subset_samples=12),
        seed=7,
        false_target_index=2,
    )
    write_design_artifacts(root, design, task_id="task_001", candidate_id=1, seed=7)
    return root, design


def _config(symbolic_task, *, mode="bayesian", rho=1.0, rounds=3, sampling="full"):
    root, _ = symbolic_task
    config = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/"
        "relational_imitation_round_feedback_no_control_smoke.yaml", environment={},
    )
    options = {
        **dict(config.game.options),
        "task_family": "musr_team_allocation", "task_dataset_dir": str(root),
        "task_id": "task_001", "agent_decision_mode": mode,
        "social_mode": "board", "social_group_size": 2,
        "prompt_version": 5, "rounds": rounds, "epistemic_persistence": rho,
        "board": {
            "sampling": sampling, "message_lifetime_rounds": 1,
            "allow_no_post": True, "communication_profile": "report_only",
        },
        "initialization": {"mode": "local_vote"},
    }
    return replace(
        config, game=replace(config.game, horizon=rounds, options=options),
        prompt=replace(config.prompt, prompt_family="relational_blackboard_ballot", prompt_version=5),
    )


class NoInference:
    async def complete(self, request):
        raise AssertionError("Bayesian agent contacted a provider")


def _run(config, **kwargs):
    return asyncio.run(run_relational_imitation_round_feedback_game(
        create_game(config.game), config, NoInference(), **kwargs,
    ))


def _ballot(game, config, state, policy, *, sources=(), seed=123):
    agent = state.agents[0]
    sources = tuple({
        "label": "Agent 2", "vote": "ALLOCATION_0", "text": "Observed fact.",
        "source_type": "peer", **source,
    } for source in sources)
    request = game.ballot_request(state, agent.agent_id, sources, config.game)
    context = game.citation_context(state, agent, sources, config.game)
    response, audit = policy.ballot(
        request, context, agent.memory,
        {fact_id: state.fact_text(fact_id) for fact_id in context.citable_fact_ids}, seed=seed,
    )
    request.prompt.compile().response_contract.validate(response).raise_for_errors()
    action = game.parse_action(request, response)
    game.validate_action(state, request, action, config.game).raise_for_errors()
    return action, audit


def test_provider_free_runtime_learns_and_records_exact_decisions(symbolic_task):
    config = _config(symbolic_task)
    result = _run(config)
    assert len(result.initial_decisions) == 24
    assert len(result.interactions) == 72
    assert result.validation_attempts == 0
    assert result.final_state.blackboard.messages
    assert sum(len(agent.known_fact_ids) for agent in result.final_state.agents) > 24
    assert result.runtime_state["initialization_context"]["initialization_source"] == "bayesian_local_vote"
    _, design = symbolic_task
    index = TeamAllocationCompletionIndex()
    facts = {fact.fact_id: fact for fact in design.facts}
    decisions = (*result.initial_decisions, *(row.decisions[0] for row in result.interactions))
    for decision in decisions:
        assert decision.attempts == ()
        audit = decision.action.metadata["bayesian"]
        expected = index.metrics_for_facts(tuple(facts[key] for key in audit["evidence_fact_ids"]))
        assert list(audit["posterior"].values()) == list(expected.probabilities)
        assert audit["posterior"][decision.action.value] == max(expected.probabilities)
    assert _run(config).to_dict() == result.to_dict()


def test_experiment_recorder_persists_bayesian_metadata_without_provider_audits(symbolic_task, tmp_path):
    config = _config(symbolic_task, rounds=1)
    run = run_experiment_sync(config, tmp_path, resume=False, show_progress=False)
    assert all(episode.status == "completed" for episode in run.outcomes)
    assert run.preflight.total_provider_requests.conservative == 0
    assert run.budget_status["used_and_reserved"]["requests"] == 0
    paths = list(run.output_dir.rglob("trajectory.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 24
    assert all(row["decisions"][0]["action"]["metadata"]["agent_decision_mode"] == "bayesian" for row in rows)
    assert all(row["decisions"][0]["validation_attempts"] == 0 for row in rows)
    assert all(not path.read_text().strip() for path in run.output_dir.rglob("audit_traces.jsonl"))


def test_grounded_sample_used_immediately_but_other_claims_ignored(symbolic_task):
    config = _config(symbolic_task, rho=0.75)
    game = create_game(config.game)
    state = _run(replace(config, game=replace(config.game, options={
        **dict(config.game.options), "initialization_only": True,
    }))).initial_state
    _, design = symbolic_task
    policy = BayesianAgentPolicy(design.facts)
    fact = next(key for key in state.fact_ids if key not in state.agents[0].known_fact_ids)
    source = {
        "message_id": "m000123", "message_type": "REPORT", "source_id": "agent-0002",
        "shared_fact_id": fact, "shared_fact_text": state.fact_text(fact),
        "round_created": 0, "expires_after_round": 0,
    }
    baseline, audit = _ballot(game, config, state, policy)
    action, with_report = _ballot(game, config, state, policy, sources=(source,))
    assert set(with_report["bayesian"]["evidence_fact_ids"]) == set(audit["bayesian"]["evidence_fact_ids"]) | {fact}
    transition = game.apply_round_event_transition(
        state, focal=state.agents[0].agent_id, action=action, config=config.game,
        social_sources=({"source_type": "peer", **source},), round_fields={"round_index": 0},
    )
    assert fact in transition.next_state.agents[0].active_fact_ids
    for changes in (
        {"message_type": "REQUEST"}, {"message_type": "DIRECTIVE"},
        {"shared_fact_text": "unsupported claim"}, {"expires_after_round": -1},
        {"source_id": str(state.agents[0].agent_id)}, {"vote": "ALLOCATION_2"},
    ):
        changed = {**source, **changes}
        _, observed = _ballot(game, config, state, policy, sources=(changed,))
        expected = with_report if set(changes) == {"vote"} else audit
        assert observed == expected
    # Hidden task labels have no influence on the policy's vote.
    altered = replace(state, data={**dict(state.data), "task": {
        **dict(state.task), "correct_relation": "ALLOCATION_2", "controller_target": "ALLOCATION_0",
    }})
    assert _ballot(game, config, altered, policy)[0].value == baseline.value


def test_forgotten_facts_excluded_until_reobserved_and_empty_agent_holds_tie(symbolic_task):
    config = _config(symbolic_task, rho=0.0)
    game = create_game(config.game)
    state = _run(replace(config, game=replace(config.game, options={
        **dict(config.game.options), "initialization_only": True,
    }))).initial_state
    faded, _ = apply_epistemic_persistence(state, persistence=0.0, rng=random.Random(1))
    policy = BayesianAgentPolicy(symbolic_task[1].facts)
    action, audit = _ballot(game, config, faded, policy)
    assert audit["bayesian"]["evidence_fact_ids"] == []
    assert len(set(audit["bayesian"]["posterior"].values())) == 1
    assert action.value == faded.agents[0].committed_action
    assert action.metadata["public_message"]["type"] == "NONE"
    fact = faded.agents[0].known_fact_ids[0]
    source = {
        "message_id": "m000456", "message_type": "REPORT", "source_id": "agent-0002",
        "shared_fact_id": fact, "shared_fact_text": faded.fact_text(fact),
        "round_created": 0, "expires_after_round": 0,
    }
    revived, audit = _ballot(game, config, faded, policy, sources=(source,))
    assert audit["bayesian"]["evidence_fact_ids"] == [fact]
    transition = game.apply_round_event_transition(
        faded, focal=faded.agents[0].agent_id, action=revived, config=config.game,
        social_sources=({"source_type": "peer", **source},), round_fields={"round_index": 0},
    )
    assert transition.next_state.agents[0].active_fact_ids == (fact,)
    assert transition.next_state.agents[0].known_fact_ids == faded.agents[0].known_fact_ids


def test_seeded_ties_and_posting_novelty_use_only_agent_view(symbolic_task):
    config = _config(symbolic_task, rho=0.75)
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    policy = BayesianAgentPolicy(symbolic_task[1].facts)
    empty = replace(state.agents[0], attributes={
        **dict(state.agents[0].attributes), "active_fact_ids": [],
    })
    empty_state = replace(state, agents=(empty, *state.agents[1:]))
    votes = {_ballot(game, config, empty_state, policy, seed=seed)[0].value for seed in range(30)}
    assert votes == {"ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2"}
    candidates = state.fact_ids[:2]
    agent = replace(state.agents[0], attributes={
        **dict(state.agents[0].attributes), "known_fact_ids": list(candidates),
        "active_fact_ids": list(candidates),
    }, memory=({"own_shared_fact_id": candidates[0]},))
    own_state = replace(state, agents=(agent, *state.agents[1:]))
    action, _ = _ballot(game, config, own_state, policy)
    assert action.metadata["shared_fact_id"] == candidates[1]
    source = {
        "message_id": "m000789", "message_type": "REPORT", "source_id": "agent-0002",
        "shared_fact_id": candidates[1], "shared_fact_text": own_state.fact_text(candidates[1]),
    }
    action, _ = _ballot(game, config, own_state, policy, sources=(source,))
    assert action.metadata["shared_fact_id"] == candidates[0]


def test_round_boundary_resume_matches_uninterrupted_run(symbolic_task):
    config = _config(symbolic_task, rho=0.75, sampling="uniform")
    whole = _run(config)
    parent = _run(config, continuation_length=1)
    child = _run(config, initial_state=parent.final_state, start_round=1,
                 restored_runtime_state=parent.runtime_state)
    assert child.final_state.to_dict() == whole.final_state.to_dict()
    assert [row.to_dict() for row in child.interactions] == [row.to_dict() for row in whole.interactions[24:]]
    llm = replace(config, game=replace(config.game, options={
        **dict(config.game.options), "agent_decision_mode": "llm",
    }))
    with pytest.raises(ValueError, match="agent decision mode"):
        _run(llm, initial_state=parent.final_state, start_round=1)


def test_mode_is_optional_isolates_artifacts_and_plans_no_agent_calls(symbolic_task):
    config = _config(symbolic_task)
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    assert all(stage.requests_per_interaction == 0 for stage in plan.decision_stages)
    assert sum(stage.provider_free_decisions_per_interaction for stage in plan.decision_stages) == 96
    llm = replace(config, game=replace(config.game, options={
        **dict(config.game.options), "agent_decision_mode": "llm",
    }))
    omitted = replace(llm, game=replace(llm.game, options={
        key: value for key, value in llm.game.options.items() if key != "agent_decision_mode"
    }))
    assert game.rules(omitted.game).to_dict() == game.rules(llm.game).to_dict()
    assert "agent_decision_mode" not in game.rules(omitted.game).to_dict()
    assert initialization_compatibility_key(game, config, 7) != initialization_compatibility_key(game, llm, 7)
    assert initialization_compatibility_key(game, omitted, 7) == initialization_compatibility_key(game, llm, 7)
    assert sum(stage.requests_per_interaction for stage in game.call_plan(llm.game).decision_stages) == 96


@pytest.mark.parametrize("override,match", [
    ({"agent_decision_mode": "unknown"}, "agent_decision_mode"),
    ({"task_family": "spatial_relational"}, "Bayesian agents require"),
    ({"social_mode": "peer"}, "Bayesian agents require"),
    ({"board": {"allow_no_post": False}}, "allow_no_post"),
])
def test_invalid_bayesian_configs_fail_early(symbolic_task, override, match):
    config = _config(symbolic_task)
    with pytest.raises(ValueError, match=match):
        create_game(config.game).rules(replace(config.game, options={**dict(config.game.options), **override}))


def test_missing_canonical_dictionary_is_not_a_silent_llm_fallback(tmp_path):
    with pytest.raises(ValueError, match="canonical predicates"):
        BayesianAgentPolicy.load(str(tmp_path), "task_001", ())


def test_observed_fact_changes_vote_even_when_active_only_prevents_relay(symbolic_task):
    config = _config(symbolic_task, rho=0.75)
    config = replace(config, game=replace(config.game, options={
        **dict(config.game.options), "board": {
            **dict(config.game.options["board"]), "report_citation_scope": "active_only",
        },
    }))
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    empty = replace(state.agents[0], attributes={
        **dict(state.agents[0].attributes), "active_fact_ids": [],
    })
    state = replace(state, agents=(empty, *state.agents[1:]))
    fact = state.fact_ids[0]
    action, audit = _ballot(game, config, state, BayesianAgentPolicy(symbolic_task[1].facts), sources=({
        "message_id": "m001234", "message_type": "REPORT", "source_id": "agent-0002",
        "shared_fact_id": fact, "shared_fact_text": state.fact_text(fact),
    },))
    assert audit["bayesian"]["evidence_fact_ids"] == [fact]
    assert action.metadata["public_message"]["type"] == "NONE"


def test_llm_controller_still_uses_provider_and_has_separate_call_estimate(symbolic_task):
    config = _config(symbolic_task, rounds=2)
    options = {
        "target": "ALLOCATION_2", "sensor_sample_size": 6, "intervention_budget": 3,
        "advocacy_schedule": "always", "controller_actuation_mode": "adaptive_communication",
        "controller_timing": "dawn_only", "controller_authoring": "llm_authored",
    }
    config = replace(config, control=replace(config.control, mechanism="relational_round_budgeted", options=options))
    game = create_game(config.game)
    task = game.load_task(config.game)
    calls = []

    def response(request):
        assert request.metadata["decision_stage"] == "controller_communication"
        facts = task.controller_reportable_fact_ids[len(calls) * 3:(len(calls) + 1) * 3]
        calls.append(request)
        return json.dumps({
            "mode": "REPORT", "fact_ids": list(facts), "text": None,
            "report_texts": [task.controller_report_texts[fact] for fact in facts], "reason": "evidence",
        })

    result = asyncio.run(run_relational_imitation_round_feedback_game(
        game, config, MockLLMProvider(config.llm_provider, response_factory=response),
        control=RelationalRoundBudgetedControl.from_options(options),
    ))
    assert len(calls) == 2
    assert all(decision.attempts == () for row in result.interactions for decision in row.decisions)
    plan = call_plan_for_run(game, config)
    assert sum(stage.requests_per_interaction for stage in plan.decision_stages) == 2
    ensemble = replace(config, ensemble=replace(config.ensemble, enabled=True))
    ensemble_plan = call_plan_for_run(game, ensemble)
    agents = next(stage for stage in ensemble_plan.decision_stages if stage.name == "relational_ballot_update")
    assert agents.requests_per_interaction == 0
    assert agents.provider_free_decisions_per_interaction == ensemble_plan.metadata["focal_updates_per_parent"]


def test_recovery_replays_bayesian_actions_and_rejects_backend_change(symbolic_task):
    config = _config(symbolic_task)
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    request = game.initial_vote_requests(state, config.game)[0]
    policy = BayesianAgentPolicy(symbolic_task[1].facts)

    class Observer:
        payload = None
        def load_failure_checkpoint(self):
            return self.payload
        def record_failure_checkpoint(self, *, runtime):
            self.payload = runtime

    observer = Observer()
    ledger = _RecoveryLedger(observer)
    original = asyncio.run(_execute_decision(
        game, request, state, config, NoInference(), RegexTokenCounter(), Seed(7),
        observer, ledger, policy,
    ))
    ledger.checkpoint({}, interruption_type="test")
    replay = _RecoveryLedger(observer)
    restored = replay.replay_decision(
        game=game, logical=request, state=state, config=config, prompt=original.compiled_prompt,
    )
    assert restored.to_dict() == original.to_dict()
    llm = replace(config, game=replace(config.game, options={**dict(config.game.options), "agent_decision_mode": "llm"}))
    with pytest.raises(RecoveryCheckpointError, match="agent decision mode"):
        replay.replay_decision(game=game, logical=request, state=state, config=llm, prompt=original.compiled_prompt)
