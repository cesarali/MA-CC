import asyncio
import json

import pandas as pd
import pytest

from naming_game.analysis.empowerment import AnalysisConfig, analyze_histories
from naming_game.api_client import MockAsyncLLMClient, OpenAIAsyncLLMClient
from naming_game.empowerment_experiment import (
    CommitteeSchedule,
    ConventionRolesConfig,
    EmpowermentExperimentConfig,
    EpisodeSpec,
    ReplicationConfig,
    build_episode_specs,
    derive_episode,
    convention_prompt_version,
    run_episode,
    run_experiment,
)
from naming_game.local_model_types import ChoiceScore, ConstrainedDecisionResponse
from naming_game.models import LLMResponse, TokenUsage
from naming_game.naming_convention_game import ConventionGameConfig, NamingConventionGame


def always_a(messages):
    return json.dumps({"value": "A", "reason": "ordinary"})


def test_openai_provider_has_explicit_identity_without_network_access():
    client = OpenAIAsyncLLMClient(model="gpt-4o-mini", api_key="test-only")
    try:
        assert client.provider_name == "openai"
        assert client.model == "gpt-4o-mini"
    finally:
        client.close()


def test_forced_actions_update_memory_then_release_to_ordinary_policy():
    schedule = CommitteeSchedule((1, 2), "alternative_pulse", "B", 1)
    game = NamingConventionGame(
        client=MockAsyncLLMClient(artificial_latency=0, response_factory=always_a),
        config=ConventionGameConfig(num_agents=2, actions=("A", "B"), memory_size=5),
        intervention=schedule,
    )
    result = asyncio.run(game.run(2, stop_on_convergence=False))
    assert result.interactions[0].player_1_decision.forced
    assert result.interactions[0].player_1_action == "B"
    assert not result.interactions[1].player_1_decision.forced
    assert result.interactions[1].player_1_action == "A"
    assert result.interactions[1].player_1_memory_before[-1].own_action == "B"


def test_replication_unit_supports_per_policy_and_split_strata():
    common = dict(
        population_size=2,
        max_population_rounds=1,
        committee_sizes=(0,),
        regimes=("neutral",),
    )
    per_policy = build_episode_specs(
        EmpowermentExperimentConfig(
            **common, replications=ReplicationConfig("per_policy", 2)
        )
    )
    per_stratum = build_episode_specs(
        EmpowermentExperimentConfig(
            **common, replications=ReplicationConfig("per_stratum", 5)
        )
    )
    assert len(per_policy) == 6
    assert len(per_stratum) == 5
    assert {spec.committee_policy for spec in per_stratum} == {
        "always_A", "always_B", "no_committee"
    }


def test_convention_roles_are_validated_against_configured_names():
    config = EmpowermentExperimentConfig(
        convention_roles=ConventionRolesConfig("A", "B", "calibration")
    )
    assert config.convention_roles is not None
    assert config.convention_roles.strong_name == "A"
    with pytest.raises(Exception, match="must match names"):
        EmpowermentExperimentConfig(
            convention_roles=ConventionRolesConfig("A", "C", "calibration")
        )


def test_rolling_window_events_and_censored_fields_are_derived():
    config = EmpowermentExperimentConfig(
        population_size=2,
        max_population_rounds=2,
        committee_sizes=(1,),
        pulse_rounds=(1,),
        regimes=("pulse",),
        replications=ReplicationConfig("per_policy", 1),
        window_interactions=2,
    )
    spec = EpisodeSpec(
        "episode", 1, "pulse", 1, "alternative_pulse", "consensus_A", "A", "B", 1, 0
    )
    outputs = [("B", "B"), ("B", "B"), ("A", "A"), ("A", "A")]
    rows = []
    for index, (left, right) in enumerate(outputs, 1):
        rows.append(
            {
                "interaction_index": index,
                "output_i": left,
                "output_j": right,
                "forced_i": index <= 2,
                "forced_j": False,
                "provider": "mock",
                "model": "mock/model",
                "prompt_hash": "hash",
                "committee_ids": "[1]",
                "population_round": (index + 1) // 2,
            }
        )
    trajectory, summary = derive_episode(rows, spec, config)
    assert trajectory[0]["insufficient_window"] is True
    assert trajectory[1]["resolved_state"] == "B"
    assert summary["takeover"] is True
    assert summary["recovery_time_interactions"] == 2
    assert summary["recovery_censored"] is False
    assert summary["final_convention"] == "A"


def test_neutral_episode_without_promoted_name_is_never_a_takeover():
    config = EmpowermentExperimentConfig(
        population_size=2,
        max_population_rounds=1,
        committee_sizes=(0,),
        regimes=("neutral",),
        window_interactions=2,
    )
    spec = EpisodeSpec(
        "episode", 1, "neutral", 0, "no_committee", "empty", None, None, None, 0
    )
    rows = [
        {
            "interaction_index": index,
            "output_i": "A",
            "output_j": "A",
            "forced_i": False,
            "forced_j": False,
            "provider": "mock",
            "model": "mock/model",
            "prompt_hash": "hash",
            "committee_ids": "[]",
            "population_round": (index + 1) // 2,
        }
        for index in range(1, 3)
    ]
    _, summary = derive_episode(rows, spec, config)
    assert summary["takeover"] is False
    assert summary["ever_crossed"] is False
    assert summary["terminal_takeover"] is False


def test_recovery_is_immediate_or_censored_after_pulse_removal():
    config = EmpowermentExperimentConfig(
        population_size=2, max_population_rounds=2, committee_sizes=(1,),
        pulse_rounds=(1,), regimes=("pulse",), window_interactions=2,
    )
    spec = EpisodeSpec(
        "episode", 1, "pulse", 1, "alternative_pulse", "consensus_A", "A", "B", 1, 0
    )

    def make_rows(action):
        return [
            {
                "interaction_index": index, "output_i": action, "output_j": action,
                "forced_i": False, "forced_j": False, "provider": "mock",
                "model": "mock/model", "prompt_hash": "hash", "committee_ids": "[1]",
                "population_round": (index + 1) // 2,
            }
            for index in range(1, 5)
        ]

    _, immediate = derive_episode(make_rows("A"), spec, config)
    _, censored = derive_episode(make_rows("B"), spec, config)
    assert immediate["recovery_time_interactions"] == 0
    assert immediate["recovery_censored"] is False
    assert censored["recovery_time_interactions"] is None
    assert censored["recovery_censored"] is True


def test_mock_parquet_experiment_and_offline_analysis(tmp_path):
    history = tmp_path / "history"
    config = EmpowermentExperimentConfig(
        population_size=2,
        memory_length=1,
        max_population_rounds=2,
        committee_sizes=(0, 1),
        pulse_rounds=(1,),
        regimes=("pulse",),
        replications=ReplicationConfig("per_policy", 1),
        window_interactions=2,
        episode_concurrency=2,
        model="mock/model",
        convention_roles=ConventionRolesConfig("A", "B", "calibration"),
    )
    result = asyncio.run(
        run_experiment(
            config,
            MockAsyncLLMClient(model="mock/model", artificial_latency=0),
            history,
        )
    )
    assert result["episodes"] == 8
    interactions = pd.read_parquet(history / "interactions.parquet")
    episodes = pd.read_parquet(history / "episodes.parquet")
    persisted_config = json.loads((history / "experiment_config.json").read_text())
    assert len(interactions) == 32
    assert len(episodes) == 8
    assert {
        "memory_i_before",
        "rolling_share_A",
        "macrostate_binary",
        "strong_name",
        "weak_name",
        "incumbent_name",
        "promoted_name",
        "attack_direction",
        "decision_output_format_i",
        "decision_method_i",
        "allowed_choices_i",
    } <= set(interactions)
    assert {"terminal_takeover", "ever_crossed", "incumbent_survives"} <= set(episodes)
    assert set(episodes["attack_direction"]) == {"strong_to_weak", "weak_to_strong"}
    assert persisted_config["decision_output_format"] == "json_reason"
    assert persisted_config["choice_selection_policy"] == "argmax"
    assert persisted_config["choice_temperature"] == 1.0
    analysis = analyze_histories(
        history,
        tmp_path / "analysis",
        AnalysisConfig((1,), bootstrap_resamples=2, null_permutations=1),
    )
    assert analysis["estimates"] > 0
    assert (tmp_path / "analysis" / "empowerment_estimates.parquet").exists()
    assert (tmp_path / "analysis" / "summary.md").exists()
    assert (tmp_path / "analysis" / "plots" / "experiment_summary.png").stat().st_size > 0
    assert (tmp_path / "analysis" / "plots" / "pulse_summary.png").stat().st_size > 0


class EmpowermentCombinedClient:
    model = "fake/combined"
    provider_name = "capable"
    concurrency = 2

    def __init__(self):
        self.calls = []

    async def complete_decision(self, messages, *, choices, **kwargs):
        self.calls.append((messages, tuple(choices), kwargs))
        scores = (
            ChoiceScore(choices[0], (1,), -0.25, 0.75),
            ChoiceScore(choices[1], (2,), -1.25, 0.25),
        )
        output_format = kwargs["output_format"]
        reason = "recent coordination" if output_format == "choice_reason" else None
        content = choices[0] if reason is None else f"{choices[0]}\nReason: {reason}"
        return ConstrainedDecisionResponse(
            choices[0],
            scores,
            content,
            reason,
            True if reason else None,
            output_format,
            self.model,
            0.0,
            TokenUsage(4, 2, 6),
            kwargs["choice_temperature"],
            kwargs["selection_policy"],
        )

    async def complete(self, messages, **kwargs):
        raise AssertionError("combined path expected")

    def close(self):
        pass


def test_combined_rows_and_summary_preserve_scientific_decision_metadata():
    config = EmpowermentExperimentConfig(
        population_size=2,
        memory_length=1,
        max_population_rounds=1,
        committee_sizes=(0,),
        regimes=("neutral",),
        replications=ReplicationConfig("per_stratum", 1),
        decision_output_format="choice_reason",
        choice_selection_policy="sample",
        choice_temperature=0.5,
    )
    spec = EpisodeSpec(
        "episode", 2, "neutral", 0, "no_committee", "empty", None, None, None, 0
    )
    client = EmpowermentCombinedClient()
    rows, summary = asyncio.run(run_episode(spec, config, client))
    assert len(client.calls) == 4
    assert summary["prompt_version"] == convention_prompt_version("choice_reason")
    assert summary["decision_output_format"] == "choice_reason"
    assert summary["choice_selection_policy"] == "sample"
    assert summary["choice_temperature"] == 0.5
    for row in rows:
        assert row["prompt_version"] == convention_prompt_version("choice_reason")
        assert row["decision_output_format"] == "choice_reason"
        for suffix in ("i", "j"):
            assert row[f"decision_output_format_{suffix}"] == "choice_reason"
            assert row[f"decision_method_{suffix}"] == "constrained_decision"
            assert row[f"reason_{suffix}"] == "recent coordination"
            assert row[f"reason_valid_{suffix}"]
            allowed = json.loads(row[f"allowed_choices_{suffix}"])
            probabilities = json.loads(row[f"choice_probabilities_{suffix}"])
            log_likelihoods = json.loads(row[f"choice_log_likelihoods_{suffix}"])
            assert list(probabilities) == allowed
            assert list(log_likelihoods) == allowed
            assert sum(probabilities.values()) == 1.0
            assert row[f"selected_choice_probability_{suffix}"] == 0.75
            assert row[f"choice_entropy_{suffix}"] > 0


def test_generated_remote_rows_keep_displayed_choices_without_probabilities():
    config = EmpowermentExperimentConfig(
        population_size=2,
        max_population_rounds=1,
        committee_sizes=(0,),
        regimes=("neutral",),
        replications=ReplicationConfig("per_stratum", 1),
        decision_output_format="choice_only",
    )
    spec = EpisodeSpec(
        "episode", 2, "neutral", 0, "no_committee", "empty", None, None, None, 0
    )
    client = MockAsyncLLMClient(
        artificial_latency=0, response_factory=lambda messages: "A"
    )
    rows, _ = asyncio.run(run_episode(spec, config, client))
    for row in rows:
        for suffix in ("i", "j"):
            assert set(json.loads(row[f"allowed_choices_{suffix}"])) == {"A", "B"}
            assert row[f"choice_probabilities_{suffix}"] is None
            assert row[f"choice_log_likelihoods_{suffix}"] is None


def test_forced_rows_have_null_rationale_and_probability_fields():
    config = EmpowermentExperimentConfig(
        population_size=2,
        max_population_rounds=1,
        committee_sizes=(2,),
        regimes=("neutral",),
        replications=ReplicationConfig("per_stratum", 1),
        decision_output_format="choice_reason",
    )
    spec = EpisodeSpec(
        "episode", 2, "neutral", 2, "always_A", "empty", None, None, None, 0
    )
    client = EmpowermentCombinedClient()
    rows, _ = asyncio.run(run_episode(spec, config, client))
    assert client.calls == []
    for row in rows:
        for suffix in ("i", "j"):
            assert row[f"decision_method_{suffix}"] == "forced"
            assert row[f"reason_{suffix}"] is None
            assert row[f"reason_valid_{suffix}"] is None
            assert row[f"allowed_choices_{suffix}"] is None
            assert row[f"choice_probabilities_{suffix}"] is None
