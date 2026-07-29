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
    run_experiment,
)
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
    } <= set(interactions)
    assert {"terminal_takeover", "ever_crossed", "incumbent_survives"} <= set(episodes)
    assert set(episodes["attack_direction"]) == {"strong_to_weak", "weak_to_strong"}
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
