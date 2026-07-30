import json
import math

import pytest
import yaml

from naming_game.empowerment_experiment import (
    EmpowermentExperimentConfig,
    _experiment_fingerprint,
    _prompt_hash,
    build_episode_specs,
    convention_prompt_version,
    load_experiment_config,
)
from naming_game.models import ConfigurationError
from naming_game.naming_convention_game import (
    ConventionAgent,
    ConventionGameConfig,
    build_convention_context,
    build_convention_messages,
    build_convention_response_instruction,
    parse_convention_decision,
)


FORMATS = ("json_reason", "choice_reason", "choice_only")
POLICIES = ("argmax", "sample")


def _config_file(tmp_path, **overrides):
    values = {
        "population_size": 2,
        "names": ["A", "B"],
        "memory_length": 1,
        "max_population_rounds": 1,
        "committee_sizes": [0],
        "pulse_rounds": [1],
        "regimes": ["neutral"],
        "replications": {"unit": "per_stratum", "count": 1},
        "decision_output_format": "json_reason",
        "choice_selection_policy": "argmax",
        "choice_temperature": 1.0,
    }
    values.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_json_reason_is_the_public_default():
    assert ConventionGameConfig().decision_output_format == "json_reason"
    assert EmpowermentExperimentConfig().decision_output_format == "json_reason"


@pytest.mark.parametrize("output_format", FORMATS)
@pytest.mark.parametrize("policy", POLICIES)
def test_every_output_format_and_policy_loads(tmp_path, output_format, policy):
    config = load_experiment_config(
        _config_file(
            tmp_path,
            decision_output_format=output_format,
            choice_selection_policy=policy,
        )
    )
    assert config.decision_output_format == output_format
    assert config.choice_selection_policy == policy


@pytest.mark.parametrize(
    ("field", "value"),
    (("decision_output_format", "xml"), ("choice_selection_policy", "last")),
)
def test_unknown_format_and_policy_fail_while_loading(tmp_path, field, value):
    with pytest.raises(ConfigurationError):
        load_experiment_config(_config_file(tmp_path, **{field: value}))


@pytest.mark.parametrize("value", (0.0, -1.0, math.inf, -math.inf, math.nan))
def test_choice_temperature_must_be_finite_and_positive(tmp_path, value):
    with pytest.raises(ConfigurationError, match="choice_temperature"):
        load_experiment_config(_config_file(tmp_path, choice_temperature=value))
    with pytest.raises(ConfigurationError, match="choice_temperature"):
        ConventionGameConfig(choice_temperature=value)


def test_prompts_share_context_and_only_the_response_contract_changes():
    agent = ConventionAgent(agent_id=3)
    agent.remember(
        interaction_index=4,
        own_action="LONG",
        partner_action="B",
        payoff=-50,
        partner_id=9,
    )
    order = ("LONG", "B")
    context = build_convention_context(
        agent=agent,
        action_order=order,
        memory_size=1,
        success_reward=100,
        failure_payoff=-50,
    )
    messages = {
        output_format: build_convention_messages(
            agent=agent,
            action_order=order,
            memory_size=1,
            success_reward=100,
            failure_payoff=-50,
            output_format=output_format,
        )
        for output_format in FORMATS
    }
    for output_format, value in messages.items():
        assert value[0]["content"] == context + "\n" + build_convention_response_instruction(
            output_format, order
        )
        assert value[1] == {
            "role": "user",
            "content": "Answer saying which action Player 1 should play.",
        }
        assert '["LONG", "B"]' in value[0]["content"]
    assert '{"value":"<VALUE_OF_PLAYER_1>"' in messages["json_reason"][0]["content"]
    assert "first line" in messages["choice_reason"][0]["content"]
    assert "Reason:" in messages["choice_reason"][0]["content"]
    assert "Return only the action" in messages["choice_only"][0]["content"]


def test_json_reason_keeps_legacy_parser_behavior():
    assert parse_convention_decision(
        'preface {"value":"A","reason":"because"} suffix',
        ("A", "B"),
        "json_reason",
    ) == ("A", "because")
    assert parse_convention_decision(
        "```json\n{\"value\":\"B\",\"reason\":\"legacy\"}\n```",
        ("A", "B"),
        "json_reason",
    ) == ("B", "legacy")


def test_choice_reason_accepts_only_action_first_with_nonblank_reason():
    actions = ("A", "AA")
    assert parse_convention_decision(
        "\n AA \nReason: coordinated recently\n", actions, "choice_reason"
    ) == ("AA", "coordinated recently")
    invalid = (
        "I choose A\nReason: x",
        "Reason: A",
        "A",
        "A\nReason:",
        "A\nReason:   ",
        '{"value":"A","reason":"x"}',
        "```\nA\nReason: x\n```",
        "Action: A\nReason: x",
        "A is best\nReason: x",
        "A\n\nReason: x",
    )
    for content in invalid:
        with pytest.raises(ValueError):
            parse_convention_decision(content, actions, "choice_reason")


def test_choice_only_accepts_only_a_stripped_exact_action():
    actions = ("A", "AA")
    assert parse_convention_decision(" \nAA\n ", actions, "choice_only") == ("AA", None)
    for content in (
        "A because",
        "Action: A",
        '"A"',
        '{"value":"A"}',
        "```A```",
        "AA.",
        "Reason: AA",
    ):
        with pytest.raises(ValueError):
            parse_convention_decision(content, actions, "choice_only")


def test_prompt_versions_hashes_and_fingerprints_distinguish_scientific_inputs():
    versions = {convention_prompt_version(output_format) for output_format in FORMATS}
    hashes = {
        _prompt_hash(EmpowermentExperimentConfig(decision_output_format=output_format))
        for output_format in FORMATS
    }
    fingerprints = {
        _experiment_fingerprint(
            EmpowermentExperimentConfig(
                decision_output_format=output_format,
                choice_selection_policy=policy,
            )
        )
        for output_format in FORMATS
        for policy in POLICIES
    }
    assert len(versions) == 3
    assert len(hashes) == 3
    assert len(fingerprints) == 6
    assert all(output_format in convention_prompt_version(output_format) for output_format in FORMATS)
    episode_ids = {
        build_episode_specs(
            EmpowermentExperimentConfig(
                population_size=2,
                max_population_rounds=1,
                committee_sizes=(0,),
                regimes=("neutral",),
                decision_output_format=output_format,
                choice_selection_policy=policy,
            )
        )[0].episode_id
        for output_format in FORMATS
        for policy in POLICIES
    }
    assert len(episode_ids) == 6


def test_displayed_action_order_has_a_deterministic_serialization():
    order = ("AA", "A", "B")
    instruction = build_convention_response_instruction("choice_only", order)
    assert json.dumps(list(order), ensure_ascii=False) in instruction
