from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import PromptConfig, load_component_config
from mas_cc.prompts import (
    PromptMarkdownLogger,
    PromptRegistry,
    PromptVersion,
    RegexTokenCounter,
    ResponseContract,
    create_default_prompt_registry as create_v3_prompt_registry,
)
from mas_cc.prompts.compatibility import PromptComposer, PromptContext, PromptDefinition
from mas_cc.prompts.examples import (
    hiddenbench_example_context,
    social_conventions_example_context,
)


def create_default_prompt_registry():
    return create_v3_prompt_registry(include_legacy=True)


def _context() -> PromptContext:
    return PromptContext(
        task_description="Choose an action that coordinates with the other player.",
        game_rules=("Choose A or B.", "Matching actions earn a positive payoff."),
        private_state={"available_actions": ["A", "B"], "score": 0},
        recent_memory=({"own_action": "A", "other_action": "B", "payoff": -50},),
        current_interaction={"number": 2, "other_action_visible": False},
        decision_instruction="Choose your action now.",
    )


def _config(**updates) -> PromptConfig:
    values = {
        "prompt_family": "basic_binary_choice",
        "prompt_version": 1,
        "blocks": (
            "task_description",
            "game_rules",
            "private_state",
            "recent_memory",
            "current_interaction",
            "decision_instruction",
            "output_contract",
        ),
        "response_contract": {"type": "choice_only", "allowed_values": ["A", "B"]},
        "schema_version": 1,
    }
    values.update(updates)
    return PromptConfig(**values)


def test_prompt_context_is_deeply_immutable_and_serializable():
    context = _context()
    with pytest.raises(TypeError):
        context.private_state["score"] = 10
    with pytest.raises(TypeError):
        context.private_state["available_actions"][0] = "B"
    with pytest.raises(TypeError):
        context.recent_memory[0]["payoff"] = 100
    assert context.to_dict()["private_state"]["available_actions"] == ["A", "B"]


def test_response_contract_renders_and_validates_choice_only():
    contract = ResponseContract.from_mapping(
        {"type": "choice_only", "allowed_values": ["A", "B"]}
    )
    assert "A, B" in contract.instruction()
    assert contract.validate("A").is_valid
    result = contract.validate("I choose A")
    assert not result.is_valid
    assert result.issues[0].field == "response"
    with pytest.raises(ValueError, match="must be unique"):
        ResponseContract("choice_only", ("A", "A"))


def test_registry_has_explicit_family_version_and_rejects_duplicates():
    registry = create_default_prompt_registry()
    assert registry.versions() == (
        PromptVersion("basic_choice", 1),
        PromptVersion("hidden_profile_discussion", 2),
        PromptVersion("hidden_profile_vote", 2),
    )
    definition = registry.get("basic_choice", 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)
    with pytest.raises(ValueError, match=r"prompt\.version"):
        registry.get("basic_choice", 2)


def test_yaml_order_is_the_message_and_block_order():
    config = _config(
        blocks=(
            "task_description",
            "private_state",
            "game_rules",
            "decision_instruction",
            "output_contract",
        )
    )
    instance = PromptComposer(create_default_prompt_registry()).compose(config, _context())
    assert tuple(block.name for block in instance.blocks) == config.blocks
    assert tuple(message.metadata["block"] for message in instance.messages) == config.blocks
    assert [message.role.value for message in instance.messages] == [
        "system",
        "user",
        "system",
        "user",
        "user",
    ]


def test_changing_private_state_changes_only_its_block():
    composer = PromptComposer(create_default_prompt_registry(), RegexTokenCounter())
    first = composer.compose(_config(), _context())
    changed_context = replace(
        _context(), private_state={"available_actions": ["A", "B"], "score": 100}
    )
    second = composer.compose(_config(), changed_context)
    changed = [
        before.name
        for before, after in zip(first.blocks, second.blocks, strict=True)
        if before.content != after.content
    ]
    assert changed == ["private_state"]


def test_compilation_records_per_block_tokens_and_normalized_messages():
    instance = PromptComposer(
        create_default_prompt_registry(), RegexTokenCounter()
    ).compose(_config(), _context())
    assert all(block.token_count and block.token_count > 0 for block in instance.blocks)
    assert instance.total_tokens == sum(block.token_count for block in instance.blocks)
    assert [item["content"] for item in instance.messages_as_dicts()] == [
        block.content for block in instance.blocks
    ]
    rendered = instance.rendered_text()
    assert "basic_binary_choice@1" in rendered
    assert "## 7. Output contract" in rendered


def test_compilation_without_tokenizer_marks_counts_unavailable():
    instance = PromptComposer(create_default_prompt_registry()).compose(_config(), _context())
    assert instance.total_tokens is None
    assert all(block.token_count is None for block in instance.blocks)


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        (("task_description", "task_description", "decision_instruction", "output_contract"), "duplicate"),
        (("task_description", "decision_instruction", "unknown", "output_contract"), "prompt.blocks[2]"),
        (("task_description", "output_contract"), "required block 'decision_instruction'"),
    ],
)
def test_composer_reports_invalid_block_configuration(blocks, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        PromptComposer(create_default_prompt_registry()).compose(
            _config(blocks=blocks), _context()
        )


def test_repository_prompt_component_compiles_without_provider_data():
    config = load_component_config(
        "configs/components/prompts/basic_binary_choice.yaml", "prompt", environment={}
    )
    instance = PromptComposer(create_default_prompt_registry()).compose(config, _context())
    serialized = str(instance.messages_as_dicts()).lower()
    assert "provider" not in serialized
    assert "committee" not in serialized
    assert "population" not in serialized


def test_consecutive_roles_merge_into_exact_paper_request_shape():
    config = load_component_config(
        "configs/components/prompts/social_conventions_paper.yaml",
        "prompt",
        environment={},
    )
    instance = PromptComposer(create_default_prompt_registry()).compose(
        config, social_conventions_example_context()
    )
    assert [message.role.value for message in instance.messages] == ["system", "user"]
    assert "simultaneously pick an action" in instance.messages[0].content
    assert "{'value': <F or J>" in instance.messages[0].content
    assert instance.messages[1].content == "Answer saying which action Player 1 should play."
    assert instance.messages[0].metadata["blocks"] == (
        "partnership_context",
        "payoff_rules",
        "bounded_memory",
        "round_state",
        "output_contract",
    )


def test_hiddenbench_fixture_uses_one_agents_packet_without_audit_answer():
    path = Path(
        "scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/"
        "scaled/exact_replication/N_32.json"
    )
    context = hiddenbench_example_context(path, task_id=1, agent_id=0)
    serialized = context.to_dict()
    assert len(serialized["private_state"]["information"]) == 5
    assert serialized["metadata"]["audit_answer_included"] is False
    assert "correct_answer" not in str(serialized)
    assert len(serialized["recent_memory"]) == 2


def test_hiddenbench_discussion_and_vote_match_two_message_templates():
    path = Path(
        "scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/"
        "scaled/exact_replication/N_32.json"
    )
    context = hiddenbench_example_context(path, task_id=1, agent_id=0)
    registry = create_default_prompt_registry()
    discussion_config = load_component_config(
        "configs/components/prompts/hidden_profile_discussion_paper.yaml",
        "prompt",
        environment={},
    )
    vote_config = load_component_config(
        "configs/components/prompts/hidden_profile_vote_paper.yaml",
        "prompt",
        environment={},
    )
    discussion = PromptComposer(registry).compose(discussion_config, context)
    vote = PromptComposer(registry).compose(vote_config, context)
    assert [message.role.value for message in discussion.messages] == ["system", "user"]
    assert "It's your turn to speak." in discussion.messages[1].content
    assert '"vote": <A string' in vote.messages[1].content
    assert vote.response_contract.validate(
        '{"vote": "West City", "rationale": "The bridge remains usable."}'
    ).is_valid


def test_markdown_logger_shows_every_exact_message_in_one_file(tmp_path: Path):
    config = load_component_config(
        "configs/components/prompts/social_conventions_paper.yaml",
        "prompt",
        environment={},
    )
    instance = PromptComposer(create_default_prompt_registry()).compose(
        config, social_conventions_example_context()
    )
    logger = PromptMarkdownLogger(tmp_path)
    path = logger.log(instance, "interaction-0001", metadata={"round": 4})
    rendered = path.read_text(encoding="utf-8")
    assert "## Exact messages sent to the LLM" in rendered
    assert "### Message 1 — `system`" in rendered
    assert "### Message 2 — `user`" in rendered
    assert all(message.content in rendered for message in instance.messages)
    with pytest.raises(FileExistsError):
        logger.log(instance, "interaction-0001")
