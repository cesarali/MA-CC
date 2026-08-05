from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from mas_cc.config import PromptConfig, load_component_config, load_run_config, resolved_config_yaml
from mas_cc.config.schema import config_schema
from mas_cc.games.naming_convention.prompts import bind_naming_convention_prompt
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.messages import MessageRole
from mas_cc.llm_runtime.validation import ValidationIssue
from mas_cc.llm_runtime.prompts import (
    UNBOUND,
    FullPrompt,
    PromptBlock,
    RegexTokenCounter,
    ResponseContract,
    Unbound,
)


@dataclass(frozen=True, slots=True)
class TextBlock(PromptBlock[object]):
    def value_issues(self, value):
        if not isinstance(value, (str, tuple)):
            return (
                ValidationIssue(
                    f"prompt.blocks.{self.name}.value", "must be text or a tuple", value
                ),
            )
        return ()

    def render(self):
        if isinstance(self.value, tuple):
            return ",".join(map(str, self.value)) or "<empty>"
        return str(self.value)


class KernelTestFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "kernel_test"


def _prompt(*, version=1, separator="\n\n"):
    return KernelTestFullPrompt(
        "kernel_test",
        version,
        (
            TextBlock(
                "fixed", "Fixed", MessageRole.SYSTEM, "constant", binding="fixed"
            ),
            TextBlock("dynamic", "Dynamic", MessageRole.USER, UNBOUND, sensitive=True),
            TextBlock(
                "optional", "Optional", MessageRole.USER, UNBOUND, required=False
            ),
        ),
        ResponseContract("choice_only", ("A", "B")),
        "merge_consecutive_roles",
        separator,
    )


def test_unbound_empty_and_fixed_binding_semantics():
    prompt = _prompt()
    issue = prompt.validate().issues[0]
    assert issue.field == "prompt.blocks.dynamic.value"
    with pytest.raises(ValueError, match="fixed block cannot be rebound"):
        prompt.bind(fixed="changed")
    bound = prompt.bind(dynamic=(), optional=())
    compiled = bound.compile()
    assert bound is not prompt
    assert bound.block("dynamic").value == ()
    assert compiled.omitted_blocks == ()
    assert compiled.blocks[1].content == "<empty>"


def test_optional_unbound_is_omitted_and_invalid_type_has_exact_field():
    compiled = _prompt().bind(dynamic="value").compile()
    assert compiled.omitted_blocks == ("optional",)
    invalid = _prompt().bind(dynamic={"not": "valid"}).validate()
    assert invalid.issues[0].field == "prompt.blocks.dynamic.value"


def test_binding_is_deeply_immutable_and_concurrent_instances_do_not_leak():
    source = ["first"]
    prompt = _prompt()

    def bind(value):
        return prompt.bind(dynamic=value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(bind, (source, ["second"])))
    source[0] = "mutated"
    assert first.block("dynamic").value == ("first",)
    assert second.block("dynamic").value == ("second",)
    assert prompt.block("dynamic").value is UNBOUND


def test_order_tokens_grouping_and_fingerprints_are_deterministic():
    counter = RegexTokenCounter()
    first = _prompt().bind(dynamic="one").compile(counter)
    again = _prompt().bind(dynamic="one").compile(counter)
    changed = _prompt().bind(dynamic="two").compile(counter)
    definition_changed = _prompt(separator="\n---\n").bind(dynamic="one").compile(counter)
    assert [block.name for block in first.blocks] == ["fixed", "dynamic"]
    assert first.total_tokens == sum(block.token_count for block in first.blocks)
    assert first.message_token_total == sum(first.message_token_counts)
    assert first == again
    assert first.definition_hash == changed.definition_hash
    assert first.instance_hash != changed.instance_hash
    assert first.definition_hash != definition_changed.definition_hash


def test_unknown_binding_is_rejected_at_dotted_field():
    with pytest.raises(ValueError, match=r"prompt\.bind\.invented"):
        _prompt().bind(invented="value")


def test_full_prompt_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        FullPrompt(  # type: ignore[abstract]
            "invalid",
            1,
            (TextBlock("value", "Value", MessageRole.USER, "bound"),),
            ResponseContract("free_text"),
        )


def test_naming_definition_hash_is_order_independent_but_instance_hash_is_not():
    first = bind_naming_convention_prompt(
        presented_actions=("Q", "M"), visible_memory=(), visible_score=0, local_round=1
    )
    second = bind_naming_convention_prompt(
        presented_actions=("M", "Q"), visible_memory=(), visible_score=0, local_round=1
    )
    assert first.definition_hash == second.definition_hash
    assert first.instance_hash != second.instance_hash
    assert "exactly Q or M" in first.compile().messages[0].content
    assert "exactly M or Q" in second.compile().messages[0].content


def test_naming_prompt_rejects_incomplete_memory_and_contract_mismatch():
    incomplete = bind_naming_convention_prompt(
        presented_actions=("Q", "M"),
        visible_memory=({"own_action": "Q", "payoff": 100},),
        visible_score=100,
        local_round=2,
    )
    assert incomplete.validate().issues[0].field == (
        "prompt.blocks.visible_memory.value[0]"
    )
    mismatch = bind_naming_convention_prompt(
        presented_actions=("Q", "X"),
        visible_memory=(),
        visible_score=0,
        local_round=1,
    )
    assert mismatch.validate().issues[0].field == (
        "prompt.blocks.presented_actions.value"
    )


def test_active_games_planning_and_runtime_do_not_import_legacy_context():
    roots = (
        Path("src/mas_cc/games"),
        Path("src/mas_cc/planning"),
        Path("src/mas_cc/runtime"),
    )
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "PromptContext" not in text, path


def test_provider_adapters_do_not_import_prompt_or_game_implementations():
    for path in Path("src/mas_cc/llm_runtime/providers/adapters").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mas_cc.llm_runtime.prompts" not in text, path
        assert "mas_cc.games" not in text, path


def test_prompt_schema_v2_exports_registered_order_and_v1_has_diagnostics():
    component = load_component_config(
        "configs/components/prompts/naming_convention_v3.yaml",
        "prompt",
        environment={},
    )
    assert component.schema_version == 2
    assert component.blocks == ()
    config = load_run_config(
        "configs/runs/naming_convention_smoke_test.yaml", environment={}
    )
    rendered = resolved_config_yaml(config)
    assert "resolved_block_manifest:" in rendered
    assert rendered.index("name: description") < rendered.index("name: visible_score")
    legacy = PromptConfig(
        "legacy", 1, ("hand_ordered",), {"type": "free_text"}, schema_version=1
    )
    assert "remove prompt.blocks" in " ".join(legacy.migration_diagnostics())


def test_prompt_schema_v2_rejects_hand_maintained_block_order(tmp_path):
    component = tmp_path / "invalid_prompt.yaml"
    for block_value in ("[task]", "[]"):
        component.write_text(
            "schema_version: 2\n"
            "prompt_family: basic_choice\n"
            "prompt_version: 1\n"
            f"blocks: {block_value}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="authoritative registered FullPrompt order"):
            load_component_config(component, "prompt", environment={})


def test_prompt_json_schema_encodes_v1_and_v2_block_order_rules():
    prompt_schema = config_schema()["properties"]["prompt"]
    assert prompt_schema["allOf"][0]["then"]["required"] == [
        "blocks", "response_contract"
    ]
    assert prompt_schema["allOf"][1]["then"] == {
        "not": {"required": ["blocks"]}
    }
    with pytest.raises(ValueError, match="forbidden in schema version 2"):
        PromptConfig("basic_choice", 1, ("task",))


def test_prescribed_v3_run_configs_load_without_environment_or_provider_work():
    for path in (
        "configs/runs/provider_smoke_test_v3.yaml",
        "configs/runs/toy_game_smoke_test_v3.yaml",
        "configs/runs/naming_convention_smoke_test_v3.yaml",
    ):
        assert load_run_config(path, environment={}).prompt.schema_version == 2
