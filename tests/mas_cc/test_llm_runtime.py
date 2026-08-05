"""Coverage for the portable mas_cc.llm_runtime component.

test_llm_providers.py, test_prompts.py, test_prompts_v3.py, and
test_provider_economics.py already exercise these objects directly (they
import mas_cc.llm_runtime.providers/mas_cc.llm_runtime.prompts, the sole
canonical location now that mas_cc.llm_providers and mas_cc.prompts have
been removed). This module adds the checks specific to llm_runtime's
portability contract: it imports cleanly without optional provider
dependencies or any mas_cc.games/experiments/config/cli code, and a prompt
built from its kernel can flow through a mocked provider end to end.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from mas_cc.llm_runtime.config import LLMProviderConfig, PromptConfig
from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.prompts import (
    UNBOUND,
    FullPrompt,
    PromptBlock,
    PromptRegistry,
    RegexTokenCounter,
    ResponseContract,
    create_default_prompt_registry,
)
from mas_cc.llm_runtime.exceptions import ValidationError
from mas_cc.llm_runtime.providers import CompletionRequest, create_llm_provider
from mas_cc.llm_runtime.validation import ValidationIssue, ValidationResult


def test_llm_runtime_imports_without_optional_provider_dependencies():
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import mas_cc.llm_runtime.providers\n"
        "import mas_cc.llm_runtime.prompts\n"
        "added = set(sys.modules) - before\n"
        "print(json.dumps(sorted(added & {'comet_ml', 'dotenv', 'openai', 'requests', 'torch', 'transformers'})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


def test_llm_runtime_providers_and_prompts_are_independent_siblings():
    """Neither submodule imports the other; both are usable standalone."""

    script = (
        "import json, sys\n"
        "import mas_cc.llm_runtime.prompts\n"
        "print(json.dumps(sorted(\n"
        "    name for name in sys.modules if name.startswith('mas_cc.llm_runtime.providers')\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []

    script = (
        "import json, sys\n"
        "import mas_cc.llm_runtime.providers\n"
        "print(json.dumps(sorted(\n"
        "    name for name in sys.modules if name.startswith('mas_cc.llm_runtime.prompts')\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


def test_llm_runtime_does_not_import_games_experiments_or_cli():
    script = (
        "import json, sys\n"
        "import mas_cc.llm_runtime.providers\n"
        "import mas_cc.llm_runtime.prompts\n"
        "print(json.dumps(sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('mas_cc.games')\n"
        "    or name.startswith('mas_cc.experiments')\n"
        "    or name.startswith('mas_cc.cli')\n"
        "    or name.startswith('mas_cc.observability')\n"
        "    or name.startswith('mas_cc.config')\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


def test_default_prompt_registry_ships_no_builtin_content():
    registry = create_default_prompt_registry()
    assert registry.versions() == ()


class _ChoiceBlock(PromptBlock[object]):
    def value_issues(self, value):
        if not isinstance(value, str):
            return (ValidationIssue(f"prompt.blocks.{self.name}.value", "must be text", value),)
        return ()

    def render(self):
        return str(self.value)


class _PortableFullPrompt(FullPrompt):
    def concrete_prompt_type(self) -> str:
        return "llm_runtime_test"


def _build_prompt():
    return _PortableFullPrompt(
        "llm_runtime_test",
        1,
        (
            _ChoiceBlock(
                "instructions", "Instructions", MessageRole.SYSTEM, "Choose A or B.", binding="fixed"
            ),
            _ChoiceBlock("question", "Question", MessageRole.USER, UNBOUND),
        ),
        ResponseContract("choice_only", ("A", "B")),
        "merge_consecutive_roles",
    )


def test_prompt_kernel_to_mocked_provider_end_to_end():
    """Construct a prompt with the portable kernel, then complete it via the mock adapter."""

    prompt = _build_prompt().bind(question="A or B?")
    compiled = prompt.compile(RegexTokenCounter())
    assert [message.role for message in compiled.messages] == [MessageRole.SYSTEM, MessageRole.USER]

    registry = PromptRegistry()
    registry.register(lambda: _build_prompt())
    assert registry.get("llm_runtime_test", 1).family == "llm_runtime_test"

    provider = create_llm_provider(
        LLMProviderConfig(type="mock", model="deterministic-v1", options={"response": "A"})
    )
    request = CompletionRequest(compiled.messages, max_output_tokens=4, seed=1)
    response = asyncio.run(provider.complete(request))

    assert response.provider == "mock"
    assert response.content == "A"
    assert response.usage.input_tokens > 0


def test_prompt_config_is_connection_free():
    config = PromptConfig(prompt_family="llm_runtime_test", prompt_version=1)
    assert config.schema_version == 2
    assert config.to_dict()["prompt_family"] == "llm_runtime_test"


def test_message_is_immutable_and_serializes_provider_independently():
    message = Message(
        role="user",
        content="Choose A or B",
        metadata={"prompt_version": 1, "labels": ["A", "B"]},
    )
    assert message.role is MessageRole.USER
    assert message.to_dict() == {
        "role": "user",
        "content": "Choose A or B",
        "metadata": {"prompt_version": 1, "labels": ["A", "B"]},
    }
    with pytest.raises(TypeError):
        message.metadata["prompt_version"] = 2
    with pytest.raises(TypeError):
        message.metadata["labels"][0] = "B"


def test_validation_result_preserves_exact_fields():
    issue = ValidationIssue("game.population_size", "must be at least 2", {"value": [1]})
    result = ValidationResult.failure(issue)
    assert not result.is_valid
    with pytest.raises(ValidationError, match=r"game\.population_size"):
        result.raise_for_errors(context="game config")
    with pytest.raises(TypeError):
        issue.invalid_value["value"][0] = 2
