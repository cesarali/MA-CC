"""Compile ordered prompt blocks into normalized messages and readable text."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol

from mas_cc.config.models import PromptConfig
from mas_cc.core import Message

from .blocks import RenderedPromptBlock
from .context import PromptContext
from .contracts import ResponseContract
from .registry import PromptRegistry
from .versions import PromptVersion


class TokenCounter(Protocol):
    """Minimal tokenizer surface accepted by the composer."""

    name: str

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RegexTokenCounter:
    """Dependency-free deterministic token estimate used for inspection.

    Provider adapters can inject their model tokenizer through ``TokenCounter``
    later without changing prompt composition.
    """

    name: str = "mas_cc_regex_v1_estimate"

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class PromptInstance:
    """A fully compiled prompt that every provider can consume unchanged."""

    prompt_version: PromptVersion
    blocks: tuple[RenderedPromptBlock, ...]
    messages: tuple[Message, ...]
    response_contract: ResponseContract
    tokenizer_name: str | None = None

    @property
    def total_tokens(self) -> int | None:
        counts = tuple(block.token_count for block in self.blocks)
        if any(count is None for count in counts):
            return None
        return sum(count for count in counts if count is not None)

    def messages_as_dicts(self) -> list[dict[str, object]]:
        return [message.to_dict() for message in self.messages]

    def blocks_as_dicts(self) -> list[dict[str, object]]:
        return [replace(block, order=index).to_dict() for index, block in enumerate(self.blocks, start=1)]

    def rendered_text(self) -> str:
        lines = [
            "# Compiled prompt",
            "",
            f"- Prompt: `{self.prompt_version}`",
            f"- Tokenizer: `{self.tokenizer_name or 'unavailable'}`",
            f"- Total tokens: `{self.total_tokens if self.total_tokens is not None else 'unavailable'}`",
        ]
        for index, block in enumerate(self.blocks, start=1):
            lines.extend(
                [
                    "",
                    f"## {index}. {block.title}",
                    "",
                    f"Role: `{block.role.value}`  ",
                    f"Block: `{block.name}@{block.version}`  ",
                    f"Tokens: `{block.token_count if block.token_count is not None else 'unavailable'}`",
                    "",
                    block.content,
                ]
            )
        return "\n".join(lines) + "\n"


class PromptComposer:
    """Compile a config and context without importing an LLM provider."""

    def __init__(self, registry: PromptRegistry, token_counter: TokenCounter | None = None) -> None:
        self._registry = registry
        self._token_counter = token_counter

    def compose(self, config: PromptConfig, context: PromptContext) -> PromptInstance:
        definition = self._registry.get_legacy(config.prompt_family, config.prompt_version)
        response_contract = ResponseContract.from_mapping(config.response_contract)
        names = tuple(config.blocks)
        if len(set(names)) != len(names):
            duplicate = next(name for index, name in enumerate(names) if name in names[:index])
            raise ValueError(f"prompt.blocks: duplicate block {duplicate!r}")
        missing = tuple(name for name in definition.required_blocks if name not in names)
        if missing:
            raise ValueError(f"prompt.blocks: required block {missing[0]!r} is missing")

        rendered: list[RenderedPromptBlock] = []
        for index, name in enumerate(names):
            try:
                block = definition.block(name)
            except KeyError as exc:
                raise ValueError(f"prompt.blocks[{index}]: unknown block {name!r}") from exc
            item = block.render(context, response_contract)
            item = replace(item, order=index + 1)
            if self._token_counter is not None:
                item = replace(item, token_count=self._token_counter.count_tokens(item.content))
            rendered.append(item)

        message_mode = config.options.get("message_mode", "per_block")
        if message_mode not in {"per_block", "merge_consecutive_roles"}:
            raise ValueError(
                "prompt.options.message_mode: must be 'per_block' or "
                "'merge_consecutive_roles'"
            )
        separator = config.options.get("block_separator", "\n\n")
        if not isinstance(separator, str):
            raise ValueError("prompt.options.block_separator: must be a string")
        messages = self._compile_messages(
            rendered,
            config=config,
            mode=message_mode,
            separator=separator,
        )
        return PromptInstance(
            prompt_version=definition.prompt_version,
            blocks=tuple(rendered),
            messages=messages,
            response_contract=response_contract,
            tokenizer_name=self._token_counter.name if self._token_counter is not None else None,
        )

    @staticmethod
    def _compile_messages(
        rendered: list[RenderedPromptBlock],
        *,
        config: PromptConfig,
        mode: str,
        separator: str,
    ) -> tuple[Message, ...]:
        if mode == "per_block":
            return tuple(
                Message(
                    role=item.role,
                    content=item.content,
                    metadata={
                        "prompt_family": config.prompt_family,
                        "prompt_version": config.prompt_version,
                        "block": item.name,
                        "block_version": item.version,
                        "block_order": index + 1,
                    },
                )
                for index, item in enumerate(rendered)
            )

        grouped: list[list[RenderedPromptBlock]] = []
        for item in rendered:
            if grouped and grouped[-1][-1].role == item.role:
                grouped[-1].append(item)
            else:
                grouped.append([item])
        return tuple(
            Message(
                role=group[0].role,
                content=separator.join(item.content for item in group),
                metadata={
                    "prompt_family": config.prompt_family,
                    "prompt_version": config.prompt_version,
                    "blocks": [item.name for item in group],
                    "block_versions": [item.version for item in group],
                    "message_order": index + 1,
                },
            )
            for index, group in enumerate(grouped)
        )
