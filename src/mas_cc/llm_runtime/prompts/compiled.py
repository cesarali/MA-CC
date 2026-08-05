"""Provider-neutral compiled prompt artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..messages import Message
from .blocks import RenderedPromptBlock
from .contracts import ResponseContract


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    family: str
    version: int
    definition_hash: str
    instance_hash: str
    blocks: tuple[RenderedPromptBlock, ...]
    omitted_blocks: tuple[str, ...]
    messages: tuple[Message, ...]
    response_contract: ResponseContract
    tokenizer_name: str | None
    message_token_counts: tuple[int, ...]

    @property
    def prompt_family(self) -> str:
        return self.family

    @property
    def prompt_version(self) -> int:
        return self.version

    @property
    def block_token_total(self) -> int | None:
        counts = tuple(block.token_count for block in self.blocks)
        if any(item is None for item in counts):
            return None
        return sum(item for item in counts if item is not None)

    @property
    def message_token_total(self) -> int | None:
        return sum(self.message_token_counts) if self.tokenizer_name is not None else None

    @property
    def total_tokens(self) -> int | None:
        return self.block_token_total

    @property
    def total_token_estimate(self) -> int | None:
        """Estimated provider input text, including response instructions."""

        return self.message_token_total

    def messages_as_dicts(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def blocks_as_dicts(self) -> list[dict[str, Any]]:
        return [block.to_dict() for block in self.blocks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "prompt_family": self.family,
            "prompt_version": self.version,
            "definition_hash": self.definition_hash,
            "instance_hash": self.instance_hash,
            "rendered_blocks": self.blocks_as_dicts(),
            "omitted_optional_blocks": list(self.omitted_blocks),
            "messages": self.messages_as_dicts(),
            "response_contract": self.response_contract.to_dict(),
            "tokenizer": self.tokenizer_name,
            "block_token_total": self.block_token_total,
            "message_token_counts": list(self.message_token_counts),
            "message_token_total": self.message_token_total,
            "total_token_estimate": self.total_token_estimate,
        }

    def rendered_text(self) -> str:
        lines = [
            "# Compiled prompt",
            "",
            f"- Prompt: `{self.family}@{self.version}`",
            f"- Definition hash: `{self.definition_hash}`",
            f"- Instance hash: `{self.instance_hash}`",
            f"- Tokenizer: `{self.tokenizer_name or 'unavailable'}`",
            f"- Block tokens: `{self.total_tokens if self.total_tokens is not None else 'unavailable'}`",
        ]
        for block in self.blocks:
            lines.extend(
                [
                    "",
                    f"## {block.order}. {block.title}",
                    "",
                    f"Role: `{block.role.value}`  ",
                    f"Block: `{block.name}@{block.version}`  ",
                    f"Tokens: `{block.token_count if block.token_count is not None else 'unavailable'}`",
                    "",
                    block.content,
                ]
            )
        if self.omitted_blocks:
            lines.extend(["", "## Omitted optional blocks", "", ", ".join(self.omitted_blocks)])
        return "\n".join(lines) + "\n"
