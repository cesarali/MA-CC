"""Ordered immutable full-prompt mechanics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, Self, runtime_checkable

from ..messages import Message
from ..validation import ValidationIssue, ValidationResult
from ._values import thaw
from .blocks import UNBOUND, PromptBlock, RenderedPromptBlock
from .compiled import CompiledPrompt
from .contracts import ResponseContract
from .fingerprints import fingerprint
from .tokenization import TokenCounter

MessageMode = Literal["per_block", "merge_consecutive_roles"]


@runtime_checkable
class CompilablePrompt(Protocol):
    family: str
    version: int

    def compile(self, token_counter: TokenCounter | None = None) -> CompiledPrompt: ...


@dataclass(frozen=True, slots=True)
class FullPrompt(ABC):
    """Authoritative ordered collection of semantic blocks and output contract."""

    family: str
    version: int
    blocks: tuple[PromptBlock[Any], ...]
    response_contract: ResponseContract
    message_mode: MessageMode = "merge_consecutive_roles"
    block_separator: str = "\n\n"

    @abstractmethod
    def concrete_prompt_type(self) -> str:
        """Return a stable concrete definition label for inspection."""

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("FullPrompt.family must be a non-empty stable name")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("FullPrompt.version must be a positive integer")
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("FullPrompt.blocks must not be empty")
        if any(not isinstance(block, PromptBlock) for block in blocks):
            raise TypeError("FullPrompt.blocks must contain PromptBlock values")
        names = tuple(block.name for block in blocks)
        if len(set(names)) != len(names):
            duplicate = next(name for index, name in enumerate(names) if name in names[:index])
            raise ValueError(f"FullPrompt.blocks contains duplicate name {duplicate!r}")
        if not isinstance(self.response_contract, ResponseContract):
            raise TypeError("FullPrompt.response_contract must be a ResponseContract")
        if self.message_mode not in {"per_block", "merge_consecutive_roles"}:
            raise ValueError(
                "FullPrompt.message_mode must be 'per_block' or 'merge_consecutive_roles'"
            )
        if not isinstance(self.block_separator, str):
            raise TypeError("FullPrompt.block_separator must be a string")
        object.__setattr__(self, "blocks", blocks)

    def block(self, name: str) -> PromptBlock[Any]:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)

    def bind(self, **values: object) -> Self:
        known = {block.name for block in self.blocks}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"prompt.bind.{unknown[0]}: unknown block binding")
        rebound = tuple(
            block.bind(values[block.name]) if block.name in values else block
            for block in self.blocks
        )
        return replace(self, blocks=rebound)

    def validate(self) -> ValidationResult:
        issues = tuple(
            issue for block in self.blocks for issue in block.validate().issues
        )
        return ValidationResult(issues)

    def definition_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family": self.family,
            "version": self.version,
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "concrete_prompt_type": self.concrete_prompt_type(),
            "message_mode": self.message_mode,
            "block_separator": self.block_separator,
            "blocks": [block.definition_dict() for block in self.blocks],
            "response_contract": self.response_contract.to_dict(),
        }

    def instance_dict(self) -> dict[str, Any]:
        """Secret-safe serialized binding state."""

        return {
            "definition_hash": self.definition_hash,
            "bindings": [block.to_dict() for block in self.blocks],
        }

    @property
    def definition_hash(self) -> str:
        payload = self.definition_dict()
        payload["blocks"] = [
            block.definition_fingerprint_dict() for block in self.blocks
        ]
        return fingerprint(payload)

    @property
    def instance_hash(self) -> str:
        return fingerprint(
            {
                "definition_hash": self.definition_hash,
                "bindings": [block.fingerprint_dict() for block in self.blocks],
            }
        )

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        return {
            **self.definition_dict(),
            "definition_hash": self.definition_hash,
            "instance_hash": self.instance_hash,
            "blocks": [
                block.to_dict(include_value=include_values) for block in self.blocks
            ],
        }

    def compile(self, token_counter: TokenCounter | None = None) -> CompiledPrompt:
        validation = self.validate()
        validation.raise_for_errors(context=f"prompt {self.family}@{self.version}")
        rendered: list[RenderedPromptBlock] = []
        omitted: list[str] = []
        for block_order, block in enumerate(self.blocks, start=1):
            if block.value is UNBOUND:
                omitted.append(block.name)
                continue
            content = block.render()
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"prompt.blocks.{block.name}.render: must return non-empty text"
                )
            normalized = content.strip()
            rendered.append(
                RenderedPromptBlock(
                    name=block.name,
                    title=block.title,
                    role=block.role,
                    content=normalized,
                    version=block.version,
                    order=block_order,
                    token_count=(
                        None if token_counter is None else token_counter.count_tokens(normalized)
                    ),
                )
            )
        messages = list(self._block_messages(tuple(rendered)))
        for index, (role, content) in enumerate(
            self.response_contract.instruction_messages(), start=1
        ):
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"prompt.response_contract.instructions[{index - 1}]: must be non-empty"
                )
            messages.append(
                Message(
                    role,
                    content.strip(),
                    metadata={
                        "prompt_family": self.family,
                        "prompt_version": self.version,
                        "response_contract": self.response_contract.type,
                        "instruction_order": index,
                    },
                )
            )
        normalized_messages = self._merge_messages(tuple(messages))
        message_counts = (
            ()
            if token_counter is None
            else tuple(token_counter.count_tokens(message.content) for message in normalized_messages)
        )
        return CompiledPrompt(
            family=self.family,
            version=self.version,
            definition_hash=self.definition_hash,
            instance_hash=self.instance_hash,
            blocks=tuple(rendered),
            omitted_blocks=tuple(omitted),
            messages=normalized_messages,
            response_contract=self.response_contract,
            tokenizer_name=None if token_counter is None else token_counter.name,
            message_token_counts=message_counts,
        )

    def _block_messages(
        self, rendered: tuple[RenderedPromptBlock, ...]
    ) -> tuple[Message, ...]:
        return tuple(
            Message(
                block.role,
                block.content,
                metadata={
                    "prompt_family": self.family,
                    "prompt_version": self.version,
                    "block": block.name,
                    "block_version": block.version,
                    "block_order": block.order,
                },
            )
            for block in rendered
        )

    def _merge_messages(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        if self.message_mode == "per_block":
            return messages
        grouped: list[list[Message]] = []
        for message in messages:
            if grouped and grouped[-1][-1].role == message.role:
                grouped[-1].append(message)
            else:
                grouped.append([message])
        return tuple(
            Message(
                role=group[0].role,
                content=self.block_separator.join(item.content for item in group),
                metadata={
                    "prompt_family": self.family,
                    "prompt_version": self.version,
                    "sources": [thaw(item.metadata) for item in group],
                    "message_order": index,
                },
            )
            for index, group in enumerate(grouped, start=1)
        )
