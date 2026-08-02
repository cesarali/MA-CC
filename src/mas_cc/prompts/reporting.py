"""Readable Markdown logging for the exact compiled request sent to an LLM."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mas_cc.config import assert_secret_free

from .compiled import CompiledPrompt

_LOG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _fenced(content: str) -> str:
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}text\n{content}\n{fence}"


def render_prompt_request_markdown(
    instance: CompiledPrompt,
    *,
    title: str = "Compiled LLM request",
    metadata: Mapping[str, Any] | None = None,
    include_block_index: bool = True,
) -> str:
    """Render the final ordered messages verbatim in one Markdown document."""

    family = getattr(instance, "prompt_family", None)
    version = instance.prompt_version
    identity = f"{family}@{version}" if family is not None else str(version)
    lines = [
        f"# {title}",
        "",
        f"- Prompt version: `{identity}`",
        f"- Messages sent: `{len(instance.messages)}`",
        f"- Token counter: `{instance.tokenizer_name or 'unavailable'}`",
        f"- Estimated block tokens: `{instance.total_tokens if instance.total_tokens is not None else 'unavailable'}`",
    ]
    if metadata:
        assert_secret_free(metadata)
        lines.extend(
            [
                "",
                "## Request metadata",
                "",
                "```json",
                json.dumps(dict(metadata), indent=2, sort_keys=True, ensure_ascii=False),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Exact messages sent to the LLM",
            "",
            "The messages below are shown in transmission order. Text inside each fence is the exact message content.",
        ]
    )
    for index, message in enumerate(instance.messages, start=1):
        lines.extend(
            [
                "",
                f"### Message {index} — `{message.role.value}`",
                "",
                _fenced(message.content),
            ]
        )
    lines.extend(
        [
            "",
            "## Response contract",
            "",
            "```json",
            json.dumps(instance.response_contract.to_dict(), indent=2, sort_keys=True),
            "```",
        ]
    )
    if include_block_index:
        lines.extend(
            [
                "",
                "## Block provenance",
                "",
                "| Order | Block | Version | Role | Estimated tokens |",
                "|---:|---|---:|---|---:|",
            ]
        )
        for index, block in enumerate(instance.blocks, start=1):
            token_count = block.token_count if block.token_count is not None else "—"
            lines.append(
                f"| {index} | `{block.name}` | {block.version} | `{block.role.value}` | {token_count} |"
            )
    return "\n".join(lines) + "\n"


class PromptMarkdownLogger:
    """Write one exact compiled request per interaction as a Markdown file."""

    def __init__(self, output_dir: str | Path, *, overwrite: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite

    def log(
        self,
        instance: CompiledPrompt,
        interaction_id: str,
        *,
        title: str = "Compiled LLM request",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not isinstance(interaction_id, str) or not _LOG_NAME.fullmatch(interaction_id):
            raise ValueError(
                "interaction_id must contain only letters, digits, dots, underscores, and hyphens"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{interaction_id}.md"
        if destination.exists() and not self.overwrite:
            raise FileExistsError(f"prompt log already exists: {destination}")
        destination.write_text(
            render_prompt_request_markdown(
                instance,
                title=title,
                metadata=metadata,
            ),
            encoding="utf-8",
        )
        return destination
