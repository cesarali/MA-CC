"""Generate readable, paper-grounded prompt inspection bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mas_cc.config import PromptConfig, load_component_config
from mas_cc.prompts import (
    PromptComposer,
    PromptMarkdownLogger,
    RegexTokenCounter,
    create_default_prompt_registry,
    hiddenbench_example_context,
    social_conventions_example_context,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prompt(path: Path) -> PromptConfig:
    loaded = load_component_config(path, "prompt", environment={})
    if not isinstance(loaded, PromptConfig):
        raise ValueError(f"prompt: {path} did not resolve to PromptConfig")
    return loaded


def generate_paper_prompt_examples(
    output_dir: str | Path,
    *,
    hiddenbench_data: str | Path,
    task_id: int = 1,
    agent_id: int = 0,
) -> Path:
    """Compile all paper examples without making an LLM call."""

    root = _root()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data_path = Path(hiddenbench_data).resolve()
    hidden_context = hiddenbench_example_context(
        data_path, task_id=task_id, agent_id=agent_id
    )
    social_context = social_conventions_example_context()

    discussion_config = _load_prompt(
        root / "configs/components/prompts/hidden_profile_discussion_paper.yaml"
    )
    vote_config = _load_prompt(
        root / "configs/components/prompts/hidden_profile_vote_paper.yaml"
    )
    configs = {
        "social_conventions": _load_prompt(
            root / "configs/components/prompts/social_conventions_paper.yaml"
        ),
        "hiddenbench_first_speaker": discussion_config,
        "hiddenbench_discussion": discussion_config,
        "hiddenbench_pre_vote": vote_config,
        "hiddenbench_post_vote": vote_config,
    }
    possible_answers = tuple(hidden_context.private_state["possible_answers"])
    resolved_vote_config = replace(
        vote_config,
        response_contract={"type": "json_vote", "allowed_values": possible_answers},
    )
    configs["hiddenbench_pre_vote"] = resolved_vote_config
    configs["hiddenbench_post_vote"] = resolved_vote_config
    first_context = replace(
        hidden_context,
        recent_memory=(),
        current_interaction={"phase": "public_discussion", "first_speaker": True},
    )
    contexts = {
        "social_conventions": social_context,
        "hiddenbench_first_speaker": first_context,
        "hiddenbench_discussion": hidden_context,
        "hiddenbench_pre_vote": replace(
            hidden_context,
            recent_memory=(),
            current_interaction={"phase": "pre_discussion_vote"},
        ),
        "hiddenbench_post_vote": hidden_context,
    }
    titles = {
        "social_conventions": "Social conventions paper — one agent decision",
        "hiddenbench_first_speaker": "HiddenBench paper — first public speaker",
        "hiddenbench_discussion": "HiddenBench paper — one public discussion turn",
        "hiddenbench_pre_vote": "HiddenBench paper — one pre-discussion vote",
        "hiddenbench_post_vote": "HiddenBench paper — one post-discussion vote",
    }
    composer = PromptComposer(create_default_prompt_registry(), RegexTokenCounter())
    consolidated: list[str] = [
        "# Paper-grounded prompt examples",
        "",
        "Each section shows one complete request exactly as it would be passed to an LLM provider.",
    ]
    checks: dict[str, bool] = {}
    for name in (
        "social_conventions",
        "hiddenbench_first_speaker",
        "hiddenbench_discussion",
        "hiddenbench_pre_vote",
        "hiddenbench_post_vote",
    ):
        example_dir = destination / name
        example_dir.mkdir(parents=True, exist_ok=True)
        config = configs[name]
        context = contexts[name]
        instance = composer.compose(config, context)
        metadata = context.to_dict()["metadata"]
        logger = PromptMarkdownLogger(example_dir, overwrite=True)
        request_path = logger.log(
            instance,
            "request",
            title=titles[name],
            metadata=metadata,
        )
        _write_json(example_dir / "compiled_messages.json", instance.messages_as_dicts())
        _write_json(example_dir / "prompt_blocks.json", instance.blocks_as_dicts())
        _write_json(example_dir / "prompt_context.json", context.to_dict())
        (example_dir / "prompt_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        markdown = request_path.read_text(encoding="utf-8")
        consolidated.extend(["", "---", "", markdown])
        checks[f"{name}_two_message_request"] = (
            len(instance.messages) == 2
            and [message.role.value for message in instance.messages] == ["system", "user"]
        )
        checks[f"{name}_markdown_matches_messages"] = all(
            message.content in markdown for message in instance.messages
        )

    (destination / "all_requests.md").write_text(
        "\n".join(consolidated).rstrip() + "\n", encoding="utf-8"
    )
    no_answer_key = "correct_answer" not in (
        destination / "hiddenbench_discussion/prompt_context.json"
    ).read_text(encoding="utf-8")
    checks["hiddenbench_audit_answer_key_excluded"] = no_answer_key

    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Paper prompt example report

- Status: **{status.upper()}**
- No LLM was called.
- Social-conventions source: `pdfs/Emergence of social conventions supplementary.pdf`, section **Prompting → Example Prompt**.
- HiddenBench source: `pdfs/Systematic Failures in Collective Reasoning under Distributed Information in.pdf`, Appendix **A.4 Prompts and Communication Templates**.
- HiddenBench fixture: `{data_path}`, task `{task_id}`, agent `{agent_id}`.
- Token counts are dependency-free estimates from `mas_cc_regex_v1_estimate`.

## What is adapted

- The social-conventions wording, F/J actions, simultaneous choice, +100/−50 payoffs, bounded memory, answer-first response, and final user request follow the supplementary example. The concrete score and three memory rows are an inspection fixture.
- HiddenBench uses the downloaded scenario, shared facts, and the selected agent's private fact. Fact order is deterministically shuffled. The two public transcript lines are inspection fixtures constructed from other agents' private packets.
- The HiddenBench `correct_answer` audit field is never copied into the prompt context or request.
- Lego blocks are merged by consecutive role so the final transmission is exactly one `system` message followed by one `user` message, matching both papers' request shape.

## Readable requests

- [`all_requests.md`](all_requests.md) — all five complete requests in one document.
- [`social_conventions/request.md`](social_conventions/request.md) — one convention decision.
- [`hiddenbench_first_speaker/request.md`](hiddenbench_first_speaker/request.md) — the first discussion turn.
- [`hiddenbench_discussion/request.md`](hiddenbench_discussion/request.md) — a later discussion turn with public transcript.
- [`hiddenbench_pre_vote/request.md`](hiddenbench_pre_vote/request.md) — a vote before discussion.
- [`hiddenbench_post_vote/request.md`](hiddenbench_post_vote/request.md) — one post-discussion vote.

Each example directory also contains the source prompt config, context, rendered blocks, and compiled JSON messages for machine inspection.
"""
    (destination / "report.md").write_text(report, encoding="utf-8")

    artifacts = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "manifest_version": 1,
        "kind": "paper_prompt_examples",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "checks": checks,
        "sources": {
            "social_conventions_paper": "pdfs/Emergence of social conventions supplementary.pdf",
            "hiddenbench_paper": "pdfs/Systematic Failures in Collective Reasoning under Distributed Information in.pdf",
            "hiddenbench_data": str(data_path),
            "task_id": task_id,
            "agent_id": agent_id,
        },
        "artifacts": artifacts,
    }
    _write_json(destination / "manifest.json", manifest)
    return destination
