"""Generate readable examples from concrete bound FullPrompt values."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mas_cc.games.naming_convention.prompts import bind_naming_convention_prompt
from mas_cc.prompts import PromptMarkdownLogger, RegexTokenCounter
from mas_cc.prompts.plugins.hidden_profile_v3 import (
    hidden_profile_discussion_prompt,
    hidden_profile_vote_prompt,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hiddenbench_values(
    data_path: Path, *, task_id: int, agent_id: int, shuffle_seed: int = 1026
) -> dict[str, Any]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    try:
        task = next(
            item for item in data.get("tasks", []) if int(item.get("task_id", -1)) == task_id
        )
    except StopIteration as exc:
        raise ValueError(f"hiddenbench.task_id: {task_id} was not found") from exc
    agents = task.get("agents")
    if not isinstance(agents, list):
        agents = [
            {
                "agent_id": int(item["evidence_type"]),
                "evidence_type": int(item["evidence_type"]),
                "private_information": [item["source_text"]],
            }
            for item in task.get("hidden_information", [])
        ]
    try:
        selected = next(item for item in agents if int(item["agent_id"]) == agent_id)
    except StopIteration as exc:
        raise ValueError(
            f"hiddenbench.agent_id: {agent_id} was not found for task {task_id}"
        ) from exc
    information = [
        *map(str, task.get("shared_information", [])),
        *map(str, selected.get("private_information", [])),
    ]
    random.Random(shuffle_seed).shuffle(information)
    transcript: list[dict[str, Any]] = []
    seen = {selected.get("evidence_type")}
    for agent in agents:
        private = agent.get("private_information", [])
        evidence_type = agent.get("evidence_type")
        if int(agent["agent_id"]) == agent_id or evidence_type in seen or not private:
            continue
        transcript.append(
            {"speaker_id": int(agent["agent_id"]), "message": f"I was told: {private[0]}"}
        )
        seen.add(evidence_type)
        if len(transcript) == 2:
            break
    return {
        "scenario": str(task.get("scenario_description", task.get("source_description", ""))),
        "information": tuple(information),
        "transcript": tuple(transcript),
        "answers": tuple(map(str, task.get("possible_answers", []))),
        "metadata": {
            "source_data": str(data_path),
            "task_id": task_id,
            "agent_id": agent_id,
            "shuffle_seed": shuffle_seed,
            "audit_answer_included": False,
        },
    }


def generate_paper_prompt_examples(
    output_dir: str | Path,
    *,
    hiddenbench_data: str | Path,
    task_id: int = 1,
    agent_id: int = 0,
) -> Path:
    """Compile concrete paper-oriented fixtures without making an LLM call."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    values = _hiddenbench_values(
        Path(hiddenbench_data).resolve(), task_id=task_id, agent_id=agent_id
    )
    discussion = hidden_profile_discussion_prompt().bind(
        scenario=values["scenario"],
        private_information={"information": values["information"]},
        transcript=values["transcript"],
    )
    vote = hidden_profile_vote_prompt()
    vote = type(vote)(
        vote.family,
        vote.version,
        vote.blocks,
        type(vote.response_contract)("json_vote", values["answers"]),
        vote.message_mode,
        vote.block_separator,
    ).bind(
        scenario=values["scenario"],
        private_information={"information": values["information"]},
        transcript=values["transcript"],
    )
    social = bind_naming_convention_prompt(
        presented_actions=("F", "J"),
        visible_memory=(
            {"own_action": "F", "partner_action": "J", "payoff": -50},
            {"own_action": "J", "partner_action": "J", "payoff": 100},
            {"own_action": "J", "partner_action": "J", "payoff": 100},
        ),
        visible_score=150,
        local_round=4,
        allowed_actions=("F", "J"),
    )
    examples = {
        "social_conventions": social,
        "hiddenbench_first_speaker": discussion.bind(transcript=()),
        "hiddenbench_discussion": discussion,
        "hiddenbench_pre_vote": vote.bind(transcript=()),
        "hiddenbench_post_vote": vote,
    }
    consolidated = [
        "# Paper-grounded concrete FullPrompt examples",
        "",
        "Social conventions paper and HiddenBench paper inspection fixtures.",
    ]
    checks: dict[str, bool] = {}
    counter = RegexTokenCounter()
    for name, prompt in examples.items():
        example_dir = destination / name
        example_dir.mkdir(parents=True, exist_ok=True)
        compiled = prompt.compile(counter)
        markdown = PromptMarkdownLogger(example_dir, overwrite=True).log(
            compiled,
            "request",
            title=name.replace("_", " ").title(),
            metadata=values["metadata"] if name.startswith("hiddenbench") else {},
        )
        _write_json(example_dir / "full_prompt.json", prompt.to_dict())
        _write_json(example_dir / "compiled_messages.json", compiled.messages_as_dicts())
        _write_json(example_dir / "prompt_blocks.json", compiled.blocks_as_dicts())
        _write_json(example_dir / "bound_prompt.json", prompt.to_dict())
        (example_dir / "prompt_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "prompt_family": prompt.family,
                    "prompt_version": prompt.version,
                    "message_mode": prompt.message_mode,
                    "block_separator": prompt.block_separator,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        text = markdown.read_text(encoding="utf-8")
        consolidated.extend(["", "---", "", text])
        checks[f"{name}_compiled"] = bool(compiled.messages)
        checks[f"{name}_markdown_matches"] = all(
            message.content in text for message in compiled.messages
        )
    (destination / "all_requests.md").write_text(
        "\n".join(consolidated).rstrip() + "\n", encoding="utf-8"
    )
    checks["hiddenbench_audit_answer_key_excluded"] = "correct_answer" not in (
        destination / "hiddenbench_discussion/bound_prompt.json"
    ).read_text(encoding="utf-8")
    status = "pass" if all(checks.values()) else "fail"
    (destination / "report.md").write_text(
        "# Paper prompt example report\n\n"
        f"- Status: **{status.upper()}**\n"
        "- All examples use concrete bound FullPrompt objects.\n"
        "- No provider was constructed and no LLM was called.\n",
        encoding="utf-8",
    )
    artifacts = [
        {
            "path": str(path.relative_to(destination)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        destination / "manifest.json",
        {
            "manifest_version": 2,
            "kind": "full_prompt_examples",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": status,
            "checks": checks,
            "artifacts": artifacts,
        },
    )
    return destination
