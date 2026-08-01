"""Deterministic paper-grounded contexts for prompt inspection only."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .context import PromptContext


def social_conventions_example_context() -> PromptContext:
    """Concrete F/J fixture following the supplementary example prompt."""

    return PromptContext(
        task_description=(
            "Context: Player 1 is playing a multi-round partnership game with "
            "Player 2 for 100 rounds. At each round, Player 1 and Player 2 "
            "simultaneously pick an action from the following values: [F, J]. "
            "The payoff that both players get is determined by the following rule:"
        ),
        game_rules=(
            "If Players play the SAME action as each other, they will both be "
            "REWARDED with payoff 100 points.",
            "If Players play DIFFERENT actions to each other, they will both be "
            "PUNISHED with payoff -50 points.",
        ),
        private_state={
            "player": "Player 1",
            "available_actions": ["F", "J"],
            "score": 150,
            "memory_limit": 5,
        },
        recent_memory=(
            {"round": 1, "own_action": "F", "other_action": "J", "payoff": -50},
            {"round": 2, "own_action": "J", "other_action": "J", "payoff": 100},
            {"round": 3, "own_action": "J", "other_action": "J", "payoff": 100},
        ),
        current_interaction={"round": 4, "simultaneous": True},
        decision_instruction="Answer saying which action Player 1 should play.",
        metadata={
            "fixture": "social_conventions_supplement_example_v1",
            "source": "pdfs/Emergence of social conventions supplementary.pdf",
        },
    )


def _find_task(data: dict[str, Any], task_id: int) -> dict[str, Any]:
    for task in data.get("tasks", []):
        if int(task.get("task_id", -1)) == task_id:
            return task
    raise ValueError(f"hiddenbench.task_id: {task_id} was not found")


def _agents(task: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(task.get("agents"), list):
        return list(task["agents"])
    return [
        {
            "agent_id": int(item["evidence_type"]),
            "evidence_type": int(item["evidence_type"]),
            "private_information": [item["source_text"]],
        }
        for item in task.get("hidden_information", [])
    ]


def hiddenbench_example_context(
    data_path: str | Path,
    *,
    task_id: int = 1,
    agent_id: int = 0,
    shuffle_seed: int = 1026,
) -> PromptContext:
    """Build one private-agent fixture without exposing the audit answer field."""

    source = Path(data_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    task = _find_task(data, task_id)
    agents = _agents(task)
    try:
        selected = next(agent for agent in agents if int(agent["agent_id"]) == agent_id)
    except StopIteration as exc:
        raise ValueError(
            f"hiddenbench.agent_id: {agent_id} was not found for task {task_id}"
        ) from exc

    information = [
        *[str(item) for item in task.get("shared_information", [])],
        *[str(item) for item in selected.get("private_information", [])],
    ]
    random.Random(shuffle_seed).shuffle(information)

    transcript = []
    seen_types = {selected.get("evidence_type")}
    for agent in agents:
        if int(agent["agent_id"]) == agent_id:
            continue
        evidence_type = agent.get("evidence_type")
        if evidence_type in seen_types:
            continue
        private = agent.get("private_information", [])
        if not private:
            continue
        transcript.append(
            {
                "speaker_id": int(agent["agent_id"]),
                "message": f"I was told: {private[0]}",
            }
        )
        seen_types.add(evidence_type)
        if len(transcript) == 2:
            break

    return PromptContext(
        task_description=str(
            task.get("scenario_description", task.get("source_description", ""))
        ),
        game_rules=(
            "Use the shared and private evidence to support the group's decision.",
        ),
        private_state={
            "information": information,
            "possible_answers": [str(item) for item in task.get("possible_answers", [])],
        },
        recent_memory=tuple(transcript),
        current_interaction={"phase": "public_discussion", "first_speaker": False},
        decision_instruction="It's your turn to speak.",
        metadata={
            "fixture": "hiddenbench_downloaded_task_v1",
            "source_data": str(source),
            "task_id": task_id,
            "agent_id": agent_id,
            "shuffle_seed": shuffle_seed,
            "audit_answer_included": False,
            "transcript_is_inspection_fixture": True,
        },
    )
