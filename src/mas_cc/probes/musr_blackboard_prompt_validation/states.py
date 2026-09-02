"""Freeze deterministic S0/S1/S2 states through the real board runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from mas_cc.core import AgentId, InteractionId, Seed
from mas_cc.games import create_game
from mas_cc.games.protocols import Action
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    BOARD_PROMPT_FAMILY,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    _message_source,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.state import (
    ACTIVE_FACT_IDS,
    COMMITTED_ACTION,
    FOCAL_UPDATE,
    BlackboardMessage,
    BlackboardState,
    RelationalAgentState,
    RelationalGameState,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object
from mas_cc.probes.musr_symbolic_ambiguity.analysis import write_csv

from .config import BlackboardValidationConfig


@dataclass(frozen=True, slots=True)
class FrozenState:
    definition: Mapping[str, Any]
    state: RelationalGameState
    social_sources: tuple[Mapping[str, Any], ...]
    request: Any
    compiled_prompt: Any


def _previous_votes(source_root: Path) -> dict[tuple[str, int], str]:
    rows = json.loads(
        (source_root / "preflight/behavioral_call_plan.json").read_text(
            encoding="utf-8"
        )
    )
    call_ids = {
        str(row["call_id"])
        for row in rows
        if row["packet_variant"] == "Private" and int(row["repetition"]) == 0
    }
    latest: dict[str, Mapping[str, Any]] = {}
    for line in (
        (source_root / "behavioral_validation/raw_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        row = json.loads(line)
        if row.get("event") == "call_finished" and str(row.get("call_id")) in call_ids:
            latest[str(row["call_id"])] = row
    output = {}
    for call_id, row in latest.items():
        if row.get("parsed_semantic_answer") is None:
            raise RuntimeError(f"frozen previous-vote source is unparsable: {call_id}")
        output[(str(row["task_id"]), int(row["agent_id"]))] = str(
            row["parsed_semantic_answer"]
        )
    return output


def _latent_ids(task: RelationalTask, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(evidence_ids)
    return tuple(
        latent
        for latent, cards in (task.supporting_fact_groups or {}).items()
        if selected.intersection(cards)
    )


def _with_focal_vote(
    state: RelationalGameState, focal: AgentId, vote: str | None
) -> RelationalGameState:
    agents = []
    for agent in state.agents:
        assert isinstance(agent, RelationalAgentState)
        standing_vote = vote if agent.agent_id == focal else state.possible_answers[0]
        agent = replace(
            agent,
            attributes={
                **dict(agent.attributes),
                COMMITTED_ACTION: standing_vote,
                "public_reason": (
                    "PRIVATE_SENTINEL_MUST_NOT_RENDER"
                    if agent.agent_id == focal
                    else None
                ),
            },
        )
        agents.append(agent)
    return replace(
        state, agents=tuple(agents), data={**dict(state.data), "phase": FOCAL_UPDATE}
    )


def _acquire(
    game: Any,
    state: RelationalGameState,
    config: Any,
    focal: AgentId,
    message: BlackboardMessage,
    round_index: int,
) -> RelationalGameState:
    source = _message_source(state, message, round_index)
    before = state.relational_agent(focal).committed_action or state.possible_answers[0]
    action = Action(
        focal,
        before,
        FOCAL_UPDATE,
        {
            "reason": "private acquisition transition",
            "shared_fact_id": None,
            "public_message": None,
        },
    )
    return game.apply_round_event_transition(
        state,
        focal=focal,
        action=action,
        config=config,
        social_sources=(source,),
        round_fields={"round_index": round_index, "within_round_index": 0},
        board_fields={"sampled_message_ids": [message.message_id]},
    ).next_state


def _state_definition(
    config: BlackboardValidationConfig,
    task: RelationalTask,
    task_id: str,
    agent_id: str,
    state_id: str,
    previous_vote: str,
) -> FrozenState:
    game_config = config.game_config(task_id)
    game = create_game(game_config)
    state = game.initialize(
        game_config, int(Seed(config.state_seed).derive(f"state:{task_id}:{agent_id}"))
    )
    focal = AgentId(agent_id)
    state = _with_focal_vote(state, focal, None if state_id == "S0" else previous_vote)
    original = task.known_facts(agent_id)
    original_latents = set(_latent_ids(task, original))
    missing = [
        latent
        for latent in sorted(task.supporting_fact_groups or {})
        if latent not in original_latents
    ]
    target_coverage = {"S0": 4, "S1": 6, "S2": 9}[state_id]
    acquired: list[str] = []
    acquisition_messages: list[BlackboardMessage] = []
    for index, latent in enumerate(missing[: max(0, target_coverage - 4)], 1):
        fact_id = next(
            card
            for card in task.fact_order
            if card in set((task.supporting_fact_groups or {})[latent])
        )
        message = BlackboardMessage(
            message_id=f"{task_id}-{agent_id}-{state_id}-acq-{index:02d}",
            author_id=f"agent_{12 - index + 1:03d}",
            message_type="REPORT",
            text=f"I checked one exact source item relevant to {latent}.",
            vote=previous_vote,
            shared_fact_id=fact_id,
            reply_to=None,
            round_created=index - 1,
            micro_step_created=index,
            expires_after_round=index - 1,
        )
        state = _acquire(game, state, game_config, focal, message, index - 1)
        acquired.append(fact_id)
        acquisition_messages.append(message)
    live_round = len(acquisition_messages) + 1
    parent = BlackboardMessage(
        message_id=f"{task_id}-{agent_id}-{state_id}-parent",
        author_id="agent_011",
        message_type="REPORT",
        text="I compared the team allocations using the evidence available to me.",
        vote=previous_vote,
        shared_fact_id=None,
        reply_to=None,
        round_created=live_round,
        micro_step_created=100,
        expires_after_round=live_round,
    )
    reply = BlackboardMessage(
        message_id=f"{task_id}-{agent_id}-{state_id}-reply",
        author_id="agent_012",
        message_type="REPORT",
        text="That comparison also depends on cooperation within the paired role.",
        vote=previous_vote,
        shared_fact_id=None,
        reply_to=parent.message_id,
        round_created=live_round,
        micro_step_created=101,
        expires_after_round=live_round,
    )
    history = tuple(acquisition_messages)
    sampled: tuple[BlackboardMessage, ...] = ()
    if state_id == "S1":
        history = (*history, parent)
        sampled = (parent,)
    elif state_id == "S2":
        history = (*history, parent, reply)
        sampled = (reply,)
    state = replace(
        state,
        turn=live_round * 12,
        data={
            **dict(state.data),
            "phase": FOCAL_UPDATE,
            "blackboard": BlackboardState(history).to_list(),
        },
    )
    social_sources = tuple(
        _message_source(state, message, live_round) for message in sampled
    )
    request = game.ballot_request(
        state,
        focal,
        social_sources,
        game_config,
        stage=FOCAL_UPDATE,
        interaction_id=InteractionId(f"validation-{task_id}-{agent_id}-{state_id}"),
    )
    compiled = request.prompt.compile(RegexTokenCounter())
    total_evidence = state.relational_agent(focal).known_fact_ids
    latent_covered = _latent_ids(task, total_evidence)
    definition = {
        "task_id": task_id,
        "agent_id": agent_id,
        "state_id": state_id,
        "episode_seed": int(state.data["seed"]),
        "state_turn": state.turn,
        "current_vote": state.relational_agent(focal).committed_action,
        "original_evidence_ids": list(original),
        "acquired_evidence_ids": acquired,
        "total_evidence_ids": list(total_evidence),
        "latent_values_covered": list(latent_covered),
        "latent_coverage_count": len(latent_covered),
        "sampled_message_ids": [message.message_id for message in sampled],
        "sampled_message_types": [message.message_type for message in sampled],
        "sampled_message_texts": [message.text for message in sampled],
        "sampled_shared_fact_ids": [message.shared_fact_id for message in sampled],
        "reply_to_structure": {
            message.message_id: message.reply_to for message in sampled
        },
        "acquisition_message_ids": [
            message.message_id for message in acquisition_messages
        ],
        "expired_message_ids": [message.message_id for message in acquisition_messages],
        "board_history": [message.to_dict() for message in history],
        "social_sources": [dict(source) for source in social_sources],
        "option_letters": dict(request.observation.visible_state["option_letters"]),
        "prompt_family": compiled.family,
        "prompt_version": compiled.version,
        "prompt_definition_hash": compiled.definition_hash,
        "prompt_instance_hash": compiled.instance_hash,
        "prompt_messages": [message.to_dict() for message in compiled.messages],
        "response_contract_type": request.prompt.response_contract.type,
        "state_sha256": "",
    }
    definition["state_sha256"] = sha256_object(
        {key: value for key, value in definition.items() if key != "state_sha256"}
    )
    return FrozenState(definition, state, social_sources, request, compiled)


def build_frozen_states(
    config: BlackboardValidationConfig,
    tasks: Mapping[str, RelationalTask],
) -> dict[str, FrozenState]:
    votes = _previous_votes(config.calibration_root)
    states: dict[str, FrozenState] = {}
    for task_id in config.task_ids:
        for agent_id in config.agents_by_task[task_id]:
            agent_number = int(agent_id.rsplit("_", 1)[1])
            vote = votes[(task_id, agent_number)]
            for state_id in config.state_ids:
                frozen = _state_definition(
                    config, tasks[task_id], task_id, agent_id, state_id, vote
                )
                key = f"{task_id}:{agent_id}:{state_id}"
                states[key] = frozen
    if len(states) != config.state_count:
        raise RuntimeError("frozen state count differs from the configured design")
    return states


def write_state_artifacts(root: Path, states: Mapping[str, FrozenState]) -> None:
    definitions = [states[key].definition for key in sorted(states)]
    from mas_cc.musr_team_allocation_generator.io_utils import write_json_atomic

    write_json_atomic(root / "states/frozen_state_definitions.json", definitions)
    summary = [
        {
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "state_id": row["state_id"],
            "current_vote": row["current_vote"],
            "original_evidence_count": len(row["original_evidence_ids"]),
            "acquired_evidence_count": len(row["acquired_evidence_ids"]),
            "total_evidence_count": len(row["total_evidence_ids"]),
            "latent_coverage_count": row["latent_coverage_count"],
            "sampled_message_count": len(row["sampled_message_ids"]),
            "sampled_message_types": "|".join(row["sampled_message_types"]),
            "prompt_instance_hash": row["prompt_instance_hash"],
        }
        for row in definitions
    ]
    write_csv(root / "states/state_summary.csv", summary)
    examples = ["# Frozen rendered blackboard prompt examples", ""]
    for state_id in ("S0", "S1", "S2"):
        row = next(item for item in definitions if item["state_id"] == state_id)
        examples.extend(
            [
                f"## {state_id}: {row['task_id']} / {row['agent_id']}",
                "",
                f"Prompt family: `{row['prompt_family']}`",
                "",
                "```text",
                "\n\n".join(message["content"] for message in row["prompt_messages"]),
                "```",
                "",
            ]
        )
    (root / "states/rendered_prompt_examples.md").write_text(
        "\n".join(examples), encoding="utf-8"
    )


def load_frozen_states(
    config: BlackboardValidationConfig,
    tasks: Mapping[str, RelationalTask],
    path: Path,
) -> dict[str, FrozenState]:
    definitions = json.loads(path.read_text(encoding="utf-8"))
    states: dict[str, FrozenState] = {}
    for definition in definitions:
        task_id = str(definition["task_id"])
        agent_id = str(definition["agent_id"])
        state_id = str(definition["state_id"])
        if (
            task_id not in config.task_ids
            or agent_id not in config.agents_by_task[task_id]
            or state_id not in config.state_ids
        ):
            continue
        game_config = config.game_config(task_id)
        game = create_game(game_config)
        state = game.initialize(game_config, int(definition["episode_seed"]))
        focal = AgentId(agent_id)
        agents = []
        for agent in state.agents:
            if agent.agent_id == focal:
                assert isinstance(agent, RelationalAgentState)
                total = tuple(str(value) for value in definition["total_evidence_ids"])
                provenance = dict(agent.fact_provenance)
                for fact_id, message_id in zip(
                    definition["acquired_evidence_ids"],
                    definition["acquisition_message_ids"],
                    strict=True,
                ):
                    provenance[str(fact_id)] = {
                        "source": "peer",
                        "round_index": None,
                        "within_round_index": None,
                        "from": None,
                        "message_id": str(message_id),
                    }
                agent = replace(
                    agent,
                    attributes={
                        **dict(agent.attributes),
                        COMMITTED_ACTION: definition["current_vote"],
                        "public_reason": "PRIVATE_SENTINEL_MUST_NOT_RENDER",
                        "known_fact_ids": list(total),
                        ACTIVE_FACT_IDS: list(total),
                        "fact_provenance": provenance,
                    },
                )
            agents.append(agent)
        state = replace(
            state,
            turn=int(definition["state_turn"]),
            agents=tuple(agents),
            data={
                **dict(state.data),
                "phase": FOCAL_UPDATE,
                "blackboard": definition["board_history"],
            },
        )
        social_sources = tuple(dict(value) for value in definition["social_sources"])
        request = game.ballot_request(
            state,
            focal,
            social_sources,
            game_config,
            stage=FOCAL_UPDATE,
            interaction_id=InteractionId(f"validation-{task_id}-{agent_id}-{state_id}"),
        )
        compiled = request.prompt.compile(RegexTokenCounter())
        if (
            compiled.family != BOARD_PROMPT_FAMILY
            or compiled.instance_hash != definition["prompt_instance_hash"]
            or dict(request.observation.visible_state["option_letters"])
            != definition["option_letters"]
        ):
            raise RuntimeError(
                f"frozen state prompt identity changed: {task_id}/{agent_id}/{state_id}"
            )
        key = f"{task_id}:{agent_id}:{state_id}"
        states[key] = FrozenState(definition, state, social_sources, request, compiled)
    if len(states) != config.state_count:
        raise RuntimeError(
            "loaded frozen state count differs from the configured design"
        )
    return states


__all__ = [
    "FrozenState",
    "build_frozen_states",
    "load_frozen_states",
    "write_state_artifacts",
]
