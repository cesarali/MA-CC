"""Safe reconstruction of blackboard episode state from retained artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mas_cc.storage.dashboard_semantic import (
    read_semantic_stream,
    semantic_seal_path,
    semantic_stream_path,
    validate_semantic_stream,
)


SUPPORTED_GAME = "relational_imitation_round_feedback"
SUPPORTED_RUN_SCHEMA_VERSIONS = {1}


@dataclass(frozen=True, slots=True)
class BlackboardSourceDescriptor:
    """Explicit description of one supported dashboard source."""

    source_kind: str
    game_type: str
    run_schema_version: int
    artifact_profile: str
    protocol: str


def _jsonl(path: Path, *, completed: bool = False) -> list[dict[str, Any]]:
    """Read complete JSONL records without consuming a partial writer tail."""

    if not path.is_file():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n") and completed:
        raise ValueError(
            f"completed JSONL artifact has a partial trailing record: {path}"
        )
    if raw and not raw.endswith(b"\n"):
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    rows = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL record {path}:{number}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _event(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("event", row)
    return dict(value) if isinstance(value, dict) else {}


def _safe_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"required dashboard artifact is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(value)


def _detect_source(
    run_dir: Path,
    episode_dir: Path,
    run_manifest: dict[str, Any],
    episode_manifest: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> BlackboardSourceDescriptor:
    """Reject unknown layouts instead of displaying plausible-but-wrong data."""

    game_type_value = (
        run_manifest.get("game_type")
        or episode_manifest.get("game_type")
        or (rounds[0].get("record_type") if rounds else "")
    )
    game_type = "" if game_type_value is None else str(game_type_value)
    if game_type and game_type != SUPPORTED_GAME:
        raise ValueError(f"unsupported dashboard game type: {game_type!r}")
    schema = int(run_manifest.get("schema_version", 1))
    if schema not in SUPPORTED_RUN_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported dashboard run schema version: {schema}")
    profile = str(run_manifest.get("artifact_profile", "full"))
    if profile not in {"full", "dashboard_semantic"}:
        raise ValueError(
            "blackboard dashboard requires artifact_profile 'full'; "
            f"{profile!r} does not retain complete prompts and evidence state"
        )
    if profile == "full" and not (episode_dir / "trajectory.jsonl").is_file():
        raise ValueError(f"full blackboard trajectory is missing beneath {episode_dir}")
    protocols = {str(row.get("protocol")) for row in rounds if row.get("protocol")}
    if len(protocols) > 1:
        raise ValueError("round records contain multiple controller protocols")
    protocol = next(iter(protocols), "legacy")
    return BlackboardSourceDescriptor(
        source_kind=(
            "relational_blackboard_semantic"
            if profile == "dashboard_semantic"
            else "relational_blackboard_full"
        ),
        game_type=game_type or SUPPORTED_GAME,
        run_schema_version=schema,
        artifact_profile=profile,
        protocol=protocol,
    )


def _validate_records(
    trajectory: list[dict[str, Any]], rounds: list[dict[str, Any]], *, completed: bool
) -> None:
    if not completed:
        return
    global_ids = [
        int(row.get("global_update_index", index))
        for index, row in enumerate(trajectory)
    ]
    if len(global_ids) != len(set(global_ids)):
        raise ValueError(
            "completed trajectory contains duplicate global update indices"
        )
    if global_ids and global_ids != list(
        range(global_ids[0], global_ids[0] + len(global_ids))
    ):
        raise ValueError(
            "completed trajectory global update indices are not contiguous"
        )
    cursor_ids = [
        (int(row.get("round_index", -1)), int(row.get("within_round_index", -1)))
        for row in trajectory
    ]
    if len(cursor_ids) != len(set(cursor_ids)):
        raise ValueError("completed trajectory contains duplicate round/update cursors")
    round_ids = [int(row.get("round_index", index)) for index, row in enumerate(rounds)]
    if len(round_ids) != len(set(round_ids)):
        raise ValueError("completed round trajectory contains duplicate round indices")


def resolve_episode_dir(run_dir: str | Path, episode_id: str | None = None) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if (root / "trajectory.jsonl").is_file() or root.name == episode_id:
        return root
    candidates = sorted(
        {
            path.resolve()
            for episodes_root in (
                root / "data" / "episodes",
                *(root / "cells").glob("*/data/episodes"),
                root / "round_records",
                *(root / "cells").glob("*/round_records"),
            )
            for path in episodes_root.glob("*")
            if path.is_dir()
            and (
                (path / "trajectory.jsonl").is_file()
                or semantic_stream_path(path).is_file()
            )
        },
        key=str,
    )
    if episode_id is not None:
        candidates = [path for path in candidates if path.name == episode_id]
    if not candidates:
        raise ValueError(f"no episode artifacts found beneath {root}")
    if len(candidates) > 1:
        raise ValueError("run contains multiple episodes; pass --episode-id")
    return candidates[0]


class BlackboardRunReader:
    """Produce versioned dashboard snapshots without modifying run artifacts."""

    def __init__(self, run_dir: str | Path, episode_id: str | None = None) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.episode_dir = resolve_episode_dir(self.run_dir, episode_id)
        self._cache_signature: tuple[tuple[str, int, int], ...] | None = None
        self._cache: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        semantic_path = semantic_stream_path(self.episode_dir)
        semantic_seal = semantic_seal_path(self.episode_dir)
        paths = (
            self.episode_dir / "trajectory.jsonl",
            self.episode_dir / "round_trajectory.jsonl",
            self.episode_dir / "audit_traces.jsonl",
            self.episode_dir / "api_call_status.jsonl",
            self.episode_dir / "usage_cost.jsonl",
            self.episode_dir / "budget_events.jsonl",
            self.episode_dir / ".checkpoints" / "checkpoint.json",
            self.episode_dir / "manifest.json",
            self.run_dir / "manifest.json",
            semantic_path,
            semantic_seal,
        )
        signature = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
            if path.is_file()
        )
        if signature == self._cache_signature and self._cache is not None:
            return self._cache
        episode_manifest = _safe_json(self.episode_dir / "manifest.json")
        run_manifest = _safe_json(self.run_dir / "manifest.json")
        if semantic_path.is_file():
            loaded = self._load_semantic(run_manifest)
            self._cache_signature = signature
            self._cache = loaded
            return loaded
        completed = episode_manifest.get("status") in {"completed", "skipped_resumed"}
        trajectory = [
            _event(row)
            for row in _jsonl(
                self.episode_dir / "trajectory.jsonl", completed=completed
            )
        ]
        rounds = [
            _event(row)
            for row in _jsonl(
                self.episode_dir / "round_trajectory.jsonl", completed=completed
            )
        ]
        audits = _jsonl(self.episode_dir / "audit_traces.jsonl", completed=completed)
        api_calls = _jsonl(
            self.episode_dir / "api_call_status.jsonl", completed=completed
        )
        usage = _jsonl(self.episode_dir / "usage_cost.jsonl", completed=completed)
        budget = _jsonl(self.episode_dir / "budget_events.jsonl", completed=completed)
        checkpoint = _safe_json(self.episode_dir / ".checkpoints" / "checkpoint.json")
        source = _detect_source(
            self.run_dir, self.episode_dir, run_manifest, episode_manifest, rounds
        )
        _validate_records(trajectory, rounds, completed=completed)
        state = checkpoint.get("state", {})
        if not isinstance(state, dict):
            state = {}
        task = state.get("task", {})
        if not isinstance(task, dict):
            task = {}
        loaded = {
            "trajectory": trajectory,
            "rounds": rounds,
            "audits": audits,
            "api_calls": api_calls,
            "usage": usage,
            "budget": budget,
            "source": source,
            "checkpoint": checkpoint,
            "state": state,
            "task": task,
            "episode_manifest": episode_manifest,
            "run_manifest": run_manifest,
            "message_ledger": {},
            "message_conflicts": [],
            "attempt_support": {
                "status": "selected_full_artifact",
                "selected_attempt": "durable" if completed else "current",
                "warning": (
                    "Historical episode attempts have no stable retained attempt ID; "
                    "they were not combined."
                ),
            },
        }
        self._cache_signature = signature
        self._cache = loaded
        return loaded

    def _load_semantic(self, run_manifest: dict[str, Any]) -> dict[str, Any]:
        """Adapt the lean semantic stream to the established replay model."""

        seal = _safe_json(semantic_seal_path(self.episode_dir))
        completed = seal.get("status") == "completed"
        rows = (
            validate_semantic_stream(self.episode_dir)
            if completed
            else read_semantic_stream(semantic_stream_path(self.episode_dir))
        )
        header = next((row for row in rows if row.get("record_type") == "header"), {})
        initialization = next(
            (row for row in rows if row.get("record_type") == "initialization"), {}
        )
        trajectory = [dict(row) for row in rows if row.get("record_type") == "update"]
        validations = [
            {
                "agent_id": row.get("agent_id"),
                "attempt": row.get("attempt"),
                "decision_stage": "focal_update",
                "interaction_id": row.get("interaction_id"),
                "valid": row.get("valid"),
                "validation_issues": row.get("issue_codes", []),
                "semantic_only": True,
            }
            for row in rows
            if row.get("record_type") == "validation"
        ]
        initial_agents = [
            value
            for value in initialization.get("agents", ())
            if isinstance(value, dict)
        ]
        agent_ids = [str(value.get("agent_id")) for value in initial_agents]
        vote_by_agent = {
            str(value.get("agent_id")): str(value.get("vote"))
            for value in initial_agents
        }
        for row in trajectory:
            before = [vote_by_agent.get(agent) for agent in agent_ids]
            focal = str(row.get("focal_agent_id"))
            vote_by_agent[focal] = str(row.get("focal_vote_after"))
            row["population_state_before"] = before
            row["population_state_after"] = [
                vote_by_agent.get(agent) for agent in agent_ids
            ]
        round_ends = {
            int(row.get("round_index", 0)): row
            for row in rows
            if row.get("record_type") == "round_end"
        }
        round_starts = {
            int(row.get("round_index", 0)): row
            for row in rows
            if row.get("record_type") == "round_start"
        }
        message_ledger: dict[str, dict[str, Any]] = {}
        message_conflicts: set[str] = set()
        for row in rows:
            candidates: list[Any] = []
            if row.get("record_type") in {"initialization", "round_start", "round_end"}:
                candidates.extend(row.get("board", ()))
            if row.get("record_type") == "update":
                candidates.extend(
                    (row.get("controller_message"), row.get("new_message"))
                )
            for value in candidates:
                if not isinstance(value, dict) or not value.get("message_id"):
                    continue
                message_id = str(value["message_id"])
                normalized = dict(value)
                previous = message_ledger.get(message_id)
                if previous is not None and previous != normalized:
                    message_conflicts.add(message_id)
                    continue
                message_ledger[message_id] = normalized
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in trajectory:
            grouped.setdefault(int(row.get("round_index", 0)), []).append(row)
        rounds = []
        observed_rounds = sorted(set(grouped) | set(round_starts) | set(round_ends))
        for round_index in observed_rounds:
            updates = grouped.get(round_index, [])
            end = round_ends.get(round_index, {})
            start = round_starts.get(round_index, {})
            controller = start.get("controller") or {}
            end_agents = {
                str(value.get("agent_id")): value
                for value in end.get("agents", ())
                if isinstance(value, dict)
            }
            first = updates[0] if updates else start
            last = updates[-1] if updates else end or start
            rounds.append(
                {
                    **last,
                    "round_index": round_index,
                    "N": len(agent_ids),
                    "agent_ids": agent_ids,
                    "population_state_before": first.get(
                        "population_state_before",
                        initialization.get("population_votes", []),
                    ),
                    "population_state_after": last.get(
                        "population_state_after", end.get("population_votes", [])
                    ),
                    "initial_active_fact_ids_by_agent": [
                        list(value.get("active_fact_ids", ()))
                        for value in initial_agents
                    ],
                    "initial_known_fact_ids_by_agent": [
                        list(value.get("known_fact_ids", ()))
                        for value in initial_agents
                    ],
                    "active_fact_ids_by_agent_after": {
                        agent: list(value.get("active_fact_ids", ()))
                        for agent, value in end_agents.items()
                    },
                    "known_fact_ids_by_agent_after": {
                        agent: list(value.get("known_fact_ids", ()))
                        for agent, value in end_agents.items()
                    },
                    "persistence_deactivated_pairs": start.get("deactivated_pairs", []),
                    "semantic_board_after": end.get("board", []),
                    "controller_enabled": controller.get("enabled"),
                    "controller_action": controller.get("action"),
                    "controller_target": controller.get("target"),
                    "controller_probability_U1_given_Y": controller.get("probability"),
                    "controller_sensor_Y": controller.get("sensor"),
                    "controller_post_ids": controller.get(
                        "post_ids", controller.get("directive_ids", [])
                    ),
                    "controller_request_ids": controller.get("request_ids", []),
                    "controller_report_ids": controller.get("report_ids", []),
                    "controller_directive_ids": controller.get("directive_ids", []),
                    "controller_selected_fact_ids": controller.get(
                        "selected_fact_ids", []
                    ),
                    "controller_message_mode": controller.get("chosen_message_mode"),
                    "controller_requested_b": controller.get("requested_b"),
                    "controller_actual_posts": controller.get("actual_posts"),
                    "controller_choice_source": controller.get(
                        "communication_choice_source"
                    ),
                    "controller_llm_fallback_used": controller.get("llm_fallback_used"),
                    "protocol": header.get("protocol_version"),
                }
            )
        fact_ids = sorted(
            {
                str(fact)
                for agent in initial_agents
                for key in ("active_fact_ids", "known_fact_ids")
                for fact in agent.get(key, ())
            }
        )
        state = {
            "agents": [
                {
                    "agent_id": value.get("agent_id"),
                    "committed_action": value.get("vote"),
                    "active_fact_ids": value.get("active_fact_ids", []),
                    "known_fact_ids": value.get("known_fact_ids", []),
                }
                for value in initial_agents
            ],
            "blackboard": initialization.get("board", []),
            "rules": {
                "n_agents": header.get("population_size"),
                "rounds": header.get("rounds"),
            },
            "task": {
                "correct_answer": initialization.get("correct_answer"),
                "facts": {
                    fact: {"text": "Not retained by dashboard_semantic profile"}
                    for fact in fact_ids
                },
                "supporting_fact_groups": {fact: [fact] for fact in fact_ids},
            },
        }
        terminal = next(
            (row for row in reversed(rows) if row.get("record_type") == "completion"),
            {},
        )
        semantic_manifest = {
            "episode_id": header.get("episode_id", self.episode_dir.name),
            "status": "completed" if completed else terminal.get("status", "running"),
            "seed": header.get("episode_seed"),
        }
        semantic_run_manifest = {
            **run_manifest,
            "schema_version": int(run_manifest.get("schema_version", 1)),
            "game_type": header.get("game_type", SUPPORTED_GAME),
            "artifact_profile": "dashboard_semantic",
        }
        source = _detect_source(
            self.run_dir,
            self.episode_dir,
            semantic_run_manifest,
            semantic_manifest,
            rounds,
        )
        _validate_records(trajectory, rounds, completed=completed)
        return {
            "trajectory": trajectory,
            "rounds": rounds,
            "audits": validations,
            "api_calls": [],
            "usage": [],
            "budget": [],
            "source": source,
            "checkpoint": {},
            "state": state,
            "task": state["task"],
            "episode_manifest": semantic_manifest,
            "run_manifest": semantic_run_manifest,
            "message_ledger": message_ledger,
            "message_conflicts": sorted(message_conflicts),
            "attempt_support": {
                "status": "durable_single_stream"
                if completed
                else "current_single_stream",
                "selected_attempt": "durable" if completed else "current",
                "warning": (
                    "Historical episode attempts have no stable retained attempt ID; "
                    "they were not combined."
                ),
            },
        }

    @staticmethod
    def _agents(data: dict[str, Any]) -> list[str]:
        rounds = data["rounds"]
        if rounds:
            return [str(value) for value in rounds[0].get("agent_ids", [])]
        return [
            str(agent.get("agent_id"))
            for agent in data["state"].get("agents", [])
            if isinstance(agent, dict) and agent.get("agent_id") is not None
        ]

    @staticmethod
    def _initial_maps(
        data: dict[str, Any], agents: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        rounds = data["rounds"]
        if rounds:
            first = rounds[0]
            active_values = first.get("initial_active_fact_ids_by_agent", [])
            known_values = first.get("initial_known_fact_ids_by_agent", [])
            active = {
                agent: list(active_values[index]) if index < len(active_values) else []
                for index, agent in enumerate(agents)
            }
            known = {
                agent: list(known_values[index]) if index < len(known_values) else []
                for index, agent in enumerate(agents)
            }
            votes = [str(vote) for vote in first.get("population_state_before", [])]
            return active, known, votes
        state_agents = {
            str(agent.get("agent_id")): agent
            for agent in data["state"].get("agents", [])
            if isinstance(agent, dict)
        }
        active = {
            agent: list(state_agents.get(agent, {}).get("active_fact_ids", []))
            for agent in agents
        }
        known = {
            agent: list(state_agents.get(agent, {}).get("known_fact_ids", []))
            for agent in agents
        }
        votes = [
            str(state_agents.get(agent, {}).get("committed_action")) for agent in agents
        ]
        return active, known, votes

    @staticmethod
    def _author_category(message: dict[str, Any]) -> str:
        return (
            "controller"
            if str(message.get("author_kind", "")).lower() == "controller"
            or str(message.get("author_id", "")).lower() == "controller"
            else "participants"
        )

    @staticmethod
    def _empty_message_counts() -> dict[str, int]:
        return {
            "REPORT": 0,
            "REQUEST": 0,
            "DIRECTIVE": 0,
            "replies": 0,
            "no_message_actions": 0,
            "total_messages": 0,
        }

    def _message_ledger(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        ledger = dict(data.get("message_ledger", {}))
        for message in data["state"].get("blackboard", ()):
            if isinstance(message, dict) and message.get("message_id"):
                ledger.setdefault(str(message["message_id"]), dict(message))
        for row in data["rounds"]:
            for message in row.get("semantic_board_after", ()):
                if isinstance(message, dict) and message.get("message_id"):
                    ledger.setdefault(str(message["message_id"]), dict(message))
        for event in data["trajectory"]:
            for key in ("controller_message", "new_message"):
                message = event.get(key)
                if isinstance(message, dict) and message.get("message_id"):
                    ledger.setdefault(str(message["message_id"]), dict(message))
        return ledger

    def _controller_event_types(
        self, row: dict[str, Any], data: dict[str, Any]
    ) -> list[str]:
        ledger = self._message_ledger(data)
        types = []
        for message_id in row.get("controller_post_ids", ()):
            message_type = ledger.get(str(message_id), {}).get("message_type")
            if message_type and str(message_type) not in types:
                types.append(str(message_type))
        mode = row.get("controller_message_mode")
        if not types and mode in {"REPORT", "REQUEST", "DIRECTIVE"}:
            types.append(str(mode))
        if not types and str(row.get("controller_action")) == "NO_OP":
            types.append("NO_OP")
        return types

    def communication(self) -> dict[str, Any]:
        """Return deduplicated communication counts and exact retained messages."""

        data = self._load()
        agents = self._agents(data)
        ledger = self._message_ledger(data)
        round_ids = sorted(
            {int(row.get("round_index", 0)) for row in data["rounds"]}
            | {int(row.get("round_index", 0)) for row in data["trajectory"]}
        )
        scope_ids = [
            "all",
            "participants",
            "controller",
            *[f"agent:{a}" for a in agents],
        ]
        by_round = {
            index: {scope: self._empty_message_counts() for scope in scope_ids}
            for index in round_ids
        }

        def increment(round_index: int, scopes: list[str], key: str) -> None:
            by_round.setdefault(
                round_index,
                {scope: self._empty_message_counts() for scope in scope_ids},
            )
            for scope in scopes:
                by_round[round_index][scope][key] += 1

        messages = []
        for message in sorted(
            ledger.values(),
            key=lambda item: (
                int(item.get("round_created") or 0),
                int(item.get("micro_step_created") or 0),
                str(item.get("message_id")),
            ),
        ):
            category = self._author_category(message)
            author = str(message.get("author_id", ""))
            message_type = str(message.get("message_type") or "").upper()
            round_index = int(message.get("round_created") or 0)
            scopes = ["all", category]
            if category == "participants" and author in agents:
                scopes.append(f"agent:{author}")
            if message_type in {"REPORT", "REQUEST", "DIRECTIVE"}:
                increment(round_index, scopes, message_type)
                increment(round_index, scopes, "total_messages")
            if message.get("reply_to"):
                increment(round_index, scopes, "replies")
            messages.append({**message, "author_category": category})

        for event in data["trajectory"]:
            message = event.get("new_message")
            message_type = (
                str(message.get("message_type", "")).upper()
                if isinstance(message, dict)
                else "NONE"
            )
            if not isinstance(message, dict) or message_type in {"", "NONE"}:
                author = str(event.get("focal_agent_id", ""))
                scopes = ["all", "participants"]
                if author in agents:
                    scopes.append(f"agent:{author}")
                increment(
                    int(event.get("round_index", 0)), scopes, "no_message_actions"
                )
        for row in data["rounds"]:
            if str(row.get("controller_action")) == "NO_OP":
                increment(
                    int(row.get("round_index", 0)),
                    ["all", "controller"],
                    "no_message_actions",
                )

        totals = {scope: self._empty_message_counts() for scope in scope_ids}
        for scopes in by_round.values():
            for scope, counts in scopes.items():
                for key, value in counts.items():
                    totals[scope][key] += value
        observed = len(by_round)
        return {
            "schema_version": 1,
            "available": bool(by_round or messages),
            "warning": (
                "Conflicting duplicate message IDs were retained only once: "
                + ", ".join(data["message_conflicts"])
                if data.get("message_conflicts")
                else None
            ),
            "participant_scopes": [
                {"id": "all", "label": "All participants"},
                {"id": "participants", "label": "Participants only"},
                {"id": "controller", "label": "Controller only"},
                *[
                    {"id": f"agent:{agent}", "label": agent.replace("agent_", "Agent ")}
                    for agent in agents
                ],
            ],
            "observed_rounds": observed,
            "counts_by_round": [
                {"round_index": index, "scopes": scopes}
                for index, scopes in sorted(by_round.items())
            ],
            "totals_by_scope": totals,
            "totals_by_author_category": {
                category: {
                    **totals[category],
                    "messages_per_observed_round": (
                        totals[category]["total_messages"] / observed
                        if observed
                        else None
                    ),
                }
                for category in ("participants", "controller")
            },
            "messages": messages,
        }

    def controller_timeline(self) -> dict[str, Any]:
        """Return round controller actions and ordered descriptive responses."""

        data = self._load()
        ledger = self._message_ledger(data)
        result = []
        for row in data["rounds"]:
            round_index = int(row.get("round_index", 0))
            updates = [
                event
                for event in data["trajectory"]
                if int(event.get("round_index", 0)) == round_index
            ]
            exposed = [
                event
                for event in updates
                if event.get("sampled_controller_message_ids", ())
            ]
            target = row.get("controller_target")
            post_ids = [str(value) for value in row.get("controller_post_ids", ())]
            direct_exposure_values = [
                event.get("controller_message_directly_exposed") for event in updates
            ]
            result.append(
                {
                    "round_index": round_index,
                    "enabled": row.get("controller_enabled"),
                    "action": row.get("controller_action"),
                    "target": target,
                    "message_types": self._controller_event_types(row, data),
                    "post_ids": post_ids,
                    "posts": [ledger[value] for value in post_ids if value in ledger],
                    "requested_posts": row.get("controller_requested_b"),
                    "realized_posts": row.get("controller_actual_posts")
                    if row.get("controller_actual_posts") is not None
                    else len(post_ids),
                    "llm_fallback_used": row.get("controller_llm_fallback_used"),
                    "recorded_exposure_events": len(exposed),
                    "direct_exposure_events": (
                        sum(bool(value) for value in direct_exposure_values)
                        if any(value is not None for value in direct_exposure_values)
                        else None
                    ),
                    "exposed_agents": len(
                        {str(event.get("focal_agent_id")) for event in exposed}
                    ),
                    "next_votes_moved_to_target": (
                        sum(
                            str(event.get("focal_vote_before")) != str(target)
                            and str(event.get("focal_vote_after")) == str(target)
                            for event in exposed
                        )
                        if target is not None
                        else None
                    ),
                }
            )
        controller_rounds = [
            row
            for row in result
            if row["enabled"] is True or row["action"] not in {None, "", "DISABLED"}
        ]
        if not controller_rounds:
            return {
                "schema_version": 1,
                "available": False,
                "summary": {},
                "rounds": [],
            }
        posts = [post for row in controller_rounds for post in row["posts"]]
        count = len(controller_rounds)
        advocate = sum(
            row["action"] in {"ADVOCATE", "ADVOCATE_Z"} for row in controller_rounds
        )
        no_op = sum(row["action"] == "NO_OP" for row in controller_rounds)
        requested = [
            row["requested_posts"]
            for row in controller_rounds
            if row["requested_posts"] is not None
        ]
        realized = [
            row["realized_posts"]
            for row in controller_rounds
            if row["realized_posts"] is not None
        ]
        return {
            "schema_version": 1,
            "available": True,
            "summary": {
                "rounds_observed": count,
                "advocate_count": advocate,
                "advocate_fraction": advocate / count,
                "no_op_count": no_op,
                "no_op_fraction": no_op / count,
                "reports": sum(
                    str(post.get("message_type")) == "REPORT" for post in posts
                ),
                "requests": sum(
                    str(post.get("message_type")) == "REQUEST" for post in posts
                ),
                "directives": sum(
                    str(post.get("message_type")) == "DIRECTIVE" for post in posts
                ),
                "total_posts": len(posts),
                "requested_posts": sum(int(value) for value in requested)
                if requested
                else None,
                "realized_posts": sum(int(value) for value in realized)
                if realized
                else None,
                "llm_fallback_count": sum(
                    row["llm_fallback_used"] is True for row in controller_rounds
                ),
                "recorded_exposure_events": sum(
                    row["recorded_exposure_events"] for row in controller_rounds
                ),
                "direct_exposure_events": (
                    sum(
                        int(row["direct_exposure_events"])
                        for row in controller_rounds
                        if row["direct_exposure_events"] is not None
                    )
                    if any(
                        row["direct_exposure_events"] is not None
                        for row in controller_rounds
                    )
                    else None
                ),
                "unique_exposed_agents": len(
                    {
                        str(event.get("focal_agent_id"))
                        for event in data["trajectory"]
                        if event.get("sampled_controller_message_ids", ())
                    }
                ),
            },
            "rounds": controller_rounds,
        }

    def timeline(self) -> dict[str, Any]:
        data = self._load()
        trajectory = data["trajectory"]
        agents = self._agents(data)
        truth = data["state"].get("task", {}).get("correct_answer")
        controller_target = next(
            (
                row.get("controller_target")
                for row in data["rounds"]
                if row.get("controller_target") is not None
            ),
            None,
        )
        rounds: dict[int, int] = {}
        for event in trajectory:
            round_index = int(event.get("round_index", 0))
            rounds[round_index] = max(
                rounds.get(round_index, 0), int(event.get("within_round_index", 0)) + 1
            )
        cursors = [
            {
                "phase": "round",
                "round_index": int(event.get("round_index", 0)),
                "step": int(event.get("within_round_index", 0)) + 1,
                "global_update_index": int(event.get("global_update_index", index)),
            }
            for index, event in enumerate(trajectory)
        ]
        time_series = []
        controller_markers = {
            int(row["round_index"]): self._controller_event_types(row, data)
            for row in data["rounds"]
        }
        for index, event in enumerate(trajectory):
            votes = [str(value) for value in event.get("population_state_after", ())]
            total = len(votes)
            time_series.append(
                {
                    "global_update_index": int(event.get("global_update_index", index)),
                    "round_index": int(event.get("round_index", 0)),
                    "step": int(event.get("within_round_index", 0)) + 1,
                    "truth_share": (
                        votes.count(str(truth)) / total
                        if truth is not None and total
                        else None
                    ),
                    "controller_target_share": (
                        votes.count(str(controller_target)) / total
                        if controller_target is not None and total
                        else None
                    ),
                    "other_share": (
                        sum(
                            vote not in {str(truth), str(controller_target)}
                            for vote in votes
                        )
                        / total
                        if total and truth is not None and controller_target is not None
                        else None
                    ),
                    "truth_and_target_same": (
                        truth is not None
                        and controller_target is not None
                        and str(truth) == str(controller_target)
                    ),
                    "board_size": event.get("board_size_after"),
                    "controller_post": bool(event.get("controller_message_posted")),
                    "controller_markers": controller_markers.get(
                        int(event.get("round_index", 0)), []
                    )
                    if int(event.get("within_round_index", 0)) == 0
                    else [],
                    "controller_exposures": sum(
                        1 for value in event.get("sampled_controller_message_ids", ())
                    ),
                }
            )
        return {
            "schema_version": 1,
            "source": asdict(data["source"]),
            "episode_id": self.episode_dir.name,
            "agents": agents,
            "rounds": [
                {"round_index": key, "available_steps": value}
                for key, value in sorted(rounds.items())
            ],
            "available_cursors": cursors,
            "time_series": time_series,
            "initialization_attempts": sum(
                row.get("decision_stage") == "initial_vote" for row in data["audits"]
            ),
            "statistics": self.statistics(),
            "communication": self.communication(),
            "controller_timeline": self.controller_timeline(),
            "attempt_support": data["attempt_support"],
        }

    def statistics(self) -> dict[str, Any]:
        """Compact descriptive diagnostics from this episode's retained records."""

        data = self._load()
        updates = data["trajectory"]
        rounds = data["rounds"]
        actions = [str(row.get("controller_action", "")) for row in rounds]
        posts = sum(len(row.get("controller_post_ids", ())) for row in rounds)
        exposures = sum(
            len(row.get("sampled_controller_message_ids", ())) for row in updates
        )
        readers = {
            str(row.get("focal_agent_id"))
            for row in updates
            if row.get("sampled_controller_message_ids", ())
        }
        board_sizes = [
            int(value)
            for row in updates
            for value in (row.get("board_size_before"), row.get("board_size_after"))
            if value is not None
        ]
        acquired = sum(
            len(row.get("new_peer_fact_ids", ()))
            + len(row.get("new_controller_fact_ids", ()))
            for row in updates
        )
        reactivated = sum(
            len(row.get("reactivated_peer_fact_ids", ()))
            + len(row.get("reactivated_controller_fact_ids", ()))
            for row in updates
        )
        opportunities = sum(bool(row.get("controller_enabled")) for row in rounds)
        return {
            "microscopic_updates": len(updates),
            "controller_opportunities": opportunities,
            "controller_advocate_rounds": sum(
                action in {"ADVOCATE", "ADVOCATE_Z"} for action in actions
            ),
            "controller_no_op_rounds": sum(action == "NO_OP" for action in actions),
            "controller_posts": posts,
            "controller_message_exposures": exposures,
            "controller_unique_readers": len(readers),
            "controller_exposed_update_fraction": exposures / len(updates)
            if updates
            else None,
            "board_peak_occupancy": max(board_sizes) if board_sizes else None,
            "board_mean_occupancy": sum(board_sizes) / len(board_sizes)
            if board_sizes
            else None,
            "fact_acquisitions": acquired,
            "fact_reactivations": reactivated,
            "controller_report_fact_acquisitions": sum(
                len(row.get("new_controller_fact_ids", ())) for row in updates
            ),
            "controller_report_fact_reactivations": sum(
                len(row.get("reactivated_controller_fact_ids", ())) for row in updates
            ),
            "controller_report_target_adoptions": sum(
                bool(row.get("sampled_controller_report_ids"))
                and bool(row.get("focal_adopted_target"))
                for row in updates
            ),
            "validation_repairs": sum(
                int(row.get("attempt", 1)) > 1 for row in data["audits"]
            ),
            "malformed_terminal": bool(
                data["audits"] and not data["audits"][-1].get("valid", True)
            ),
            "actuation_semantics": (
                "ordinary truthful REPORT publication"
                if any(
                    row.get("controller_actuation_mode") == "truthful_strategic_report"
                    for row in rounds
                )
                else "shared-board publication; saturation/attention competition"
            ),
        }

    def status(self) -> dict[str, Any]:
        data = self._load()
        trajectory = data["trajectory"]
        manifest = data["episode_manifest"]
        last = trajectory[-1] if trajectory else {}
        expected = 0
        rules = data["state"].get("rules", {})
        if isinstance(rules, dict) and rules.get("n_agents") is not None:
            expected = int(rules.get("n_agents", 0)) * (
                0 if rules.get("initialization_only") else int(rules.get("rounds", 0))
            )
        elif data["rounds"]:
            expected = int(data["rounds"][0].get("N", 0)) * max(
                len(data["rounds"]), int(last.get("round_index", -1)) + 1
            )
        return {
            "schema_version": 1,
            "source": asdict(data["source"]),
            "episode_id": self.episode_dir.name,
            "status": manifest.get(
                "status", "running" if trajectory else "waiting_for_writer"
            ),
            "completed_updates": len(trajectory),
            "expected_updates": expected or None,
            "prompt_attempts": len(data["audits"]),
            "provider_attempts": len(data["api_calls"]),
            "invalid_attempts": sum(
                not bool(row.get("valid", False))
                for row in (
                    data["audits"]
                    if data["source"].artifact_profile == "dashboard_semantic"
                    else data["api_calls"]
                )
            ),
            "provider_status": {
                "available": bool(data["api_calls"]),
                "attempts": len(data["api_calls"]),
                "usage_records": len(data["usage"]),
                "budget_records": len(data["budget"]),
            },
            "latest_cursor": None
            if not trajectory
            else {
                "round_index": int(last.get("round_index", 0)),
                "step": int(last.get("within_round_index", 0)) + 1,
                "global_update_index": int(
                    last.get("global_update_index", len(trajectory) - 1)
                ),
            },
        }

    def snapshot(
        self,
        round_index: int | None = None,
        step: int | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        semantic = data["source"].artifact_profile == "dashboard_semantic"
        unavailable_reason = "Not retained by dashboard_semantic profile"
        trajectory = data["trajectory"]
        if not trajectory:
            return {
                "schema_version": 1,
                "run": self.status(),
                "cursor": None,
                "population": {},
                "blackboard": [],
                "coverage": {},
                "agent": {},
                "controller": {},
                "evidence_events": [],
                "available_cursors": [],
            }
        if round_index is None:
            target = trajectory[-1]
        else:
            candidates = [
                event
                for event in trajectory
                if int(event.get("round_index", -1)) == round_index
                and (
                    step is None or int(event.get("within_round_index", -1)) + 1 <= step
                )
            ]
            if not candidates:
                raise ValueError(
                    f"no recorded update for round {round_index + 1}, step {step}"
                )
            target = candidates[-1]
        target_round = int(target.get("round_index", 0))
        target_step = int(target.get("within_round_index", 0)) + 1
        target_global = int(target.get("global_update_index", trajectory.index(target)))
        agents = self._agents(data)
        active, known, votes = self._initial_maps(data, agents)
        rounds_by_index = {
            int(record.get("round_index", index)): record
            for index, record in enumerate(data["rounds"])
        }
        for previous_round in range(target_round):
            record = rounds_by_index.get(previous_round, {})
            active.update(
                {
                    str(key): list(value)
                    for key, value in record.get(
                        "active_fact_ids_by_agent_after", {}
                    ).items()
                }
            )
            known.update(
                {
                    str(key): list(value)
                    for key, value in record.get(
                        "known_fact_ids_by_agent_after", {}
                    ).items()
                }
            )
            votes = [
                str(value) for value in record.get("population_state_after", votes)
            ]
        round_record = rounds_by_index.get(target_round, {})
        if round_record.get("controller_timing") == "dawn_only" or (
            semantic and round_record.get("persistence_deactivated_pairs")
        ):
            for pair in round_record.get("persistence_deactivated_pairs", []):
                if not isinstance(pair, dict):
                    continue
                deactivated_agent = str(pair.get("agent_id"))
                fact_id = str(pair.get("fact_id"))
                active[deactivated_agent] = [
                    value
                    for value in active.get(deactivated_agent, [])
                    if value != fact_id
                ]
        selected_events = [
            event
            for event in trajectory
            if int(event.get("round_index", -1)) == target_round
            and int(event.get("within_round_index", -1)) + 1 <= target_step
        ]
        for event in selected_events:
            focal = str(event.get("focal_agent_id"))
            active[focal] = [
                str(value) for value in event.get("focal_active_fact_ids_after", [])
            ]
            known[focal] = [
                str(value) for value in event.get("focal_known_fact_ids_after", [])
            ]
            votes = [str(value) for value in event.get("population_state_after", votes)]

        task = data["task"]
        facts = task.get("facts", {}) if isinstance(task.get("facts", {}), dict) else {}
        groups = task.get("supporting_fact_groups", {})
        latent_for = (
            {
                str(fact): str(latent)
                for latent, fact_ids in groups.items()
                for fact in fact_ids
            }
            if isinstance(groups, dict)
            else {}
        )
        latents = sorted(set(latent_for.values()))
        coverage_agents = []
        for index, agent in enumerate(agents):
            active_ids = active.get(agent, [])
            known_ids = known.get(agent, [])
            coverage_agents.append(
                {
                    "agent_id": agent,
                    "vote": votes[index] if index < len(votes) else None,
                    "active_fact_ids": active_ids,
                    "historical_fact_ids": known_ids,
                    "active_latent_ids": sorted(
                        {latent_for.get(item, item) for item in active_ids}
                    ),
                    "historical_latent_ids": sorted(
                        {latent_for.get(item, item) for item in known_ids}
                    ),
                }
            )

        messages_by_id = self._message_ledger(data)
        semantic_messages = rounds_by_index.get(target_round, {}).get(
            "semantic_board_after", []
        )
        if semantic_messages:
            for message in semantic_messages:
                if isinstance(message, dict) and message.get("message_id"):
                    messages_by_id.setdefault(str(message["message_id"]), dict(message))
        for event in trajectory:
            if int(event.get("global_update_index", -1)) > target_global:
                break
            for key in ("controller_message", "new_message"):
                message = event.get(key)
                if isinstance(message, dict) and message.get("message_id"):
                    messages_by_id[str(message["message_id"])] = dict(message)
        messages = []
        for message in messages_by_id.values():
            created_global = int(message.get("micro_step_created") or 0)
            created_round = int(message.get("round_created") or 0)
            if created_round > target_round or created_global > target_global + 1:
                continue
            expires = int(message.get("expires_after_round") or created_round)
            messages.append(
                {
                    **message,
                    "live": created_round <= target_round <= expires,
                    "new_at_cursor": created_global == target_global + 1,
                }
            )
        messages.sort(
            key=lambda item: (
                int(item.get("round_created") or 0),
                int(item.get("micro_step_created") or 0),
                str(item.get("message_id")),
            )
        )

        counts = Counter(votes)
        correct = str(target.get("correct_answer", task.get("correct_answer", "")))
        cumulative_messages = Counter(
            message.get("message_type") for message in messages
        )
        evidence_events = []
        if semantic:
            for pair in round_record.get("persistence_deactivated_pairs", []):
                if isinstance(pair, dict):
                    evidence_events.append(
                        {
                            "type": "deactivation",
                            "fact_id": pair.get("fact_id"),
                            "agent_id": pair.get("agent_id"),
                            "round_index": target_round,
                        }
                    )
        for event in trajectory:
            if int(event.get("global_update_index", -1)) > target_global:
                break
            for fact_id in event.get("new_peer_fact_ids", []):
                evidence_events.append(
                    {
                        "type": "acquisition",
                        "fact_id": fact_id,
                        "agent_id": event.get("focal_agent_id"),
                        "global_update_index": event.get("global_update_index"),
                    }
                )
            for fact_id in event.get("new_controller_fact_ids", []):
                evidence_events.append(
                    {
                        "type": "acquisition",
                        "fact_id": fact_id,
                        "agent_id": event.get("focal_agent_id"),
                        "global_update_index": event.get("global_update_index"),
                        "source": "controller",
                    }
                )
            for fact_id in event.get("reactivated_peer_fact_ids", []):
                evidence_events.append(
                    {
                        "type": "reactivation" if semantic else "refresh",
                        "fact_id": fact_id,
                        "agent_id": event.get("focal_agent_id"),
                        "global_update_index": event.get("global_update_index"),
                    }
                )
            for fact_id in event.get("reactivated_controller_fact_ids", []):
                evidence_events.append(
                    {
                        "type": "reactivation" if semantic else "refresh",
                        "fact_id": fact_id,
                        "agent_id": event.get("focal_agent_id"),
                        "global_update_index": event.get("global_update_index"),
                        "source": "controller",
                    }
                )
            for fact_id in event.get("refresh_fact_ids", []):
                evidence_events.append(
                    {
                        "type": "refresh",
                        "fact_id": fact_id,
                        "agent_id": event.get("focal_agent_id"),
                        "global_update_index": event.get("global_update_index"),
                    }
                )

        selected_agent = (
            agent_id
            if agent_id in agents
            else str(target.get("focal_agent_id", agents[0]))
        )
        audit_candidates = []
        event_by_interaction = {
            int(event.get("interaction_index")): event
            for event in trajectory
            if int(event.get("global_update_index", -1)) <= target_global
        }
        for index, audit in enumerate(data["audits"]):
            if str(audit.get("agent_id")) != selected_agent:
                continue
            interaction = str(audit.get("interaction_id", ""))
            if interaction == "initial-local-votes":
                audit_candidates.append((index, -1, audit))
                continue
            suffix = interaction.rsplit("-", 1)[-1]
            interaction_number = int(suffix) if suffix.isdigit() else None
            if interaction_number in event_by_interaction:
                audit_candidates.append(
                    (
                        index,
                        int(
                            event_by_interaction[interaction_number].get(
                                "global_update_index", 0
                            )
                        ),
                        audit,
                    )
                )
        latest_audit = (
            max(
                audit_candidates,
                key=lambda item: (item[1], int(item[2].get("attempt", 0))),
            )
            if audit_candidates
            else None
        )
        audit_value = latest_audit[2] if latest_audit else {}
        latest_interaction = (
            str(audit_value.get("interaction_id")) if latest_audit else None
        )
        attempt_history = [
            {
                "audit_index": index,
                "attempt": audit.get("attempt"),
                "valid": audit.get("valid"),
                "validation_error": audit.get("validation_error"),
                "validation_issues": audit.get("validation_issues", []),
                "raw_response": (audit.get("response") or {}).get("content")
                if isinstance(audit.get("response"), dict)
                else None,
            }
            for index, audit in enumerate(data["audits"])
            if latest_interaction is not None
            and str(audit.get("interaction_id")) == latest_interaction
            and str(audit.get("agent_id")) == selected_agent
        ]
        response = (
            audit_value.get("response", {})
            if isinstance(audit_value.get("response", {}), dict)
            else {}
        )
        parsed_response: Any = None
        try:
            parsed_response = json.loads(str(response.get("content", "")))
        except json.JSONDecodeError:
            parsed_response = None
        selected_coverage = next(
            (row for row in coverage_agents if row["agent_id"] == selected_agent), {}
        )
        timeline = [
            {
                "global_update_index": event.get("global_update_index"),
                "round_index": event.get("round_index"),
                "step": int(event.get("within_round_index", 0)) + 1,
                "vote_before": event.get("focal_vote_before", event.get("vote_before")),
                "vote_after": event.get("focal_vote_after", event.get("vote_after")),
                "message_type": event.get("new_message_type"),
            }
            for event in trajectory
            if str(event.get("focal_agent_id")) == selected_agent
            and int(event.get("global_update_index", -1)) <= target_global
        ]
        return {
            "schema_version": 1,
            "source": asdict(data["source"]),
            "capabilities": {
                "microscopic_updates": {"available": True},
                "prompts": {
                    "available": bool(data["audits"]) and not semantic,
                    "reason": unavailable_reason
                    if semantic
                    else None
                    if data["audits"]
                    else "not retained",
                },
                "raw_response": {
                    "available": bool(data["audits"]) and not semantic,
                    "reason": unavailable_reason
                    if semantic
                    else None
                    if data["audits"]
                    else "not retained",
                },
                "validation_attempts": {"available": bool(data["audits"])},
                "provider_attempts": {
                    "available": bool(data["api_calls"]),
                    "reason": unavailable_reason
                    if semantic
                    else None
                    if data["api_calls"]
                    else "not retained",
                },
                "active_evidence": {"available": True},
                "historical_evidence": {"available": True},
                "controller": {
                    "available": bool(data["rounds"]),
                    "reason": None if data["rounds"] else "not retained",
                },
            },
            "run": self.status(),
            "cursor": {
                "phase": "round",
                "round_index": target_round,
                "step": target_step,
                "global_update_index": target_global,
                "state_semantics": "after_selected_update",
            },
            "population": {
                "votes": votes,
                "vote_counts": dict(counts),
                "correct_answer": correct,
                "truth_vote_share": (counts.get(correct, 0) / len(votes))
                if votes
                else None,
                "blackboard_live_size": sum(
                    bool(message["live"]) for message in messages
                ),
                "message_counts": dict(cumulative_messages),
                "mean_active": sum(
                    len(row["active_fact_ids"]) for row in coverage_agents
                )
                / len(coverage_agents)
                if coverage_agents
                else 0,
                "mean_historical": sum(
                    len(row["historical_fact_ids"]) for row in coverage_agents
                )
                / len(coverage_agents)
                if coverage_agents
                else 0,
                "exact_acquisitions": sum(
                    item["type"] == "acquisition" for item in evidence_events
                ),
                "refreshes": sum(item["type"] == "refresh" for item in evidence_events),
            },
            "blackboard": messages,
            "coverage": {"latents": latents, "agents": coverage_agents, "facts": facts},
            "agent": {
                **selected_coverage,
                "is_focal_at_cursor": selected_agent
                == str(target.get("focal_agent_id")),
                "latest_decision_global_update": latest_audit[1]
                if latest_audit
                else None,
                "audit_index": latest_audit[0] if latest_audit else None,
                "attempt": audit_value.get("attempt"),
                "valid": audit_value.get("valid"),
                "validation_error": audit_value.get("validation_error"),
                "visible_state": (audit_value.get("observation") or {}).get(
                    "visible_state", {}
                ),
                "compiled_messages": audit_value.get("compiled_messages", []),
                "raw_response": response.get("content"),
                "parsed_response": (
                    target.get("public_action") if semantic else parsed_response
                ),
                "attempt_history": attempt_history,
                "timeline": timeline,
            },
            "controller": {
                "availability": "available" if data["rounds"] else "unavailable",
                "decision": (
                    "disabled"
                    if round_record.get("controller_enabled") is False
                    else "acted"
                    if round_record.get("controller_sampled_U")
                    in {1, "ACT", "ADVOCATE"}
                    else "no_action"
                    if round_record.get("controller_sampled_U") in {0, "NO_OP"}
                    else "unknown"
                ),
                "enabled": round_record.get(
                    "controller_enabled", target.get("controller_enabled")
                ),
                "sensor": round_record.get("controller_sensor_Y")
                or {
                    "sampled_agent_ids": target.get("sensor_agent_ids", []),
                    "sampled_votes": target.get("sensor_observed_opinions", []),
                    "count_vector": target.get("sensor_count_vector", []),
                },
                "action": round_record.get(
                    "controller_action", target.get("round_controller_action")
                ),
                "target": round_record.get(
                    "controller_target", target.get("round_controller_target")
                ),
                "probability": round_record.get("controller_probability_U1_given_Y"),
                "sampled_action": round_record.get("controller_sampled_U"),
                "controlled_positions": round_record.get("controlled_positions", []),
                "post_ids": round_record.get("controller_post_ids", []),
                "directive_ids": round_record.get("directive_message_ids", []),
                "report_ids": round_record.get("controller_report_ids", []),
                "report_fact_ids": round_record.get("controller_report_fact_ids", []),
                "direct_replies": round_record.get("controller_direct_replies"),
                "unique_readers": round_record.get("controller_unique_readers"),
            },
            "evidence_events": evidence_events,
            "available_cursors": self.timeline()["available_cursors"],
        }

    def prompt(self, audit_index: int) -> dict[str, Any]:
        audits = self._load()["audits"]
        if audit_index < 0 or audit_index >= len(audits):
            raise ValueError("prompt audit index is outside the available range")
        return {
            "schema_version": 1,
            "audit_index": audit_index,
            "audit": audits[audit_index],
        }
