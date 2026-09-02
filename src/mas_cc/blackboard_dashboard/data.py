"""Safe reconstruction of blackboard episode state from retained artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records without consuming a partial writer tail."""

    if not path.is_file():
        return []
    raw = path.read_bytes()
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


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def resolve_episode_dir(run_dir: str | Path, episode_id: str | None = None) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if (root / "trajectory.jsonl").is_file() or root.name == episode_id:
        return root
    episodes_root = root / "data" / "episodes"
    candidates = sorted(path for path in episodes_root.glob("*") if path.is_dir())
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
        paths = (
            self.episode_dir / "trajectory.jsonl",
            self.episode_dir / "round_trajectory.jsonl",
            self.episode_dir / "audit_traces.jsonl",
            self.episode_dir / "api_call_status.jsonl",
            self.episode_dir / ".checkpoints" / "checkpoint.json",
            self.episode_dir / "manifest.json",
            self.run_dir / "manifest.json",
        )
        signature = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
            if path.is_file()
        )
        if signature == self._cache_signature and self._cache is not None:
            return self._cache
        trajectory = [_event(row) for row in _jsonl(self.episode_dir / "trajectory.jsonl")]
        rounds = [_event(row) for row in _jsonl(self.episode_dir / "round_trajectory.jsonl")]
        audits = _jsonl(self.episode_dir / "audit_traces.jsonl")
        api_calls = _jsonl(self.episode_dir / "api_call_status.jsonl")
        checkpoint = _safe_json(self.episode_dir / ".checkpoints" / "checkpoint.json")
        episode_manifest = _safe_json(self.episode_dir / "manifest.json")
        run_manifest = _safe_json(self.run_dir / "manifest.json")
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
            "checkpoint": checkpoint,
            "state": state,
            "task": task,
            "episode_manifest": episode_manifest,
            "run_manifest": run_manifest,
        }
        self._cache_signature = signature
        self._cache = loaded
        return loaded

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
        votes = [str(state_agents.get(agent, {}).get("committed_action")) for agent in agents]
        return active, known, votes

    def timeline(self) -> dict[str, Any]:
        data = self._load()
        trajectory = data["trajectory"]
        agents = self._agents(data)
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
        return {
            "schema_version": 1,
            "episode_id": self.episode_dir.name,
            "agents": agents,
            "rounds": [
                {"round_index": key, "available_steps": value}
                for key, value in sorted(rounds.items())
            ],
            "available_cursors": cursors,
            "initialization_attempts": sum(
                row.get("decision_stage") == "initial_vote" for row in data["audits"]
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
            "episode_id": self.episode_dir.name,
            "status": manifest.get("status", "running" if trajectory else "waiting_for_writer"),
            "completed_updates": len(trajectory),
            "expected_updates": expected or None,
            "prompt_attempts": len(data["audits"]),
            "provider_attempts": len(data["api_calls"]),
            "invalid_attempts": sum(not bool(row.get("valid", False)) for row in data["api_calls"]),
            "latest_cursor": None
            if not trajectory
            else {
                "round_index": int(last.get("round_index", 0)),
                "step": int(last.get("within_round_index", 0)) + 1,
                "global_update_index": int(last.get("global_update_index", len(trajectory) - 1)),
            },
        }

    def snapshot(
        self,
        round_index: int | None = None,
        step: int | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        data = self._load()
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
                and (step is None or int(event.get("within_round_index", -1)) + 1 <= step)
            ]
            if not candidates:
                raise ValueError(f"no recorded update for round {round_index + 1}, step {step}")
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
                {str(key): list(value) for key, value in record.get("active_fact_ids_by_agent_after", {}).items()}
            )
            known.update(
                {str(key): list(value) for key, value in record.get("known_fact_ids_by_agent_after", {}).items()}
            )
            votes = [str(value) for value in record.get("population_state_after", votes)]
        selected_events = [
            event
            for event in trajectory
            if int(event.get("round_index", -1)) == target_round
            and int(event.get("within_round_index", -1)) + 1 <= target_step
        ]
        for event in selected_events:
            focal = str(event.get("focal_agent_id"))
            active[focal] = [str(value) for value in event.get("focal_active_fact_ids_after", [])]
            known[focal] = [str(value) for value in event.get("focal_known_fact_ids_after", [])]
            votes = [str(value) for value in event.get("population_state_after", votes)]

        task = data["task"]
        facts = task.get("facts", {}) if isinstance(task.get("facts", {}), dict) else {}
        groups = task.get("supporting_fact_groups", {})
        latent_for = {
            str(fact): str(latent)
            for latent, fact_ids in groups.items()
            for fact in fact_ids
        } if isinstance(groups, dict) else {}
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
                    "active_latent_ids": sorted({latent_for.get(item, item) for item in active_ids}),
                    "historical_latent_ids": sorted({latent_for.get(item, item) for item in known_ids}),
                }
            )

        checkpoint_messages = data["state"].get("blackboard", [])
        messages_by_id: dict[str, dict[str, Any]] = {}
        for message in checkpoint_messages:
            if isinstance(message, dict) and message.get("message_id"):
                messages_by_id[str(message["message_id"])] = dict(message)
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
            if created_global > target_global + 1 or created_round > target_round:
                continue
            expires = int(message.get("expires_after_round") or created_round)
            messages.append(
                {
                    **message,
                    "live": created_round <= target_round <= expires,
                    "new_at_cursor": created_global == target_global + 1,
                }
            )
        messages.sort(key=lambda item: (int(item.get("micro_step_created") or 0), str(item.get("message_id"))))

        counts = Counter(votes)
        correct = str(target.get("correct_answer", task.get("correct_answer", "")))
        cumulative_messages = Counter(message.get("message_type") for message in messages)
        evidence_events = []
        for event in trajectory:
            if int(event.get("global_update_index", -1)) > target_global:
                break
            for fact_id in event.get("new_peer_fact_ids", []):
                evidence_events.append({"type": "acquisition", "fact_id": fact_id, "agent_id": event.get("focal_agent_id"), "global_update_index": event.get("global_update_index")})
            for fact_id in event.get("new_controller_fact_ids", []):
                evidence_events.append({"type": "acquisition", "fact_id": fact_id, "agent_id": event.get("focal_agent_id"), "global_update_index": event.get("global_update_index"), "source": "controller"})
            for fact_id in event.get("reactivated_peer_fact_ids", []):
                evidence_events.append({"type": "refresh", "fact_id": fact_id, "agent_id": event.get("focal_agent_id"), "global_update_index": event.get("global_update_index")})
            for fact_id in event.get("reactivated_controller_fact_ids", []):
                evidence_events.append({"type": "refresh", "fact_id": fact_id, "agent_id": event.get("focal_agent_id"), "global_update_index": event.get("global_update_index"), "source": "controller"})

        selected_agent = agent_id if agent_id in agents else str(target.get("focal_agent_id", agents[0]))
        audit_candidates = []
        event_by_interaction = {
            int(event.get("interaction_index")): event for event in trajectory
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
                audit_candidates.append((index, int(event_by_interaction[interaction_number].get("global_update_index", 0)), audit))
        latest_audit = max(audit_candidates, key=lambda item: (item[1], int(item[2].get("attempt", 0)))) if audit_candidates else None
        audit_value = latest_audit[2] if latest_audit else {}
        response = audit_value.get("response", {}) if isinstance(audit_value.get("response", {}), dict) else {}
        parsed_response: Any = None
        try:
            parsed_response = json.loads(str(response.get("content", "")))
        except json.JSONDecodeError:
            parsed_response = None
        selected_coverage = next((row for row in coverage_agents if row["agent_id"] == selected_agent), {})
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
        round_record = rounds_by_index.get(target_round, {})
        return {
            "schema_version": 1,
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
                "truth_vote_share": (counts.get(correct, 0) / len(votes)) if votes else None,
                "blackboard_live_size": sum(bool(message["live"]) for message in messages),
                "message_counts": dict(cumulative_messages),
                "mean_active": sum(len(row["active_fact_ids"]) for row in coverage_agents) / len(coverage_agents) if coverage_agents else 0,
                "mean_historical": sum(len(row["historical_fact_ids"]) for row in coverage_agents) / len(coverage_agents) if coverage_agents else 0,
                "exact_acquisitions": sum(item["type"] == "acquisition" for item in evidence_events),
                "refreshes": sum(item["type"] == "refresh" for item in evidence_events),
            },
            "blackboard": messages,
            "coverage": {"latents": latents, "agents": coverage_agents, "facts": facts},
            "agent": {
                **selected_coverage,
                "is_focal_at_cursor": selected_agent == str(target.get("focal_agent_id")),
                "latest_decision_global_update": latest_audit[1] if latest_audit else None,
                "audit_index": latest_audit[0] if latest_audit else None,
                "attempt": audit_value.get("attempt"),
                "valid": audit_value.get("valid"),
                "validation_error": audit_value.get("validation_error"),
                "visible_state": (audit_value.get("observation") or {}).get("visible_state", {}),
                "compiled_messages": audit_value.get("compiled_messages", []),
                "raw_response": response.get("content"),
                "parsed_response": parsed_response,
                "timeline": timeline,
            },
            "controller": {
                "enabled": round_record.get("controller_enabled", target.get("controller_enabled")),
                "sensor": round_record.get("controller_sensor_Y") or {
                    "sampled_agent_ids": target.get("sensor_agent_ids", []),
                    "sampled_votes": target.get("sensor_observed_opinions", []),
                    "count_vector": target.get("sensor_count_vector", []),
                },
                "action": round_record.get("controller_action", target.get("round_controller_action")),
                "target": round_record.get("controller_target", target.get("round_controller_target")),
                "probability": round_record.get("controller_probability_U1_given_Y"),
                "sampled_action": round_record.get("controller_sampled_U"),
                "controlled_positions": round_record.get("controlled_positions", []),
                "directive_ids": round_record.get("controller_post_ids", []),
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
        return {"schema_version": 1, "audit_index": audit_index, "audit": audits[audit_index]}
