"""Lean append-only semantic retention for the blackboard dashboard."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DASHBOARD_SEMANTIC_SCHEMA_VERSION = 1
STREAM_FILENAME = "dashboard_semantic.jsonl"
SEAL_FILENAME = "dashboard_semantic_complete.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def semantic_stream_path(round_records_dir: str | Path) -> Path:
    return Path(round_records_dir) / STREAM_FILENAME


def semantic_seal_path(round_records_dir: str | Path) -> Path:
    return Path(round_records_dir) / SEAL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_semantic_stream(
    path: str | Path, *, completed: bool = False
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    raw = source.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if completed:
            raise ValueError(
                f"completed semantic stream has a partial trailing record: {source}"
            )
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    rows = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed semantic record {source}:{number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"semantic record must be an object: {source}:{number}")
        if int(value.get("schema_version", 0)) != DASHBOARD_SEMANTIC_SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic schema in {source}:{number}")
        rows.append(value)
    return rows


def validate_semantic_stream(
    directory: str | Path, *, require_completed: bool = True
) -> list[dict[str, Any]]:
    root = Path(directory)
    stream = semantic_stream_path(root)
    seal_path = semantic_seal_path(root)
    if not stream.is_file():
        raise ValueError(f"dashboard semantic stream is missing: {stream}")
    if not seal_path.is_file():
        if require_completed:
            raise ValueError(
                f"dashboard semantic completion seal is missing: {seal_path}"
            )
        return read_semantic_stream(stream)
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read semantic completion seal {seal_path}") from exc
    rows = read_semantic_stream(stream, completed=True)
    if seal.get("status") != "completed":
        raise ValueError(f"semantic completion seal is not completed: {seal_path}")
    if int(seal.get("row_count", -1)) != len(rows):
        raise ValueError(f"semantic stream row count does not match its seal: {stream}")
    if seal.get("stream_sha256") != _sha256(stream):
        raise ValueError(f"semantic stream hash does not match its seal: {stream}")
    if not rows or rows[0].get("record_type") != "header":
        raise ValueError(f"semantic stream has no header: {stream}")
    if rows[-1].get("record_type") != "completion":
        raise ValueError(f"semantic stream has no completion record: {stream}")
    if sum(row.get("record_type") == "header" for row in rows) != 1:
        raise ValueError(f"semantic stream must contain exactly one header: {stream}")
    if sum(row.get("record_type") == "initialization" for row in rows) != 1:
        raise ValueError(
            f"semantic stream must contain exactly one initialization: {stream}"
        )
    if sum(row.get("record_type") == "completion" for row in rows) != 1:
        raise ValueError(
            f"semantic stream must contain exactly one completion: {stream}"
        )
    if rows[-1].get("status") != "completed":
        raise ValueError(f"semantic completion record is not completed: {stream}")
    for key in ("run_id", "cell_id", "episode_id", "episode_seed"):
        if str(seal.get(key)) != str(rows[0].get(key)):
            raise ValueError(f"semantic seal {key} does not match its stream")
        if any(str(row.get(key)) != str(rows[0].get(key)) for row in rows):
            raise ValueError(f"semantic row {key} does not match its header")
    updates = [row for row in rows if row.get("record_type") == "update"]
    global_ids = [int(row.get("global_update_index", -1)) for row in updates]
    if global_ids != list(range(len(global_ids))):
        raise ValueError(
            f"semantic update indices are not contiguous from zero: {stream}"
        )
    cursors = [
        (int(row.get("round_index", -1)), int(row.get("within_round_index", -1)))
        for row in updates
    ]
    if len(cursors) != len(set(cursors)):
        raise ValueError(f"semantic update cursors are duplicated: {stream}")
    expected = rows[0].get("expected_updates")
    if expected is not None and len(updates) != int(expected):
        raise ValueError(
            f"semantic stream has {len(updates)} updates; expected {expected}: {stream}"
        )
    return rows


def reset_semantic_attempt(directory: str | Path) -> None:
    """Remove only an uncommitted semantic attempt before episode rerun."""

    root = Path(directory)
    for path in (semantic_stream_path(root), semantic_seal_path(root)):
        if path.exists():
            path.unlink()


def _message(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = (
        "message_id",
        "author_id",
        "author_kind",
        "message_type",
        "text",
        "vote",
        "shared_fact_id",
        "reply_to",
        "round_created",
        "micro_step_created",
        "expires_after_round",
    )
    return {key: value.get(key) for key in allowed}


def _agent(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "agent_id": value.get("agent_id"),
        "vote": value.get("committed_action"),
        "active_fact_ids": list(value.get("active_fact_ids", ())),
        "known_fact_ids": list(value.get("known_fact_ids", ())),
    }


@dataclass
class SemanticDashboardWriter:
    directory: Path
    identity: Mapping[str, Any]
    header: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream_path = semantic_stream_path(self.directory)
        self.seal_path = semantic_seal_path(self.directory)
        if self.seal_path.is_file():
            validate_semantic_stream(self.directory)
            raise ValueError(
                f"completed semantic stream cannot be appended: {self.stream_path}"
            )
        if self.stream_path.is_file():
            self.stream_path.unlink()
        self._rows = 0
        self._last_cursor: dict[str, Any] | None = None

    def append(self, record_type: str, **payload: Any) -> None:
        row = {
            "schema_version": DASHBOARD_SEMANTIC_SCHEMA_VERSION,
            "record_type": record_type,
            "timestamp": _now(),
            **dict(self.identity),
            **payload,
        }
        encoded = (
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self.stream_path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._rows += 1
        if record_type == "update":
            self._last_cursor = {
                "round_index": row.get("round_index"),
                "within_round_index": row.get("within_round_index"),
                "global_update_index": row.get("global_update_index"),
            }

    def initialization(self, state: Mapping[str, Any]) -> None:
        agents = [
            _agent(value)
            for value in state.get("agents", ())
            if isinstance(value, Mapping)
        ]
        board = state.get("blackboard", ())
        messages = board.get("messages", ()) if isinstance(board, Mapping) else board
        task = (
            state.get("task", {}) if isinstance(state.get("task", {}), Mapping) else {}
        )
        rules = (
            state.get("rules", {})
            if isinstance(state.get("rules", {}), Mapping)
            else {}
        )
        if not self.stream_path.is_file() or self.stream_path.stat().st_size == 0:
            header = {
                **dict(self.header),
                "task_id": task.get("task_id", self.header.get("task_id")),
                "task_family": task.get("task_family"),
                "semantic_world_sha256": task.get("semantic_world_sha256"),
                "truth_option": task.get("correct_answer"),
                "controller_target": rules.get("controller_target"),
                "game_protocol_version": (
                    "night_dawn_autonomous_day_v1"
                    if rules.get("controller_timing") == "dawn_only"
                    else "legacy"
                ),
            }
            self.append("header", **header)
        self.append(
            "initialization",
            agents=agents,
            population_votes=[agent["vote"] for agent in agents],
            board=[
                item for item in (_message(value) for value in messages or ()) if item
            ],
            correct_answer=task.get("correct_answer"),
            possible_answers=list(task.get("possible_answers", ())),
        )

    def attempt(
        self,
        *,
        round_index: int,
        interaction_id: str,
        agent_id: str,
        attempt: int,
        valid: bool,
        validation_issues: Sequence[Any],
        repair: bool,
    ) -> None:
        issue_codes = sorted(
            {str(getattr(issue, "field", "response")) for issue in validation_issues}
        )
        self.append(
            "validation",
            round_index=round_index,
            interaction_id=interaction_id,
            agent_id=agent_id,
            attempt=attempt,
            valid=bool(valid),
            issue_codes=issue_codes,
            repair=bool(repair),
        )

    def update(
        self, event: Mapping[str, Any], public_action: Mapping[str, Any] | None = None
    ) -> None:
        exposed = {
            str(value)
            for value in (
                *event.get("peer_exposed_fact_ids", ()),
                *(
                    [event.get("controller_fact_id")]
                    if event.get("controller_fact_id")
                    else []
                ),
            )
        }
        acquired = {
            str(value)
            for value in (
                *event.get("new_peer_fact_ids", ()),
                *event.get("new_controller_fact_ids", ()),
            )
        }
        reactivated = {
            str(value)
            for value in (
                *event.get("reactivated_peer_fact_ids", ()),
                *event.get("reactivated_controller_fact_ids", ()),
            )
        }
        refresh = sorted(exposed - acquired - reactivated)
        fields = (
            "round_index",
            "within_round_index",
            "global_update_index",
            "interaction_index",
            "focal_agent_id",
            "focal_vote_before",
            "focal_vote_after",
            "occupation_counts_before",
            "occupation_counts_after",
            "population_shares_before",
            "population_shares_after",
            "q_requested",
            "q_effective",
            "sampled_message_ids",
            "sampled_message_authors",
            "sampled_message_types",
            "sampled_message_ages",
            "sampled_controller_message_ids",
            "focal_known_fact_ids_before",
            "focal_known_fact_ids_after",
            "focal_active_fact_ids_before",
            "focal_active_fact_ids_after",
            "peer_exposed_fact_ids",
            "controller_fact_id",
            "new_peer_fact_ids",
            "new_controller_fact_ids",
            "reactivated_peer_fact_ids",
            "reactivated_controller_fact_ids",
            "controller_enabled",
            "controller_action",
            "controller_target",
            "controller_policy",
            "controller_threshold",
            "controller_beta",
            "controller_advocacy_probability",
            "sensor_sample_size",
            "sensor_agent_ids",
            "sensor_observed_opinions",
            "sensor_count_vector",
            "round_controller_action",
            "round_controller_target",
            "controlled_slot",
            "intervention_budget",
            "controller_message_posted",
            "controller_message_id",
            "controller_message_directly_exposed",
            "possible_answers",
            "correct_answer",
            "board_size_before",
            "board_size_after",
        )
        retained = {key: event.get(key) for key in fields}
        for key in (
            "sampled_message_ids",
            "sampled_message_authors",
            "sampled_message_types",
            "sampled_message_ages",
            "sampled_controller_message_ids",
            "focal_known_fact_ids_before",
            "focal_known_fact_ids_after",
            "focal_active_fact_ids_before",
            "focal_active_fact_ids_after",
            "peer_exposed_fact_ids",
            "new_peer_fact_ids",
            "new_controller_fact_ids",
            "reactivated_peer_fact_ids",
            "reactivated_controller_fact_ids",
            "sensor_agent_ids",
            "sensor_observed_opinions",
        ):
            if retained.get(key) is None:
                retained[key] = []
        self.append(
            "update",
            **retained,
            new_message=_message(event.get("new_message")),
            public_action=None if public_action is None else dict(public_action),
            refresh_fact_ids=refresh,
        )

    def round_start(
        self,
        *,
        round_index: int,
        state: Mapping[str, Any],
        expired_message_ids: Sequence[str],
        deactivated_pairs: Sequence[Mapping[str, Any]],
        controller: Mapping[str, Any],
    ) -> None:
        agents = [
            _agent(value)
            for value in state.get("agents", ())
            if isinstance(value, Mapping)
        ]
        board = state.get("blackboard", ())
        messages = board.get("messages", ()) if isinstance(board, Mapping) else board
        self.append(
            "round_start",
            round_index=round_index,
            agents=agents,
            population_votes=[agent["vote"] for agent in agents],
            board=[
                item for item in (_message(value) for value in messages or ()) if item
            ],
            expired_message_ids=list(expired_message_ids),
            deactivated_pairs=[dict(value) for value in deactivated_pairs],
            controller=dict(controller),
        )

    def round_end(self, *, round_index: int, state: Mapping[str, Any]) -> None:
        agents = [
            _agent(value)
            for value in state.get("agents", ())
            if isinstance(value, Mapping)
        ]
        board = state.get("blackboard", ())
        messages = board.get("messages", ()) if isinstance(board, Mapping) else board
        retained_messages = [
            item
            for item in (_message(value) for value in messages or ())
            if item
            and int(item.get("round_created") or 0)
            <= round_index
            <= int(item.get("expires_after_round") or item.get("round_created") or 0)
        ]
        self.append(
            "round_end",
            round_index=round_index,
            agents=agents,
            population_votes=[agent["vote"] for agent in agents],
            board=retained_messages,
        )

    def finalize(self, status: str, *, error_type: str | None = None) -> Path | None:
        self.append(
            "completion",
            status=status,
            final_cursor=self._last_cursor,
            error_type=error_type,
        )
        if status != "completed":
            return None
        seal = {
            "schema_version": DASHBOARD_SEMANTIC_SCHEMA_VERSION,
            "status": "completed",
            **dict(self.identity),
            "row_count": len(read_semantic_stream(self.stream_path, completed=True)),
            "stream_sha256": _sha256(self.stream_path),
            "final_cursor": self._last_cursor,
            "completed_at": _now(),
        }
        temporary = self.seal_path.with_suffix(self.seal_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, self.seal_path)
        try:
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        try:
            validate_semantic_stream(self.directory)
        except ValueError:
            self.seal_path.unlink(missing_ok=True)
            raise
        return self.seal_path


__all__ = [
    "DASHBOARD_SEMANTIC_SCHEMA_VERSION",
    "SEAL_FILENAME",
    "STREAM_FILENAME",
    "SemanticDashboardWriter",
    "read_semantic_stream",
    "semantic_seal_path",
    "semantic_stream_path",
    "reset_semantic_attempt",
    "validate_semantic_stream",
]
