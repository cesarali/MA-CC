"""Atomic, versioned local checkpoints without prompt-object restoration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Checkpoint:
    run_id: str
    completed_rounds: int
    resolved_config_hash: str
    state: Mapping[str, Any]
    budget: Mapping[str, Any]
    prompt_definitions: Mapping[str, str]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "completed_rounds": self.completed_rounds,
            "resolved_config_hash": self.resolved_config_hash,
            "state": dict(self.state),
            "budget": dict(self.budget),
            "prompt_definitions": dict(self.prompt_definitions),
            "prompt_state_restored": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Checkpoint":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint schema version")
        return cls(
            run_id=str(value["run_id"]), completed_rounds=int(value["completed_rounds"]),
            resolved_config_hash=str(value["resolved_config_hash"]),
            state=dict(value["state"]), budget=dict(value["budget"]),
            prompt_definitions={str(k): str(v) for k, v in dict(value["prompt_definitions"]).items()},
        )


class AtomicCheckpointStore:
    """Writes a single replaceable checkpoint and rejects incompatible resumes."""

    filename = "checkpoint.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / self.filename

    def write(self, checkpoint: Checkpoint) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(checkpoint.to_dict(), stream, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        return self.path

    def load(self) -> Checkpoint | None:
        if not self.path.exists():
            return None
        return Checkpoint.from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def require_compatible(self, *, resolved_config_hash: str, prompt_definitions: Mapping[str, str]) -> Checkpoint:
        checkpoint = self.load()
        if checkpoint is None:
            raise ValueError("no checkpoint exists")
        if checkpoint.resolved_config_hash != resolved_config_hash:
            raise ValueError("checkpoint resolved configuration does not match this run")
        if dict(checkpoint.prompt_definitions) != dict(prompt_definitions):
            raise ValueError("checkpoint prompt definition hashes do not match this run")
        return checkpoint
