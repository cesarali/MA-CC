"""Crash-safe, credential-free request and sampled trace logging."""

from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditTraceConfig:
    enabled: bool = True
    inspect_every_percent: int = 10
    examples_per_selected_round: int = 3
    seed: int = 1
    include_request: bool = True
    include_raw_response: bool = True
    include_parsed_response: bool = True
    include_agent_memory: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.inspect_every_percent <= 100:
            raise ValueError("inspect_every_percent must be between 1 and 100.")
        if self.examples_per_selected_round < 0:
            raise ValueError("examples_per_selected_round cannot be negative.")


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    level: str = "INFO"
    api_status_log: bool = True
    audit_traces: AuditTraceConfig = AuditTraceConfig()


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


class ExperimentAuditLogger:
    """Append-only logger.  Logging errors are loud and never affect choices."""

    def __init__(self, directory: Path, config: LoggingConfig, run_id: str, population_size: int, max_rounds: int) -> None:
        self.directory, self.config, self.run_id = directory, config, run_id
        self.population_size, self.max_rounds = population_size, max_rounds
        self._lock = threading.Lock()
        directory.mkdir(parents=True, exist_ok=True)
        self.status_path = directory / "api_call_status.jsonl"
        self.trace_path = directory / "audit_traces.jsonl"
        self.report_path = directory / "audit_report.md"
        self._completed = self._load_completed()
        if not self.report_path.exists():
            self.report_path.write_text("# Sampled API audit traces\n\n", encoding="utf-8")

    def _load_completed(self) -> set[tuple[str, int]]:
        found: set[tuple[str, int]] = set()
        if self.status_path.exists():
            try:
                for line in self.status_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("status") in {"success", "failed", "skipped", "forced_no_api_call"}:
                        found.add((row["call_id"], int(row["attempt_number"])))
            except Exception:
                LOGGER.exception("Logging failure while reading existing API status log; append will continue.")
        return found

    def append_status(self, row: dict[str, Any]) -> None:
        if not self.config.enabled or not self.config.api_status_log:
            return
        key = (str(row["call_id"]), int(row["attempt_number"]))
        if key in self._completed:
            return
        try:
            with self._lock, self.status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                if row.get("status") in {"success", "failed", "skipped", "forced_no_api_call"}:
                    self._completed.add(key)
        except Exception:
            LOGGER.exception("AUDIT LOGGING FAILURE: could not append API status; experiment continues unchanged.")

    def selected(self, episode_id: str, population_round: int, interaction_index: int, slot: int) -> bool:
        cfg = self.config.audit_traces
        if not self.config.enabled or not cfg.enabled or cfg.examples_per_selected_round == 0:
            return False
        checkpoints = {1, self.max_rounds}
        checkpoints.update(max(1, round(self.max_rounds * p / 100)) for p in range(0, 101, cfg.inspect_every_percent))
        if population_round not in checkpoints:
            return False
        local = (interaction_index - 1) % self.population_size * 2 + slot
        rng = random.Random(f"{cfg.seed}:{episode_id}:{population_round}")
        chosen = set(rng.sample(range(self.population_size * 2), min(cfg.examples_per_selected_round, self.population_size * 2)))
        return local in chosen

    def append_trace(self, row: dict[str, Any]) -> None:
        try:
            with self._lock:
                with self.trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); handle.flush()
                with self.report_path.open("a", encoding="utf-8") as report:
                    report.write(self._markdown(row)); report.flush()
        except Exception:
            LOGGER.exception("AUDIT LOGGING FAILURE: could not append sampled trace; experiment continues unchanged.")

    @staticmethod
    def _markdown(row: dict[str, Any]) -> str:
        def block(value: Any) -> str:
            return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n```\n"
        return (
            f"## Episode `{row['episode_id']}` — population round {row['population_round']}\n\n"
            f"### Call `{row['call_id']}`\n\n"
            f"**Experiment and agent metadata**\n\n{block({k: row.get(k) for k in ('run_id','episode_id','population_round','interaction_index','agent_id','partner_id','provider','requested_model','returned_model','committee_policy','forced_action')})}"
            f"**Full prompt sent to the model**\n\n{block(row.get('messages'))}"
            f"**Agent memory before**\n\n{block(row.get('agent_memory_before'))}"
            f"**Raw model response**\n\n{block(row.get('raw_provider_response'))}"
            f"**Parsed action**\n\n{block(row.get('parsed_model_output'))}"
            f"**Interaction result**\n\n{block({k: row.get(k) for k in ('selected_convention','partner_output','interaction_success','payoff','agent_memory_after')})}"
            f"**Request status**: {row.get('status')}; latency={row.get('latency_seconds')}s; tokens={json.dumps(row.get('token_usage'))}; retries={row.get('retries')}\n\n---\n\n"
        )


def memory_dict(memory: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in memory]
