"""Local-first event recording with optional aggregate-only Comet reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mas_cc.config import RetentionPolicy
from mas_cc.metrics import FinalMetric, Metric, StreamingMetric
from mas_cc.metrics.interactions import (
    PRODUCTION_PROBABILITY_FIELDS,
    SUCCESS_RATE_FIELDS,
    BinnedTrajectoryPolicy,
    InteractionOutcome,
    binned_trajectory_tables,
)
from mas_cc.observability.audit import DetailedAuditPolicy, DetailedAuditSelector
from mas_cc.storage import (
    AtomicCheckpointStore,
    Checkpoint,
    ScientificIdentity,
    canonical_hash,
    compact_imitation_event,
    empty_compact_row,
    write_completed_episode,
)

STREAMING_METRIC_FIELDS = [
    "round_index",
    "episode_id",
    "agent_id",
    "series",
    "metric_name",
    "value",
]
"""Column order of ``metrics/streaming.csv``, and the marker for its schema.

``agent_id`` is set for agent-scope metrics, ``series`` for option-scope ones,
and both are empty for population-scope metrics. Any change here changes the
on-disk schema, which is exactly what the header comparison in
``_record_round_metrics`` detects when it meets a file from an older run.
"""


_SECRET_FIELD = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth_token|authorization|bearer|client_secret|"
    r"private_key|secret|password|credential|credentials)(?:$|_)",
    re.IGNORECASE,
)
"""Parameter names that must never leave the machine, even as a config value.

Deliberately a copy of the loader's check rather than an import of it: the
config layer refuses to *load* such a field, and this refuses to *send* one.
Two independent guards on the same class of mistake is the point.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


class CometMetricSink:
    """Lazy Comet bridge that never receives prompt content or arbitrary metadata."""

    def __init__(self, enabled: bool, *, project_name: str, run_name: str) -> None:
        self.enabled = enabled
        self._experiment: Any | None = None
        self.status = "disabled" if not enabled else "unavailable"
        self.reason: str | None = None
        self.reference: str | None = None
        self.url: str | None = None
        if not enabled:
            return
        key = os.environ.get("COMET_API_KEY") or self._key_from_local_dotenv()
        if not key:
            self.reason = "COMET_API_KEY is not set"
            return
        try:
            from comet_ml import Experiment  # imported only for an enabled run

            # The remote surface is intentionally aggregate-only. In particular,
            # do not let SDK defaults collect source files, git patches, command
            # arguments, output, or machine/environment details.
            self._experiment = Experiment(
                api_key=key,
                project_name=project_name,
                log_code=False,
                log_graph=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                parse_args=False,
                auto_output_logging=False,
                log_env_details=False,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                auto_log_co2=False,
                display_summary_level=0,
            )
            self._experiment.set_name(run_name)
            self.status = "active"
            self.reference = getattr(self._experiment, "get_key", lambda: None)()
            # The clickable link, so a run says where it is being watched rather
            # than leaving you to find it. Defensive because it is the one piece
            # of this that is pure convenience: an SDK without it must not turn
            # a working run into a failed one.
            try:
                self.url = self._experiment.url
            except Exception:
                self.url = None
        except Exception as exc:  # local runs must remain useful without Comet
            # SDK exceptions can include request URLs or headers; never persist
            # their text because the connector may have seen the credential.
            self.reason = f"{type(exc).__name__}: Comet initialization failed"

    @staticmethod
    def _key_from_local_dotenv() -> str | None:
        """Read only the documented key from an untracked repository .env file."""
        candidate = Path.cwd() / ".env"
        if not candidate.is_file():
            return None
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if line.startswith("COMET_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                return value or None
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if self._experiment is not None:
            self._experiment.log_metrics(dict(metrics), step=step)

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Send scalar run parameters — never prompt content, never secrets.

        Callers are expected to have flattened whatever they want recorded into
        scalars already; anything else is dropped here rather than serialized,
        so a nested structure can never smuggle a prompt or a credential onto
        the remote surface by accident.
        """

        if self._experiment is None:
            return
        safe = {
            str(key): value
            for key, value in parameters.items()
            if isinstance(value, (str, int, float, bool))
            and not _SECRET_FIELD.search(str(key))
        }
        if safe:
            self._experiment.log_parameters(safe)

    def log_figure(self, figure: Any, *, name: str, step: int) -> None:
        """Log a rendered matplotlib figure as a stepped image asset.

        Comet's Image Panel exposes a stepper over same-named images, so a
        figure re-logged under one name at increasing steps scrubs as an
        animation rather than piling up as unrelated assets.
        """

        if self._experiment is not None:
            self._experiment.log_figure(figure_name=name, figure=figure, step=step)

    def log_image(self, path: str | Path, *, name: str, step: int) -> None:
        """Upload an already-rendered aggregate plot as a stepped image asset."""

        if self._experiment is not None:
            self._experiment.log_image(image_data=str(path), name=name, step=step)

    def log_asset(self, path: str | Path, *, name: str) -> None:
        """Upload an arbitrary already-written file (e.g. a report) as a run asset."""

        if self._experiment is not None:
            self._experiment.log_asset(str(path), file_name=name)

    def add_tags(self, tags: Sequence[str]) -> None:
        if self._experiment is not None and tags:
            self._experiment.add_tags([str(tag) for tag in tags])

    def close(self) -> dict[str, Any]:
        if self._experiment is not None:
            try:
                self._experiment.end()
            except Exception as exc:
                self.reason = f"end failed: {type(exc).__name__}"
                self.status = "error"
        return {
            "schema_version": 1,
            "status": self.status,
            "reference": self.reference,
            # The clickable link, not only the opaque key. A run publishes to
            # more than one experiment - the master, one per cell, one for the
            # post-run analysis - and only the master's URL was ever shown, so
            # everything uploaded to the others looked like it had not been
            # uploaded at all.
            "url": self.url,
            "reason": self.reason,
            "privacy": "aggregate metrics, requested plots, and requested reports only; prompts, blocks, messages, and responses are never sent",
        }


class RunRecorder:
    """Incremental Phase 7 event and audit writer.

    The runtime gives this object rich in-process records.  Its normal event files
    contain only operational/provenance fields; prompt content is copied only to
    local detailed files after the policy has selected the attempt.
    """

    schema_version = 1
    malformed_response_row_cap = 100

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        resolved_config: Mapping[str, Any],
        policy: DetailedAuditPolicy,
        comet_enabled: bool = False,
        project_name: str = "mas-cc",
        run_name: str | None = None,
        checkpoint_enabled: bool = True,
        price_snapshot_hash: str | None = None,
        metrics: Sequence[Metric] = (),
        to_round_view: Callable[[Any], Any] | None = None,
        comet_metric_export: Sequence[str] = (),
        binning: BinnedTrajectoryPolicy | None = None,
        retention_policy: RetentionPolicy | None = None,
        scientific_identity: ScientificIdentity | None = None,
        scientific_path: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config_hash = canonical_hash(resolved_config)
        self.selector = DetailedAuditSelector(policy)
        self.price_snapshot_hash = price_snapshot_hash
        self.checkpoint_enabled = checkpoint_enabled
        self.retention_policy = retention_policy or RetentionPolicy.for_profile("full")
        self.scientific_identity = scientific_identity
        self.scientific_path = (
            None if scientific_path is None else Path(scientific_path)
        )
        if self.retention_policy.compact_scientific and (
            self.scientific_identity is None or self.scientific_path is None
        ):
            raise ValueError(
                "results_only recorder requires scientific_identity and scientific_path"
            )
        self._started_at = _now()
        self._compact_rows: dict[int, dict[str, Any]] = {}
        self._episode_usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self._event_path = self.output_dir / "events.jsonl"
        self._log_path = self.output_dir / "experiment.log"
        self._api_path = self.output_dir / "api_call_status.jsonl"
        self._usage_path = self.output_dir / "usage_cost.jsonl"
        self._budget_path = self.output_dir / "budget_events.jsonl"
        self._audit_path = self.output_dir / "audit_traces.jsonl"
        self._blocks_path = self.output_dir / "prompt_block_traces.jsonl"
        self._trajectory_path = self.output_dir / "trajectory.jsonl"
        self._malformed_path = self.output_dir / "runtime" / "malformed_responses.jsonl"
        self._malformed_rows = 0
        if self.retention_policy.compact_scientific:
            assert self.scientific_identity is not None
            auxiliary_dir = (
                self.output_dir.parent.parent
                / "round_records"
                / self.scientific_identity.episode_id
            )
            auxiliary_dir.mkdir(parents=True, exist_ok=True)
            self._round_trajectory_path = auxiliary_dir / "round_trajectory.jsonl"
            self._micro_slot_trajectory_path = (
                auxiliary_dir / "micro_slot_trajectory.jsonl"
            )
        else:
            self._round_trajectory_path = self.output_dir / "round_trajectory.jsonl"
            self._micro_slot_trajectory_path = (
                self.output_dir / "micro_slot_trajectory.jsonl"
            )
        self._detailed_created = False
        self._metric_rows: list[dict[str, Any]] = []
        self._checkpoint_store = AtomicCheckpointStore(self.output_dir / ".checkpoints")
        self.comet = CometMetricSink(
            comet_enabled, project_name=project_name, run_name=run_name or run_id
        )
        self._logger = logging.getLogger(f"mas_cc.run.{run_id}")
        self._streaming_metrics = tuple(
            m for m in metrics if isinstance(m, StreamingMetric)
        )
        self._final_metrics = tuple(m for m in metrics if isinstance(m, FinalMetric))
        self._to_round_view = to_round_view
        self._comet_metric_export = frozenset(comet_metric_export)
        self._round_views: list[Any] = []
        self._streaming_metrics_path = self.output_dir / "metrics" / "streaming.csv"
        self._final_metrics_path = self.output_dir / "metrics" / "final.csv"
        self._binning = binning
        self._success_rate_path = self.output_dir / "metrics" / "success_rate.csv"
        self._production_probability_path = (
            self.output_dir / "metrics" / "production_probability.csv"
        )
        self._streaming_metrics_header_written = False

    def event(self, event_type: str, **payload: Any) -> None:
        row = {
            "schema_version": self.schema_version,
            "timestamp": _now(),
            "run_id": self.run_id,
            "event_type": event_type,
            **payload,
        }
        if self.retention_policy.verbose_episode_history:
            _jsonl(self._event_path, row)
            with self._log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{row['timestamp']} {event_type} {json.dumps({k: v for k, v in payload.items() if k not in {'prompt', 'response'}}, sort_keys=True, default=str)}\n"
                )
        self._logger.info(
            "%s %s",
            event_type,
            {k: v for k, v in payload.items() if k not in {"prompt", "response"}},
        )

    def record_attempt(
        self,
        *,
        round_index: int,
        game_id: str,
        request: Any,
        prompt: Any,
        response: Any | None,
        attempt: int,
        valid: bool,
        validation_error: str | None,
        validation_issues: Sequence[Any] = (),
        provider_error: Exception | None = None,
        budget_status: Mapping[str, Any] | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        metadata = request.metadata
        common = {
            "round_index": round_index,
            "interaction_id": str(metadata.get("interaction_id")),
            "agent_id": str(metadata.get("agent_id")),
            "decision_stage": metadata.get("decision_stage"),
            "attempt": attempt,
            "provider": None if response is None else response.provider,
            "model": None if response is None else response.model,
            "prompt_family": prompt.family,
            "prompt_version": prompt.version,
            "prompt_definition_hash": prompt.definition_hash,
            "prompt_instance_hash": prompt.instance_hash,
            "valid": valid,
            "validation_error": validation_error,
            "provider_error": None if provider_error is None else str(provider_error),
        }
        if response is not None:
            self._episode_usage["requests"] += 1
            self._episode_usage["input_tokens"] += int(response.usage.input_tokens or 0)
            self._episode_usage["output_tokens"] += int(
                response.usage.output_tokens or 0
            )
        if response is not None and validation_issues:
            self._record_malformed_response(
                common=common,
                response=response,
                request=request,
                issues=validation_issues,
            )
        if not self.retention_policy.verbose_episode_history:
            return
        _jsonl(
            self._api_path,
            {"schema_version": self.schema_version, "run_id": self.run_id, **common},
        )
        usage = {} if response is None else response.usage.to_dict()
        _jsonl(
            self._usage_path,
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                **common,
                "usage": usage,
                "price_snapshot_hash": self.price_snapshot_hash,
            },
        )
        selection = self.selector.select(
            game_id=game_id,
            round_index=round_index,
            provider_error=provider_error is not None,
            invalid_response=not valid,
        )
        if selection.selected:
            self._detailed_created = True
            _jsonl(
                self._audit_path,
                {
                    "schema_version": self.schema_version,
                    "run_id": self.run_id,
                    "selection_reason": selection.reason,
                    **common,
                    "observation": None if observation is None else dict(observation),
                    "compiled_messages": prompt.messages_as_dicts(),
                    "response": None if response is None else response.to_dict(),
                },
            )
            for block in prompt.blocks_as_dicts():
                _jsonl(
                    self._blocks_path,
                    {
                        "schema_version": self.schema_version,
                        "run_id": self.run_id,
                        "round_index": round_index,
                        "agent_id": common["agent_id"],
                        "prompt_definition_hash": prompt.definition_hash,
                        "prompt_instance_hash": prompt.instance_hash,
                        "selection_reason": selection.reason,
                        "binding_state": "bound",
                        **block,
                    },
                )
        self.event(
            "provider_attempt",
            **common,
            detailed_audit=selection.selected,
            detailed_audit_reason=selection.reason,
        )
        if attempt > 1:
            self.event("retry", **common)
        if not valid:
            self.event(
                "invalid_response" if provider_error is None else "provider_error",
                **common,
            )
        if budget_status is not None:
            self.record_budget("reconciliation", budget_status)

    def _record_malformed_response(
        self,
        *,
        common: Mapping[str, Any],
        response: Any,
        request: Any,
        issues: Sequence[Any],
    ) -> None:
        """Retain bounded field-level operational evidence, never full responses."""

        self._malformed_path.parent.mkdir(parents=True, exist_ok=True)
        if self._malformed_rows >= self.malformed_response_row_cap:
            if self._malformed_rows == self.malformed_response_row_cap:
                _jsonl(
                    self._malformed_path,
                    {
                        "schema_version": 1,
                        "timestamp": _now(),
                        "run_id": self.run_id,
                        "truncated": True,
                        "row_cap": self.malformed_response_row_cap,
                    },
                )
            self._malformed_rows += 1
            return
        issue = issues[0]
        usage = response.usage
        _jsonl(
            self._malformed_path,
            {
                "schema_version": 1,
                "timestamp": _now(),
                "run_id": self.run_id,
                "cell_id": (
                    None
                    if self.scientific_identity is None
                    else self.scientific_identity.cell_id
                ),
                "episode_id": (
                    None
                    if self.scientific_identity is None
                    else self.scientific_identity.episode_id
                ),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                **common,
                "validation_attempt": request.metadata.get("validation_attempt"),
                "validation_repair": request.metadata.get("validation_repair", False),
                "repair_schema_version": request.metadata.get("repair_schema_version"),
                "effective_messages_hash": request.metadata.get(
                    "effective_messages_hash"
                ),
                "correction_message_hash": request.metadata.get(
                    "correction_message_hash"
                ),
                "validation_issue_path": getattr(issue, "field", None),
                "validation_issue_category": (
                    "response_contract"
                    if str(getattr(issue, "field", "")).startswith("response")
                    else "game_action"
                ),
                "validation_issue_message": getattr(issue, "message", None),
                "received_value": getattr(issue, "invalid_value", None),
                "received_type": type(getattr(issue, "invalid_value", None)).__name__,
                "finish_reason": response.finish_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "repair_outcome": (
                    "correction_requested"
                    if int(request.metadata.get("validation_attempt", 1))
                    <= int(request.metadata.get("validation_retry_bound", 0))
                    else "exhausted"
                ),
            },
        )
        self._malformed_rows += 1

    def record_interaction(
        self,
        *,
        round_index: int,
        interaction: Any,
        budget_status: Mapping[str, Any],
        state: Mapping[str, Any],
        prompt_definitions: Mapping[str, str],
    ) -> None:
        metrics = {
            "round_index": round_index,
            "successful_interaction": int(
                bool(
                    getattr(
                        interaction.transition,
                        "success",
                        getattr(interaction.transition, "matched", False),
                    )
                )
            ),
            "payoff": getattr(interaction.transition, "payoff", 0.0),
            "provider_attempts": sum(
                getattr(decision, "validation_attempts", 0)
                for decision in getattr(interaction, "decisions", ())
            ),
        }
        self._metric_rows.append(metrics)
        self.comet.log_metrics(
            {k: float(v) for k, v in metrics.items() if k != "round_index"}, round_index
        )
        population_metrics, option_metrics = self._record_round_metrics(
            round_index, interaction.transition.next_state
        )
        if self.retention_policy.compact_scientific:
            assert self.scientific_identity is not None
            row = empty_compact_row(self.scientific_identity, round_index)
            row["population_metrics_json"] = json.dumps(
                population_metrics, sort_keys=True, ensure_ascii=False
            )
            row["option_metrics_json"] = json.dumps(
                option_metrics, sort_keys=False, ensure_ascii=False
            )
            self._compact_rows[round_index] = row
        self.event(
            "interaction_completed",
            interaction_id=str(interaction.interaction_id),
            **metrics,
        )
        self.event("heartbeat", completed_rounds=round_index, status="running")
        self.record_budget("interaction_checkpoint", budget_status)
        if self.checkpoint_enabled and self.retention_policy.verbose_episode_history:
            self._checkpoint_store.write(
                Checkpoint(
                    self.run_id,
                    round_index,
                    self.config_hash,
                    state,
                    budget_status,
                    prompt_definitions,
                )
            )
            self.event("checkpoint_written", completed_rounds=round_index)

    def record_trajectory(self, *, record: Any) -> None:
        """Persist a game-supplied rich trajectory row locally.

        Only runtimes that need more than the shared interaction metrics call
        this optional hook.  It deliberately never reaches Comet.
        """

        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        if self.retention_policy.compact_scientific:
            assert self.scientific_identity is not None
            event = payload.get("event")
            if not isinstance(event, Mapping):
                return
            if "within_round_index" in event:
                retained_fields = (
                    "round_index",
                    "within_round_index",
                    "microscopic_event_index",
                    "round_controller_action",
                    "round_controller_target",
                    "round_controller_advocate_probability",
                    "controlled_slot",
                    "intervention_budget",
                    "controlled_positions_hash_or_id",
                    "focal_opinion_before",
                    "focal_opinion_after",
                    "occupation_counts_before",
                    "occupation_counts_after",
                    "possible_answers",
                    "delta_m_ctrl",
                    "delta_m_truth",
                    "delta_m_order",
                    "truth_current_increment",
                    "social_mode",
                    "q_requested",
                    "q_effective",
                    "sampled_message_ids",
                    "sampled_message_authors",
                    "sampled_message_types",
                    "sampled_message_ages",
                    "sampled_controller_message_ids",
                    "focal_posted_message",
                    "new_message",
                    "new_message_id",
                    "new_message_type",
                    "new_message_reply_to",
                    "new_message_shared_fact_id",
                    "board_size_before",
                    "board_size_after",
                    "controller_actuation_mode",
                    "controller_message_posted",
                    "controller_message_id",
                    "controller_message_directly_exposed",
                    "theory_status",
                )
                _jsonl(
                    self._micro_slot_trajectory_path,
                    {
                        "schema_version": self.schema_version,
                        "run_id": self.run_id,
                        "cell_id": self.scientific_identity.cell_id,
                        "episode_id": self.scientific_identity.episode_id,
                        **{field: event.get(field) for field in retained_fields},
                    },
                )
            compact = compact_imitation_event(event, self.scientific_identity)
            index = int(compact["interaction_index"])
            existing = self._compact_rows.get(index, {})
            compact["population_metrics_json"] = existing.get("population_metrics_json")
            compact["option_metrics_json"] = existing.get("option_metrics_json")
            self._compact_rows[index] = compact
            return
        _jsonl(
            self._trajectory_path,
            {"schema_version": self.schema_version, "run_id": self.run_id, **payload},
        )

    def record_round_trajectory(self, *, record: Any) -> None:
        """Persist one coarse-grained game record independently of micro rows.

        Round records are compact, provider-free scientific data and are kept
        under every artifact profile so post-hoc round analysis never has to
        reconstruct its main channel by summing microscopic events.
        """

        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        if self.retention_policy.compact_scientific:
            assert self.scientific_identity is not None
            payload["cell_id"] = self.scientific_identity.cell_id
            payload["episode_id"] = self.scientific_identity.episode_id
        _jsonl(
            self._round_trajectory_path,
            {"schema_version": self.schema_version, "run_id": self.run_id, **payload},
        )

    def record_round_boundary(
        self,
        *,
        round_index: int,
        state: Mapping[str, Any],
        budget_status: Mapping[str, Any],
        prompt_definitions: Mapping[str, str],
    ) -> None:
        """Replace the micro-step checkpoint with the canonical round-end state."""

        if self.checkpoint_enabled and self.retention_policy.verbose_episode_history:
            completed = int(state.get("turn", round_index))
            self._checkpoint_store.write(
                Checkpoint(
                    self.run_id,
                    completed,
                    self.config_hash,
                    state,
                    budget_status,
                    prompt_definitions,
                )
            )
            self.event(
                "round_boundary_checkpoint_written",
                round_index=round_index,
                completed_rounds=completed,
            )

    def _existing_streaming_header(self) -> list[str] | None:
        """The header of an already-written streaming.csv, or None if there isn't one."""

        if not self._streaming_metrics_path.exists():
            return None
        with self._streaming_metrics_path.open(newline="", encoding="utf-8") as stream:
            return next(csv.reader(stream), None)

    def _record_round_metrics(
        self, round_index: int, game_state: Any
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Compute the game's declared streaming metrics for one round.

        No-op unless the caller supplied both metric instances and a
        ``to_round_view`` adapter; keeps this opt-in so runs that don't wire a
        game's metrics (e.g. the frozen Phase 7 gate) are unaffected.
        """
        if not self._streaming_metrics or self._to_round_view is None:
            return {}, {}
        view = self._to_round_view(game_state)
        # Compact scientific runs persist their canonical trajectory as it is
        # produced. Retain views only when a final metric actually consumes the
        # whole sequence; otherwise a view's cumulative state history makes
        # memory grow quadratically with episode length.
        if self._final_metrics or not self.retention_policy.compact_scientific:
            self._round_views.append(view)
        rows: list[dict[str, Any]] = []
        comet_values: dict[str, float] = {}
        population_values: dict[str, float] = {}
        option_values: dict[str, dict[str, float]] = {}
        for metric in self._streaming_metrics:
            for key, value in metric.compute_round(view).items():
                # `key` means different things per scope, so it lands in a
                # different column: agent ids identify a row's agent, option
                # labels identify one curve within a single metric's family.
                rows.append(
                    {
                        "round_index": round_index,
                        "episode_id": self.run_id,
                        "agent_id": str(key) if metric.scope == "agent" else "",
                        "series": str(key) if metric.scope == "option" else "",
                        "metric_name": metric.name,
                        "value": value,
                    }
                )
                if metric.scope != "agent" and metric.name in self._comet_metric_export:
                    # Comet keys must be unique, so each option curve is exported
                    # under its own suffixed key while staying one metric locally.
                    comet_key = (
                        metric.name
                        if metric.scope == "population"
                        else f"{metric.name}_{key}"
                    )
                    comet_values[comet_key] = float(value)
                if metric.scope == "population" and value is not None:
                    try:
                        population_values[metric.name] = float(value)
                    except (TypeError, ValueError):
                        pass
                elif metric.scope == "option" and value is not None:
                    try:
                        option_values.setdefault(metric.name, {})[str(key)] = float(
                            value
                        )
                    except (TypeError, ValueError):
                        pass
        if not rows:
            return population_values, option_values
        if not self.retention_policy.verbose_episode_history:
            return population_values, option_values
        self._streaming_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        mode, write_header = "a", False
        if not self._streaming_metrics_header_written:
            # First write of this recorder's life. The run id is derived from
            # the config (experiment name + seed), so re-running one config
            # lands in the directory a previous run already used - append only
            # when that file is genuinely this same schema (a resumed episode).
            # A header from an older metrics schema means a stale file whose
            # rows can no longer be parsed alongside ours, so it is replaced
            # rather than appended to, which would interleave two row widths
            # under one header and make every reader mis-parse the file.
            existing = self._existing_streaming_header()
            if existing is None:
                write_header = True
            elif existing != STREAMING_METRIC_FIELDS:
                mode, write_header = "w", True
                self._logger.warning(
                    "replacing %s: its header %s predates the current metrics schema %s",
                    self._streaming_metrics_path,
                    existing,
                    list(STREAMING_METRIC_FIELDS),
                )
        with self._streaming_metrics_path.open(
            mode, newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=STREAMING_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self._streaming_metrics_header_written = True
        if comet_values:
            self.comet.log_metrics(comet_values, round_index)
        return population_values, option_values

    def _final_metric_values(self) -> dict[str, float | None]:
        if not self._final_metrics:
            return {}
        views = tuple(self._round_views)
        values = {
            metric.name: metric.compute_final(views) for metric in self._final_metrics
        }
        return values

    def _write_final_metrics(self) -> None:
        values = self._final_metric_values()
        if not values:
            return
        rows = [
            {"episode_id": self.run_id, "metric_name": name, "value": value}
            for name, value in values.items()
        ]
        self._final_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._final_metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["episode_id", "metric_name", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_binned_metrics(self) -> None:
        """The two binned trajectory tables, from this episode's interaction records.

        Opt-in: no ``binning`` policy means no tables, same stance as the
        streaming metrics. Both are derived from the persisted per-interaction
        record rather than accumulated during the run, so re-deriving them from
        a finished episode gives byte-identical output.
        """

        if self._binning is None or not self._round_views:
            return
        view = self._round_views[-1]
        # Only rounds that actually *are* pairwise interaction outcomes. A game
        # with a uniform per-round outcome (naming_convention) declares
        # `interaction_index` and `actions` on every entry and is unaffected; a
        # phase-structured game (hidden_bench) declares them only on its
        # outcome-bearing rounds, so its discussion turns are skipped instead of
        # being binned into a success rate that would mix votes with messages.
        records = [
            InteractionOutcome.from_evaluator_entry(entry)
            for entry in getattr(view, "recent_history", ())
            if "interaction_index" in entry and "actions" in entry
        ]
        if not records:
            return
        success_rows, production_rows = binned_trajectory_tables(
            records,
            episode_id=self.run_id,
            policy=self._binning,
            action_space=getattr(view, "options", ()) or None,
        )
        for path, fields, rows in (
            (self._success_rate_path, SUCCESS_RATE_FIELDS, success_rows),
            (
                self._production_probability_path,
                PRODUCTION_PROBABILITY_FIELDS,
                production_rows,
            ),
        ):
            if not rows:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def record_budget(self, event: str, status: Mapping[str, Any]) -> None:
        if not self.retention_policy.verbose_episode_history:
            return
        _jsonl(
            self._budget_path,
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "event": event,
                "price_snapshot_hash": self.price_snapshot_hash,
                "budget": dict(status),
            },
        )

    def finalize(
        self,
        *,
        status: str,
        budget_status: Mapping[str, Any],
        termination_reason: str | None = None,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        if self.retention_policy.compact_scientific:
            comet = self.comet.close()
            if status != "completed":
                return {
                    "audit": self.selector.summary(),
                    "comet": comet,
                    "checkpoint": None,
                    "detailed_files_created": False,
                    "scientific_path": None,
                    "error_type": None if error is None else type(error).__name__,
                }
            final_metrics = json.dumps(
                self._final_metric_values(), sort_keys=True, ensure_ascii=False
            )
            rows = [self._compact_rows[index] for index in sorted(self._compact_rows)]
            for row in rows:
                row["final_metrics_json"] = final_metrics
            assert self.scientific_identity is not None
            assert self.scientific_path is not None
            path = write_completed_episode(
                self.scientific_path,
                rows,
                self.scientific_identity,
                termination_reason=termination_reason,
                started_at=self._started_at,
                usage=self._episode_usage,
            )
            return {
                "audit": self.selector.summary(),
                "comet": comet,
                "checkpoint": {"mode": "episode", "path": str(path)},
                "detailed_files_created": False,
                "scientific_path": str(path),
            }
        self.record_budget("run_completed", budget_status)
        self.event("run_completed", status=status, audit=self.selector.summary())
        metrics_path = self.output_dir / "local_metrics.csv"
        fields = [
            "round_index",
            "successful_interaction",
            "payoff",
            "provider_attempts",
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._metric_rows)
        self._write_final_metrics()
        self._write_binned_metrics()
        checkpoint = self._checkpoint_store.load()
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "checkpoint_path": f".checkpoints/{self._checkpoint_store.filename}",
            "completed_rounds": None
            if checkpoint is None
            else checkpoint.completed_rounds,
            "resolved_config_hash": self.config_hash,
            "prompt_state_restored": False,
        }
        (self.output_dir / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        comet = self.comet.close()
        (self.output_dir / "comet_summary.json").write_text(
            json.dumps(comet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "audit": self.selector.summary(),
            "comet": comet,
            "checkpoint": manifest,
            "detailed_files_created": self._detailed_created,
        }


def price_snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
