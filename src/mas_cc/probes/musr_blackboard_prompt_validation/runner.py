"""Prepare, execute, analyze, and seal blackboard prompt validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    BlackboardBallotContract,
)
from mas_cc.llm_runtime.providers import UniversityPricingSource
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)
from mas_cc.probes.musr_symbolic_ambiguity.analysis import write_csv
from mas_cc.probes.musr_symbolic_ambiguity_replication.runner import (
    load_tasks as load_source_tasks,
)
from mas_cc.probes.musr_symbolic_ambiguity_replication.runner import (
    verify_source_reference,
)

from .analysis import build_outputs
from .config import BlackboardValidationConfig
from .execution import call_plan, execute, read, terminal
from .states import build_frozen_states, load_frozen_states, write_state_artifacts


def _git() -> dict[str, Any]:
    try:
        return {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _source_config(config: BlackboardValidationConfig):
    from mas_cc.probes.musr_symbolic_ambiguity_replication.config import (
        ReplicationConfig,
    )

    return ReplicationConfig(
        source_path=config.source_path,
        calibration_root=config.calibration_root,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
        expected_manifest_content_sha256=config.expected_manifest_content_sha256,
        expected_symbolic_selection_sha256=config.expected_symbolic_selection_sha256,
        behavioral_provider=config.provider,
        population_size=12,
        seed=20260901,
        prompt_variant="P2",
        full_profile="F9",
        private_breadth=4,
        max_predictability=0.45,
        min_normalized_entropy=0.90,
        min_score_margin=2,
        private_existing_repetitions=3,
        private_additional_repetitions=3,
        endpoint_existing_repetitions=10,
        endpoint_additional_repetitions=10,
        zero_max_truth_rate=0.45,
        private_max_truth_rate=0.45,
        full_min_truth_rate=0.80,
        borderline_private_max=0.50,
        minimum_full_private_separation=0.25,
        task_pathology_full_below=0.50,
        behavioral_workers=min(4, config.provider.request_concurrency),
        output_dir=Path("unused"),
        max_behavioral_requests=360,
        max_behavioral_input_tokens=4_000_000,
        max_behavioral_output_tokens=1_500_000,
        max_behavioral_cost=5.0,
        accounting_unit=config.accounting_unit,
    )


def _tasks(config: BlackboardValidationConfig):
    all_tasks = load_source_tasks(_source_config(config))
    return {task_id: all_tasks[task_id] for task_id in config.task_ids}


def _sanity(
    root: Path, states: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_rows = []
    schema_rows = []
    lifetime_rows = []
    forbidden = (
        "skill_matrix",
        "cooperation_matrix",
        "candidate_scores",
        "gold_answer",
        "hidden_claim",
        "max_predictability",
        "normalized_entropy",
    )
    for key in sorted(states):
        frozen = states[key]
        row = frozen.definition
        prompt = "\n".join(
            message.content for message in frozen.compiled_prompt.messages
        )
        evidence_rows.append(
            {
                "state_key": key,
                "original_preserved": set(row["original_evidence_ids"]).issubset(
                    row["total_evidence_ids"]
                ),
                "acquired_persisted": set(row["acquired_evidence_ids"]).issubset(
                    row["total_evidence_ids"]
                ),
                "semantic_only_added_no_exact_evidence": all(
                    value is None for value in row["sampled_shared_fact_ids"]
                ),
                "private_reason_not_rendered": "PRIVATE_SENTINEL_MUST_NOT_RENDER"
                not in prompt,
                "hidden_metadata_absent": all(term not in prompt for term in forbidden),
                "coverage_expected": row["latent_coverage_count"]
                == {"S0": 4, "S1": 6, "S2": 9}[row["state_id"]],
            }
        )
        contract = frozen.request.prompt.response_contract
        assert isinstance(contract, BlackboardBallotContract)
        visible = tuple(row["sampled_message_ids"])
        vote = next(iter(row["option_letters"]))
        for message_type in (
            "CLAIM",
            "QUESTION",
            "REQUEST",
            "RESULT",
            "REPLY",
            "CORRECTION",
        ):
            reply_to = (
                visible[0]
                if message_type in {"REPLY", "CORRECTION"} and visible
                else None
            )
            response = json.dumps(
                {
                    "vote": vote,
                    "reason": "private",
                    "shared_fact_id": "none",
                    "public_message": {
                        "type": message_type,
                        "text": "public",
                        "reply_to": reply_to,
                    },
                }
            )
            valid = contract.validate(response).valid
            expected_valid = message_type not in {"REPLY", "CORRECTION"} or bool(
                visible
            )
            schema_rows.append(
                {
                    "state_key": key,
                    "message_type": message_type,
                    "expected_valid": expected_valid,
                    "observed_valid": valid,
                    "passed": valid == expected_valid,
                    "reply_to": reply_to,
                }
            )
        if visible:
            invalid = json.dumps(
                {
                    "vote": vote,
                    "reason": "private",
                    "shared_fact_id": "none",
                    "public_message": {
                        "type": "REPLY",
                        "text": "public",
                        "reply_to": "missing",
                    },
                }
            )
            schema_rows.append(
                {
                    "state_key": key,
                    "message_type": "REPLY_INVALID",
                    "expected_valid": False,
                    "observed_valid": contract.validate(invalid).valid,
                    "passed": not contract.validate(invalid).valid,
                    "reply_to": "missing",
                }
            )
        live_round = int(row["state_turn"]) // 12
        board = frozen.state.blackboard
        lifetime_rows.append(
            {
                "state_key": key,
                "expired_absent_from_live": all(
                    message_id
                    not in {
                        message.message_id
                        for message in board.live_messages(live_round)
                    }
                    for message_id in row["expired_message_ids"]
                ),
                "acquired_persists_after_expiry": set(
                    row["acquired_evidence_ids"]
                ).issubset(
                    frozen.state.relational_agent(
                        frozen.request.agent_id
                    ).known_fact_ids
                ),
                "active_equals_known_at_rho_1": frozen.state.relational_agent(
                    frozen.request.agent_id
                ).active_fact_ids
                == frozen.state.relational_agent(
                    frozen.request.agent_id
                ).known_fact_ids,
                "sample_count_q1": len(row["sampled_message_ids"])
                in ({0} if row["state_id"] == "S0" else {1}),
            }
        )
    return evidence_rows, schema_rows, lifetime_rows


def _approval(config: BlackboardValidationConfig, payload: Mapping[str, Any]) -> str:
    return sha256_object({"config": config.to_dict(), "preflight": payload})


def prepare(
    config: BlackboardValidationConfig, output_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = verify_source_reference(_source_config(config))
    tasks = _tasks(config)
    states = build_frozen_states(config, tasks)
    if config.mode == "full":
        write_state_artifacts(root, states)
    else:
        smoke_states = root / "states/smoke_state_definitions.json"
        write_json_atomic(
            smoke_states,
            [states[key].definition for key in sorted(states)],
        )
    calls = call_plan(config, states)
    evidence, schema, lifetime = _sanity(root, states)
    write_csv(root / "sanity/evidence_memory_checks.csv", evidence)
    write_csv(root / "sanity/message_schema_checks.csv", schema)
    write_csv(root / "sanity/board_lifetime_checks.csv", lifetime)
    all_checks = [
        *(
            bool(value)
            for row in (*evidence, *lifetime)
            for key, value in row.items()
            if key != "state_key"
        ),
        *(bool(row["passed"]) for row in schema),
    ]
    tokens = sum(
        states[call.state_key].compiled_prompt.total_token_estimate or 0
        for call in calls
    )
    maximum_validated_attempts = config.invalid_response_retries + 1
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    latency = float(config.provider.options.get("estimated_latency_seconds", 10.0))
    effective_concurrency = min(
        config.max_concurrency, config.provider.request_concurrency
    )
    payload = {
        "probe": "musr_blackboard_prompt_validation",
        "mode": config.mode,
        "passed": (
            all(all_checks)
            and quote.status == "known"
            and tokens * maximum_validated_attempts <= config.max_input_tokens
            and len(calls) * maximum_validated_attempts <= config.max_provider_requests
            and len(calls)
            * config.provider.max_output_tokens
            * maximum_validated_attempts
            <= config.max_output_tokens_total
        ),
        "source": source,
        "states": len(states),
        "logical_calls": len(calls),
        "provider_attempt_ceiling": config.max_provider_requests,
        "rendered_input_estimate": tokens * maximum_validated_attempts,
        "output_token_ceiling": len(calls)
        * config.provider.max_output_tokens
        * maximum_validated_attempts,
        "execution": {
            "local_workers": config.local_workers,
            "max_concurrency": config.max_concurrency,
            "provider_request_concurrency": config.provider.request_concurrency,
            "max_rpm": config.max_rpm,
            "fallback_concurrency": list(config.fallback_concurrency),
            "expected_wall_seconds": len(calls) / effective_concurrency * latency,
            "conservative_wall_seconds": len(calls)
            / effective_concurrency
            * config.provider.timeout_seconds,
        },
        "pricing": quote.to_dict(),
        "cost": {
            "conservative": quote.pricing.cost(
                config.max_input_tokens, config.max_output_tokens_total
            ).to_dict()
            if quote.pricing
            else None,
            "accounting_unit": config.accounting_unit,
        },
    }
    preflight = root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    write_json_atomic(preflight / f"{config.mode}_preflight.json", payload)
    write_json_atomic(
        preflight / f"{config.mode}_call_plan.json", [call.to_dict() for call in calls]
    )
    (preflight / f"{config.mode}_preflight_id.txt").write_text(
        _approval(config, payload) + "\n", encoding="utf-8"
    )
    (preflight / f"{config.mode}_report.md").write_text(
        f"# Blackboard prompt validation {config.mode} preflight\n\n- Passed: **{payload['passed']}**\n- Frozen states: **{len(states)}**\n- Logical calls: **{len(calls)}**\n- Maximum validated provider attempts: **{len(calls) * maximum_validated_attempts}**\n- Model: `{config.provider.model}`\n- Local workers: **{config.local_workers}**\n- Provider concurrency: **{config.provider.request_concurrency}**\n- Global cap: **{config.max_concurrency} concurrent / {config.max_rpm} RPM**\n- Input estimate including validation retries: **{tokens * maximum_validated_attempts:,} tokens**\n- Expected wall time: **{payload['execution']['expected_wall_seconds'] / 60:.1f} minutes**\n- Conservative wall time: **{payload['execution']['conservative_wall_seconds'] / 3600:.2f} hours**\n- No task or evidence generation is performed.\n",
        encoding="utf-8",
    )
    config_name = "config_smoke.yaml" if config.mode == "smoke" else "config_full.yaml"
    (root / config_name).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    write_json_atomic(root / "source_reference.json", source)
    (root / "README.md").write_text(
        "# MuSR blackboard prompt validation 01\n\nDevelopment harness and smoke test. The full 360-call run is intentionally not executed here.\n",
        encoding="utf-8",
    )
    build_outputs(root, config)
    return root, payload


def _verify_approval(
    root: Path, config: BlackboardValidationConfig, value: Path | str | None
) -> None:
    approved = (
        Path(value).read_text(encoding="utf-8").strip()
        if value is not None and Path(str(value)).is_file()
        else str(value or "").strip()
    )
    expected = (
        (root / f"preflight/{config.mode}_preflight_id.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if approved != expected:
        raise RuntimeError(
            "blackboard validation run requires the matching preflight approval ID"
        )


async def run(
    config: BlackboardValidationConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    _verify_approval(root, config, approve_preflight)
    verify_source_reference(_source_config(config))
    states_path = (
        root / "states/smoke_state_definitions.json"
        if config.mode == "smoke"
        else root / "states/frozen_state_definitions.json"
    )
    states = load_frozen_states(config, _tasks(config), states_path)
    calls = call_plan(config, states)
    journal = (
        root / "behavioral/smoke_raw_calls.jsonl"
        if config.mode == "smoke"
        else root / "behavioral/full_raw_calls.jsonl"
    )
    archived_rows = read(journal)
    attempts = sum(
        len(row.get("attempts") or ())
        for row in archived_rows
        if row.get("event") in {"call_finished", "call_failed"}
    )
    latest = terminal(journal)
    outstanding = sum(
        call.call_id not in latest or latest[call.call_id].get("event") == "call_failed"
        for call in calls
    )
    maximum_new_attempts = outstanding * (config.invalid_response_retries + 1)
    if attempts + maximum_new_attempts > config.max_provider_requests:
        raise RuntimeError("cumulative provider attempts would exceed the smoke budget")
    execution = await execute(
        config, states, calls, journal, root / "runtime/provider-control"
    )
    prior_manifest = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_execution = prior_manifest.get("execution") or {}
    if execution["attempted_now"] == 0:
        execution["wall_clock_seconds"] = float(
            prior_execution.get("wall_clock_seconds", execution["wall_clock_seconds"])
        )
        execution["observed_peak_concurrency"] = int(
            prior_execution.get(
                "observed_peak_concurrency", execution["observed_peak_concurrency"]
            )
        )
        execution["observed_peak_rolling_60s_dispatches"] = int(
            prior_execution.get(
                "observed_peak_rolling_60s_dispatches",
                execution["observed_peak_rolling_60s_dispatches"],
            )
        )
        execution["observed_sustained_rpm"] = float(
            prior_execution.get(
                "observed_sustained_rpm", execution["observed_sustained_rpm"]
            )
        )
        execution["final_fallback_concurrency"] = int(
            prior_execution.get(
                "final_fallback_concurrency",
                execution["final_fallback_concurrency"],
            )
        )
    outputs = build_outputs(root, config, execution)
    journal_rows = read(journal)
    execution["logical_request_starts"] = sum(
        row.get("event") == "request_started" for row in journal_rows
    )
    execution["archived_failed_runs"] = sum(
        row.get("event") == "call_failed" for row in journal_rows
    )
    execution["provider_attempts"] = sum(
        len(row.get("attempts") or ())
        for row in journal_rows
        if row.get("event") in {"call_finished", "call_failed"}
    )
    artifacts = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "runtime" not in path.relative_to(root).parts
    }
    manifest = {
        "schema_version": 1,
        "probe": "musr_blackboard_prompt_validation",
        "mode": config.mode,
        "status": "complete"
        if execution["successful"] == execution["scheduled"]
        else "incomplete",
        "decision": outputs["decision"],
        "config_sha256": sha256_object(config.to_dict()),
        "execution": execution,
        "artifact_hashes": artifacts,
        "mas_cc_git": _git(),
    }
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {"output": str(root), "execution": execution, **outputs}


def analyze(
    config: BlackboardValidationConfig, output_dir: Path | None = None
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    manifest_path = root / "manifest.json"
    execution = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("execution", {})
        if manifest_path.is_file()
        else {}
    )
    return build_outputs(root, config, execution)


def install_full_config(
    smoke_config: BlackboardValidationConfig, full_config_path: Path, root: Path
) -> None:
    if not full_config_path.is_file():
        raise RuntimeError(f"full config does not exist: {full_config_path}")
    shutil.copyfile(full_config_path, root / "config_full.yaml")


__all__ = ["analyze", "install_full_config", "prepare", "run"]
