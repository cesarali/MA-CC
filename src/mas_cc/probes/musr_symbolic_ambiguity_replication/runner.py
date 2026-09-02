"""Preflight, execute, analyze, and seal the frozen replication study."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.games.relational_reasoning.data import (
    RelationalTask,
    load_musr_team_allocation_task,
)
from mas_cc.llm_runtime.providers import UniversityPricingSource
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)
from mas_cc.probes.musr_prompt_solvability.execution import execute, read, terminal
from mas_cc.probes.musr_prompt_solvability.prompting import render
from mas_cc.probes.musr_symbolic_ambiguity.analysis import summarize, write_csv

from .analysis import build_outputs, existing_task_diagnostic, load_source_observations
from .config import ReplicationConfig
from .design import additional_call_plan

_REQUIRED_ARTIFACTS = (
    "symbolic_scan/frozen_selection.json",
    "accepted_tasks/generation_manifest.json",
    "accepted_tasks/full_profile_packets.json",
    "accepted_tasks/private_assignments.json",
    "preflight/behavioral_call_plan.json",
    "behavioral_validation/raw_calls.jsonl",
    "behavioral_validation/observation_level_results.csv",
    "config.yaml",
    *(
        f"accepted_tasks/task_{index:03d}/{name}"
        for index in range(1, 7)
        for name in ("base_task.json", "distribution_N12.json")
    ),
)


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


def _approval(config: ReplicationConfig, payload: Mapping[str, Any]) -> str:
    return sha256_object({"config": config.to_dict(), "preflight": payload})


def verify_source_reference(config: ReplicationConfig) -> dict[str, Any]:
    root = config.calibration_root
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"source calibration manifest does not exist: {manifest_path}"
        )
    if sha256_file(manifest_path) != config.expected_manifest_file_sha256:
        raise RuntimeError("source calibration manifest file hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("acceptance_decision") != "FAIL"
    ):
        raise RuntimeError(
            "source calibration is not the expected completed FAIL result"
        )
    if (
        manifest.get("manifest_content_sha256")
        != config.expected_manifest_content_sha256
    ):
        raise RuntimeError("source manifest content identity changed")
    if (
        manifest.get("symbolic_selection_sha256")
        != config.expected_symbolic_selection_sha256
    ):
        raise RuntimeError("source symbolic selection identity changed")
    artifact_hashes = manifest.get("artifact_hashes") or {}
    verified: dict[str, str] = {}
    for relative in _REQUIRED_ARTIFACTS:
        expected = artifact_hashes.get(relative)
        path = root / relative
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source artifact failed hash verification: {relative}")
        verified[relative] = str(expected)
    frozen = json.loads(
        (root / "symbolic_scan/frozen_selection.json").read_text(encoding="utf-8")
    )
    rule = frozen["construction_rule"]
    if (
        int(rule["private_breadth"]),
        float(rule["max_predictability"]),
        float(rule["min_normalized_entropy"]),
        int(rule["min_score_margin"]),
    ) != (
        config.private_breadth,
        config.max_predictability,
        config.min_normalized_entropy,
        config.min_score_margin,
    ):
        raise RuntimeError(
            "source construction rule differs from the frozen replication rule"
        )
    generation = json.loads(
        (root / "accepted_tasks/generation_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "calibration_root": str(root),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "symbolic_selection_sha256": manifest["symbolic_selection_sha256"],
        "generation_fingerprint_sha256": generation["fingerprint_sha256"],
        "verified_artifact_hashes": verified,
    }


def load_tasks(config: ReplicationConfig) -> dict[str, RelationalTask]:
    root = config.calibration_root / "accepted_tasks"
    return {
        path.name: load_musr_team_allocation_task(
            root, path.name, population_size=config.population_size
        )
        for path in sorted(root.glob("task_*"))
        if path.is_dir()
    }


def _specs(config: ReplicationConfig, tasks: Mapping[str, RelationalTask]):
    return additional_call_plan(
        tasks,
        private_start=config.private_existing_repetitions,
        private_count=config.private_additional_repetitions,
        endpoint_start=config.endpoint_existing_repetitions,
        endpoint_count=config.endpoint_additional_repetitions,
        seed=config.seed,
    )


def _verify_prompts(
    config: ReplicationConfig, tasks: Mapping[str, RelationalTask], specs: Any
):
    rendered = {spec.call_id: render(tasks[spec.task_id], spec) for spec in specs}
    forbidden = (
        "skill_matrix",
        "cooperation_matrix",
        "candidate_scores",
        "hidden_claim",
        "gold_answer",
        "max_predictability",
        "normalized_entropy",
    )
    if any(
        term in "\n".join(message.content for message in prompt.messages)
        for prompt in rendered.values()
        for term in forbidden
    ):
        raise RuntimeError("hidden symbolic metadata leaked into a replication prompt")
    return rendered


def prepare(
    config: ReplicationConfig, output_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_reference = verify_source_reference(config)
    tasks = load_tasks(config)
    if len(tasks) != 6:
        raise RuntimeError(f"expected six frozen tasks, found {len(tasks)}")
    specs = _specs(config, tasks)
    if len(specs) != 336:
        raise RuntimeError("additional call plan is not exactly 336 calls")
    source_plan = json.loads(
        (config.calibration_root / "preflight/behavioral_call_plan.json").read_text(
            encoding="utf-8"
        )
    )
    source_ids = {str(row["call_id"]) for row in source_plan}
    new_ids = {spec.call_id for spec in specs}
    rendered = _verify_prompts(config, tasks, specs)
    exact_input_tokens = sum(prompt.token_estimate for prompt in rendered.values())
    quote = UniversityPricingSource(config.behavioral_provider).fetch(
        config.behavioral_provider.type, config.behavioral_provider.model
    )
    checks = [
        {"check": "source_hashes", "passed": True},
        {"check": "six_frozen_tasks", "passed": len(tasks) == 6},
        {"check": "additional_calls", "passed": len(specs) == 336},
        {"check": "disjoint_call_ids", "passed": not source_ids.intersection(new_ids)},
        {"check": "combined_unique_ids", "passed": len(source_ids | new_ids) == 672},
        {
            "check": "rendered_input_budget",
            "passed": exact_input_tokens <= config.max_behavioral_input_tokens,
        },
        {
            "check": "output_budget",
            "passed": len(specs) * config.provider.max_output_tokens
            <= config.max_behavioral_output_tokens,
        },
        {
            "check": "request_budget",
            "passed": len(specs) <= config.max_behavioral_requests,
        },
        {
            "check": "known_pricing",
            "passed": quote.status == "known" and quote.pricing is not None,
        },
    ]
    latency = float(config.provider.options.get("estimated_latency_seconds", 10.0))
    expected_wall = len(specs) / config.workers * latency
    conservative_wall = len(specs) / config.workers * config.provider.timeout_seconds
    payload = {
        "probe": "musr_symbolic_ambiguity_replication",
        "passed": all(bool(row["passed"]) for row in checks),
        "checks": checks,
        "source": source_reference,
        "calls": {
            "private": 216,
            "zero": 60,
            "full": 60,
            "new_total": 336,
            "final_total": 672,
        },
        "tokens": {
            "rendered_input_estimate": exact_input_tokens,
            "input_budget": config.max_behavioral_input_tokens,
            "output_ceiling": len(specs) * config.provider.max_output_tokens,
            "output_budget": config.max_behavioral_output_tokens,
        },
        "concurrency": {
            "workers": config.workers,
            "request_concurrency": config.provider.request_concurrency,
            "effective_ceiling": min(
                config.workers, config.provider.request_concurrency
            ),
        },
        "wall_time": {
            "expected_seconds": expected_wall,
            "expected_assumption": "configured latency per call divided by workers",
            "conservative_seconds": conservative_wall,
            "conservative_assumption": "every call consumes its full timeout",
        },
        "pricing": quote.to_dict(),
        "cost": {
            "conservative": quote.pricing.cost(
                config.max_behavioral_input_tokens, config.max_behavioral_output_tokens
            ).to_dict()
            if quote.pricing
            else None,
            "accounting_unit": config.accounting_unit,
            "interpretation": "hard configured bounds, not a spend prediction",
        },
        "recommendation_rule": config.to_dict()["recommendation_rule"],
    }
    preflight = root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "existing_data_reference.json", source_reference)
    write_json_atomic(preflight / "preflight.json", payload)
    write_json_atomic(
        preflight / "behavioral_call_plan.json", [spec.to_dict() for spec in specs]
    )
    write_csv(
        preflight / "existing_per_task_diagnostic.csv",
        existing_task_diagnostic(config.calibration_root),
    )
    existing_private_agents = summarize(
        [
            row
            for row in load_source_observations(config.calibration_root)
            if row["condition"] == "Private"
        ],
        ("task_id", "agent_id", "symbolic_M", "symbolic_Hbar"),
    )
    write_csv(
        preflight / "existing_per_task_agent_private_diagnostic.csv",
        existing_private_agents,
    )
    (preflight / "preflight_id.txt").write_text(
        _approval(config, payload) + "\n", encoding="utf-8"
    )
    (preflight / "report.md").write_text(
        f"""# MuSR symbolic ambiguity replication preflight

- Passed: **{payload["passed"]}**
- Frozen source observations inspected first: **336**
- New calls: **336** (216 Private + 60 Zero + 60 Full)
- Final observations: **672**
- Model: `{config.provider.model}`
- Prompt / full packet: **P2 / F9**
- Request concurrency: **{config.workers}**
- Rendered input estimate: **{exact_input_tokens:,} tokens**
- Output-token ceiling: **{len(specs) * config.provider.max_output_tokens:,} tokens**
- Expected wall time: **{expected_wall / 60:.1f} minutes**
- Conservative wall time: **{conservative_wall / 3600:.2f} hours**
- Cost units: **{config.accounting_unit}**; the reported cost is a hard configured bound
- No symbolic scan or evidence generation is part of this plan.
""",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# MuSR symbolic ambiguity replication 01\n\n"
        "This appends 336 observations to the immutable calibration data. "
        "See `analysis/symbolic_ambiguity_replication_report.md`.\n",
        encoding="utf-8",
    )
    write_json_atomic(
        root / "manifest.json",
        {
            "schema_version": 1,
            "probe": "musr_symbolic_ambiguity_replication",
            "status": "planned",
            "config_sha256": sha256_object(config.to_dict()),
            "source": source_reference,
            "behavioral_provider": config.provider.type,
            "behavioral_model": config.provider.model,
            "prompt_variant": config.prompt_variant,
            "full_profile": config.full_profile,
            "mas_cc_git": _git(),
        },
    )
    return root, payload


def _verify_approval(root: Path, value: Path | str | None) -> None:
    approved = (
        Path(value).read_text(encoding="utf-8").strip()
        if value is not None and Path(str(value)).is_file()
        else str(value or "").strip()
    )
    expected = (root / "preflight/preflight_id.txt").read_text(encoding="utf-8").strip()
    if approved != expected:
        raise RuntimeError(
            "replication run requires the matching preflight approval ID"
        )


def _cumulative_attempts(path: Path) -> int:
    return sum(row.get("event") == "request_started" for row in read(path))


async def run(
    config: ReplicationConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    _verify_approval(root, approve_preflight)
    preflight = json.loads(
        (root / "preflight/preflight.json").read_text(encoding="utf-8")
    )
    if not preflight.get("passed"):
        raise RuntimeError("preflight failed")
    verify_source_reference(config)
    tasks = load_tasks(config)
    specs = _specs(config, tasks)
    rendered = _verify_prompts(config, tasks, specs)
    journal = root / "behavioral_validation/new_raw_calls.jsonl"
    attempts_before = _cumulative_attempts(journal)
    latest = terminal(journal)
    outstanding = sum(
        spec.call_id not in latest or latest[spec.call_id].get("event") == "call_failed"
        for spec in specs
    )
    if attempts_before + outstanding > config.max_behavioral_requests:
        raise RuntimeError(
            "cumulative replication attempts would exceed the approved request budget"
        )
    execution = await execute(
        config,
        tasks,
        specs,
        rendered,
        journal,
        probe_name="musr_symbolic_ambiguity_replication",
        retry_failed=True,
    )
    completed = terminal(journal)
    completed_calls = sum(
        completed.get(spec.call_id, {}).get("event") == "call_finished"
        for spec in specs
    )
    execution["completed"] = completed_calls
    execution["unparsed_completed"] = sum(
        completed.get(spec.call_id, {}).get("event") == "call_finished"
        and completed.get(spec.call_id, {}).get("parse_success") is not True
        for spec in specs
    )
    if completed_calls != execution["scheduled"]:
        return {
            "output": str(root),
            "report": None,
            "decision": "INCOMPLETE",
            "execution": execution,
        }
    outputs = build_outputs(root, config.calibration_root, config)
    verify_source_reference(config)
    journal_rows = read(journal)
    final = terminal(journal)
    artifact_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "strict_gate": outputs["strict_gate"],
            "acceptance_decision": outputs["decision"],
            "behavioral_execution": {
                **execution,
                "provider_attempts": sum(
                    row.get("event") == "request_started" for row in journal_rows
                ),
                "archived_transport_failures": sum(
                    row.get("event") == "call_failed" for row in journal_rows
                ),
            },
            "observed_usage": {
                "new_behavioral_observations": len(final),
                "combined_behavioral_observations": outputs["observations"],
                "new_input_tokens": sum(
                    int((row.get("usage") or {}).get("input_tokens") or 0)
                    for row in final.values()
                ),
                "new_output_tokens": sum(
                    int((row.get("usage") or {}).get("output_tokens") or 0)
                    for row in final.values()
                ),
            },
            "separations": outputs["separations"],
            "artifact_hashes": artifact_hashes,
        }
    )
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {"output": str(root), "execution": execution, **outputs}


def analyze(
    config: ReplicationConfig, output_dir: Path | None = None
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    verify_source_reference(config)
    return build_outputs(root, config.calibration_root, config)


__all__ = ["analyze", "load_tasks", "prepare", "run", "verify_source_reference"]
