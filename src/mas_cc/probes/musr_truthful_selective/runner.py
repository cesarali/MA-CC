"""Prepare, execute, analyze, and seal truthful-selective calibration."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)

from .analysis import build_outputs
from .config import TruthfulSelectiveConfig
from .design import call_plan
from .diversity import build_diversity_audit
from .execution import execute
from .generation import _load_task, generate, input_token_estimate
from .prompting import render
from .symbolic import repair_controller_ranking, run_symbolic_scan


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


def _approval(config: TruthfulSelectiveConfig, payload: Mapping[str, Any]) -> str:
    return sha256_object({"config": config.to_dict(), "preflight": payload})


def _task_roots(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in (root / "tasks").glob("task_*")
            if path.is_dir() and re.fullmatch(r"task_\d{3}", path.name)
        )
    )


def _expected_tasks(root: Path) -> dict[str, int]:
    revision = root / "analysis/task_revision_manifest.json"
    if revision.is_file():
        payload = json.loads(revision.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in payload["final_tasks"].items()}
    return {"task_001": 42, "task_002": 53, "task_003": 130}


def load_tasks(config: TruthfulSelectiveConfig, root: Path) -> dict[str, Any]:
    return {
        path.name: load_musr_team_allocation_task(
            root / "tasks", path.name, population_size=config.symbolic.population_size
        )
        for path in _task_roots(root)
    }


def prepare(
    config: TruthfulSelectiveConfig, output_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scan_path = root / "symbolic_scan/scan_summary.json"
    if scan_path.is_file():
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        expected = _expected_tasks(root)
        actual = {
            path.name: int(
                json.loads((path / "task.json").read_text(encoding="utf-8"))[
                    "candidate_id"
                ]
            )
            for path in _task_roots(root)
        }
        if (
            scan.get("candidate_worlds_scanned") != config.candidate_worlds
            or actual != expected
        ):
            raise RuntimeError(
                "existing symbolic scan does not match frozen development tasks"
            )
    else:
        scan = run_symbolic_scan(config, root / "symbolic_scan")
    task_roots = _task_roots(root)
    generation_tokens = input_token_estimate(config, task_roots)
    exact_fact_calls = sum(len(_load_task(path)[1]) for path in task_roots)
    tasks = load_tasks(config, root)
    exact_behavioral_calls = len(call_plan(config, tasks, root / "tasks"))
    checks = [
        {
            "check": "symbolic_candidate_count",
            "passed": scan["candidate_worlds_scanned"] >= 10_000,
        },
        {
            "check": "symbolic_development_tasks",
            "passed": len(scan["selected_tasks"]) == config.development_tasks,
        },
        {
            "check": "symbolic_passes_exist",
            "passed": scan["symbolic_pass_count"] >= config.development_tasks,
        },
        {
            "check": "models_frozen",
            "passed": config.generation_provider.model == "microsoft/gpt-5.6-terra"
            and config.behavioral_provider.model == "gwdg/openai-gpt-oss-120b",
        },
        {
            "check": "request_budgets",
            "passed": config.maximum_generation_calls <= config.max_generation_requests
            and config.behavioral_calls <= config.max_behavioral_requests,
        },
        {
            "check": "token_budgets",
            "passed": generation_tokens * config.semantic_retries
            <= config.max_generation_input_tokens
            and config.maximum_generation_calls
            * config.generation_provider.max_output_tokens
            <= config.max_generation_output_tokens
            and config.behavioral_calls * config.behavioral_provider.max_output_tokens
            <= config.max_behavioral_output_tokens,
        },
    ]
    generation_latency = float(
        config.generation_provider.options.get("estimated_latency_seconds", 10)
    )
    behavioral_latency = float(
        config.behavioral_provider.options.get("estimated_latency_seconds", 10)
    )
    payload = {
        "probe": "musr_truthful_selective",
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "symbolic": scan,
        "calls": {
            "generation_evidence_exact": exact_fact_calls,
            "generation_semantic_audit_exact": exact_fact_calls,
            "generation_logical_exact": exact_fact_calls * 2,
            "generation_nominal_ceiling": config.nominal_generation_calls,
            "generation_conservative": config.maximum_generation_calls,
            "behavioral_exact": exact_behavioral_calls,
            "behavioral_ceiling": config.behavioral_calls,
        },
        "tokens": {
            "generation_input_estimate": generation_tokens,
            "generation_input_conservative": generation_tokens
            * config.semantic_retries,
            "behavioral_input_budget": config.max_behavioral_input_tokens,
        },
        "concurrency": {
            "generation_workers": config.generation_workers,
            "behavioral_workers": config.behavioral_workers,
            "generation_provider_limit": config.generation_provider.request_concurrency,
            "behavioral_provider_limit": config.behavioral_provider.request_concurrency,
        },
        "wall_time": {
            "expected_seconds": config.nominal_generation_calls
            / config.generation_workers
            * generation_latency
            + config.behavioral_calls / config.behavioral_workers * behavioral_latency,
            "conservative_seconds": config.maximum_generation_calls
            / config.generation_workers
            * config.generation_provider.timeout_seconds
            + config.behavioral_calls
            / config.behavioral_workers
            * config.behavioral_provider.timeout_seconds,
        },
        "provider_io": {
            "completion_calls": 0,
            "metadata_calls": 0,
            "note": "symbolic preflight is fully offline",
        },
        "pricing": {
            "generation": {"status": "not_queried_provider_free_preflight"},
            "behavioral": {"status": "not_queried_provider_free_preflight"},
        },
        "cost": {
            "accounting_unit": config.accounting_unit,
            "generation_limit": config.max_generation_cost,
            "behavioral_limit": config.max_behavioral_cost,
            "interpretation": "hard configured bounds, not predictions",
        },
    }
    preflight = root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    write_json_atomic(preflight / "preflight.json", payload)
    (preflight / "preflight_id.txt").write_text(
        _approval(config, payload) + "\n", encoding="utf-8"
    )
    (preflight / "report.md").write_text(
        f"# Truthful-selective calibration preflight\n\n- Passed: **{payload['passed']}**\n- Candidate worlds: {config.candidate_worlds:,}\n- Symbolic passes: {scan['symbolic_pass_count']}\n- Development tasks: {config.development_tasks}\n- Exact Terra logical calls: {exact_fact_calls * 2}\n- Exact behavioral calls: {exact_behavioral_calls}\n- No provider calls were made during this preflight.\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    write_json_atomic(
        root / "manifest.json",
        {
            "schema_version": 1,
            "probe": "musr_truthful_selective",
            "status": "planned",
            "config_sha256": sha256_object(config.to_dict()),
            "symbolic_scan_sha256": scan["fingerprint_sha256"],
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
        raise RuntimeError("probe run requires the matching preflight approval ID")


async def run(
    config: TruthfulSelectiveConfig,
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
    task_roots = _task_roots(root)
    expected = _expected_tasks(root)
    actual = {
        task_root.name: int(
            json.loads((task_root / "task.json").read_text(encoding="utf-8"))[
                "candidate_id"
            ]
        )
        for task_root in task_roots
    }
    if actual != expected:
        raise RuntimeError(f"frozen development tasks changed: {actual!r}")
    if not (root / "analysis/task_revision_manifest.json").is_file():
        for task_root in task_roots:
            repair_controller_ranking(task_root)
    pre_generation_tasks = load_tasks(config, root)
    pre_generation_specs = call_plan(config, pre_generation_tasks, root / "tasks")
    write_json_atomic(
        root / "preflight/frozen_behavioral_call_plan.json",
        [spec.to_dict() for spec in pre_generation_specs],
    )
    write_json_atomic(
        root / "analysis/controller_diversity_audit.json",
        [build_diversity_audit(task_root) for task_root in task_roots],
    )
    generation = await generate(config, task_roots, root)
    if generation.get("generation_validation_status") != "PASS":
        raise RuntimeError(
            "Terra evidence generation failed semantic validation; OSS was not run"
        )
    approval_payload = {
        "generation_manifest_sha256": sha256_file(
            root / "generation/terra_generation_manifest.json"
        ),
        "task_hashes": {
            task_root.name: json.loads(
                (task_root / "task.json").read_text(encoding="utf-8")
            )["task_hash"]
            for task_root in task_roots
        },
        "validation_status": "PASS",
    }
    write_json_atomic(root / "generation/oss_approval.json", approval_payload)
    tasks = load_tasks(config, root)
    specs = call_plan(config, tasks, root / "tasks")
    if [spec.to_dict() for spec in specs] != [
        spec.to_dict() for spec in pre_generation_specs
    ]:
        raise RuntimeError("Terra generation changed the frozen behavioral fact sets")
    rendered = {spec.call_id: render(tasks[spec.task_id], spec) for spec in specs}
    forbidden = (
        "skill_matrix",
        "cooperation_matrix",
        "candidate_scores",
        "gold_target",
        "false_target",
        "controller",
    )
    if any(
        term in "\n".join(message.content for message in prompt.messages).casefold()
        for prompt in rendered.values()
        for term in forbidden
    ):
        raise RuntimeError(
            "hidden or social metadata leaked into a local behavioral prompt"
        )
    exact_tokens = sum(prompt.token_estimate for prompt in rendered.values())
    if exact_tokens > config.max_behavioral_input_tokens:
        raise RuntimeError(
            "rendered behavioral prompts exceed the approved token budget"
        )
    write_json_atomic(
        root / "behavioral_local/call_plan.json", [spec.to_dict() for spec in specs]
    )
    examples = ["# Isolated production prompt examples", ""]
    for condition in (
        "ZERO",
        "PRIVATE",
        "C3",
        "C6",
        "C12",
        "C24",
        "DECISIVE",
        "C3+D",
        "C6+D",
        "C12+D",
        "C24+D",
        "FULL",
    ):
        spec = next(item for item in specs if item.condition == condition)
        examples.extend(
            [
                f"## {condition}",
                "",
                "```text",
                "\n\n".join(
                    message.content for message in rendered[spec.call_id].messages
                ),
                "```",
                "",
            ]
        )
    (root / "behavioral_local/prompt_examples.md").write_text(
        "\n".join(examples), encoding="utf-8"
    )
    execution = await execute(
        config, tasks, specs, rendered, root / "behavioral_local/raw_oss_calls.jsonl"
    )
    outputs = build_outputs(config, root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete"
            if execution["terminal"] == execution["scheduled"]
            else "incomplete",
            "acceptance_decision": outputs["decision"],
            "generation": generation,
            "behavioral_execution": execution,
            "artifact_hashes": {
                str(path.relative_to(root)): sha256_file(path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
        }
    )
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {
        "output": str(root),
        **outputs,
        "generation": generation,
        "execution": execution,
    }


async def run_generation_only(
    config: TruthfulSelectiveConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    _verify_approval(root, approve_preflight)
    task_roots = _task_roots(root)
    expected = _expected_tasks(root)
    actual = {
        path.name: int(
            json.loads((path / "task.json").read_text(encoding="utf-8"))["candidate_id"]
        )
        for path in task_roots
    }
    if actual != expected:
        raise RuntimeError(f"frozen development tasks changed: {actual!r}")
    if not (root / "analysis/task_revision_manifest.json").is_file():
        for path in task_roots:
            repair_controller_ranking(path)
    tasks = load_tasks(config, root)
    specs = call_plan(config, tasks, root / "tasks")
    write_json_atomic(
        root / "preflight/frozen_behavioral_call_plan.json",
        [spec.to_dict() for spec in specs],
    )
    write_json_atomic(
        root / "analysis/controller_diversity_audit.json",
        [build_diversity_audit(path) for path in task_roots],
    )
    generation = await generate(config, task_roots, root)
    return {"output": str(root), "generation": generation}


async def run_behavioral_only(
    config: TruthfulSelectiveConfig,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    manifest_path = root / "generation/terra_generation_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Terra generation manifest is missing")
    generation = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generation.get("generation_validation_status") != "PASS":
        raise RuntimeError("Terra validation did not pass; refusing OSS execution")
    tasks = load_tasks(config, root)
    specs = call_plan(config, tasks, root / "tasks")
    frozen = json.loads(
        (root / "preflight/frozen_behavioral_call_plan.json").read_text(
            encoding="utf-8"
        )
    )
    if [spec.to_dict() for spec in specs] != frozen:
        raise RuntimeError("behavioral call plan changed after Terra generation")
    rendered = {spec.call_id: render(tasks[spec.task_id], spec) for spec in specs}
    exact_tokens = sum(prompt.token_estimate for prompt in rendered.values())
    if exact_tokens > config.max_behavioral_input_tokens:
        raise RuntimeError(
            "rendered behavioral prompts exceed the approved token budget"
        )
    write_json_atomic(
        root / "behavioral_local/call_plan.json", [spec.to_dict() for spec in specs]
    )
    execution = await execute(
        config, tasks, specs, rendered, root / "behavioral_local/raw_oss_calls.jsonl"
    )
    outputs = build_outputs(config, root)
    return {
        "output": str(root),
        **outputs,
        "generation": generation,
        "execution": execution,
    }


def analyze(
    config: TruthfulSelectiveConfig, output_dir: Path | None = None
) -> dict[str, Any]:
    return build_outputs(config, Path(output_dir or config.output_dir))


__all__ = [
    "analyze",
    "load_tasks",
    "prepare",
    "run",
    "run_behavioral_only",
    "run_generation_only",
]
