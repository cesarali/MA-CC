"""Offline preparation, live execution, and analysis for isolated OSS evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.musr_team_allocation_generator.io_utils import sha256_file, sha256_object, write_json_atomic

from .isolated_analysis import aggregate
from .isolated_config import IsolatedOSSConfig
from .isolated_design import build_manifest, load_selected_tasks, smoke_ids
from .isolated_execution import execute
from .isolated_prompting import render_isolated


def _git() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
        return {"commit": commit, "dirty": bool(diff), "working_tree_patch_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "working_tree_patch_sha256": None}


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _copy_frozen_tasks(config: IsolatedOSSConfig, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for task_id in sorted(config.expected_tasks):
        source = config.calibration_root / "tasks" / task_id
        target = destination / task_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _checksum_tree(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def prepare(config: IsolatedOSSConfig, output_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    prep = root / "preparation"
    prep.mkdir(parents=True, exist_ok=True)
    tasks = load_selected_tasks(config)
    manifest = build_manifest(config, tasks)
    rendered = {row.request_id: render_isolated(tasks[row.task_id], row, prompt_variant=config.prompt_variant) for row in manifest}
    forbidden = ("strategic", "random", "controller", "false target", "gold answer", "candidate_scores", "skill_matrix", "cooperation_matrix", "symbolic")
    for request in manifest:
        visible = "\n".join(message.content for message in rendered[request.request_id].messages).casefold()
        if any(term in visible for term in forbidden):
            raise RuntimeError(f"forbidden metadata leaked into {request.request_id}")
    manifest_rows = [
        {
            **row.to_dict(),
            "prompt_definition_hash": rendered[row.request_id].definition_hash,
            "prompt_instance_hash": rendered[row.request_id].instance_hash,
            "estimated_input_tokens": rendered[row.request_id].token_estimate,
        }
        for row in manifest
    ]
    _write_jsonl(prep / "evaluation_manifest.jsonl", manifest_rows)
    prompt_by_hash = {}
    for row in manifest:
        prompt = rendered[row.request_id]
        prompt_by_hash.setdefault(prompt.instance_hash, prompt.to_dict())
    _write_jsonl(
        prep / "prompt_archive.jsonl",
        ({"prompt_instance_hash": key, **value} for key, value in sorted(prompt_by_hash.items())),
    )
    write_json_atomic(prep / "smoke_request_ids.json", list(smoke_ids(manifest)))
    (prep / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    _copy_frozen_tasks(config, prep / "frozen_tasks")
    git = _git(); write_json_atomic(prep / "source_commit.json", git)
    counts = {
        condition: sum(row.condition == condition for row in manifest)
        for condition in ("full", "private", "strategic", "random")
    }
    total_input = sum(prompt.token_estimate for prompt in rendered.values())
    pricing = config.provider.options.get("offline_pricing_per_million_tokens")
    cost = None
    if isinstance(pricing, Mapping):
        input_rate = float(pricing.get("input", 0))
        output_rate = float(pricing.get("output", 0))
        cost = {
            "amount": total_input / 1_000_000 * input_rate
            + len(manifest) * config.provider.max_output_tokens / 1_000_000 * output_rate,
            "unit": config.accounting_unit,
            "source": "frozen config offline_pricing_per_million_tokens",
            "interpretation": "conservative output-token ceiling, not predicted spend",
        }
    preflight = {
        "passed": True,
        "provider_calls_made": 0,
        "model": config.provider.model,
        "temperature": config.provider.temperature,
        "prompt_variant": config.prompt_variant,
        "tasks": dict(config.expected_tasks),
        "population_size": config.assignment_population,
        "budgets": list(config.budgets),
        "answer_orders": config.answer_orders,
        "random_replicates": config.random_replicates,
        "logical_calls": len(manifest),
        "condition_counts": counts,
        "schema_retry_allowance": config.invalid_response_retries,
        "provider_attempt_ceiling": config.max_provider_attempts,
        "estimated_input_tokens": total_input,
        "output_token_ceiling": len(manifest) * config.provider.max_output_tokens,
        "cost": cost or "not estimated locally; query current provider pricing on cluster before launch",
        "duration": {
            name: {
                "assumed_seconds": (
                    len(smoke_ids(manifest)) if name == "smoke" else len(manifest)
                )
                * config.assumed_latency_seconds
                / profile.concurrency,
                "assumption": f"{config.assumed_latency_seconds}s mean latency, concurrency={profile.concurrency}, RPM cap={profile.requests_per_minute}",
            }
            for name, profile in (("smoke", config.smoke), ("cluster", config.cluster))
        },
        "output_dir": str(root),
    }
    write_json_atomic(prep / "preflight.json", preflight)
    identity = sha256_object({"config": config.to_dict(), "manifest": manifest_rows, "git": git})
    (prep / "preflight_id.txt").write_text(identity + "\n", encoding="utf-8")
    checksums = _checksum_tree(root, [path for path in prep.rglob("*") if path.is_file() and path.name not in {"checksum_manifest.json", "transfer_bundle.tar.gz"}])
    write_json_atomic(prep / "checksum_manifest.json", checksums)
    handoff = {
        "schema_version": 1,
        "result_root": str(root),
        "config": str(Path(config.source_path)),
        "preflight_approval": str(prep / "preflight_id.txt"),
        "commands": {
            "verify": f"python scripts/exploratory/verify_musr_isolated_bundle.py {root}",
            "preflight": f"mas-cc probe preflight --config {config.source_path} --output-dir {root}",
            "smoke": f"mas-cc probe run --config {config.source_path} --output-dir {root} --approve-preflight {prep / 'preflight_id.txt'} --request-set smoke --execution-profile smoke",
            "full": f"mas-cc probe run --config {config.source_path} --output-dir {root} --approve-preflight {prep / 'preflight_id.txt'} --request-set full --execution-profile cluster",
            "resume": f"mas-cc probe run --config {config.source_path} --output-dir {root} --approve-preflight {prep / 'preflight_id.txt'} --request-set full --execution-profile cluster",
            "analyze": f"mas-cc probe analyze --config {config.source_path} --output-dir {root}",
        },
        "cluster": {"conda": "/home/ojedamarin/.local/share/miniforge3/bin/conda", "generic_job": "scripts/Potsdam/SLURM/run_probe.job", "gpu_required": False},
        "cluster_result_root": "/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/musr_truthful_selective_isolated_oss_01",
    }
    write_json_atomic(prep / "cluster_handoff.json", handoff)
    (prep / "offline_test_report.md").write_text(
        f"# Offline preparation verification\n\n- Calls: {len(manifest):,}\n- Counts: `{json.dumps(counts, sort_keys=True)}`\n- Tasks: `{json.dumps(config.expected_tasks, sort_keys=True)}`\n- Live provider calls: **0**\n",
        encoding="utf-8",
    )
    write_json_atomic(prep / "offline_test_report.json", {"passed": True, **preflight})
    # Recompute checksums after all metadata files exist.
    write_json_atomic(
        prep / "checksum_manifest.json",
        _checksum_tree(root, [path for path in prep.rglob("*") if path.is_file() and path.name not in {"checksum_manifest.json", "transfer_bundle.tar.gz"}]),
    )
    bundle = prep / "transfer_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted(prep.rglob("*")):
            if path.is_file() and path != bundle:
                archive.add(path, arcname=str(path.relative_to(root)))
    return root, preflight


def _load_manifest(root: Path):
    from .isolated_design import IsolatedRequest
    rows = []
    for line in (root / "preparation/evaluation_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        rows.append(IsolatedRequest(
            request_id=raw["request_id"], paired_unit_id=raw["paired_unit_id"], task_id=raw["task_id"], candidate_id=int(raw["candidate_id"]), task_artifact_sha256=raw["task_artifact_sha256"], assignment_sha256=raw["assignment_sha256"], condition=raw["condition"], budget=raw.get("budget"), agent_id=raw.get("agent_id"), answer_order_id=raw["answer_order_id"], option_mapping=dict(raw["semantic_option_mapping"]), private_fact_ids=tuple(raw["private_fact_ids"]), report_fact_ids=tuple(raw["report_fact_ids"]), evidence_ids=tuple(raw["evidence_ids"]), random_replicate=raw.get("random_replicate"), random_seed=raw.get("random_seed"), unique_visible_fact_count=int(raw["unique_visible_fact_count"]), private_report_overlap_count=int(raw["private_report_overlap_count"]), latent_coverage_count=int(raw["latent_coverage_count"]), predicate_family_count=int(raw["predicate_family_count"]), compatible_worlds_prefix_only=raw.get("compatible_worlds_prefix_only"), compatible_worlds_private_plus_reports=raw.get("compatible_worlds_private_plus_reports"), compatible_world_reduction_prefix_only=raw.get("compatible_world_reduction_prefix_only"), compatible_world_reduction_private_plus_reports=raw.get("compatible_world_reduction_private_plus_reports"), report_character_count=int(raw.get("report_character_count", 0)),
        ))
    return tuple(rows)


async def run(
    config: IsolatedOSSConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: Path | str | None = None,
    request_set: str = "full",
    execution_profile: str = "cluster",
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    approved = Path(approve_preflight).read_text(encoding="utf-8").strip() if approve_preflight else ""
    expected = (root / "preparation/preflight_id.txt").read_text(encoding="utf-8").strip()
    if approved != expected:
        raise RuntimeError("isolated OSS run requires matching preflight approval")
    tasks = load_selected_tasks(config)
    manifest = _load_manifest(root)
    if request_set == "smoke":
        ids = set(json.loads((root / "preparation/smoke_request_ids.json").read_text(encoding="utf-8")))
        manifest = tuple(row for row in manifest if row.request_id in ids)
    elif request_set != "full":
        raise ValueError("request_set must be smoke or full")
    prompts = {row.request_id: render_isolated(tasks[row.task_id], row, prompt_variant=config.prompt_variant) for row in manifest}
    return await execute(config, tasks, manifest, prompts, root, execution_profile=execution_profile)


def analyze(config: IsolatedOSSConfig, output_dir: Path | None = None) -> dict[str, Any]:
    return aggregate(Path(output_dir or config.output_dir))


__all__ = ["analyze", "prepare", "run"]
