"""Prepare, execute, analyze, and seal symbolic ambiguity calibration 01."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
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
from mas_cc.probes.musr_prompt_solvability.design import packet
from mas_cc.probes.musr_prompt_solvability.execution import execute, read, terminal
from mas_cc.probes.musr_prompt_solvability.prompting import render

from .analysis import build_outputs
from .config import SymbolicAmbiguityConfig
from .design import call_plan
from .symbolic import run_symbolic_scan
from .tasks import generate_task_pack, generation_input_estimate


def _git() -> dict[str, Any]:
    try:
        return {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
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


def _approval(config: SymbolicAmbiguityConfig, payload: Mapping[str, Any]) -> str:
    return sha256_object({"config": config.to_dict(), "preflight": payload})


def _load_frozen(root: Path) -> dict[str, Any]:
    return json.loads((root / "symbolic_scan/frozen_selection.json").read_text(encoding="utf-8"))


def load_tasks(config: SymbolicAmbiguityConfig, root: Path) -> dict[str, RelationalTask]:
    return {
        path.name: load_musr_team_allocation_task(
            root / "accepted_tasks", path.name, population_size=config.population_size
        )
        for path in sorted((root / "accepted_tasks").glob("task_*"))
        if path.is_dir()
    }


def prepare(
    config: SymbolicAmbiguityConfig, output_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frozen = run_symbolic_scan(config, root / "symbolic_scan")
    selections = frozen["selected_worlds"]
    generation_tokens = generation_input_estimate(selections, config)
    generation_quote = UniversityPricingSource(config.generation_provider).fetch(
        config.generation_provider.type, config.generation_provider.model
    )
    behavioral_quote = UniversityPricingSource(config.behavioral_provider).fetch(
        config.behavioral_provider.type, config.behavioral_provider.model
    )
    rule = frozen["construction_rule"]
    checks = [
        {
            "check": "symbolic_candidate_count",
            "passed": frozen["candidate_worlds_scanned"] >= 10_000,
        },
        {
            "check": "preferred_criterion_selected",
            "passed": rule["criterion"] == "preferred",
        },
        {
            "check": "exact_gold_balance",
            "passed": Counter(item["gold_index"] for item in selections)
            == Counter({0: config.final_tasks // 3, 1: config.final_tasks // 3, 2: config.final_tasks // 3}),
        },
        {
            "check": "realized_private_views",
            "passed": all(
                len(item["private_views"]) == config.population_size
                and all(
                    row["max_predictability"] <= rule["max_predictability"]
                    and row["normalized_entropy"] >= rule["min_normalized_entropy"]
                    for row in item["private_views"]
                )
                for item in selections
            ),
        },
        {
            "check": "frozen_behavioral_model",
            "passed": config.behavioral_provider.model == "gwdg/openai-gpt-oss-120b",
        },
        {
            "check": "separate_generation_model",
            "passed": config.generation_provider.model != config.behavioral_provider.model,
        },
        {
            "check": "request_budgets",
            "passed": config.maximum_generation_calls <= config.max_generation_requests
            and config.behavioral_calls <= config.max_behavioral_requests,
        },
        {
            "check": "token_budgets",
            "passed": generation_tokens * config.semantic_retries <= config.max_generation_input_tokens
            and config.maximum_generation_calls * config.generation_provider.max_output_tokens
            <= config.max_generation_output_tokens
            and config.behavioral_calls * config.behavioral_provider.max_output_tokens
            <= config.max_behavioral_output_tokens,
        },
        {
            "check": "known_pricing",
            "passed": generation_quote.status == "known"
            and behavioral_quote.status == "known"
            and generation_quote.pricing is not None
            and behavioral_quote.pricing is not None,
        },
    ]
    latency_generation = float(config.generation_provider.options.get("estimated_latency_seconds", 10))
    latency_behavioral = float(config.behavioral_provider.options.get("estimated_latency_seconds", 10))
    expected_wall = (
        config.nominal_generation_calls / config.generation_workers * latency_generation
        + config.behavioral_calls / config.behavioral_workers * latency_behavioral
    )
    conservative_wall = (
        config.maximum_generation_calls / config.generation_workers * config.generation_provider.timeout_seconds
        + config.behavioral_calls / config.behavioral_workers * config.behavioral_provider.timeout_seconds
    )
    payload = {
        "probe": "musr_symbolic_ambiguity",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "symbolic": {
            "candidate_worlds": frozen["candidate_worlds_scanned"],
            "selected_rule": rule,
            "final_tasks": len(selections),
        },
        "calls": {
            "generation_nominal": config.nominal_generation_calls,
            "generation_expected": config.nominal_generation_calls,
            "generation_conservative": config.maximum_generation_calls,
            "behavioral": config.behavioral_calls,
            "nominal_total": config.nominal_generation_calls + config.behavioral_calls,
            "conservative_total": config.maximum_generation_calls + config.behavioral_calls,
        },
        "tokens": {
            "generation_input_nominal": generation_tokens,
            "generation_input_conservative": generation_tokens * config.semantic_retries,
            "generation_output_ceiling": config.maximum_generation_calls
            * config.generation_provider.max_output_tokens,
            "behavioral_input_budget": config.max_behavioral_input_tokens,
            "behavioral_output_ceiling": config.behavioral_calls
            * config.behavioral_provider.max_output_tokens,
        },
        "concurrency": {
            "generation_workers": config.generation_workers,
            "generation_request_concurrency": config.generation_provider.request_concurrency,
            "behavioral_workers": config.behavioral_workers,
            "behavioral_request_concurrency": config.behavioral_provider.request_concurrency,
            "effective_ceiling": max(config.generation_workers, config.behavioral_workers),
            "note": "generation and behavioral stages execute sequentially",
        },
        "wall_time": {
            "expected_seconds": expected_wall,
            "expected_assumption": "configured latency per call divided by stage workers",
            "conservative_seconds": conservative_wall,
            "conservative_assumption": "every call consumes its full timeout",
        },
        "pricing": {
            "generation": generation_quote.to_dict(),
            "behavioral": behavioral_quote.to_dict(),
        },
        "cost": {
            "generation_conservative": generation_quote.pricing.cost(
                config.max_generation_input_tokens, config.max_generation_output_tokens
            ).to_dict()
            if generation_quote.pricing
            else None,
            "behavioral_conservative": behavioral_quote.pricing.cost(
                config.max_behavioral_input_tokens, config.max_behavioral_output_tokens
            ).to_dict()
            if behavioral_quote.pricing
            else None,
            "accounting_unit": config.accounting_unit,
            "interpretation": "hard configured bounds, not spend predictions",
        },
    }
    preflight = root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    write_json_atomic(preflight / "preflight.json", payload)
    write_json_atomic(
        preflight / "pricing_snapshot.json",
        {"generation": generation_quote.to_dict(), "behavioral": behavioral_quote.to_dict()},
    )
    (preflight / "preflight_id.txt").write_text(_approval(config, payload) + "\n", encoding="utf-8")
    (preflight / "report.md").write_text(
        f"""# MuSR symbolic ambiguity calibration preflight

- Passed: **{payload['passed']}**
- Candidate worlds: {frozen['candidate_worlds_scanned']:,}
- Frozen rule: k={rule['private_breadth']}, M<={rule['max_predictability']}, Hbar>={rule['min_normalized_entropy']}, margin>={rule['min_score_margin']}
- Final balanced worlds: {len(selections)}
- Nominal calls: {payload['calls']['nominal_total']} ({config.nominal_generation_calls} generation + {config.behavioral_calls} behavioral)
- Conservative call ceiling: {payload['calls']['conservative_total']}
- Generation concurrency: {config.generation_workers}
- Behavioral concurrency: {config.behavioral_workers}
- Expected wall time: {expected_wall / 60:.1f} minutes
- Conservative wall time: {conservative_wall / 3600:.2f} hours
- Cost units: {config.accounting_unit}; reported costs are hard configured bounds
""",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# MuSR symbolic ambiguity calibration 01\n\n"
        "See `analysis/symbolic_ambiguity_calibration_report.md`.\n",
        encoding="utf-8",
    )
    write_json_atomic(
        root / "manifest.json",
        {
            "schema_version": 1,
            "probe": "musr_symbolic_ambiguity",
            "status": "planned",
            "config_sha256": sha256_object(config.to_dict()),
            "symbolic_selection_sha256": frozen["fingerprint_sha256"],
            "generation_provider": config.generation_provider.type,
            "generation_model": config.generation_provider.model,
            "behavioral_provider": config.behavioral_provider.type,
            "behavioral_model": config.behavioral_provider.model,
            "prompt_variant": "P2",
            "full_profile": "F9",
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


def _archive_packets(root: Path, tasks: Mapping[str, RelationalTask]) -> None:
    write_json_atomic(
        root / "accepted_tasks/full_profile_packets.json",
        {task_id: list(packet(task, 1)) for task_id, task in tasks.items()},
    )
    assignments = {}
    for task_id in tasks:
        distribution = json.loads(
            (root / f"accepted_tasks/{task_id}/distribution_N12.json").read_text(encoding="utf-8")
        )
        assignments[task_id] = {
            "agent_evidence_ids": distribution["agent_evidence_ids"],
            "agent_diagnostics": distribution["agent_diagnostics"],
            "latent_holder_counts": distribution["latent_holder_counts"],
            "fingerprint_sha256": distribution["fingerprint_sha256"],
        }
    write_json_atomic(root / "accepted_tasks/private_assignments.json", assignments)


async def run(
    config: SymbolicAmbiguityConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    _verify_approval(root, approve_preflight)
    preflight = json.loads((root / "preflight/preflight.json").read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("preflight failed")
    frozen = _load_frozen(root)
    generation = await generate_task_pack(
        config, frozen["selected_worlds"], root / "accepted_tasks"
    )
    tasks = load_tasks(config, root)
    if len(tasks) != config.final_tasks:
        raise RuntimeError("generated task pack is incomplete")
    _archive_packets(root, tasks)
    specs = call_plan(
        tasks,
        private_repetitions=config.private_repetitions,
        endpoint_repetitions=config.endpoint_repetitions,
        seed=config.seed,
    )
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
        raise RuntimeError("hidden symbolic metadata leaked into a behavioral prompt")
    exact_input_tokens = sum(prompt.token_estimate for prompt in rendered.values())
    if exact_input_tokens > config.max_behavioral_input_tokens:
        raise RuntimeError("rendered behavioral prompts exceed the approved input-token budget")
    write_json_atomic(root / "preflight/behavioral_call_plan.json", [spec.to_dict() for spec in specs])
    execution = await execute(
        config,
        tasks,
        specs,
        rendered,
        root / "behavioral_validation/raw_calls.jsonl",
        probe_name="musr_symbolic_ambiguity",
        retry_failed=True,
    )
    outputs = build_outputs(root, frozen)
    behavioral_journal = read(root / "behavioral_validation/raw_calls.jsonl")
    behavioral_terminal = terminal(root / "behavioral_validation/raw_calls.jsonl")
    execution["provider_attempts"] = sum(
        row.get("event") == "request_started" for row in behavioral_journal
    )
    execution["archived_transport_failures"] = sum(
        row.get("event") == "call_failed" for row in behavioral_journal
    )
    artifact_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete"
            if execution["successful"] == execution["scheduled"]
            else "incomplete",
            "acceptance_decision": outputs["decision"],
            "construction_rule": frozen["construction_rule"],
            "generation": generation,
            "behavioral_execution": execution,
            "observed_usage": {
                "generation_requests": generation["calls"],
                "generation_input_tokens": generation["usage"]["input_tokens"],
                "generation_output_tokens": generation["usage"]["output_tokens"],
                "behavioral_provider_attempts": execution["provider_attempts"],
                "behavioral_observations": execution["terminal"],
                "behavioral_input_tokens": sum(
                    int((row.get("usage") or {}).get("input_tokens") or 0)
                    for row in behavioral_terminal.values()
                ),
                "behavioral_output_tokens": sum(
                    int((row.get("usage") or {}).get("output_tokens") or 0)
                    for row in behavioral_terminal.values()
                ),
                "behavioral_transport_retries": sum(
                    int(row.get("transport_retries") or 0)
                    for row in behavioral_terminal.values()
                ),
            },
            "artifact_hashes": artifact_hashes,
        }
    )
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {
        "output": str(root),
        "report": outputs["report"],
        "decision": outputs["decision"],
        "construction_rule": frozen["construction_rule"],
        "generation": generation,
        "execution": execution,
        "pooled": outputs["pooled"],
    }


def analyze(
    config: SymbolicAmbiguityConfig, output_dir: Path | None = None
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    return build_outputs(root, _load_frozen(root))


__all__ = ["analyze", "load_tasks", "prepare", "run"]
