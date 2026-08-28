"""Credential-free validation and exact call accounting for the focused probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

from mas_cc.games.relational_reasoning.data import (
    RelationalTask,
    list_relational_task_ids,
    load_relational_task,
)

from . import vignette as vignette_module
from .config import ProbeConfig
from .design import ARMS_BY_Q, NO_OP, ONE_SLOT, TWO_SLOTS, Vignette, build_vignettes

SOCIAL_BLOCK = "social_information"


@dataclass(slots=True)
class Preflight:
    tasks: dict[int, tuple[RelationalTask, ...]]
    fingerprints: dict[tuple[int, str], str]
    vignettes: tuple[Vignette, ...]
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check["ok"] for check in self.checks)

    def failures(self) -> tuple[dict[str, Any], ...]:
        return tuple(check for check in self.checks if not check["ok"])

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})


def load_tasks(
    config: ProbeConfig,
) -> tuple[dict[int, tuple[RelationalTask, ...]], dict[tuple[int, str], str]]:
    tasks: dict[int, tuple[RelationalTask, ...]] = {}
    fingerprints: dict[tuple[int, str], str] = {}
    for depth in config.design.reasoning_depths:
        directory = config.dataset_dirs[depth]
        explicit = config.design.tasks.get(depth)
        ids = explicit or list_relational_task_ids(directory)[: config.design.tasks_per_depth]
        loaded = tuple(load_relational_task(directory, task_id) for task_id in ids)
        tasks[depth] = loaded
        for task in loaded:
            fingerprints[(depth, task.task_id)] = _task_fingerprint(task.source_path)
    return tasks, fingerprints


def _task_fingerprint(path: str | Path) -> str:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_preflight(config: ProbeConfig) -> Preflight:
    tasks, fingerprints = load_tasks(config)
    vignettes = build_vignettes(config.design, tasks, fingerprints)
    result = Preflight(tasks=tasks, fingerprints=fingerprints, vignettes=vignettes)

    result.record(
        "models_configured",
        bool(config.models),
        f"{len(config.models)} model specification(s) configured in YAML",
    )
    result.record(
        "twelve_tasks_per_depth",
        all(len(tasks.get(depth, ())) == 12 for depth in (1, 2)),
        ", ".join(f"L={depth}:{len(tasks.get(depth, ())) }" for depth in (1, 2)),
    )
    result.record(
        "task_depths_match",
        all(task.reasoning_depth == depth for depth, group in tasks.items() for task in group),
        "every loaded task matches its configured reasoning depth",
    )
    result.record(
        "task_fingerprints_recomputed",
        all(fingerprints.get((task.reasoning_depth, task.task_id)) for group in tasks.values() for task in group),
        "SHA-256 fingerprints were recomputed from all selected frozen task files",
    )
    result.record(
        "focused_axes_only",
        config.design.q_values == (2, 3)
        and config.design.reasoning_depths == (1, 2)
        and config.design.receivers == ("naive",)
        and config.design.targets == ("truth", "false")
        and config.design.replicates == 1,
        "q=[2,3], L=[1,2], naive receiver, truth/false targets, one replicate",
    )
    result.record(
        "base_vignette_count",
        len(vignettes) == 96,
        f"{len(vignettes)} base vignettes; expected 96",
    )
    result.record(
        "recommendation_only",
        all(item.message_mode == "recommendation_only" for item in vignettes),
        "every vignette uses recommendation_only with no controller fact",
    )
    _check_arms(result, tasks, vignettes)
    return result


def _check_arms(
    result: Preflight,
    tasks: Mapping[int, Sequence[RelationalTask]],
    vignettes: Sequence[Vignette],
) -> None:
    by_task = {
        (task.reasoning_depth, task.task_id): task
        for group in tasks.values()
        for task in group
    }
    slots_ok = True
    shared_ok = True
    social_only = True
    for item in vignettes:
        task = by_task[(item.reasoning_depth, item.task_id)]
        baseline = vignette_module.rendered_blocks(task, item, NO_OP)
        expected = {2: {NO_OP: 0, ONE_SLOT: 1}, 3: {NO_OP: 0, ONE_SLOT: 1, TWO_SLOTS: 2}}[item.q]
        for arm in ARMS_BY_Q[item.q]:
            sources = vignette_module.social_sources(task, item, arm)
            controls = sum(source["source_type"] == "control" for source in sources)
            slots_ok &= len(sources) == item.q and controls == expected[arm]
            rendered = vignette_module.rendered_blocks(task, item, arm)
            differing = {name for name in baseline if baseline[name] != rendered.get(name)}
            if arm != NO_OP and differing != {SOCIAL_BLOCK}:
                social_only = False
        if item.q == 3:
            one = vignette_module.social_sources(task, item, ONE_SLOT)
            two = vignette_module.social_sources(task, item, TWO_SLOTS)
            shared_ok &= one[0]["source_type"] == two[0]["source_type"] == "control"
            shared_ok &= one[2]["vote"] == two[2]["vote"]
    result.record("arm_slot_counts", slots_ok, "NOOP/one_slot/two_slots use 0/1/2 controller slots")
    result.record("matched_prompts_differ_only_socially", social_only, "all arms share every non-social prompt block")
    result.record("q3_two_slots_adds_one_controller", shared_ok, "q=3 two_slots preserves one_slot and replaces one additional peer")


def call_counts(
    config: ProbeConfig,
    vignettes: Sequence[Vignette],
    completed: set[str] | None = None,
) -> dict[str, Any]:
    done = completed or set()
    per_model: dict[str, dict[str, int]] = {}
    q_counts: dict[int, int] = {2: 0, 3: 0}
    total = 0
    for model in config.models:
        scheduled = completed_count = 0
        identity = model.call_identity
        for item in vignettes:
            for arm in ARMS_BY_Q[item.q]:
                scheduled += 1
                total += 1
                q_counts[item.q] += 1
                if item.call_id(identity, arm) in done:
                    completed_count += 1
        per_model[model.label] = {
            "scheduled": scheduled,
            "completed": completed_count,
            "remaining": scheduled - completed_count,
        }
    expected_ids = {
        item.call_id(model.call_identity, arm)
        for model in config.models
        for item in vignettes
        for arm in ARMS_BY_Q[item.q]
    }
    return {
        "vignettes": len(vignettes),
        "calls_q2": q_counts[2],
        "calls_q3": q_counts[3],
        "calls_per_model": 240,
        "calls_total": total,
        "calls_completed": len(done & expected_ids),
        "calls_remaining": sum(item["remaining"] for item in per_model.values()),
        "per_model": per_model,
    }


def token_estimate(
    config: ProbeConfig,
    tasks: Mapping[int, Sequence[RelationalTask]],
    vignettes: Sequence[Vignette],
) -> dict[str, Any]:
    by_task = {
        (task.reasoning_depth, task.task_id): task
        for group in tasks.values()
        for task in group
    }
    counts: list[int] = []
    for item in vignettes[:24]:
        task = by_task[(item.reasoning_depth, item.task_id)]
        for arm in ARMS_BY_Q[item.q]:
            compiled = vignette_module.build_prompt(task, item, arm).compile()
            counts.append(
                compiled.total_token_estimate
                or sum(max(1, len(message.content) // 4) for message in compiled.messages)
            )
    mean_input = sum(counts) / len(counts) if counts else 0.0
    per_model = {
        model.label: {
            "calls": 240,
            "estimated_input_tokens": int(mean_input * 240),
            "max_output_tokens": model.max_output_tokens * 240,
        }
        for model in config.models
    }
    return {
        "mean_input_tokens_per_call": round(mean_input, 1),
        "sampled_prompts": len(counts),
        "per_model": per_model,
        "estimated_input_tokens_total": sum(row["estimated_input_tokens"] for row in per_model.values()),
        "max_output_tokens_total": sum(row["max_output_tokens"] for row in per_model.values()),
    }


def preflight_payload(
    config: ProbeConfig, preflight: Preflight, completed: set[str] | None = None
) -> dict[str, Any]:
    from .execution import effective_workers

    counts = call_counts(config, preflight.vignettes, completed)
    checks = list(preflight.checks)
    checks.append({
        "check": "exact_call_count",
        "ok": all(row["scheduled"] == 240 for row in counts["per_model"].values()),
        "detail": f"{counts['calls_per_model']} calls/model; {counts['calls_total']} total",
    })
    return {
        "probe": "controller_retention",
        "config": config.to_dict(),
        "models": len(config.models),
        "tasks": {f"L={depth}": [task.task_id for task in group] for depth, group in sorted(preflight.tasks.items())},
        "task_fingerprints": {f"L={depth}:{task_id}": value for (depth, task_id), value in sorted(preflight.fingerprints.items())},
        "calls": counts,
        "tokens": token_estimate(config, preflight.tasks, preflight.vignettes),
        "concurrency": {
            "backend": config.execution.backend,
            "requested_workers": config.execution.workers,
            "effective_workers": effective_workers(config),
            "provider_concurrency_caps": dict(config.execution.provider_concurrency_caps),
        },
        "checks": checks,
        "passed": preflight.passed and all(check["ok"] for check in checks),
    }


def format_preflight(payload: Mapping[str, Any]) -> str:
    calls = payload["calls"]
    tokens = payload["tokens"]
    concurrency = payload["concurrency"]
    lines = [
        "Controller-retention focused local probe - preflight",
        "",
        f"  models                       {payload['models']} (configured in YAML)",
        f"  frozen L=1 tasks             {len(payload['tasks'].get('L=1', []))}",
        f"  frozen L=2 tasks             {len(payload['tasks'].get('L=2', []))}",
        f"  base vignettes / model       {calls['vignettes']}",
        f"  q=2 calls / model            96",
        f"  q=3 calls / model            144",
        f"  calls per model              {calls['calls_per_model']}",
        f"  calls total                  {calls['calls_total']}",
        f"  already completed            {calls['calls_completed']}",
        f"  remaining                    {calls['calls_remaining']}",
        "",
        f"  mean input tokens / call     {tokens['mean_input_tokens_per_call']}",
        f"  estimated input tokens       {tokens['estimated_input_tokens_total']:,}",
        f"  max output tokens (ceiling)  {tokens['max_output_tokens_total']:,}",
        "",
        f"  execution backend            {concurrency['backend']}",
        f"  requested workers            {concurrency['requested_workers']}",
        f"  effective workers            {concurrency['effective_workers']}",
        "",
        "  checks:",
    ]
    for check in payload["checks"]:
        lines.append(f"    [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
    lines += ["", f"  preflight {'passed' if payload['passed'] else 'FAILED'}"]
    return "\n".join(lines)


__all__ = [
    "Preflight",
    "call_counts",
    "format_preflight",
    "load_tasks",
    "preflight_payload",
    "run_preflight",
]
