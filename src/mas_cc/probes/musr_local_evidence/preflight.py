"""Credential-free design checks and exact rendered-token accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.musr_team_allocation_generator.io_utils import sha256_file, sha256_object

from .config import LocalEvidenceProbeConfig
from .design import CallSpec, build_call_plan
from .prompting import RenderedCall, render_call


@dataclass(frozen=True, slots=True)
class ProbePlan:
    task: Any
    calls: tuple[CallSpec, ...]
    rendered: dict[str, RenderedCall]
    dose_definitions: tuple[dict[str, Any], ...]
    checks: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def build_plan(config: LocalEvidenceProbeConfig) -> ProbePlan:
    task = load_musr_team_allocation_task(
        config.task_dir, config.task_id, population_size=config.population_size
    )
    calls, doses = build_call_plan(
        task,
        config.agents,
        config.pair_repetitions,
        config.doses,
        config.dose_repetitions,
        config.seed,
    )
    rendered = {spec.call_id: render_call(task, spec) for spec in calls}
    pair_groups: dict[str, list[CallSpec]] = {}
    for spec in calls:
        if spec.pair_id:
            pair_groups.setdefault(spec.pair_id, []).append(spec)
    nested_ok = True
    coverage_ok = True
    for agent in config.agents:
        rows = [row for row in doses if row["agent_id"] == agent]
        rows.sort(key=lambda row: row["dose"])
        for left, right in zip(rows, rows[1:]):
            nested_ok &= set(left["evidence_ids"]) < set(right["evidence_ids"])
        coverage = {row["dose"]: row["distinct_latent_fact_count"] for row in rows}
        coverage_ok &= all(coverage[dose] == dose for dose in (0, 3, 6, 9))
    checks = (
        {
            "check": "provider_model",
            "passed": config.provider.type == "university"
            and config.provider.model == "gwdg/openai-gpt-oss-120b",
            "detail": f"{config.provider.type}/{config.provider.model}",
        },
        {
            "check": "task_contract",
            "passed": task.correct_relation == "ALLOCATION_2"
            and len(task.fact_order) == 27
            and len(task.supporting_fact_groups or {}) == 9,
            "detail": "task_001: 27 cards, 9 latent facts, truth ALLOCATION_2",
        },
        {
            "check": "call_count",
            "passed": len(calls) == 123,
            "detail": f"{len(calls)} logical calls",
        },
        {
            "check": "paired_mappings",
            "passed": all(
                len(group) == 2
                and group[0].option_mapping == group[1].option_mapping
                and group[0].evidence_ids == group[1].evidence_ids
                for group in pair_groups.values()
            ),
            "detail": f"{len(pair_groups)} matched pairs",
        },
        {
            "check": "nested_doses",
            "passed": nested_ok,
            "detail": "strict prefix nesting for every probe agent",
        },
        {
            "check": "breadth_first",
            "passed": coverage_ok,
            "detail": "0/3/6/9 cards cover 0/3/6/9 latent facts",
        },
        {
            "check": "hidden_metadata_absent",
            "passed": all(
                all(
                    term not in "\n".join(message.content for message in item.messages)
                    for term in (
                        "skill_matrix",
                        "cooperation_matrix",
                        "candidate_scores",
                        "hidden_claim",
                        "gold_answer",
                    )
                )
                for item in rendered.values()
            ),
            "detail": "no evaluation-only latent metadata in any prompt",
        },
        {
            "check": "budgets",
            "passed": len(calls) <= config.max_requests
            and sum(item.token_estimate for item in rendered.values())
            <= config.max_input_tokens
            and len(calls) * config.provider.max_output_tokens
            <= config.max_output_tokens_total,
            "detail": "conservative request/input/output ceilings fit configured budgets",
        },
    )
    return ProbePlan(task, calls, rendered, doses, checks)


def preflight_payload(
    config: LocalEvidenceProbeConfig, plan: ProbePlan
) -> dict[str, Any]:
    input_tokens = sum(item.token_estimate for item in plan.rendered.values())
    return {
        "probe": "musr_local_evidence",
        "passed": plan.passed,
        "checks": list(plan.checks),
        "calls": {
            "prompt_equivalence": sum(
                spec.experiment == "prompt_equivalence" for spec in plan.calls
            ),
            "evidence_dose": sum(
                spec.experiment == "evidence_dose" for spec in plan.calls
            ),
            "total": len(plan.calls),
            "maximum_http_attempts": len(plan.calls),
        },
        "tokens": {
            "estimated_input_total": input_tokens,
            "maximum_output_total": len(plan.calls) * config.provider.max_output_tokens,
        },
        "concurrency": config.workers,
        "estimated_wall_seconds": len(plan.calls)
        / config.workers
        * float(config.provider.options.get("estimated_latency_seconds", 10.0)),
        "task_hashes": {
            "base_task.json": sha256_file(Path(plan.task.source_path.split("|", 1)[0])),
            "distribution_N12.json": sha256_file(
                Path(plan.task.source_path.split("|", 1)[1])
            ),
        },
        "call_plan_sha256": sha256_object([spec.to_dict() for spec in plan.calls]),
        "dose_definitions_sha256": sha256_object(plan.dose_definitions),
        "prompt_hashes_sha256": sha256_object(
            {key: value.to_dict() for key, value in sorted(plan.rendered.items())}
        ),
    }


__all__ = ["ProbePlan", "build_plan", "preflight_payload"]
