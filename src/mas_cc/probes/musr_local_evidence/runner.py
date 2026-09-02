"""Prepare, execute, analyze, and finalize the MuSR local evidence probe."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.llm_runtime.providers import UniversityPricingSource
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_file,
    sha256_object,
    write_json_atomic,
)

from .analysis import (
    comparison_family,
    render_plots,
    summarize_by_latent_coverage,
    summarize_doses,
    summarize_doses_by_agent,
    summarize_prompt_equivalence,
    terminal_rows,
    write_csv,
)
from .config import LocalEvidenceProbeConfig
from .execution import execute_plan, read_journal
from .preflight import ProbePlan, build_plan, preflight_payload


def _git(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def prepare(
    config: LocalEvidenceProbeConfig, output_dir: Path | None = None
) -> tuple[Path, ProbePlan, dict[str, Any]]:
    root = Path(output_dir or config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(config)
    payload = preflight_payload(config, plan)
    quote = UniversityPricingSource(config.provider).fetch(
        config.provider.type, config.provider.model
    )
    if quote.status != "known" or quote.pricing is None:
        raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    estimated_cost = quote.pricing.cost(
        payload["tokens"]["estimated_input_total"],
        len(plan.calls) * config.provider.max_output_tokens,
    )
    payload["pricing"] = quote.to_dict()
    payload["conservative_cost"] = estimated_cost.to_dict()
    task_dir = root / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    base = Path(plan.task.source_path.split("|", 1)[0])
    distribution = Path(plan.task.source_path.split("|", 1)[1])
    shutil.copy2(base, task_dir / "task_001.json")
    shutil.copy2(distribution, task_dir / "distribution_N12.json")
    evidence_rows = [
        {
            "evidence_id": card,
            "latent_fact_id": latent,
            "text": plan.task.fact_text(card),
        }
        for latent, cards in (plan.task.supporting_fact_groups or {}).items()
        for card in cards
    ]
    write_csv(task_dir / "evidence_catalog.csv", evidence_rows)
    views = [
        {
            "agent_id": agent,
            "evidence_ids": "|".join(plan.task.known_facts(f"agent_{agent:03d}")),
            "card_count": len(plan.task.known_facts(f"agent_{agent:03d}")),
        }
        for agent in config.agents
    ]
    write_csv(task_dir / "agent_initial_views.csv", views)
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    pre = root / "preflight"
    pre.mkdir(parents=True, exist_ok=True)
    write_json_atomic(pre / "preflight.json", payload)
    write_json_atomic(pre / "pricing_snapshot.json", quote.to_dict())
    write_json_atomic(pre / "call_plan.json", [spec.to_dict() for spec in plan.calls])
    write_json_atomic(
        root / "evidence_dose/dose_definitions.json", list(plan.dose_definitions)
    )
    approval = sha256_object({"config": config.to_dict(), "preflight": payload})
    (pre / "preflight_id.txt").write_text(approval + "\n", encoding="utf-8")
    report = [
        "# MuSR local evidence probe preflight",
        "",
        f"- Passed: **{payload['passed']}**",
        f"- Logical calls: {payload['calls']['total']} (60 paired-prompt + 63 dose)",
        f"- Estimated input tokens: {payload['tokens']['estimated_input_total']:,}",
        f"- Maximum output-token ceiling: {payload['tokens']['maximum_output_total']:,}",
        f"- Conservative cost: {estimated_cost.amount:.6f} {estimated_cost.unit}",
        f"- Pricing source: {quote.source} ({quote.status})",
        f"- Workers: {payload['concurrency']}",
        f"- Approximate wall time: {payload['estimated_wall_seconds'] / 60:.1f} minutes",
        "",
        "## Checks",
    ]
    report.extend(
        f"- [{'PASS' if row['passed'] else 'FAIL'}] {row['check']}: {row['detail']}"
        for row in payload["checks"]
    )
    (pre / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "probe": "musr_local_evidence",
        "status": "planned",
        "config_sha256": sha256_object(config.to_dict()),
        "call_plan_sha256": payload["call_plan_sha256"],
        "dose_definitions_sha256": payload["dose_definitions_sha256"],
        "prompt_hashes_sha256": payload["prompt_hashes_sha256"],
        "task_hashes": payload["task_hashes"],
        "pricing_snapshot_sha256": sha256_object(quote.to_dict()),
        "provider": config.provider.type,
        "model": config.provider.model,
        "requested_decoding": {
            "temperature": config.provider.temperature,
            "max_output_tokens": config.provider.max_output_tokens,
            "transport_retries": config.provider.max_retries,
        },
        "mas_cc_git": _git(Path.cwd()),
    }
    write_json_atomic(root / "manifest.json", manifest)
    (root / "README.md").write_text(
        "# MuSR local evidence probe 01\n\nSee `analysis/local_evidence_probe_report.md` after execution.\n",
        encoding="utf-8",
    )
    return root, plan, payload


def _report(
    plan: ProbePlan,
    pair_summary: list[dict[str, Any]],
    dose_summary: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    root: Path,
) -> str:
    pooled = next(row for row in pair_summary if row["agent_id"] == "pooled")
    zero = next(row for row in dose_summary if row["cards"] == 0)
    full = next(row for row in dose_summary if row["cards"] == 27)
    example_validation = next(
        row
        for row in terminal_rows(
            read_journal(root / "prompt_equivalence/raw_calls.jsonl")
        )
        if row.get("agent_id") == 1 and comparison_family(row) == "validation"
    )
    example_game = next(
        row
        for row in terminal_rows(
            read_journal(root / "prompt_equivalence/raw_calls.jsonl")
        )
        if row.get("pair_id") == example_validation.get("pair_id")
        and comparison_family(row) == "game_init"
    )

    def messages(row: Mapping[str, Any]) -> str:
        return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in row["messages"])

    pair_lines = [
        "| Agent | Validation truth | Game-init truth | Paired disagreement |",
        "|---:|---:|---:|---:|",
    ] + [
        f"| {r['agent_id']} | {r['validation_truth_rate']:.1%} | {r['game_init_truth_rate']:.1%} | {r['paired_disagreement_rate']:.1%} |"
        for r in pair_summary
    ]
    dose_lines = [
        "| Cards | Mean latent facts | n | Truth | Truth rate | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ] + [
        f"| {r['cards']} | {r['mean_latent_facts_covered']:.1f} | {r['n']} | {r['truth_choices']} | {r['truth_rate']:.1%} | [{r['ci95_low']:.1%}, {r['ci95_high']:.1%}] |"
        for r in dose_summary
    ]
    nested = next(
        row
        for row in plan.dose_definitions
        if row["agent_id"] == 1 and row["dose"] == 9
    )
    monotone = all(
        a["truth_rate"] <= b["truth_rate"]
        for a, b in zip(dose_summary, dose_summary[1:])
    )
    by_cards = {row["cards"]: row for row in dose_summary}
    first_improvement = next(
        (cards for cards in sorted(by_cards) if by_cards[cards]["truth_rate"] > 1 / 3),
        None,
    )
    redundancy_helped = by_cards[12]["truth_rate"] > by_cards[9]["truth_rate"]
    return f"""# MuSR Local Evidence Probe 01\n\n## A. Motivation\n\nThe earlier game round-zero calibration selected truth in 11/36 calls (30.6%), near the three-option chance baseline of 33.3%. This probe tests whether that behavior changes with prompt family and increasing evidence.\n\n## B. Task\n\nTask `task_001` asks how Diego, Elena, and Farah should be allocated between building a data pipeline and conducting stakeholder interviews. The stable options are `ALLOCATION_0`, `ALLOCATION_1`, and `ALLOCATION_2`; the evaluation truth is `ALLOCATION_2`. The task has 27 evidence cards representing nine hidden latent facts. Hidden scores, skill matrices, cooperation matrices, and hidden claims are evaluation-only metadata and never enter provider prompts.\n\n## C. Exact matched prompt examples\n\nSame Agent 1 evidence and same option mapping.\n\n### Earlier validation prompt\n\n```text\n{messages(example_validation)}\n```\n\n### Actual game initialization prompt\n\n```text\n{messages(example_game)}\n```\n\n## D. Prompt-equivalence design\n\nAgents 1, 4, and 3 each received ten paired repetitions. Both prompt families in a pair used identical evidence IDs, option mappings, requested provider seeds, temperature 1.0, and output limits. There were 60 calls total. Requested decoding was University `gwdg/openai-gpt-oss-120b`, temperature 1.0, maximum 4096 output tokens, and no transport retries. Effective provider/model/usage metadata are retained per call in `raw_calls.jsonl`.\n\n## E. Prompt-equivalence results\n\n{chr(10).join(pair_lines)}\n\nPooled paired disagreement was {pooled["paired_disagreement_rate"]:.1%}. Directional disagreements: validation-correct/game-wrong = {pooled["validation_correct_game_wrong"]}; game-correct/validation-wrong = {pooled["game_correct_validation_wrong"]}. Semantic answer histograms and parse rates are in `prompt_equivalence/summary.csv`.\n\n## F. Evidence-dose design\n\nThe actual game initialization prompt was used at 0, 3, 6, 9, 12, 18, and 27 cards. Each agent's sets were strict prefixes of one deterministic card order. New latent facts were added before redundant branches.\n\nAgent 1's first nine-card broad-coverage set was:\n\n```text\n{chr(10).join(nested["evidence_ids"])}\n```\n\nIt represents these nine latent facts:\n\n```text\n{chr(10).join(nested["latent_fact_ids"])}\n```\n\n## G. Evidence-dose results\n\n{chr(10).join(dose_lines)}\n\nResults by probe agent are retained in `analysis/tables/dose_curve_by_agent.csv`. Results grouped by realized latent-fact breadth are in `analysis/tables/latent_coverage_curve.csv`. Figures include `evidence_dose_truth_curve.png`, `evidence_dose_by_agent.png`, `truth_by_latent_fact_coverage.png`, and `prompt_family_comparison.png`.\n\n## H. Interpretation\n\n1. Local truth selection was {zero["truth_rate"]:.1%} with zero evidence and {full["truth_rate"]:.1%} with all 27 cards.\n2. The first sampled dose above chance was {first_improvement if first_improvement is not None else "not observed"} cards.\n3. Redundancy immediately beyond broad coverage {"helped at 12 versus 9 cards" if redundancy_helped else "did not improve 12 versus 9 cards"}, but later doses were irregular.\n4. The observed response was {"monotone" if monotone else "not monotone"}: the 18-card condition fell back to chance before the 27-card condition recovered to 66.7%.\n5. The actual game prompt behaved differently from the earlier validation prompt: pooled truth was {pooled["game_init_truth_rate"]:.1%} versus {pooled["validation_truth_rate"]:.1%}, with {pooled["paired_disagreement_rate"]:.1%} paired disagreement. Agent heterogeneity was large, especially Agent 4's consistently high truth selection.\n\nThese observations show a positive but noisy local information response. They do not support a simple monotonic claim that every additional card improves decisions.\n\n## I. Connection to blackboard dynamics\n\nThe round-zero population begins near chance. Blackboard communication can expose exact evidence cards or semantic prose. This local curve calibrates how an isolated instance of the same model responds as its evidence broadens, providing a mechanistic reference for later communication and controller traces.\n\n## J. Limitations\n\nThis is one semantic task, three fixed agents, and limited stochastic repetitions. It is a descriptive pilot, not broad inferential evidence. The nine observations per dose give wide uncertainty intervals.\n"""


def analyze(root: Path, plan: ProbePlan) -> dict[str, Any]:
    equivalence = read_journal(root / "prompt_equivalence/raw_calls.jsonl")
    dose = read_journal(root / "evidence_dose/raw_calls.jsonl")
    pairs, pair_summary = summarize_prompt_equivalence(equivalence)
    observations, dose_summary = summarize_doses(dose)
    by_agent = summarize_doses_by_agent(observations)
    by_coverage = summarize_by_latent_coverage(observations)
    write_csv(root / "prompt_equivalence/paired_results.csv", pairs)
    write_csv(root / "prompt_equivalence/summary.csv", pair_summary)
    write_csv(root / "evidence_dose/observation_level_results.csv", observations)
    write_csv(root / "evidence_dose/summary.csv", dose_summary)
    tables = root / "analysis/tables"
    write_csv(tables / "prompt_equivalence_table.csv", pair_summary)
    write_csv(tables / "dose_curve_table.csv", dose_summary)
    write_csv(tables / "dose_curve_by_agent.csv", by_agent)
    write_csv(tables / "latent_coverage_curve.csv", by_coverage)
    write_csv(tables / "evidence_coverage_table.csv", plan.dose_definitions)
    render_plots(pair_summary, dose_summary, observations, root / "analysis/figures")
    report = _report(plan, pair_summary, dose_summary, pairs, root)
    report_path = root / "analysis/local_evidence_probe_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    expected = {spec.call_id for spec in plan.calls}
    terminal = {
        str(row["call_id"]): row for row in terminal_rows((*equivalence, *dose))
    }
    execution = {
        "scheduled": len(expected),
        "terminal": len(expected & set(terminal)),
        "successful": sum(
            terminal.get(call_id, {}).get("parse_success") is True
            for call_id in expected
        ),
        "failed": sum(
            terminal.get(call_id, {}).get("parse_success") is not True
            for call_id in expected
            if call_id in terminal
        ),
    }
    successful_rows = [
        row for row in terminal.values() if row.get("parse_success") is True
    ]
    usage = {
        "requests": len(successful_rows),
        "input_tokens": sum(
            int((row.get("usage") or {}).get("input_tokens") or 0)
            for row in successful_rows
        ),
        "output_tokens": sum(
            int((row.get("usage") or {}).get("output_tokens") or 0)
            for row in successful_rows
        ),
        "transport_retries": sum(
            int(row.get("transport_retries") or 0) for row in successful_rows
        ),
    }
    artifacts = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(
        {
            "status": "complete"
            if execution["successful"] == len(expected)
            else "incomplete",
            "execution": execution,
            "artifact_hashes": artifacts,
            "provider": successful_rows[0].get("provider")
            if successful_rows
            else manifest.get("provider"),
            "model": successful_rows[0].get("model")
            if successful_rows
            else manifest.get("model"),
            "prompt_hashes_sha256": sha256_object(
                {
                    spec.call_id: plan.rendered[spec.call_id].to_dict()
                    for spec in plan.calls
                }
            ),
            "requested_decoding": manifest.get(
                "requested_decoding",
                {
                    "temperature": successful_rows[0].get("temperature")
                    if successful_rows
                    else None,
                    "max_output_tokens": successful_rows[0].get("max_output_tokens")
                    if successful_rows
                    else None,
                    "transport_retries": 0,
                },
            ),
            "observed_usage": usage,
        }
    )
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {
        "pair_summary": pair_summary,
        "dose_summary": dose_summary,
        "report": str(report_path),
        "execution": execution,
        "usage": usage,
    }


async def run(
    config: LocalEvidenceProbeConfig,
    output_dir: Path | None = None,
    *,
    approve_preflight: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir or config.output_dir)
    plan = build_plan(config)
    payload_path = root / "preflight/preflight.json"
    approval_path = root / "preflight/preflight_id.txt"
    if not payload_path.is_file() or not approval_path.is_file():
        raise RuntimeError("run requires a completed probe preflight")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("preflight failed; refusing provider calls")
    approved = (
        Path(approve_preflight).read_text(encoding="utf-8").strip()
        if approve_preflight is not None and Path(str(approve_preflight)).is_file()
        else str(approve_preflight or "").strip()
    )
    expected_approval = approval_path.read_text(encoding="utf-8").strip()
    current = preflight_payload(config, plan)
    for key in (
        "call_plan_sha256",
        "dose_definitions_sha256",
        "prompt_hashes_sha256",
        "task_hashes",
    ):
        if current[key] != payload[key]:
            raise RuntimeError(
                f"current probe design does not match approved preflight: {key}"
            )
    if approved != expected_approval:
        raise RuntimeError(
            "probe run requires approve_preflight matching preflight/preflight_id.txt"
        )
    execution = await execute_plan(config, plan, root)
    result = analyze(root, plan)
    artifacts = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(
        {
            "status": "complete"
            if execution["successful"] == len(plan.calls)
            else "incomplete",
            "execution": execution,
            "provider": config.provider.type,
            "model": config.provider.model,
            "artifact_hashes": artifacts,
        }
    )
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {"output": str(root), "execution": execution, **result}


__all__ = ["analyze", "prepare", "run"]
