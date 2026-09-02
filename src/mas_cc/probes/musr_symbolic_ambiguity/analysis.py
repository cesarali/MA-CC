"""Scientific tables, figures, and report for symbolic ambiguity calibration."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.latent_problem import problem_from_latent_values
from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval
from mas_cc.probes.musr_prompt_solvability.execution import read


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def terminal(path: Path) -> list[dict[str, Any]]:
    latest = {
        str(row["call_id"]): row
        for row in read(path)
        if row.get("event") in {"call_finished", "call_failed"}
    }
    return [latest[key] for key in sorted(latest)]


def observation_rows(
    raw: Sequence[Mapping[str, Any]], task_dir: Path
) -> list[dict[str, Any]]:
    diagnostics: dict[tuple[str, int], Mapping[str, Any]] = {}
    gold: dict[str, str] = {}
    for path in sorted(task_dir.glob("task_*/base_task.json")):
        task_id = path.parent.name
        base = json.loads(path.read_text(encoding="utf-8"))
        distribution = json.loads(
            (path.parent / "distribution_N12.json").read_text(encoding="utf-8")
        )
        gold[task_id] = str(base["gold_answer"])
        for row in distribution["agent_diagnostics"]:
            diagnostics[(task_id, int(row["agent_id"]) + 1)] = row
    output = []
    for row in raw:
        task_id = str(row["task_id"])
        agent_id = int(row["agent_id"]) if row.get("agent_id") is not None else None
        diagnostic = diagnostics.get((task_id, agent_id or -1), {})
        condition = str(row["packet_variant"])
        output.append(
            {
                "call_id": row["call_id"],
                "task_id": task_id,
                "condition": condition,
                "agent_id": agent_id,
                "repetition": row["repetition"],
                "symbolic_M": diagnostic.get("max_predictability"),
                "symbolic_Hbar": diagnostic.get("normalized_entropy"),
                "p_allocation_0": diagnostic.get("p_allocation_0"),
                "p_allocation_1": diagnostic.get("p_allocation_1"),
                "p_allocation_2": diagnostic.get("p_allocation_2"),
                "evidence_ids": "|".join(row.get("evidence_ids") or ()),
                "semantic_option_mapping": json.dumps(row.get("semantic_option_mapping"), sort_keys=True),
                "parsed_semantic_answer": row.get("parsed_semantic_answer"),
                "gold_answer": gold[task_id],
                "correct": bool(row.get("correct")),
                "parse_success": bool(row.get("parse_success")),
                "input_tokens": (row.get("usage") or {}).get("input_tokens"),
                "output_tokens": (row.get("usage") or {}).get("output_tokens"),
            }
        )
    return output


def summarize(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        truth = sum(bool(row["correct"]) for row in group)
        low, high = wilson_interval(truth, len(group))
        histogram = Counter(row.get("parsed_semantic_answer") for row in group)
        output.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "truth": truth,
                "truth_rate": truth / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_rate": sum(bool(row["parse_success"]) for row in group) / len(group),
                "answer_ALLOCATION_0": histogram["ALLOCATION_0"],
                "answer_ALLOCATION_1": histogram["ALLOCATION_1"],
                "answer_ALLOCATION_2": histogram["ALLOCATION_2"],
                "unparsed": histogram[None],
            }
        )
    return output


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def _dangerous_views(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    rule = frozen["construction_rule"]
    breadth = int(rule["private_breadth"])
    index = TeamAllocationCompletionIndex(min_score_margin=int(rule["min_score_margin"]))
    rows = []
    fact_kinds = ("skill",) * 6 + ("cooperation",) * 3
    for selection in frozen["selected_worlds"]:
        problem = problem_from_latent_values(selection["latent_values"])
        worst = max(index.scan(problem, breadth), key=lambda row: row.max_predictability)
        kinds = Counter(fact_kinds[position] for position in worst.visible_indices)
        rows.append(
            {
                "task_id": selection["task_id"],
                "visible_indices": json.dumps(worst.visible_indices),
                "latent_type_profile": "+".join(
                    ["skill"] * kinds["skill"] + ["cooperation"] * kinds["cooperation"]
                ),
                "M": worst.max_predictability,
                "Hbar": worst.normalized_entropy,
                "diagnostic_only": True,
            }
        )
    return rows


def _figures(
    root: Path,
    pooled: Sequence[Mapping[str, Any]],
    symbolic_empirical: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = root / "analysis/figures"
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(root / "symbolic_scan/candidate_worlds.csv")
    rule = frozen["construction_rule"]
    margin = int(rule["min_score_margin"])
    data = [
        candidates.loc[candidates["unique_optimum"] == True, f"margin_{margin}_k_{k}_max_M"].dropna().values  # noqa: E712
        for k in (2, 3, 4)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=("2", "3", "4"), showfliers=False)
    ax.axhline(0.45, color="black", linestyle="--", label="preferred threshold")
    ax.set(xlabel="Private breadth k", ylabel="Worst-case $M_I$", ylim=(1 / 3, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ambiguity_by_private_breadth.png", dpi=180)
    plt.close(fig)

    acceptance = pd.read_csv(root / "symbolic_scan/acceptance_summary.csv")
    acceptance = acceptance[(acceptance.criterion == "preferred") & (acceptance.min_score_margin == margin)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(acceptance.k.astype(str), acceptance.acceptance_rate_all)
    ax.set(xlabel="Private breadth k", ylabel="Acceptance rate")
    fig.tight_layout()
    fig.savefig(output / "candidate_acceptance_rates.png", dpi=180)
    plt.close(fig)

    k = int(rule["private_breadth"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(candidates.score_margin, candidates[f"margin_{margin}_k_{k}_max_M"], s=5, alpha=0.2)
    ax.axhline(float(rule["max_predictability"]), color="black", linestyle="--")
    ax.set(xlabel="Complete-world score margin", ylabel="Worst-case subset $M_I$")
    fig.tight_layout()
    fig.savefig(output / "score_margin_vs_ambiguity.png", dpi=180)
    plt.close(fig)

    order = ("Zero", "Private", "F9")
    selected = [next(row for row in pooled if row["condition"] == condition) for condition in order]
    values = [float(row["truth_rate"]) for row in selected]
    lows = [value - float(row["ci95_low"]) for value, row in zip(values, selected, strict=True)]
    highs = [float(row["ci95_high"]) - value for value, row in zip(values, selected, strict=True)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(("Zero", "Private", "Full F9"), values)
    ax.errorbar(range(3), values, yerr=[lows, highs], fmt="none", color="black", capsize=4)
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set(ylabel="Truth-selection rate", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output / "zero_private_full_separation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        [float(row["symbolic_M"]) for row in symbolic_empirical],
        [float(row["empirical_truth_rate"]) for row in symbolic_empirical],
        alpha=0.7,
    )
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set(xlabel="Symbolic predictability $M_I$", ylabel="Empirical truth-selection rate", ylim=(-0.02, 1.02))
    fig.tight_layout()
    fig.savefig(output / "symbolic_vs_empirical_predictability.png", dpi=180)
    plt.close(fig)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows),
        ]
    )


def build_outputs(root: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    generation_manifest = json.loads(
        (root / "accepted_tasks/generation_manifest.json").read_text(encoding="utf-8")
    )
    generation_calls = int(generation_manifest["calls"])
    generation_model = str(generation_manifest["model"])
    journal_rows = read(root / "behavioral_validation/raw_calls.jsonl")
    provider_attempts = sum(row.get("event") == "request_started" for row in journal_rows)
    transport_failures = sum(row.get("event") == "call_failed" for row in journal_rows)
    raw = terminal(root / "behavioral_validation/raw_calls.jsonl")
    observations = observation_rows(raw, root / "accepted_tasks")
    pooled = summarize(observations, ("condition",))
    per_task = summarize(observations, ("task_id", "condition"))
    write_csv(root / "behavioral_validation/observation_level_results.csv", observations)
    write_csv(root / "behavioral_validation/summary_pooled.csv", pooled)
    write_csv(root / "behavioral_validation/summary_by_task_condition.csv", per_task)

    private_views = summarize(
        [row for row in observations if row["condition"] == "Private"],
        ("task_id", "agent_id", "symbolic_M", "symbolic_Hbar"),
    )
    symbolic_empirical = [
        {
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "symbolic_M": row["symbolic_M"],
            "symbolic_Hbar": row["symbolic_Hbar"],
            "n": row["n"],
            "empirical_truth_rate": row["truth_rate"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
        }
        for row in private_views
    ]
    xs = [float(row["symbolic_M"]) for row in symbolic_empirical]
    ys = [float(row["empirical_truth_rate"]) for row in symbolic_empirical]
    correlation = _correlation(xs, ys)

    tables = root / "analysis/tables"
    acceptance = list(csv.DictReader((root / "symbolic_scan/acceptance_summary.csv").open(encoding="utf-8")))
    write_csv(tables / "symbolic_acceptance_table.csv", acceptance)
    final_tasks = []
    for row in frozen["selected_worlds"]:
        metrics = row["private_views"]
        final_tasks.append(
            {
                "task_id": row["task_id"],
                "gold_allocation": row["gold_answer"],
                "score_margin": row["score_margin"],
                "worst_private_M": max(item["max_predictability"] for item in metrics),
                "mean_private_M": statistics.fmean(item["max_predictability"] for item in metrics),
                "mean_private_Hbar": statistics.fmean(item["normalized_entropy"] for item in metrics),
                "min_private_Hbar": min(item["normalized_entropy"] for item in metrics),
            }
        )
    write_csv(tables / "final_task_table.csv", final_tasks)
    write_csv(tables / "zero_private_full_table.csv", pooled)
    write_csv(tables / "symbolic_vs_empirical.csv", symbolic_empirical)
    write_csv(tables / "dangerous_partial_views.csv", _dangerous_views(frozen))
    _figures(root, pooled, symbolic_empirical, frozen)

    by_condition = {row["condition"]: row for row in pooled}
    complete = len(raw) > 0 and all(float(row["parse_rate"]) == 1.0 for row in pooled)
    passed = bool(
        complete
        and float(by_condition["Zero"]["truth_rate"]) <= 0.45
        and float(by_condition["Private"]["truth_rate"]) <= 0.45
        and float(by_condition["F9"]["truth_rate"]) >= 0.80
    )
    decision = "PASS" if passed else "FAIL"
    rule = frozen["construction_rule"]
    ambiguity = list(csv.DictReader((root / "symbolic_scan/ambiguity_by_k.csv").open(encoding="utf-8")))
    margin_rows = [
        row
        for row in acceptance
        if row["criterion"] == "preferred" and int(row["k"]) == int(rule["private_breadth"])
    ]
    behavior_table = [
        {
            "Condition": "Full F9" if row["condition"] == "F9" else row["condition"],
            "n": row["n"],
            "Truth rate": f"{float(row['truth_rate']):.1%}",
            "95% CI": f"[{float(row['ci95_low']):.1%}, {float(row['ci95_high']):.1%}]",
        }
        for row in sorted(pooled, key=lambda item: ("Zero", "Private", "F9").index(item["condition"]))
    ]
    report = f"""# MuSR Symbolic Ambiguity Calibration 01

## A. Motivation

The preceding redistribution calibration produced Zero = 33.3%, R2 = 50.9%, R3 = 50.0%, R4 = 53.7%, and F9 = 90.0%. Structural incompleteness therefore did not guarantee decision ambiguity: a partial view could omit latent values while still strongly predicting the correct allocation.

## B. Exact world definition

Each world has six skill values and three pairwise-cooperation values. Values use the generator's authoritative support `{frozen['latent_support']}` and prior `{frozen['latent_prior']}`. The existing exact allocation scorer maps the nine-value vector to three candidate scores; tied worlds are invalid. Gold is the unique score maximum and decisiveness is its margin over the second-best score.

## C. Private ambiguity definition

For visible indices I and values z_I, the analysis exactly enumerates all completions. Completions use the independent generator prior conditioned on the observation, a unique optimum, and the tested minimum score margin; tied and sub-margin completions are excluded before renormalization. It records p_I(a) = P(A*=a given z_I), M_I = max_a p_I(a), entropy H_I = -sum_a p_I(a) log p_I(a), and normalized entropy Hbar_I = H_I / log(3).

## D. Symbolic scan

The scan sampled **{frozen['candidate_worlds_scanned']:,}** candidate matrices before any language-generation call. Every $k=2,3,4$ subset was evaluated. A candidate passes a private gate when a deterministic 12-agent assignment exists whose every realized view satisfies both thresholds, whose union covers all nine values, and whose holder count is at least two per value; dangerous non-assigned subsets remain diagnostic rather than a hand-editing rule. `candidate_worlds.csv` retains accepted and rejected sampled worlds; `subset_metrics.parquet` retains the exact subset-level metrics under the baseline margin-one completion rule. Gold balance, margins, preferred/fallback feasibility, and both conditional acceptance denominators are machine-readable in `acceptance_summary.csv`.

## E. Ambiguity versus private breadth

{_markdown_table(ambiguity, ('k','candidate_worlds','pass_M_le_0_45','pass_rate','median_worst_case_M','median_mean_Hbar'))}

## F. Score-margin tradeoff

{_markdown_table(margin_rows, ('min_score_margin','k','accepted_worlds','acceptance_rate_all','acceptance_rate_structurally_valid','gold_0','gold_1','gold_2'))}

The frozen rule maximizes private breadth first and then full-world score margin, subject to enough symbolically accepted, exactly balanced worlds. This makes margin 2 preferable only because it remained feasible; the margin-one alternative is retained above.

## G. Frozen construction rule

- Private breadth: **k={rule['private_breadth']}**
- Maximum predictability: **M <= {rule['max_predictability']}** ({rule['criterion']} criterion)
- Minimum normalized entropy: **Hbar >= {rule['min_normalized_entropy']}**
- Minimum complete-world score margin: **{rule['min_score_margin']}**
- Gold balance: **{rule['balance_rule']}**
- Prompt/model packet: **P2 / gwdg/openai-gpt-oss-120b / F9**

This rule was frozen from symbolic feasibility before evidence generation and behavioral validation.

## H. Accepted task set

{_markdown_table(final_tasks, ('task_id','gold_allocation','score_margin','worst_private_M','mean_private_M','mean_private_Hbar','min_private_Hbar'))}

The final assignments are deterministic, cover all nine latent values, give every latent value at least two holders, never increase an agent beyond the selected breadth, and archive the full posterior for every realized view. Evidence was generated only for these six frozen worlds using {generation_model} in {generation_calls} calls; it was never used as the population game-playing model. All generation prompts, raw responses, attempts, metadata, and token usage are retained under `generation/` and `accepted_tasks/generation_manifest.json`.

## I. Real-provider validation

{_markdown_table(behavior_table, ('Condition','n','Truth rate','95% CI'))}

All calls used the frozen P2 ballot prompt and `gwdg/openai-gpt-oss-120b`. F9 contains one deterministic first-sorted card per latent value. The design contains {len(raw)} distinct observations and required {provider_attempts} provider attempts, including {transport_failures} archived failed transport/schema attempt(s) that were retried with the same call identity and seed. Raw rendered messages, option permutations, responses, request metadata, and token usage are retained in `behavioral_validation/raw_calls.jsonl`.

## J. Symbolic versus empirical relation

Across {len(symbolic_empirical)} realized task-agent views, the Pearson correlation between symbolic $M_I$ and empirical truth-selection frequency was **{correlation if correlation is not None else 'undefined'}**. The view-level table and scatter plot retain the data; with only three repetitions per view, this correlation is diagnostic rather than a precise effect estimate. `dangerous_partial_views.csv` separately records each world's highest-M subset and its skill/cooperation type composition without using it to alter the frozen worlds.

## K. PASS / FAIL

**{decision} — {'symbolically ambiguous worlds create the required distributed-information benchmark' if passed else 'symbolic ambiguity filtering alone is insufficient under the prespecified behavioral gate'}**

## L. Consequence

{'Freeze the accepted tasks, assignments, P2, and F9. Proceed to blackboard no-control / direct-recommendation / coordination-request comparisons.' if passed else 'Do not scale the blackboard experiment. The next revision must modify evidence generation or task complexity.'}

## M. Limitations

The scan contains {frozen['candidate_worlds_scanned']:,} sampled candidates and {len(final_tasks)} frozen worlds. Evidence generation used {generation_calls} provider calls; behavioral validation contains {len(raw)} designed observations from {provider_attempts} provider attempts, for {generation_calls + provider_attempts} provider attempts across both stages. Results are subject to provider stochasticity and dependence among repeated calls to the same task-agent view. Completion weighting is exact under the repository's discrete independent prior; no Monte Carlo approximation is used. The subset Parquet records the baseline margin-one completion condition, while margin-two tradeoffs and the frozen assignment metrics are retained in their dedicated tables and JSON.
"""
    report_path = root / "analysis/symbolic_ambiguity_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "decision": decision,
        "pooled": pooled,
        "per_task": per_task,
        "symbolic_empirical_correlation": correlation,
        "observations": len(observations),
    }


__all__ = ["build_outputs", "observation_rows", "summarize", "terminal", "write_csv"]
