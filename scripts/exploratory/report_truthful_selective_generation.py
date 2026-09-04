"""Summarize Terra generation and its semantic validation gate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("results/studies/musr_truthful_selective_task_calibration_01")


def main() -> None:
    manifest = json.loads(
        (ROOT / "generation/terra_generation_manifest.json").read_text(encoding="utf-8")
    )
    failure_reasons: Counter[str] = Counter()
    task_rows = []
    for task_root in sorted((ROOT / "tasks").glob("task_*")):
        rows = json.loads(
            (task_root / "generation/semantic_validation.json").read_text(
                encoding="utf-8"
            )
        )
        failed = [row for row in rows if not row["passed"]]
        for row in failed:
            for name, passed in row["deterministic_checks"].items():
                if not passed:
                    failure_reasons[f"deterministic:{name}"] += 1
            for name, verdict in row["semantic_audit"].items():
                if verdict == "FAIL":
                    failure_reasons[f"semantic:{name}"] += 1
        summary = json.loads(
            (task_root / "generation/semantic_validation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        task_rows.append(
            {
                "task": task_root.name,
                "cards": summary["cards"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "controller": f"{summary['by_role']['controller-compatible']['passed']}/{summary['by_role']['controller-compatible']['cards']}",
                "decisive": f"{summary['by_role']['decisive']['passed']}/{summary['by_role']['decisive']['cards']}",
                "neutral": f"{summary['by_role']['neutral']['passed']}/{summary['by_role']['neutral']['cards']}",
                "decision": "PASS" if summary["all_passed"] else "FAIL",
            }
        )
    table = "\n".join(
        [
            "| Task | Cards | Passed | Failed | Controller | Decisive | Neutral | Decision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *(
                f"| {row['task']} | {row['cards']} | {row['passed']} | {row['failed']} | {row['controller']} | {row['decisive']} | {row['neutral']} | {row['decision']} |"
                for row in task_rows
            ),
        ]
    )
    reasons = "\n".join(
        [
            "| Failed check | Count |",
            "| --- | ---: |",
            *(f"| {key} | {value} |" for key, value in failure_reasons.most_common()),
        ]
    )
    report = f"""# MuSR Truthful-Selective Terra Generation Validation

## Outcome

**FAIL — OSS behavioral execution was not authorized.**

Terra generated evidence for only the three frozen symbolic development tasks. Structural and cross-card leakage checks passed, but the stricter per-card semantic gate found failures in every task. The pipeline therefore stopped before any OSS call.

## Usage

- Model: `{manifest["model"]}`
- Evidence-generation logical calls: **{manifest["evidence_generation_logical_calls"]}**
- Semantic-audit logical calls: **{manifest["semantic_validation_logical_calls"]}**
- Total logical calls: **{manifest["logical_calls"]}**
- Provider attempts: **{manifest["provider_attempts"]}**
- Retries: **{manifest["retry_count"]}**
- Input tokens: **{manifest["usage"]["input_tokens"]}**
- Output tokens: **{manifest["usage"]["output_tokens"]}**

## Per-task validation

{table}

## Failed checks

{reasons}

## Interpretation

The symbolic worlds remain valid. The failure is in the generated language layer: at least one generated card per task did not pass all polarity, faithfulness, strengthening, unsupported-implication, hidden-state-leakage, coherence, and deterministic leakage checks.

The audit is model-assisted rather than a formal proof of natural-language entailment. Failed items and their concrete reasons are retained in each task's `generation/semantic_validation.json`.

## Next step

Do not run OSS or a population pilot with these generated cards. Regenerate only failed cards under the same frozen symbolic tasks, or manually review and explicitly approve the failed cards. Do not change symbolic thresholds or task selection to rescue the language generation.
"""
    output = ROOT / "analysis/terra_generation_validation_report.md"
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
