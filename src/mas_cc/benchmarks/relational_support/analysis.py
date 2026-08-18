"""From rows to the three tables that answer the question.

1. ``accuracy_by_k`` - the general quantity, ``A_k`` per parameter condition.
2. ``accuracy_by_condition_id`` - the *same* data split by which subset was
   shown.  Reported before pooling because the links of a chain are not
   interchangeable: in ``A -> B -> C``, knowing the ``A``-``B`` link and knowing
   the ``B``-``C`` link are different pieces of evidence and can differ in how
   much they narrow the answer.  Pooling first would hide that.
3. ``headline_l2`` - for L = 2 only, the contrast the experiment was built to
   report, with the pooled partial accuracy and the full-minus-partial gap.
4. ``accuracy_by_correct_position`` - the same contrast split by where the
   correct relation was displayed.  With the balanced crossing every cell here
   has the same size by construction, so a position effect shows up as a
   difference between rows rather than as noise.
5. ``predicted_position_distribution`` - which *letters* the model reached for.
   This is the table that exposed the problem: at zero evidence the model chose
   ``A`` in 89% of items, which made the pre-control ``accuracy_zero`` a measure
   of how often ``A`` happened to be correct.
6. ``predicted_relation_distribution`` - the same answers read semantically, to
   distinguish a letter habit from a compass habit (a model that always says
   "north" is a different problem from one that always says "A").

Intervals are Wilson score intervals at 95%.  With 20 tasks per cell the normal
approximation is not usable near 0 or 1, which is exactly where a well-behaved
``A_L`` and a chance-level ``A_0`` will sit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """A 95% Wilson score interval; ``(0.0, 1.0)`` for an empty sample."""

    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read ``rows.jsonl`` (or a directory containing it)."""

    source = Path(path)
    if source.is_dir():
        source = source / "rows.jsonl"
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scored(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows that actually produced an answer.  Provider errors are not zeros."""

    return [row for row in rows if not row.get("error")]


def _aggregate(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, group in sorted(buckets.items(), key=lambda item: [str(v) for v in item[0]]):
        correct = sum(1 for row in group if row.get("correct"))
        unparsed = sum(1 for row in group if not row.get("parse_ok"))
        low, high = wilson_interval(correct, len(group))
        entry = dict(zip(keys, values))
        entry.update(
            {
                "num_tasks": len(group),
                "num_correct": correct,
                "accuracy": correct / len(group) if group else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "unparsed_responses": unparsed,
            }
        )
        output.append(entry)
    return output


def accuracy_by_k(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``A_k`` per parameter condition - the general-``L`` summary."""

    return _aggregate(
        _scored(rows),
        ("parameter_condition", "reasoning_depth", "distractors", "num_options",
         "num_supporting_facts_shown"),
    )


def accuracy_by_condition_id(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The same accuracy, split by *which* subset of the support was shown."""

    return _aggregate(
        _scored(rows),
        ("parameter_condition", "reasoning_depth", "num_supporting_facts_shown",
         "condition_id", "supporting_fact_ids_shown"),
    )


def headline_l2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For each L = 2 parameter condition: full, pooled partial, zero, and the gap."""

    scored = [row for row in _scored(rows) if row.get("reasoning_depth") == 2]
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        buckets[str(row.get("parameter_condition"))].append(row)

    output: list[dict[str, Any]] = []
    for label, group in sorted(buckets.items()):
        def slice_accuracy(predicate) -> tuple[float | None, int, int]:
            subset = [row for row in group if predicate(row)]
            correct = sum(1 for row in subset if row.get("correct"))
            return (
                (correct / len(subset)) if subset else None,
                correct,
                len(subset),
            )

        full_accuracy, full_correct, full_n = slice_accuracy(
            lambda row: row.get("condition") == "full"
        )
        partial_accuracy, partial_correct, partial_n = slice_accuracy(
            lambda row: row.get("condition") == "partial"
        )
        zero_accuracy, zero_correct, zero_n = slice_accuracy(
            lambda row: row.get("condition") == "zero"
        )
        singles = {
            str(row.get("supporting_fact_ids_shown"))
            for row in group
            if row.get("condition") == "partial"
        }
        per_single = {}
        for signature in sorted(singles):
            accuracy, correct, total = slice_accuracy(
                lambda row, signature=signature: row.get("condition") == "partial"
                and str(row.get("supporting_fact_ids_shown")) == signature
            )
            per_single[signature] = {
                "accuracy": accuracy,
                "num_tasks": total,
                "num_correct": correct,
            }
        full_low, full_high = wilson_interval(full_correct, full_n)
        partial_low, partial_high = wilson_interval(partial_correct, partial_n)
        output.append(
            {
                "parameter_condition": label,
                "reasoning_depth": 2,
                "distractors": group[0].get("distractors"),
                "num_options": group[0].get("num_options"),
                "chance_level": 1.0 / float(group[0].get("num_options") or 3),
                "num_tasks_full": full_n,
                "num_tasks_partial": partial_n,
                "num_tasks_zero": zero_n,
                "accuracy_full": full_accuracy,
                "accuracy_full_ci95": [full_low, full_high],
                "accuracy_partial": partial_accuracy,
                "accuracy_partial_ci95": [partial_low, partial_high],
                "accuracy_zero": zero_accuracy,
                "full_minus_partial": (
                    None
                    if full_accuracy is None or partial_accuracy is None
                    else full_accuracy - partial_accuracy
                ),
                "accuracy_per_single_fact": per_single,
            }
        )
    return output


def accuracy_by_correct_position(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy split by the label the correct relation was displayed under."""

    return _aggregate(
        _scored(rows),
        ("parameter_condition", "condition", "num_supporting_facts_shown",
         "correct_display_position"),
    )


def _distribution(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value_field: str
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, group in sorted(buckets.items(), key=lambda item: [str(v) for v in item[0]]):
        counts: dict[str, int] = defaultdict(int)
        for row in group:
            counts[str(row.get(value_field) or "<unparsed>")] += 1
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            entry = dict(zip(keys, values))
            entry.update(
                {
                    value_field: value,
                    "count": count,
                    "share": count / len(group),
                    "group_size": len(group),
                }
            )
            output.append(entry)
    return output


def predicted_position_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Which displayed label the model picked, by evidence condition."""

    return _distribution(_scored(rows), ("condition", "num_supporting_facts_shown"), "prediction")


def predicted_relation_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Which compass relation the model picked, by evidence condition."""

    return _distribution(
        _scored(rows), ("condition", "num_supporting_facts_shown"), "predicted_relation"
    )


def accuracy_by_feasible_options(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy split by how many displayed options survive elimination.

    An item where only one option is reachable from the shown facts is
    answerable *without* the missing link, so pooling it with a genuine 3-way
    choice understates how much the withheld fact actually matters.  The
    ``num_feasible_options == 3`` rows are the honest partial-evidence subset.
    """

    scored = [row for row in _scored(rows) if row.get("num_feasible_options") is not None]
    return _aggregate(
        scored, ("condition", "num_supporting_facts_shown", "num_feasible_options")
    )


def headline_overall(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The four headline numbers pooled over every distractor setting."""

    scored = [row for row in _scored(rows) if row.get("reasoning_depth") == 2]

    def slice_stats(condition: str) -> dict[str, Any]:
        subset = [row for row in scored if row.get("condition") == condition]
        correct = sum(1 for row in subset if row.get("correct"))
        low, high = wilson_interval(correct, len(subset))
        return {
            "num_items": len(subset),
            "accuracy": correct / len(subset) if subset else None,
            "ci95": [low, high],
        }

    full, partial, zero = (slice_stats(name) for name in ("full", "partial", "zero"))
    options = {int(row.get("num_options") or 3) for row in scored}
    return {
        "chance_level": 1.0 / max(options) if options else None,
        "accuracy_full": full["accuracy"],
        "accuracy_full_ci95": full["ci95"],
        "accuracy_partial": partial["accuracy"],
        "accuracy_partial_ci95": partial["ci95"],
        "accuracy_zero": zero["accuracy"],
        "accuracy_zero_ci95": zero["ci95"],
        "full_minus_partial": (
            None
            if full["accuracy"] is None or partial["accuracy"] is None
            else full["accuracy"] - partial["accuracy"]
        ),
        "partial_minus_zero": (
            None
            if partial["accuracy"] is None or zero["accuracy"] is None
            else partial["accuracy"] - zero["accuracy"]
        ),
        "num_items_full": full["num_items"],
        "num_items_partial": partial["num_items"],
        "num_items_zero": zero["num_items"],
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = _scored(rows)
    return {
        "rows_total": len(rows),
        "rows_scored": len(scored),
        "rows_with_provider_error": len(rows) - len(scored),
        "rows_unparsed": sum(1 for row in scored if not row.get("parse_ok")),
        "accuracy_by_k": accuracy_by_k(rows),
        "accuracy_by_condition_id": accuracy_by_condition_id(rows),
        "headline_l2": headline_l2(rows),
        "headline_overall": headline_overall(rows),
        "accuracy_by_correct_position": accuracy_by_correct_position(rows),
        "predicted_position_distribution": predicted_position_distribution(rows),
        "predicted_relation_distribution": predicted_relation_distribution(rows),
        "accuracy_by_feasible_options": accuracy_by_feasible_options(rows),
    }


def enrich_with_feasibility(
    rows: Sequence[Mapping[str, Any]], datasets_dir: str | Path
) -> list[dict[str, Any]]:
    """Add ``num_feasible_options`` to rows produced before it was recorded.

    Runs after the fact from the run's own ``datasets/`` tree, so an existing
    result set gains the column without being re-collected.  Rows that already
    carry the field, and rows whose dataset is missing, are returned untouched.
    """

    from .geometry import feasible_options
    from .tasks import load_benchmark_tasks

    root = Path(datasets_dir)
    if not root.is_dir():
        return [dict(row) for row in rows]

    tasks: dict[tuple[str, str], Any] = {}
    for cell in sorted(root.iterdir()):
        if cell.is_dir():
            for task in load_benchmark_tasks(cell):
                tasks[(cell.name, task.task_id)] = task

    def _ids(value: Any) -> list[str]:
        text = str(value or "none")
        return [] if text == "none" else text.split("+")

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        task = tasks.get((str(item.get("parameter_condition")), str(item.get("task_id"))))
        if task is None or item.get("num_feasible_options") is not None:
            enriched.append(item)
            continue
        shown = _ids(item.get("supporting_fact_ids_shown"))
        omitted = _ids(item.get("supporting_fact_ids_omitted"))
        options = str(item.get("displayed_option_order") or "").split("|")
        surviving = feasible_options(
            [task.fact(f) for f in [*shown, *task.distractor_fact_ids]],
            [task.fact(f) for f in omitted],
            [option for option in options if option],
            task.question_subject,
            task.question_reference,
        )
        item["num_feasible_options"] = len(surviving)
        item["feasible_options"] = "|".join(surviving)
        enriched.append(item)
    return enriched


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_no rows_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(input_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Write ``summary.json`` plus the three tables as CSV and one Markdown read."""

    source = Path(input_dir)
    destination = Path(output_dir) if output_dir is not None else source / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(source)
    rows = enrich_with_feasibility(rows, source / "datasets")
    report = summarize(rows)

    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(destination / "accuracy_by_k.csv", report["accuracy_by_k"])
    _write_csv(destination / "accuracy_by_condition_id.csv", report["accuracy_by_condition_id"])
    headline_flat = [
        {key: value for key, value in entry.items() if key != "accuracy_per_single_fact"}
        for entry in report["headline_l2"]
    ]
    _write_csv(destination / "headline_l2.csv", headline_flat)

    per_single_lines = []
    for entry in report["headline_l2"]:
        for signature, stats in entry["accuracy_per_single_fact"].items():
            per_single_lines.append(
                {
                    "parameter_condition": entry["parameter_condition"],
                    "supporting_fact_ids_shown": signature,
                    "num_tasks": stats["num_tasks"],
                    "accuracy": stats["accuracy"],
                }
            )
    _write_csv(destination / "l2_single_fact_conditions.csv", per_single_lines)
    _write_csv(destination / "headline_overall.csv", [report["headline_overall"]])
    _write_csv(
        destination / "accuracy_by_correct_position.csv",
        report["accuracy_by_correct_position"],
    )
    _write_csv(
        destination / "predicted_position_distribution.csv",
        report["predicted_position_distribution"],
    )
    _write_csv(
        destination / "predicted_relation_distribution.csv",
        report["predicted_relation_distribution"],
    )
    _write_csv(
        destination / "accuracy_by_feasible_options.csv",
        report["accuracy_by_feasible_options"],
    )

    overall = report["headline_overall"]

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    markdown = [
        "# Relational support benchmark - summary\n",
        f"- rows: {report['rows_total']} "
        f"(scored {report['rows_scored']}, provider errors "
        f"{report['rows_with_provider_error']}, unparsed {report['rows_unparsed']})\n",
        "- correctness is scored on the **semantic relation**, not the displayed "
        "letter; the correct relation is placed at every label in turn\n",
        "\n## Headline (pooled over distractor settings)\n\n",
        "| quantity | value |\n|---|---|\n"
        f"| chance level | {_fmt(overall['chance_level'])} |\n"
        f"| accuracy_full | {_fmt(overall['accuracy_full'])} "
        f"(n={overall['num_items_full']}) |\n"
        f"| accuracy_partial | {_fmt(overall['accuracy_partial'])} "
        f"(n={overall['num_items_partial']}) |\n"
        f"| accuracy_zero | {_fmt(overall['accuracy_zero'])} "
        f"(n={overall['num_items_zero']}) |\n"
        f"| full_minus_partial | {_fmt(overall['full_minus_partial'])} |\n"
        f"| partial_minus_zero | {_fmt(overall['partial_minus_zero'])} |\n",
        "\n## A_k: accuracy given k of the L supporting facts\n\n",
        _markdown_table(
            report["accuracy_by_k"],
            ["parameter_condition", "reasoning_depth", "distractors", "num_options",
             "num_supporting_facts_shown", "num_tasks", "accuracy", "ci95_low", "ci95_high"],
        ),
        "\n## Per-subset accuracy (before pooling)\n\n",
        _markdown_table(
            report["accuracy_by_condition_id"],
            ["parameter_condition", "num_supporting_facts_shown", "supporting_fact_ids_shown",
             "num_tasks", "accuracy", "ci95_low", "ci95_high"],
        ),
        "\n## L = 2 headline\n\n",
        _markdown_table(
            headline_flat,
            ["parameter_condition", "chance_level", "num_tasks_full", "accuracy_full",
             "accuracy_partial", "accuracy_zero", "full_minus_partial"],
        ),
        "\n### L = 2 single-fact conditions, separately\n\n",
        _markdown_table(
            per_single_lines,
            ["parameter_condition", "supporting_fact_ids_shown", "num_tasks", "accuracy"],
        ),
        "\n## Accuracy by the position the correct relation was displayed at\n\n",
        _markdown_table(
            report["accuracy_by_correct_position"],
            ["parameter_condition", "condition", "correct_display_position",
             "num_tasks", "accuracy", "ci95_low", "ci95_high"],
        ),
        "\n## Which label the model picked\n\n",
        _markdown_table(
            report["predicted_position_distribution"],
            ["condition", "num_supporting_facts_shown", "prediction", "count", "share"],
        ),
        "\n## Which relation the model picked\n\n",
        _markdown_table(
            report["predicted_relation_distribution"],
            ["condition", "num_supporting_facts_shown", "predicted_relation", "count", "share"],
        ),
        "\n## Accuracy by how many displayed options survive elimination\n\n",
        "An item with one feasible option is answerable without the missing "
        "link; only `num_feasible_options == 3` is a genuine 3-way choice.\n\n",
        _markdown_table(
            report["accuracy_by_feasible_options"],
            ["condition", "num_supporting_facts_shown", "num_feasible_options",
             "num_tasks", "accuracy", "ci95_low", "ci95_high"],
        ),
    ]
    (destination / "summary.md").write_text("".join(markdown), encoding="utf-8")
    return destination
