"""Self-contained Markdown report for the focused local probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import CellEffect, ModelQuality, effect_index, r_exposure
from .config import ProbeConfig
from .design import ONE_SLOT, TWO_SLOTS

NA = "N/A"


def _num(value: float | None) -> str:
    return NA if value is None or value != value else f"{value:+.3f}"


def _rate(value: float | None) -> str:
    return NA if value is None or value != value else f"{value:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return ["_No complete matched rows._", ""]
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if index else "---" for index, _ in enumerate(headers)) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def build_report(
    config: ProbeConfig,
    effects: Sequence[CellEffect],
    quality: Sequence[ModelQuality],
    plot_paths: Mapping[str, Sequence[Path]],
    preflight: Mapping[str, Any],
    output_dir: Path,
) -> str:
    index = effect_index(effects)
    lines = [
        "# Controller-retention local probe",
        "",
        "This standalone probe asks whether increasing the visible social group from",
        "`q=2` to `q=3` reduces one controller's influence on one LLM decision.",
        "It runs no population, rounds, controller policy, or trajectory.",
        "",
        "For each frozen vignette, the controlled prompt and its `NO_OP` baseline",
        "share the task, private facts, initial vote, option order, and ordinary-peer",
        "panel. Only the selected social slots are replaced by controller messages.",
        "",
        "`Delta_C = P(X' = controller target | controlled) - P(X' = controller target | NO_OP)`.",
        "Positive values mean that the controller moves responses toward its target.",
        "The `q=3` exposure rescue is `Delta_C(two_slots) - Delta_C(one_slot)`.",
        "These are descriptive local differences. No confidence intervals, p-values,",
        "mutual information, or population-level quantities are reported.",
        "",
        "## Design and execution",
        "",
    ]
    calls = preflight.get("calls", {})
    lines += _table(
        ["Setting", "Value"],
        [
            ["Models", ", ".join(f"`{model.label}` (`{model.model}`)" for model in config.models)],
            ["Reasoning depths `L`", "1, 2"],
            ["Visible slots `q`", "2, 3"],
            ["Targets", "truth, false"],
            ["Receiver", "naive"],
            ["Message mode", "recommendation_only"],
            ["Frozen vignettes per `L`", "12"],
            ["Replicates", "1"],
            ["Calls per model", str(calls.get("calls_per_model", 240))],
            ["Total calls", str(calls.get("calls_total", NA))],
            ["Workers", str(preflight.get("concurrency", {}).get("effective_workers", NA))],
        ],
    )
    lines += ["## Cross-model summary", ""]
    summary_rows: list[list[str]] = []
    for model in config.models:
        mine = [effect for effect in effects if effect.key[0] == model.label]
        q2 = _mean(effect.delta_c for effect in mine if effect.key[3:] == (2, ONE_SLOT))
        q3 = _mean(effect.delta_c for effect in mine if effect.key[3:] == (3, ONE_SLOT))
        rescue = _mean(
            value
            for effect in mine
            if effect.key[3:] == (3, TWO_SLOTS)
            and (value := r_exposure(index, effect.key)) is not None
        )
        summary_rows.append([
            f"`{model.label}`", _num(q2), _num(q3), _num(None if q2 is None or q3 is None else q3 - q2), _num(rescue)
        ])
    lines += _table(
        ["Model", "Mean Delta_C q=2", "Mean Delta_C q=3 one-slot", "q=3 minus q=2", "Mean exposure rescue"],
        summary_rows,
    )

    for number, model in enumerate(config.models, start=1):
        mine = sorted(
            (effect for effect in effects if effect.key[0] == model.label),
            key=lambda effect: (effect.key[1], effect.key[2], effect.key[3], effect.key[4]),
        )
        lines += [f"## Model {number}: `{model.label}`", "", f"Provider model: `{model.model}`", ""]
        rows = [
            [
                str(effect.key[1]), str(effect.key[2]), str(effect.key[3]), str(effect.key[4]),
                str(effect.n_pairs), _rate(effect.p_controlled), _rate(effect.p_noop), _num(effect.delta_c),
            ]
            for effect in mine
        ]
        lines += _table(
            ["L", "Target", "q", "Exposure", "N vignettes", "P target", "P target NOOP", "Delta_C"],
            rows,
        )
        rescue_rows: list[list[str]] = []
        for depth in (1, 2):
            for target in ("truth", "false"):
                q2 = index.get((model.label, depth, target, 2, ONE_SLOT))
                q3_one = index.get((model.label, depth, target, 3, ONE_SLOT))
                q3_two = index.get((model.label, depth, target, 3, TWO_SLOTS))
                rescue_rows.append([
                    str(depth), target,
                    _num(q2.delta_c if q2 else None),
                    _num(q3_one.delta_c if q3_one else None),
                    _num(q3_two.delta_c if q3_two else None),
                    _num(r_exposure(index, q3_two.key) if q3_two else None),
                ])
        lines += _table(
            ["L", "Target", "Delta_C q=2", "Delta_C q=3 one-slot", "Delta_C q=3 two-slots", "Exposure rescue"],
            rescue_rows,
        )
        for path in plot_paths.get(model.label, ()):
            lines += [f"![Controller retention]({path.relative_to(output_dir)})", ""]
        item = next((row for row in quality if row.model_label == model.label), None)
        if item:
            lines += [
                f"Execution quality: {item.successful}/{item.scheduled} valid calls; "
                f"{item.provider_errors} provider failures; {item.validation_failures} response-contract failures.",
                "",
            ]
        lines += _model_interpretation(model.label, index)

    lines += ["## Cross-model interpretation", ""]
    lines += _interpretation_table(config, index)
    lines += [
        "A later population experiment is easiest to interpret when one-slot `Delta_C`",
        "remains meaningfully positive at its chosen `q`. If it falls at `q=3` but the",
        "two-slot arm restores it, the loss is consistent with exposure dilution and the",
        "population controller design should be reconsidered before launch.",
        "",
        "## Artifacts and reproducibility",
        "",
        f"- Config: `{config.source_path}`",
        f"- Design seed: `{config.design.seed}`",
        "- `raw_calls.jsonl`: resumable provider-call journal.",
        "- `local_response_rows.csv`: individual matched-vignette rows.",
        "- `paired_controller_effects.csv`: grouped direct displacements.",
        "- `model_summary.csv`: provider and response-validation quality.",
        "- Production relational prompt construction, option shuffling, ballot contract,",
        "  ballot parser, controller rendering, and provider adapters are reused.",
        "- The relational game runtime and existing studies are unchanged.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float] | Any) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _model_interpretation(label: str, index: Mapping[tuple, CellEffect]) -> list[str]:
    lines = ["**Interpretation by condition**", ""]
    for depth in (1, 2):
        for target in ("truth", "false"):
            q2 = index.get((label, depth, target, 2, ONE_SLOT))
            q3 = index.get((label, depth, target, 3, ONE_SLOT))
            two = index.get((label, depth, target, 3, TWO_SLOTS))
            if not q2 or not q3 or not two:
                text = NA
            else:
                change = q3.delta_c - q2.delta_c
                rescue = two.delta_c - q3.delta_c
                text = f"q=3 change {_num(change)}; exposure rescue {_num(rescue)}"
            lines.append(f"- `L={depth}`, `{target}`: {text}.")
    lines.append("")
    return lines


def _interpretation_table(config: ProbeConfig, index: Mapping[tuple, CellEffect]) -> list[str]:
    rows: list[list[str]] = []
    for model in config.models:
        changes: list[float] = []
        rescues: list[float] = []
        truth: list[float] = []
        false: list[float] = []
        l1: list[float] = []
        l2: list[float] = []
        for depth in (1, 2):
            for target in ("truth", "false"):
                q2 = index.get((model.label, depth, target, 2, ONE_SLOT))
                q3 = index.get((model.label, depth, target, 3, ONE_SLOT))
                two = index.get((model.label, depth, target, 3, TWO_SLOTS))
                if q2 and q3:
                    changes.append(q3.delta_c - q2.delta_c)
                if q3 and two:
                    rescues.append(two.delta_c - q3.delta_c)
                for effect in (q2, q3):
                    if effect:
                        (truth if target == "truth" else false).append(effect.delta_c)
                        (l1 if depth == 1 else l2).append(effect.delta_c)
        mean_change = _mean(changes)
        mean_rescue = _mean(rescues)
        q2_values = [
            effect.delta_c for key, effect in index.items()
            if key[0] == model.label and key[3:] == (2, ONE_SLOT)
        ]
        q3_values = [
            effect.delta_c for key, effect in index.items()
            if key[0] == model.label and key[3:] == (3, ONE_SLOT)
        ]
        viability = (
            NA if not q2_values or not q3_values else
            "q=2 and q=3" if _mean(q3_values) is not None and _mean(q3_values) > 0.05 else
            "q=2 only" if _mean(q2_values) is not None and _mean(q2_values) > 0.05 else
            "neither clearly supported"
        )
        rows.append([
            f"`{model.label}`",
            "yes" if mean_change is not None and mean_change < 0 else "no clear decrease" if mean_change is not None else NA,
            "yes" if mean_rescue is not None and mean_rescue > 0 else "no clear rescue" if mean_rescue is not None else NA,
            _num(None if not truth or not false else _mean(truth) - _mean(false)),
            _num(None if not l1 or not l2 else _mean(l2) - _mean(l1)),
            viability,
        ])
    return _table(
        ["Model", "Decrease q=2→3?", "Exposure rescue?", "Truth minus false", "L=2 minus L=1", "Population follow-up"],
        rows,
    )


__all__ = ["build_report"]
