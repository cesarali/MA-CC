"""Direct matched local displacements for the focused probe."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_cc.games.relational_reasoning.data import RelationalTask

from .config import ProbeConfig
from .design import NO_OP, ONE_SLOT, PROBE_VERSION, TWO_SLOTS, Vignette

CELL_KEYS = ("model_label", "L", "target_semantics", "q", "exposure")


@dataclass(frozen=True, slots=True)
class ResponseRow:
    probe_version: str
    model_label: str
    provider: str
    model_id: str
    generation_settings_hash: str
    task_id: str
    task_fingerprint: str
    reasoning_depth: int
    truth_semantic: str
    controller_target_semantic: str
    target_semantics: str
    receiver_epistemic_disposition: str
    message_mode: str
    known_fact_ids: tuple[str, ...]
    initial_vote_semantic: str
    q: int
    arm: str
    controller_slots: int
    controller_exposure_fraction: float
    ordinary_peer_vote_count_vector: Mapping[str, int]
    option_permutation_seed: int
    prompt_definition_hash: str
    vignette_id: str
    replicate: int
    final_vote_semantic: str | None
    final_is_controller_target: bool
    parse_ok: bool
    provider_error: str | None
    validation_error: str | None
    latency: float
    input_tokens: int | None
    output_tokens: int | None
    call_id: str

    @property
    def condition(self) -> str:
        return self.arm

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_version": self.probe_version,
            "model_label": self.model_label,
            "provider": self.provider,
            "model_id": self.model_id,
            "generation_settings_hash": self.generation_settings_hash,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "L": self.reasoning_depth,
            "truth_semantic": self.truth_semantic,
            "controller_target_semantic": self.controller_target_semantic,
            "target_semantics": self.target_semantics,
            "receiver_epistemic_disposition": self.receiver_epistemic_disposition,
            "message_mode": self.message_mode,
            "known_fact_ids": "|".join(self.known_fact_ids),
            "initial_vote_semantic": self.initial_vote_semantic,
            "q": self.q,
            "arm": self.arm,
            "controller_slots": self.controller_slots,
            "controller_exposure_fraction": round(self.controller_exposure_fraction, 4),
            "ordinary_peer_vote_count_vector": "|".join(
                f"{key}:{value}"
                for key, value in sorted(self.ordinary_peer_vote_count_vector.items())
            ),
            "option_permutation_seed": self.option_permutation_seed,
            "prompt_definition_hash": self.prompt_definition_hash,
            "vignette_id": self.vignette_id,
            "replicate": self.replicate,
            "final_vote_semantic": self.final_vote_semantic or "",
            "final_is_controller_target": self.final_is_controller_target,
            "parse_ok": self.parse_ok,
            "provider_error": self.provider_error or "",
            "validation_error": self.validation_error or "",
            "latency": round(self.latency, 3),
            "input_tokens": self.input_tokens if self.input_tokens is not None else "",
            "output_tokens": self.output_tokens
            if self.output_tokens is not None
            else "",
            "call_id": self.call_id,
        }


def build_response_rows(
    config: ProbeConfig,
    vignettes: Sequence[Vignette],
    tasks: Mapping[int, Sequence[RelationalTask]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[ResponseRow, ...]:
    from . import vignette as vignette_module

    by_task = {
        (task.reasoning_depth, task.task_id): task
        for group in tasks.values()
        for task in group
    }
    by_vignette = {item.vignette_id: item for item in vignettes}
    rows: list[ResponseRow] = []
    for raw in raw_rows:
        item = by_vignette.get(str(raw.get("vignette_id")))
        if item is None:
            continue
        try:
            model = config.model(str(raw.get("model_label")))
        except KeyError:
            continue
        arm = str(raw.get("arm"))
        expected_call_id = item.call_id(model.call_identity, arm)
        if raw.get("call_id") != expected_call_id:
            continue
        task = by_task[(item.reasoning_depth, item.task_id)]
        vote = raw.get("final_vote_semantic")
        vote = str(vote) if isinstance(vote, str) and vote else None
        rows.append(
            ResponseRow(
                probe_version=PROBE_VERSION,
                model_label=model.label,
                provider=model.provider,
                model_id=model.model,
                generation_settings_hash=model.generation_settings_hash,
                task_id=item.task_id,
                task_fingerprint=item.task_fingerprint,
                reasoning_depth=item.reasoning_depth,
                truth_semantic=item.truth_semantic,
                controller_target_semantic=item.controller_target_semantic,
                target_semantics=item.target_semantics,
                receiver_epistemic_disposition=item.receiver_epistemic_disposition,
                message_mode=item.message_mode,
                known_fact_ids=item.known_fact_ids,
                initial_vote_semantic=item.initial_vote_semantic,
                q=item.q,
                arm=arm,
                controller_slots=item.controller_slots(arm),
                controller_exposure_fraction=item.controller_exposure_fraction(arm),
                ordinary_peer_vote_count_vector=item.vote_count_vector(
                    task.semantic_answers, arm
                ),
                option_permutation_seed=item.option_permutation_seed,
                prompt_definition_hash=vignette_module.build_prompt(
                    task, item, arm
                ).definition_hash,
                vignette_id=item.vignette_id,
                replicate=item.replicate,
                final_vote_semantic=vote,
                final_is_controller_target=vote == item.controller_target_semantic,
                parse_ok=bool(raw.get("parse_ok")),
                provider_error=str(raw["provider_error"])
                if raw.get("provider_error")
                else None,
                validation_error=str(raw["validation_error"])
                if raw.get("validation_error")
                else None,
                latency=float(raw.get("latency") or 0.0),
                input_tokens=_int_or_none(raw.get("input_tokens")),
                output_tokens=_int_or_none(raw.get("output_tokens")),
                call_id=expected_call_id,
            )
        )
    return tuple(rows)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    model_label: str
    row: ResponseRow
    controlled_hit: bool
    noop_hit: bool

    @property
    def delta(self) -> int:
        return int(self.controlled_hit) - int(self.noop_hit)


def pair_outcomes(rows: Sequence[ResponseRow]) -> tuple[PairedOutcome, ...]:
    index = {(row.model_label, row.vignette_id, row.arm): row for row in rows}
    pairs: list[PairedOutcome] = []
    for (model_label, vignette_id, arm), row in sorted(index.items()):
        if arm == NO_OP:
            continue
        baseline = index.get((model_label, vignette_id, NO_OP))
        if baseline is None or not row.parse_ok or not baseline.parse_ok:
            continue
        pairs.append(
            PairedOutcome(
                model_label=model_label,
                row=row,
                controlled_hit=row.final_is_controller_target,
                noop_hit=baseline.final_is_controller_target,
            )
        )
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class CellEffect:
    key: tuple
    n_pairs: int
    p_controlled: float
    p_noop: float
    delta_c: float

    def as_dict(self) -> dict[str, Any]:
        payload = {name: value for name, value in zip(CELL_KEYS, self.key)}
        payload.update(
            n_pairs=self.n_pairs,
            p_controlled=round(self.p_controlled, 4),
            p_noop=round(self.p_noop, 4),
            delta_c=round(self.delta_c, 4),
        )
        return payload


def cell_key(pair: PairedOutcome) -> tuple:
    row = pair.row
    return (
        pair.model_label,
        row.reasoning_depth,
        row.target_semantics,
        row.q,
        row.arm,
    )


def cell_effects(pairs: Sequence[PairedOutcome]) -> tuple[CellEffect, ...]:
    grouped: dict[tuple, list[PairedOutcome]] = {}
    for pair in pairs:
        grouped.setdefault(cell_key(pair), []).append(pair)
    effects: list[CellEffect] = []
    for key, group in sorted(grouped.items(), key=str):
        n = len(group)
        p_controlled = sum(item.controlled_hit for item in group) / n
        p_noop = sum(item.noop_hit for item in group) / n
        effects.append(CellEffect(key, n, p_controlled, p_noop, p_controlled - p_noop))
    return tuple(effects)


def effect_index(effects: Sequence[CellEffect]) -> dict[tuple, CellEffect]:
    return {effect.key: effect for effect in effects}


def r_exposure(effects: Mapping[tuple, CellEffect], key: tuple) -> float | None:
    model, depth, target, q, _ = key
    if q != 3:
        return None
    one = effects.get((model, depth, target, 3, ONE_SLOT))
    two = effects.get((model, depth, target, 3, TWO_SLOTS))
    if one is None or two is None:
        return None
    return two.delta_c - one.delta_c


@dataclass(frozen=True, slots=True)
class ModelQuality:
    model_label: str
    scheduled: int
    successful: int
    provider_errors: int
    validation_failures: int
    median_latency: float
    input_tokens: int
    output_tokens: int

    @property
    def success_rate(self) -> float:
        return self.successful / self.scheduled if self.scheduled else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "calls_scheduled": self.scheduled,
            "calls_successful": self.successful,
            "provider_errors": self.provider_errors,
            "validation_failures": self.validation_failures,
            "success_rate": round(self.success_rate, 4),
            "median_latency_seconds": round(self.median_latency, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def model_quality(
    rows: Sequence[ResponseRow], scheduled: Mapping[str, int]
) -> tuple[ModelQuality, ...]:
    grouped: dict[str, list[ResponseRow]] = {}
    for row in rows:
        grouped.setdefault(row.model_label, []).append(row)
    quality: list[ModelQuality] = []
    for label in sorted(scheduled):
        group = grouped.get(label, [])
        latencies = sorted(row.latency for row in group if row.parse_ok)
        quality.append(
            ModelQuality(
                model_label=label,
                scheduled=scheduled[label],
                successful=sum(row.parse_ok for row in group),
                provider_errors=sum(bool(row.provider_error) for row in group),
                validation_failures=sum(bool(row.validation_error) for row in group),
                median_latency=latencies[len(latencies) // 2]
                if latencies
                else float("nan"),
                input_tokens=sum(row.input_tokens or 0 for row in group),
                output_tokens=sum(row.output_tokens or 0 for row in group),
            )
        )
    return tuple(quality)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# no rows\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "CELL_KEYS",
    "CellEffect",
    "ModelQuality",
    "PairedOutcome",
    "ResponseRow",
    "build_response_rows",
    "cell_effects",
    "effect_index",
    "model_quality",
    "pair_outcomes",
    "r_exposure",
    "write_csv",
]
