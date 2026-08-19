#!/usr/bin/env python3
"""Aggregate a sharded grid study into study-level analysis tables.

A sharded study is one where every original grid cell was run in its own
process by ``run_classical_grid_cell.py``, so the study root looks like::

    <result_root>/<arm>/shard_NNNN/<game>/<experiment>/<run>/cells/<cell_id>/

Each shard is a complete, self-consistent run tree that happens to contain a
single cell. Nothing is missing scientifically -- what is missing is the
grid-level roll-up a single-process run would have written. This script builds
that roll-up and nothing else: it reads, it never recomputes dynamics, and it
never writes inside the source tree.

Outputs (one study-level folder):

    cells.csv             one row per grid cell (design matrix + completion)
    episodes.parquet      one row per episode (compact trajectory summaries)
    rounds.parquet        one row per round record (the primary substrate)
    interactions.parquet  every scientific-event row (per-agent decisions)
    source_manifest.csv   every source file read, with sha256 and row counts
    aggregation_report.json  counts, breakdowns, exclusions, integrity checks
    README.md             what each table is and how the columns were derived

Design columns are derived from the per-cell ``resolved_config.yaml`` (which is
authoritative) and cross-checked against the round records; any disagreement is
a hard failure rather than a silent coercion.

Reuse on another sharded study:

    python scripts/experiment_design/aggregate_grid_shards.py \
      --result-root /work/.../results/<study> \
      --output-dir  /work/.../results/<study>_analysis \
      --zip

Counts are asserted only when you state them (--expect-cells/-episodes/-rounds)
or when --strict-counts is combined with them; otherwise they are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

SCRIPT_VERSION = "1.0.0"

SHARD_DIR_RE = re.compile(r"^shard_(\d+)$")
DATASET_R_RE = re.compile(r"_r(\d+)(?:$|_)")
EPISODE_REP_RE = re.compile(r"^(?P<cell>.+)-(?P<rep>\d{1,6})$")

# Round-record fields used to build the per-episode summaries. "before" is read
# off round 0 and "after" off the final round, so the summary spans the whole
# episode regardless of how many rounds it ran.
TRAJECTORY_METRICS = {
    "truth_vote_share": ("truth_vote_share_before", "truth_vote_share"),
    "kappa": ("mean_supporting_fact_coverage_before", "mean_supporting_fact_coverage"),
    "phi": ("full_proof_agent_share_before", "full_proof_agent_share"),
}
FINAL_STRATUM_FIELDS = ("truth_share_k0", "truth_share_k1", "truth_share_k2")
EXPOSURE_TOTALS = (
    "peer_fact_exposures",
    "controller_fact_exposures",
    "new_peer_facts",
    "new_controller_facts",
    "controlled_update_count",
    "controlled_adoption_count",
    "controlled_off_target_count",
)
FINAL_SCALARS = (
    "m_truth_after",
    "m_ctrl_after",
    "m_order_after",
    "vote_entropy",
    "knowledge_share_k0",
    "knowledge_share_k1",
    "knowledge_share_k2",
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mean_of(values: Iterable[Any]) -> float | None:
    kept = [v for v in values if v is not None and not pd.isna(v)]
    return float(sum(kept) / len(kept)) if kept else None


def total_of(values: Iterable[Any]) -> float | None:
    kept = [v for v in values if v is not None and not pd.isna(v)]
    return float(sum(kept)) if kept else None


def delta(initial: Any, final: Any) -> float | None:
    if initial is None or final is None or pd.isna(initial) or pd.isna(final):
        return None
    return float(final) - float(initial)


def first_index_where(rows: list[dict], predicate) -> int | None:
    for row in rows:
        try:
            if predicate(row):
                return int(row.get("round_index"))
        except (TypeError, ValueError):
            continue
    return None


def json_safe(value: Any) -> Any:
    """Make one round-record value safe to put in a parquet column."""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


# --------------------------------------------------------------------------
# aggregator
# --------------------------------------------------------------------------


class ShardAggregator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.result_root: Path = args.result_root.resolve()
        self.output_dir: Path = args.output_dir.resolve()
        self.repo_root: Path | None = args.repo_root.resolve() if args.repo_root else None
        self.violations: list[str] = []
        self.warnings: list[str] = []
        self.excluded_dirs: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []
        self.dataset_manifest_cache: dict[str, dict[str, Any]] = {}

    # -- reporting -------------------------------------------------------

    def fail(self, message: str) -> None:
        self.violations.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def record_source(self, path: Path, kind: str, rows: int, **extra: Any) -> None:
        stat = path.stat()
        self.sources.append(
            {
                "kind": kind,
                "path": str(path),
                "relative_path": os.path.relpath(path, self.result_root),
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "sha256": sha256_file(path),
                "rows_contributed": rows,
                **extra,
            }
        )

    # -- discovery -------------------------------------------------------

    def discover_arms(self) -> list[str]:
        if self.args.arms:
            arms = list(self.args.arms)
            for arm in arms:
                if not (self.result_root / arm).is_dir():
                    self.fail(f"requested arm directory does not exist: {arm}")
            return arms
        arms = []
        for child in sorted(self.result_root.iterdir()):
            if not child.is_dir():
                continue
            if any(SHARD_DIR_RE.match(sub.name) for sub in child.iterdir() if sub.is_dir()):
                arms.append(child.name)
            else:
                self.excluded_dirs.append(
                    {
                        "arm": None,
                        "directory": child.name,
                        "relative_path": os.path.relpath(child, self.result_root),
                        "reason": "top-level directory contains no shard_* subdirectory",
                    }
                )
        return arms

    def discover_shards(self, arm: str) -> list[tuple[str, int, Path]]:
        """Return accepted shards; every rejected sibling is logged as excluded.

        Acceptance is by strict name match on ``shard_<digits>``. Quarantine
        directories produced during remediation (``failed503_*_shard_0000``,
        ``backup_n1_shard_0004``, ``excluded_n1_shard_0004``, ...) embed the
        word "shard" but never match at position zero, so they are excluded
        here and enumerated in the report.
        """
        arm_dir = self.result_root / arm
        shards: list[tuple[str, int, Path]] = []
        for child in sorted(arm_dir.iterdir()):
            if not child.is_dir():
                continue
            match = SHARD_DIR_RE.match(child.name)
            if match:
                shards.append((child.name, int(match.group(1)), child))
            else:
                has_payload = any(child.rglob("cell_summary.json"))
                self.excluded_dirs.append(
                    {
                        "arm": arm,
                        "directory": child.name,
                        "relative_path": os.path.relpath(child, self.result_root),
                        "reason": (
                            "name does not match ^shard_\\d+$ "
                            f"(contains run payload: {has_payload})"
                        ),
                    }
                )
        return shards

    def find_cell_dirs(self, shard_dir: Path) -> list[Path]:
        cells = sorted(
            {p.parent for p in shard_dir.rglob("cells/*/cell_summary.json")}
        )
        return cells

    # -- design derivation ----------------------------------------------

    def dataset_support_redundancy(self, dataset_dir: str) -> int | None:
        """Read support_redundancy from the dataset manifest, if reachable."""
        if dataset_dir in self.dataset_manifest_cache:
            manifest = self.dataset_manifest_cache[dataset_dir]
        else:
            if self.repo_root is None:
                return None
            manifest_path = self.repo_root / dataset_dir / "manifest.json"
            if not manifest_path.is_file():
                self.warn(
                    f"dataset manifest not found, r cross-check skipped: {manifest_path}"
                )
                self.dataset_manifest_cache[dataset_dir] = {}
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.dataset_manifest_cache[dataset_dir] = manifest
        return manifest.get("config", {}).get("support_redundancy")

    def parse_r(self, dataset_dir: str, where: str) -> int | None:
        """Parse r from the dataset directory name and verify it.

        r is the support redundancy of the task dataset. It is NOT
        game.options.social_group_size -- that field is q, and it is 1 in every
        cell of this study. r appears only in the dataset directory name, so it
        is parsed here and cross-checked against the dataset manifest.
        """
        basename = Path(dataset_dir).name
        match = DATASET_R_RE.search(basename)
        if not match:
            self.fail(f"cannot parse r from dataset directory {basename!r} at {where}")
            return None
        parsed = int(match.group(1))
        declared = self.dataset_support_redundancy(dataset_dir)
        if declared is not None and int(declared) != parsed:
            self.fail(
                f"r mismatch at {where}: dataset name says r={parsed} but "
                f"{basename}/manifest.json declares support_redundancy={declared}"
            )
        return parsed

    def derive_design(self, arm: str, shard: str, cell_dir: Path, cfg: dict) -> dict:
        """Build the design row for one cell from its resolved config."""
        where = f"{arm}/{shard}/{cell_dir.name}"
        game_options = cfg.get("game", {}).get("options", {}) or {}
        control = cfg.get("control", {}) or {}
        mechanism = control.get("mechanism", "none")
        control_options = control.get("options", {}) or {}

        dataset_dir = game_options.get("task_dataset_dir")
        if not dataset_dir:
            self.fail(f"missing game.options.task_dataset_dir at {where}")
        r_value = self.parse_r(dataset_dir, where) if dataset_dir else None

        # b: the controller's per-round intervention budget. An uncontrolled arm
        # has no control.options at all, so b is explicitly 0 rather than null;
        # a fixed-budget arm carries its budget in the config (24 for the
        # adversarial arm). Both are cross-checked against the round records.
        if mechanism == "none":
            b_value: int | None = 0
            b_source = "control.mechanism == none -> 0"
        elif "intervention_budget" in control_options:
            b_value = int(control_options["intervention_budget"])
            b_source = "control.options.intervention_budget"
        else:
            b_value = None
            b_source = "unavailable"
            self.fail(
                f"control mechanism {mechanism!r} has no intervention_budget at {where}"
            )

        target_mode = control_options.get("target") if mechanism != "none" else "none"
        message_mode = (
            control_options.get("message_mode") if mechanism != "none" else "none"
        )

        return {
            "arm": arm,
            "shard": shard,
            "cell_id": cell_dir.name,
            "task_id": game_options.get("task_id"),
            "world_id": game_options.get("task_id"),
            "dataset": Path(dataset_dir).name if dataset_dir else None,
            "dataset_dir": dataset_dir,
            "r": r_value,
            "b": b_value,
            "b_source": b_source,
            "q": game_options.get("social_group_size"),
            "rounds_configured": game_options.get("rounds"),
            "dynamics_mode": game_options.get("dynamics_mode"),
            "control_mechanism": mechanism,
            "target_mode": target_mode,
            "message_mode": message_mode,
            "controller_fact_selector": control_options.get("controller_fact_selector"),
            "controller_policy": control_options.get("policy"),
            "controller_threshold": control_options.get("threshold"),
            "controller_beta": control_options.get("beta"),
            "sensor_sample_size": control_options.get("sensor_sample_size"),
            "advocacy_schedule": control_options.get("advocacy_schedule"),
            "vote_visibility": game_options.get("vote_visibility"),
            "social_distrust": game_options.get("social_distrust"),
            "initialization_mode": (game_options.get("initialization") or {}).get("mode"),
        }

    # -- per-cell ingest -------------------------------------------------

    def load_cell(self, arm: str, shard: str, shard_index: int, cell_dir: Path) -> dict | None:
        where = f"{arm}/{shard}/{cell_dir.name}"
        run_dir = cell_dir.parent.parent

        config_path = cell_dir / "resolved_config.yaml"
        summary_path = cell_dir / "cell_summary.json"
        complete_path = cell_dir / "cell_complete.json"
        events_path = cell_dir / "scientific_events.parquet"

        if not config_path.is_file():
            self.fail(f"missing resolved_config.yaml at {where}")
            return None
        if not summary_path.is_file():
            self.fail(f"missing cell_summary.json at {where}")
            return None
        if not complete_path.is_file():
            self.fail(f"cell never wrote cell_complete.json (incomplete cell) at {where}")

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.record_source(config_path, "resolved_config.yaml", 0, arm=arm, shard=shard)
        self.record_source(summary_path, "cell_summary.json", 1, arm=arm, shard=shard)

        design = self.derive_design(arm, shard, cell_dir, cfg)
        design["shard_index"] = shard_index
        design["run_dir"] = os.path.relpath(run_dir, self.result_root)

        shard_def_path = cell_dir.parents[2].parent / "shard_definition.json"
        if not shard_def_path.is_file():
            shard_def_path = self.result_root / arm / shard / "shard_definition.json"
        if shard_def_path.is_file():
            shard_def = json.loads(shard_def_path.read_text(encoding="utf-8"))
            design["cell_index"] = shard_def.get("selected_cell_index")
            design["original_grid_id"] = shard_def.get("original_grid_id")
            self.record_source(shard_def_path, "shard_definition.json", 0, arm=arm, shard=shard)
            if shard_def.get("selected_cell_id") != cell_dir.name:
                self.fail(
                    f"shard_definition.json cell id {shard_def.get('selected_cell_id')!r} "
                    f"does not match cell directory {cell_dir.name!r} at {where}"
                )
        else:
            design["cell_index"] = None
            design["original_grid_id"] = None
            self.warn(f"no shard_definition.json for {where}")

        # completion bookkeeping straight off the cell summary
        design["episodes_completed"] = summary.get("completed")
        design["episodes_failed"] = summary.get("failed")
        design["episodes_skipped_resumed"] = summary.get("skipped_resumed")
        design["episodes_skipped_aborted"] = summary.get("skipped_aborted")
        design["failure_summary"] = json.dumps(summary.get("failures", []), default=str)
        if summary.get("failed"):
            self.fail(
                f"{where} reports {summary['failed']} failed episode(s); "
                "the study is not clean"
            )

        episodes, rounds = self.load_round_records(arm, shard, cell_dir, design)
        interactions = self.load_interactions(arm, shard, cell_dir, events_path, design)

        design["episodes_found"] = len(episodes)
        design["rounds_found"] = len(rounds)
        design["interaction_rows"] = 0 if interactions is None else len(interactions)

        if design["episodes_completed"] is not None and len(episodes) != design["episodes_completed"]:
            self.fail(
                f"{where}: cell_summary says {design['episodes_completed']} completed "
                f"episode(s) but {len(episodes)} round-record director(ies) were found"
            )

        return {
            "design": design,
            "episodes": episodes,
            "rounds": rounds,
            "interactions": interactions,
        }

    def load_round_records(
        self, arm: str, shard: str, cell_dir: Path, design: dict
    ) -> tuple[list[dict], list[dict]]:
        episodes: list[dict] = []
        rounds: list[dict] = []
        record_root = cell_dir / "round_records"
        if not record_root.is_dir():
            self.fail(f"missing round_records/ at {arm}/{shard}/{cell_dir.name}")
            return episodes, rounds

        for episode_dir in sorted(p for p in record_root.iterdir() if p.is_dir()):
            path = episode_dir / "round_trajectory.jsonl"
            where = f"{arm}/{shard}/{cell_dir.name}/{episode_dir.name}"
            if not path.is_file():
                self.fail(f"missing round_trajectory.jsonl at {where}")
                continue
            raw = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not raw:
                self.fail(f"empty round_trajectory.jsonl at {where}")
                continue
            raw.sort(key=lambda rec: rec.get("round_index", 0))
            self.record_source(path, "round_trajectory.jsonl", len(raw), arm=arm, shard=shard)

            episode_design = self.episode_design(design, episode_dir.name, raw, where)
            self.verify_against_records(design, raw, where)

            for rec in raw:
                row = {k: json_safe(v) for k, v in rec.items()}
                for key, value in episode_design.items():
                    if key in row:
                        continue
                    row[key] = value
                rounds.append(row)

            episodes.append(self.summarize_episode(episode_design, raw))

        return episodes, rounds

    def episode_design(
        self, design: dict, episode_id: str, raw: list[dict], where: str
    ) -> dict:
        match = EPISODE_REP_RE.match(episode_id)
        if not match:
            self.fail(f"cannot parse repetition index from episode id {episode_id!r} at {where}")
            repetition = None
        else:
            repetition = int(match.group("rep"))
            if match.group("cell") != design["cell_id"]:
                self.fail(
                    f"episode id {episode_id!r} does not belong to cell "
                    f"{design['cell_id']!r} at {where}"
                )

        head = raw[0]
        population_size = head.get("N")
        b_value = design.get("b")
        b_fraction = (
            float(b_value) / float(population_size)
            if b_value is not None and population_size
            else None
        )

        row = {
            "arm": design["arm"],
            "shard": design["shard"],
            "cell_id": design["cell_id"],
            "episode_id": episode_id,
            "repetition": repetition,
            "task_id": design["task_id"],
            "world_id": design["world_id"],
            "dataset": design["dataset"],
            "r": design["r"],
            "b": b_value,
            "b_fraction": b_fraction,
            "q": design["q"],
            "N": population_size,
            "L": head.get("reasoning_depth_L"),
            "K": head.get("K"),
            "control_mechanism": design["control_mechanism"],
            "target_mode": design["target_mode"],
            "message_mode": design["message_mode"],
            "controller_fact_selector": design["controller_fact_selector"],
            "controller_target": head.get("controller_target"),
            "controller_fact_id": head.get("controller_episode_fact_id")
            or head.get("controller_fact_id"),
            "correct_answer": head.get("correct_answer"),
            "analysis_target": head.get("analysis_target"),
            "controller_enabled": head.get("controller_enabled"),
            "episode_seed": head.get("seed"),
            "task_seed": head.get("task_seed"),
            "n_rounds": len(raw),
        }
        return row

    def verify_against_records(self, design: dict, raw: list[dict], where: str) -> None:
        """Cross-check config-derived design values against the round records."""
        checks = (
            ("b", "intervention_budget"),
            ("q", "social_group_size"),
        )
        for design_key, record_key in checks:
            expected = design.get(design_key)
            if expected is None:
                continue
            observed = {rec.get(record_key) for rec in raw}
            if observed != {expected}:
                self.fail(
                    f"{where}: config gives {design_key}={expected} but round records "
                    f"carry {record_key} in {sorted(observed, key=str)}"
                )
        if design["control_mechanism"] != "none":
            observed_modes = {rec.get("controller_message_mode") for rec in raw}
            if observed_modes != {design["message_mode"]}:
                self.fail(
                    f"{where}: config message_mode={design['message_mode']!r} but records "
                    f"carry {sorted(observed_modes, key=str)}"
                )
        indices = [rec.get("round_index") for rec in raw]
        if indices != list(range(len(raw))):
            self.fail(f"{where}: round_index sequence is not 0..{len(raw) - 1}: {indices}")

    def summarize_episode(self, episode_design: dict, raw: list[dict]) -> dict:
        head, tail = raw[0], raw[-1]
        row = dict(episode_design)

        for name, (before_key, after_key) in TRAJECTORY_METRICS.items():
            initial = head.get(before_key)
            final = tail.get(after_key)
            row[f"{name}_initial"] = initial
            row[f"{name}_final"] = final
            row[f"{name}_delta"] = delta(initial, final)
            row[f"{name}_mean"] = mean_of(rec.get(after_key) for rec in raw)

        for field_name in FINAL_STRATUM_FIELDS:
            row[f"{field_name}_final"] = tail.get(field_name)
        for field_name in FINAL_SCALARS:
            row[f"{field_name}_final" if not field_name.endswith("_after") else field_name] = (
                tail.get(field_name)
            )

        for field_name in EXPOSURE_TOTALS:
            row[f"{field_name}_total"] = total_of(rec.get(field_name) for rec in raw)

        row["controlled_target_adoption_rate_mean"] = mean_of(
            rec.get("controlled_target_adoption_rate") for rec in raw
        )
        row["actuation_fraction_mean"] = mean_of(rec.get("actuation_fraction") for rec in raw)
        row["delta_H_vote_total"] = total_of(rec.get("delta_H_vote") for rec in raw)
        row["vote_entropy_initial"] = head.get("vote_entropy_before")

        row["first_round_phi_positive"] = first_index_where(
            raw, lambda rec: rec.get("full_proof_agent_share") is not None
            and float(rec["full_proof_agent_share"]) > 0.0
        )
        row["first_round_truth_majority"] = first_index_where(
            raw, lambda rec: rec.get("truth_vote_share") is not None
            and float(rec["truth_vote_share"]) > 0.5
        )
        row["truth_majority_final"] = (
            bool(float(tail["truth_vote_share"]) > 0.5)
            if tail.get("truth_vote_share") is not None
            else None
        )
        return row

    def load_interactions(
        self, arm: str, shard: str, cell_dir: Path, events_path: Path, design: dict
    ) -> pd.DataFrame | None:
        if not events_path.is_file():
            self.fail(f"missing scientific_events.parquet at {arm}/{shard}/{cell_dir.name}")
            return None
        frame = pd.read_parquet(events_path)
        self.record_source(events_path, "scientific_events.parquet", len(frame), arm=arm, shard=shard)
        frame.insert(0, "shard", shard)
        frame.insert(0, "arm", arm)
        if "cell_id" in frame.columns:
            unexpected = set(frame["cell_id"].unique()) - {design["cell_id"]}
            if unexpected:
                self.fail(
                    f"{arm}/{shard}: scientific_events carries foreign cell ids {unexpected}"
                )
        return frame

    # -- assembly --------------------------------------------------------

    def run(self) -> dict:
        started = utc_now()
        arms = self.discover_arms()
        cells: list[dict] = []
        episodes: list[dict] = []
        rounds: list[dict] = []
        interaction_frames: list[pd.DataFrame] = []
        round_key_sets: set[frozenset] = set()

        for arm in arms:
            for shard_name, shard_index, shard_dir in self.discover_shards(arm):
                cell_dirs = self.find_cell_dirs(shard_dir)
                if not cell_dirs:
                    self.fail(f"{arm}/{shard_name}: no cell directory found")
                    continue
                for cell_dir in cell_dirs:
                    loaded = self.load_cell(arm, shard_name, shard_index, cell_dir)
                    if loaded is None:
                        continue
                    cells.append(loaded["design"])
                    episodes.extend(loaded["episodes"])
                    rounds.extend(loaded["rounds"])
                    if loaded["interactions"] is not None:
                        interaction_frames.append(loaded["interactions"])

        cells_df = pd.DataFrame(cells)
        episodes_df = pd.DataFrame(episodes)
        rounds_df = pd.DataFrame(rounds)
        interactions_df = (
            pd.concat(interaction_frames, ignore_index=True)
            if interaction_frames
            else pd.DataFrame()
        )

        for row in rounds:
            round_key_sets.add(frozenset(row.keys()))
        if len(round_key_sets) > 1:
            sizes = sorted(len(s) for s in round_key_sets)
            self.warn(
                f"round records do not share one schema; {len(round_key_sets)} distinct "
                f"key sets with sizes {sizes} (columns unioned, missing values are null)"
            )

        self.attach_design_to_interactions(interactions_df, episodes_df)
        self.check_uniqueness(cells_df, episodes_df, rounds_df)
        self.check_missing(cells_df, episodes_df, rounds_df, interactions_df)
        self.check_counts(cells_df, episodes_df, rounds_df)

        report = self.build_report(
            started, arms, cells_df, episodes_df, rounds_df, interactions_df
        )
        return {
            "cells": cells_df,
            "episodes": episodes_df,
            "rounds": rounds_df,
            "interactions": interactions_df,
            "report": report,
        }

    def attach_design_to_interactions(
        self, interactions: pd.DataFrame, episodes: pd.DataFrame
    ) -> None:
        """Broadcast design columns onto the interaction rows in place."""
        if interactions.empty or episodes.empty:
            return
        design_columns = [
            "arm",
            "cell_id",
            "episode_id",
            "world_id",
            "repetition",
            "r",
            "b",
            "b_fraction",
            "q",
            "target_mode",
            "message_mode",
            "controller_target",
            "controller_fact_id",
            "controller_fact_selector",
            "episode_seed",
        ]
        available = [c for c in design_columns if c in episodes.columns]
        keys = ["arm", "cell_id", "episode_id"]
        design = episodes[available].drop_duplicates(subset=keys)

        before = len(interactions)
        collisions = [c for c in available if c not in keys and c in interactions.columns]
        for column in collisions:
            merged = interactions[keys + [column]].merge(
                design[keys + [column]], on=keys, how="left", suffixes=("_events", "_design")
            )
            left, right = merged[f"{column}_events"], merged[f"{column}_design"]
            mismatch = (left.notna() | right.notna()) & (left.astype(str) != right.astype(str))
            if bool(mismatch.any()):
                self.fail(
                    f"interactions column {column!r} disagrees with the round-record design "
                    f"on {int(mismatch.sum())} row(s); refusing to overwrite"
                )
        merge_columns = [c for c in available if c in keys or c not in interactions.columns]
        merged = interactions.merge(design[merge_columns], on=keys, how="left")
        if len(merged) != before:
            self.fail(
                f"design merge changed the interaction row count: {before} -> {len(merged)}"
            )
            return
        for column in merged.columns:
            if column not in interactions.columns:
                interactions[column] = merged[column].values
        unmatched = int(merged["r"].isna().sum()) if "r" in merged.columns else 0
        if unmatched:
            self.fail(f"{unmatched} interaction row(s) did not match any episode design row")

    # -- validation ------------------------------------------------------

    def check_uniqueness(
        self, cells: pd.DataFrame, episodes: pd.DataFrame, rounds: pd.DataFrame
    ) -> None:
        specs = (
            ("cells", cells, ["arm", "cell_id"]),
            ("episodes", episodes, ["arm", "cell_id", "episode_id"]),
            ("rounds", rounds, ["arm", "cell_id", "episode_id", "round_index"]),
        )
        for name, frame, keys in specs:
            if frame.empty:
                self.fail(f"{name} table is empty")
                continue
            missing = [k for k in keys if k not in frame.columns]
            if missing:
                self.fail(f"{name} table is missing key column(s) {missing}")
                continue
            duplicated = frame.duplicated(subset=keys, keep=False)
            if bool(duplicated.any()):
                sample = (
                    frame.loc[duplicated, keys]
                    .drop_duplicates()
                    .head(10)
                    .to_dict("records")
                )
                self.fail(
                    f"{name} table has {int(duplicated.sum())} duplicate row(s) on {keys}; "
                    f"first offenders: {sample}"
                )

    def check_missing(
        self,
        cells: pd.DataFrame,
        episodes: pd.DataFrame,
        rounds: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        required = {
            "cells": (cells, ["arm", "cell_id", "task_id", "r", "b", "q"]),
            "episodes": (
                episodes,
                [
                    "arm",
                    "cell_id",
                    "episode_id",
                    "repetition",
                    "r",
                    "b",
                    "b_fraction",
                    "q",
                    "episode_seed",
                    "truth_vote_share_initial",
                    "truth_vote_share_final",
                ],
            ),
            "rounds": (rounds, ["arm", "cell_id", "episode_id", "round_index", "r", "b", "q"]),
        }
        for name, (frame, columns) in required.items():
            if frame.empty:
                continue
            for column in columns:
                if column not in frame.columns:
                    self.fail(f"{name} table is missing required column {column!r}")
                    continue
                nulls = int(frame[column].isna().sum())
                if nulls:
                    self.fail(f"{name}.{column} has {nulls} null value(s)")
        if interactions.empty:
            self.fail("interactions table is empty")

    def check_counts(
        self, cells: pd.DataFrame, episodes: pd.DataFrame, rounds: pd.DataFrame
    ) -> None:
        expectations = (
            ("cells", len(cells), self.args.expect_cells),
            ("episodes", len(episodes), self.args.expect_episodes),
            ("rounds", len(rounds), self.args.expect_rounds),
        )
        for name, actual, expected in expectations:
            if expected is None:
                continue
            if actual != expected:
                self.fail(f"expected {expected} {name} row(s), found {actual}")

    # -- report ----------------------------------------------------------

    def breakdown(self, frame: pd.DataFrame, keys: list[str]) -> list[dict]:
        if frame.empty or any(k not in frame.columns for k in keys):
            return []
        grouped = frame.groupby(keys, dropna=False).size().reset_index(name="count")
        return json.loads(grouped.to_json(orient="records"))

    def build_report(
        self,
        started: str,
        arms: list[str],
        cells: pd.DataFrame,
        episodes: pd.DataFrame,
        rounds: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> dict:
        arm_constants = []
        if not cells.empty:
            for arm, group in cells.groupby("arm"):
                arm_constants.append(
                    {
                        "arm": arm,
                        "cells": int(len(group)),
                        "control_mechanism": sorted(map(str, group["control_mechanism"].unique())),
                        "target_mode": sorted(map(str, group["target_mode"].unique())),
                        "message_mode": sorted(map(str, group["message_mode"].unique())),
                        "b_values": sorted(int(v) for v in group["b"].dropna().unique()),
                        "b_source": sorted(map(str, group["b_source"].unique())),
                        "r_values": sorted(int(v) for v in group["r"].dropna().unique()),
                        "q_values": sorted(int(v) for v in group["q"].dropna().unique()),
                        "worlds": sorted(map(str, group["task_id"].dropna().unique())),
                    }
                )

        status = "PASS" if not self.violations else "FAIL"
        return {
            "schema_version": 1,
            "script": "scripts/experiment_design/aggregate_grid_shards.py",
            "script_version": SCRIPT_VERSION,
            "status": status,
            "started_utc": started,
            "finished_utc": utc_now(),
            "result_root": str(self.result_root),
            "output_dir": str(self.output_dir),
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "arms": arms,
            "counts": {
                "cells": int(len(cells)),
                "episodes": int(len(episodes)),
                "rounds": int(len(rounds)),
                "interactions": int(len(interactions)),
                "source_files_read": len(self.sources),
            },
            "expected_counts": {
                "cells": self.args.expect_cells,
                "episodes": self.args.expect_episodes,
                "rounds": self.args.expect_rounds,
            },
            "counts_by_arm": self.breakdown(cells, ["arm"]),
            "cells_by_arm_r_b": self.breakdown(cells, ["arm", "r", "b"]),
            "episodes_by_arm_r_b": self.breakdown(episodes, ["arm", "r", "b"]),
            "rounds_by_arm": self.breakdown(rounds, ["arm"]),
            "episodes_by_arm_repetition": self.breakdown(episodes, ["arm", "repetition"]),
            "arm_constants": arm_constants,
            "excluded_directories": self.excluded_dirs,
            "excluded_directory_count": len(self.excluded_dirs),
            "quarantine_exclusion_confirmed": True,
            "integrity_checks": {
                "b_cross_checked_against": "round_trajectory.intervention_budget",
                "q_cross_checked_against": "round_trajectory.social_group_size",
                "r_parsed_from": "game.options.task_dataset_dir basename (_r<NN>)",
                "r_cross_checked_against": "dataset manifest.json config.support_redundancy",
                "shard_acceptance_regex": SHARD_DIR_RE.pattern,
                "source_tree_written_to": False,
            },
            "violations": self.violations,
            "warnings": self.warnings,
        }

    # -- writing ---------------------------------------------------------

    def guard_output_dir(self) -> None:
        if self.output_dir == self.result_root or self.result_root in self.output_dir.parents:
            raise SystemExit(
                f"refusing to write inside the source study tree: {self.output_dir}\n"
                "choose an --output-dir outside --result-root so the shards stay untouched"
            )

    def write(self, tables: dict) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        cells_path = self.output_dir / "cells.csv"
        tables["cells"].to_csv(cells_path, index=False)
        written["cells.csv"] = cells_path

        for name in ("episodes", "rounds", "interactions"):
            path = self.output_dir / f"{name}.parquet"
            frame = tables[name]
            frame.to_parquet(path, index=False)
            written[f"{name}.parquet"] = path

        manifest_path = self.output_dir / "source_manifest.csv"
        pd.DataFrame(self.sources).to_csv(manifest_path, index=False)
        written["source_manifest.csv"] = manifest_path

        readme_path = self.output_dir / "README.md"
        readme_path.write_text(self.render_readme(tables), encoding="utf-8")
        written["README.md"] = readme_path

        report = tables["report"]
        report["outputs"] = [
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(written.items())
        ]
        report_path = self.output_dir / "aggregation_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        written["aggregation_report.json"] = report_path
        return written

    def render_readme(self, tables: dict) -> str:
        report = tables["report"]
        counts = report["counts"]
        arms = ", ".join(report["arms"])
        excluded = report["excluded_directories"]
        excluded_lines = (
            "\n".join(
                f"- `{item['relative_path']}` — {item['reason']}" for item in excluded
            )
            or "- none"
        )
        return f"""# {self.result_root.name} — aggregated analysis tables

Built {report['finished_utc']} by `scripts/experiment_design/aggregate_grid_shards.py`
v{SCRIPT_VERSION} from `{self.result_root}`.

Status: **{report['status']}** — {counts['cells']} cells, {counts['episodes']} episodes,
{counts['rounds']} round records, {counts['interactions']} interaction rows,
read from {counts['source_files_read']} source files across arms: {arms}.

This folder is a **roll-up, not a rerun**. No scientific dynamics were
recomputed: every number here is read from the shard artifacts and re-keyed.
The source study tree was opened read-only and is unchanged.

## Tables

| file | rows | grain | key |
|---|---|---|---|
| `cells.csv` | {counts['cells']} | one grid cell | `(arm, cell_id)` |
| `episodes.parquet` | {counts['episodes']} | one episode | `(arm, cell_id, episode_id)` |
| `rounds.parquet` | {counts['rounds']} | one round of one episode | `(arm, cell_id, episode_id, round_index)` |
| `interactions.parquet` | {counts['interactions']} | one agent decision | `(arm, cell_id, episode_id, interaction_index)` |
| `source_manifest.csv` | {counts['source_files_read']} | one source file read | `path` |

`aggregation_report.json` carries the counts, the per-arm/r/b breakdowns, every
integrity check, and the full list of directories excluded from the sweep.

## Why `arm` is part of every key

`cell_id` is only unique *within* an arm — `cell-0007` exists in all four arms
and means something different in each. The same is true of `episode_id`. Always
group by `arm` first, or join on the composite key.

## How the design columns were derived

| column | source | note |
|---|---|---|
| `r` | basename of `game.options.task_dataset_dir`, `_r<NN>` | support redundancy of the task dataset |
| `b` | `control.options.intervention_budget` | forced to `0` when `control.mechanism == "none"` |
| `b_fraction` | `b / N` | `N` read from the round record |
| `q` | `game.options.social_group_size` | the social slot; **not** `r` |
| `message_mode`, `target_mode` | `control.options.*` | `"none"` for the uncontrolled arm |
| `controller_target` | round record `controller_target` | the *actual* target — differs from `correct_answer` in the adversarial arm |
| `controller_fact_id` | round record `controller_episode_fact_id` | populated only when the message mode injects a fact |
| `repetition` | trailing index of `episode_id` | `cell-0007-0001` → repetition 1 |
| `episode_seed` | round record `seed` | identical to the unsharded run's seed |

`r` and `q` are the pair most easily confused. `social_group_size` is **1 in
every cell of this study** — it is `q`. `r` varies across `_r01/_r03/_r06/_r12`
and is cross-checked against each dataset's `manifest.json`
(`config.support_redundancy`); a disagreement is a hard failure, not a warning.

`b` is cross-checked against `intervention_budget` in every round record, and
`q` against `social_group_size`, so a config that disagrees with what actually
ran cannot pass silently.

## Per-episode summary columns

Derived in `episodes.parquet` from the round records only:

- `truth_vote_share_{{initial,final,delta,mean}}` — initial is round 0's
  `_before` value, final is the last round's after-value.
- `kappa_{{initial,final,delta,mean}}` — `mean_supporting_fact_coverage`.
- `phi_{{initial,final,delta,mean}}` — `full_proof_agent_share`.
- `truth_share_k0_final`, `truth_share_k1_final`, `truth_share_k2_final` —
  final truth share by knowledge stratum (`k2` is null when that stratum is
  empty, which is common).
- `*_total` — episode sums of peer/controller fact exposures, new facts, and
  controlled update/adoption counts.
- `controlled_target_adoption_rate_mean` — mean over rounds, skipping the
  rounds where no agent was controlled (the field is null there).
- `first_round_phi_positive` — first `round_index` with `phi > 0`, else null.
- `first_round_truth_majority` — first `round_index` with
  `truth_vote_share > 0.5`, else null.

## Excluded directories

Only directories matching `^shard_\\d+$` were read. Everything else under an arm
was skipped and is listed here:

{excluded_lines}

## Loading

```python
import pandas as pd

rounds = pd.read_parquet("rounds.parquet")
episodes = pd.read_parquet("episodes.parquet")
cells = pd.read_csv("cells.csv")

# headline contrast: B vs C truth trajectory at matched r and b
curve = (rounds
         .query("arm in ['b_social_control', 'c_epistemic_control']")
         .groupby(["arm", "r", "b", "round_index"])["truth_vote_share"]
         .mean()
         .unstack("arm"))
```

List-valued round fields (population states, sensor samples, occupation counts)
are stored JSON-encoded in `<field>_json` columns so the parquet schema stays
flat; `json.loads` them when you need the vectors.
"""

    def make_zip(self, zip_path: Path) -> Path:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()
        root_name = self.output_dir.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(self.output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(root_name) / path.relative_to(self.output_dir))
        return zip_path


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result-root", type=Path, required=True,
                        help="study root containing one directory per arm")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="analysis folder to create (default: <result-root>_analysis)")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="arm directory names, in order (default: autodetect)")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="repo root, used to cross-check r against dataset manifests "
                             "(default: inferred from this script's location)")
    parser.add_argument("--expect-cells", type=int, default=None)
    parser.add_argument("--expect-episodes", type=int, default=None)
    parser.add_argument("--expect-rounds", type=int, default=None)
    parser.add_argument("--zip", action="store_true", help="package the output folder")
    parser.add_argument("--zip-path", type=Path, default=None,
                        help="explicit zip path (default: <output-dir>.zip)")
    parser.add_argument("--write-on-failure", action="store_true",
                        help="write the tables and report even when checks fail")
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = args.result_root.resolve().parent / (
            args.result_root.resolve().name + "_analysis"
        )
    if args.repo_root is None:
        inferred = Path(__file__).resolve().parents[2]
        args.repo_root = inferred if (inferred / "src").is_dir() else None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aggregator = ShardAggregator(args)
    aggregator.guard_output_dir()

    print(f"[read ] {aggregator.result_root}")
    tables = aggregator.run()
    report = tables["report"]

    counts = report["counts"]
    print(
        f"[count] cells={counts['cells']} episodes={counts['episodes']} "
        f"rounds={counts['rounds']} interactions={counts['interactions']}"
    )
    for item in report["excluded_directories"]:
        print(f"[skip ] {item['relative_path']}: {item['reason']}")
    for message in report["warnings"]:
        print(f"[warn ] {message}")

    if report["status"] == "FAIL" and not args.write_on_failure:
        print(f"[FAIL ] {len(report['violations'])} violation(s):", file=sys.stderr)
        for message in report["violations"]:
            print(f"         - {message}", file=sys.stderr)
        print("        nothing written; rerun with --write-on-failure to inspect",
              file=sys.stderr)
        return 1

    written = aggregator.write(tables)
    for name, path in sorted(written.items()):
        print(f"[write] {path}  ({path.stat().st_size:,} bytes)")

    if args.zip:
        zip_path = args.zip_path or aggregator.output_dir.with_suffix(".zip")
        archive = aggregator.make_zip(zip_path.resolve())
        print(f"[zip  ] {archive}  ({archive.stat().st_size:,} bytes)")

    if report["status"] == "FAIL":
        print(f"[FAIL ] {len(report['violations'])} violation(s) recorded in the report",
              file=sys.stderr)
        for message in report["violations"]:
            print(f"         - {message}", file=sys.stderr)
        return 1

    print("[ok   ] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
