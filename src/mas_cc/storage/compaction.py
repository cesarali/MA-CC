"""Explicit, path-contained compaction of legacy full experiment runs."""

from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from mas_cc.config import parse_run_config
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.metrics.aggregate import read_episode_frame

from .checkpoints import canonical_hash
from .scientific import (
    ScientificIdentity,
    compact_imitation_event,
    empty_compact_row,
    episode_shard_path,
    merge_cell_scientific_tables,
    merge_episode_artifacts,
    prompt_definition_hash,
    validate_cell_artifact,
    write_completed_episode,
)

RAW_BASENAMES = frozenset(
    {
        "trajectory.jsonl",
        "events.jsonl",
        "experiment.log",
        "budget_events.jsonl",
        "usage_cost.jsonl",
        "api_call_status.jsonl",
        "streaming.csv",
        "local_metrics.csv",
        "checkpoint_manifest.json",
        "comet_summary.json",
        "audit_traces.jsonl",
        "prompt_block_traces.jsonl",
        "final.csv",
        "success_rate.csv",
        "production_probability.csv",
    }
)

ANALYSIS_INTERMEDIATES = frozenset(
    {
        "event_metrics.csv",
        "information_nulls.csv",
        "option_share_trajectories.csv",
        "order_parameter_trajectories.csv",
        "episode_summaries.csv",
        "cell_summaries.csv",
        "controller_diagnostics.csv",
    }
)

SUMMARY_INTERMEDIATES = frozenset({"experiment_summary.json", "grid_summary.json"})


def _counts(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files)}


def _inside(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing path outside run directory: {resolved}") from exc
    return resolved


def _cell_directories(root: Path) -> list[Path]:
    cells = root / "cells"
    if cells.is_dir():
        return [path for path in sorted(cells.iterdir()) if path.is_dir()]
    return [root]


def _resolved_config(cell_dir: Path, root: Path) -> dict[str, Any]:
    for path in (
        cell_dir / "resolved_config.yaml",
        root / "resolved_config.yaml",
        root / "resolved_base_config.yaml",
    ):
        if path.is_file():
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _prompt_hash(config: Mapping[str, Any]) -> str:
    try:
        return prompt_definition_hash(parse_run_config(config))
    except ConfigurationError:
        # Very old resolved configs can predate fields the current parser
        # requires. Preserve a deterministic fingerprint for those runs while
        # current configs use the exact live-orchestrator definition hash.
        pass
    prompt = config.get("prompt", {})
    definitions = {}
    if isinstance(prompt, Mapping):
        if prompt.get("definition_hash"):
            definitions[str(prompt.get("prompt_family", "prompt"))] = str(
                prompt["definition_hash"]
            )
        else:
            definitions["configured_prompt"] = canonical_hash(dict(prompt))
    return canonical_hash(definitions)


def _identity(
    root: Path,
    cell_dir: Path,
    episode_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> ScientificIdentity:
    episode_id = episode_dir.name
    seed = int(manifest.get("seed", 0))
    per_episode = deepcopy(dict(config))
    execution = per_episode.setdefault("execution", {})
    if isinstance(execution, dict):
        execution["seed"] = seed
    game = config.get("game", {}) if isinstance(config.get("game"), Mapping) else {}
    control = config.get("control", {}) if isinstance(config.get("control"), Mapping) else {}
    root_manifest = {}
    if (root / "manifest.json").is_file():
        root_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    run_id = str(root_manifest.get("run_id") or root.name)
    cell_id = cell_dir.name if cell_dir.parent.name == "cells" else "run"
    return ScientificIdentity(
        run_id=run_id,
        cell_id=cell_id,
        episode_id=episode_id,
        episode_seed=seed,
        resolved_config_hash=canonical_hash(per_episode),
        prompt_definition_hashes_hash=_prompt_hash(config),
        pricing_snapshot_hash=str(
            manifest.get("pricing_snapshot_hash")
            or manifest.get("price_snapshot_hash")
            or "legacy-unavailable"
        ),
        game_type=str(game.get("type") or root_manifest.get("game_type") or "unknown"),
        dynamics_mode=(
            None
            if not isinstance(game.get("options"), Mapping)
            else game["options"].get("dynamics_mode")
        ),
        control_mechanism=str(control.get("mechanism") or "none"),
        task_id=(
            None
            if not isinstance(game.get("options"), Mapping)
            else game["options"].get("task_id")
        ),
    )


def _metric_payloads(episode_dir: Path) -> dict[int, tuple[str, str]]:
    frame = read_episode_frame(episode_dir)
    if frame is None:
        return {}
    payloads = {}
    for position, round_index in enumerate(frame.rounds):
        population = {
            name: values[position]
            for name, values in frame.population.items()
            if position < len(values)
        }
        options = {
            name: {
                series: values[position]
                for series, values in by_series.items()
                if position < len(values)
            }
            for name, by_series in frame.options.items()
        }
        payloads[int(round_index)] = (
            json.dumps(population, sort_keys=True),
            json.dumps(options, sort_keys=False),
        )
    return payloads


def _compact_episode_rows(
    episode_dir: Path, identity: ScientificIdentity
) -> list[dict[str, Any]]:
    trajectory = episode_dir / "trajectory.jsonl"
    metrics = _metric_payloads(episode_dir)
    final = read_episode_frame(episode_dir)
    final_json = json.dumps({} if final is None else dict(final.final), sort_keys=True)
    if not trajectory.is_file():
        rows = []
        for round_index, (population_json, option_json) in metrics.items():
            row = empty_compact_row(identity, round_index)
            row["population_metrics_json"] = population_json
            row["option_metrics_json"] = option_json
            row["final_metrics_json"] = final_json
            rows.append(row)
        if not rows:
            raise ValueError(
                f"completed episode has neither trajectory nor streaming metrics: {episode_dir}"
            )
        return rows
    rows = []
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        event = payload.get("event")
        if not isinstance(event, Mapping):
            continue
        row = compact_imitation_event(event, identity)
        population_json, option_json = metrics.get(
            int(row["interaction_index"]), ("{}", "{}")
        )
        row["population_metrics_json"] = population_json
        row["option_metrics_json"] = option_json
        row["final_metrics_json"] = final_json
        rows.append(row)
    if not rows:
        raise ValueError(f"trajectory contains no scientific imitation events: {trajectory}")
    return rows


def _episodes(cell_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    directory = cell_dir / "data" / "episodes"
    if not directory.is_dir():
        return []
    found = []
    for episode_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        manifest_path = episode_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"episode has no terminal manifest: {episode_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(
                f"refusing partial/corrupt cell: {episode_dir.name} is {manifest.get('status')!r}"
            )
        found.append((episode_dir, manifest))
    return found


def _render_legacy_prompt_sample(cell_dir: Path, episode_ids: list[str], count: int) -> None:
    if count <= 0:
        return
    for episode_id in sorted(episode_ids):
        prompts = sorted((cell_dir / "data" / "episodes" / episode_id / "prompts").glob("*.md"))
        if not prompts:
            continue
        selected = prompts[:1]
        if count > 1 and len(prompts) > 1:
            middle = prompts[1 : max(1, count - 1)]
            selected.extend(middle)
            selected.append(prompts[-1])
        selected = selected[:count]
        sections = ["# Prompt examples", "", f"Selected from `{episode_id}`.", ""]
        for index, path in enumerate(selected, start=1):
            sections.extend([f"## Example {index}", "", path.read_text(encoding="utf-8").rstrip(), ""])
        (cell_dir / "prompt_examples.md").write_text(
            "\n".join(sections).rstrip() + "\n", encoding="utf-8"
        )
        return


def _delete_raw(root: Path) -> list[str]:
    removed = []
    for path in sorted(root.rglob("*"), reverse=True):
        _inside(root, path)
        relative = path.relative_to(root)
        parts = relative.parts
        in_episode_tree = any(
            parts[index : index + 2] == ("data", "episodes")
            for index in range(max(0, len(parts) - 1))
        )
        in_analysis_tree = any("analysis" in part for part in parts[:-1])
        if path.is_file() and (
            (in_episode_tree and path.name in RAW_BASENAMES)
            or (in_analysis_tree and path.name in ANALYSIS_INTERMEDIATES)
            or (path.parent == root and path.name in SUMMARY_INTERMEDIATES)
            or (
                in_episode_tree
                and path.name in {"manifest.json", "checkpoint.json"}
            )
            or (in_episode_tree and "prompts" in parts and path.suffix == ".md")
        ):
            path.unlink()
            removed.append(str(path.relative_to(root)))
        elif path.is_dir() and not any(path.iterdir()) and path != root:
            path.rmdir()
    return removed


def compact_run_directory(
    run_dir: str | Path,
    *,
    profile: str = "results_only",
    delete_raw: bool = False,
    archive: bool = False,
) -> dict[str, Any]:
    """Preview or compact one exact run directory; never follows paths above it."""

    if profile != "results_only":
        raise ValueError("the compactor currently supports only profile results_only")
    if archive and not delete_raw:
        raise ValueError("--archive requires --delete-raw in the first compactor release")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    before = _counts(root)
    cells = _cell_directories(root)
    planned: dict[str, int] = {}
    cell_inputs: list[tuple[Path, dict[str, Any], list[tuple[Path, dict[str, Any]]]]] = []
    for cell_dir in cells:
        config = _resolved_config(cell_dir, root)
        if (
            (cell_dir / "scientific_events.parquet").is_file()
            and (cell_dir / "cell_complete.json").is_file()
        ):
            validate_cell_artifact(cell_dir)
            continue
        episodes = _episodes(cell_dir)
        if not episodes:
            raise ValueError(f"no completed episodes under {cell_dir}")
        # Parsing every row makes preview a real validation pass without writes.
        for episode_dir, manifest in episodes:
            identity = _identity(root, cell_dir, episode_dir, manifest, config)
            _compact_episode_rows(episode_dir, identity)
        planned[str(cell_dir.relative_to(root) or Path("."))] = len(episodes)
        cell_inputs.append((cell_dir, config, episodes))

    if not delete_raw:
        return {
            "run_dir": str(root),
            "profile": profile,
            "dry_run": True,
            "before": before,
            "after": before,
            "cells": planned,
            "removed": [],
        }

    for cell_dir, config, episodes in cell_inputs:
        identities = []
        for episode_dir, manifest in episodes:
            identity = _identity(root, cell_dir, episode_dir, manifest, config)
            rows = _compact_episode_rows(episode_dir, identity)
            write_completed_episode(
                episode_shard_path(cell_dir, identity.episode_id),
                rows,
                identity,
                termination_reason=manifest.get("termination_reason"),
                started_at=str(manifest.get("started_at") or "legacy-unknown"),
            )
            identities.append(identity)
        logging_options = config.get("logging", {}).get("options", {}) if isinstance(config.get("logging"), Mapping) else {}
        prompt_options = logging_options.get("prompt_examples", {}) if isinstance(logging_options, Mapping) else {}
        count = int(prompt_options.get("count", 2)) if isinstance(prompt_options, Mapping) else 2
        _render_legacy_prompt_sample(
            cell_dir, [identity.episode_id for identity in identities], count
        )
        merge_episode_artifacts(cell_dir, identities, remove_shards=True)
    merge_cell_scientific_tables(root)
    removed = _delete_raw(root)
    archive_path = None
    if archive:
        archive_path = root.with_name(root.name + "-results-only.zip")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.name.endswith(":Zone.Identifier"):
                    bundle.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    return {
        "run_dir": str(root),
        "profile": profile,
        "dry_run": False,
        "before": before,
        "after": _counts(root),
        "cells": planned,
        "removed": removed,
        "archive": None if archive_path is None else str(archive_path),
    }


__all__ = ["ANALYSIS_INTERMEDIATES", "RAW_BASENAMES", "compact_run_directory"]
