"""Provider-completion-free study preflight and strict scientific contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games.relational_reasoning.data import load_relational_task

from .manifest import StudySpec, discover_study


FALSE_TAKEOVER_CONTRACT = "relational_false_takeover_v1"
PERSISTENCE_EXPLORATORY_CONTRACT = "relational_persistence_exploratory_v1"
PERSISTENCE_REFINEMENT_CONTRACT = "relational_persistence_refinement_v1"
PERSISTENCE_TRUTH_REFINEMENT_CONTRACT = "relational_persistence_truth_refinement_v1"
PERSISTENCE_Q1_L2_FALSE_CONTRACT = "relational_persistence_q1_l2_false_v1"
PERSISTENCE_Q1_L2_TRUTH_CONTRACT = "relational_persistence_q1_l2_truth_v1"
PERSISTENCE_HIGH_STATISTICS_FALSE_CONTRACT = (
    "relational_persistence_high_statistics_false_v1"
)
PERSISTENCE_HIGH_STATISTICS_TRUTH_CONTRACT = (
    "relational_persistence_high_statistics_truth_v1"
)


def _repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


def _dataset_path(spec: StudySpec, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (_repo_root(spec.config_dir) / path).resolve()
    )


def _task_fingerprint(path: Path) -> str:
    task = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        task, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_persistence_contract(
    spec: StudySpec,
    *,
    contract: str,
    rho_values: list[float],
    repetitions_value: int,
    target_is_truth: bool = False,
    q_values_expected: list[int] | None = None,
    evidence_strategies_expected: list[str] | None = None,
    budget_values: list[int] | None = None,
    depth_value: int = 3,
    redundancy_value: int = 3,
    expected_truth: str = "NORTH",
    expected_target: str | None = None,
) -> dict[str, Any]:
    """Validate an exact finite-persistence study design."""

    errors: list[str] = []
    expected_q = [2] if q_values_expected is None else q_values_expected
    expected_strategies = (
        ["strategic"]
        if evidence_strategies_expected is None
        else evidence_strategies_expected
    )
    expected_budgets = [3, 6, 9, 12] if budget_values is None else budget_values
    _require(len(spec.configs) == 1, "study must list exactly one config", errors)
    cells: list[Any] = []
    axes: list[tuple[str, list[Any]]] = []
    if spec.configs:
        source = load_run_config_or_grid(spec.configs[0])
        _require(
            isinstance(source, GridSpec), "experiment must be a grid config", errors
        )
        if isinstance(source, GridSpec):
            axes = [(axis.path, list(axis.values)) for axis in source.axes]
            cells = [cell.config for cell in source.cells]
    expected_axes = []
    if len(expected_q) > 1:
        expected_axes.append(("game.options.social_group_size", expected_q))
    if len(expected_strategies) > 1:
        expected_axes.append(
            (
                "control.options.controller_evidence_strategy",
                expected_strategies,
            )
        )
    expected_axes.extend(
        [
            ("game.options.epistemic_persistence", rho_values),
            ("control.options.intervention_budget", expected_budgets),
        ]
    )
    _require(
        axes == expected_axes,
        (
            f"grid axes must be rho={rho_values} then b={expected_budgets}, got {axes}"
            if len(expected_q) == 1 and len(expected_strategies) == 1
            else f"grid axes must be {expected_axes}, got {axes}"
        ),
        errors,
    )

    populations: set[int] = set()
    rounds: set[int] = set()
    q_values: set[int] = set()
    depths: set[int] = set()
    redundancies: set[int] = set()
    sensors: set[int] = set()
    budgets: set[int] = set()
    persistence: set[float] = set()
    tasks: set[str] = set()
    targets: set[str] = set()
    truths: set[str] = set()
    dispositions: set[str] = set()
    strategies: set[str] = set()
    modes: set[str] = set()
    schedules: set[str] = set()
    betas: set[float] = set()
    thresholds: set[float] = set()
    repetitions: set[int] = set()
    resolved_cells: set[tuple[int, str, float, int]] = set()
    task_audit: dict[str, dict[str, Any]] = {}

    for config in cells:
        game = config.game
        options = game.options
        control_options = config.control.options
        dataset = _dataset_path(spec, options.get("task_dataset_dir"))
        manifest_path = dataset / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid frozen dataset manifest {manifest_path}: {exc}")
            continue
        dataset_config = manifest.get("config", {})
        populations.add(int(game.population_size))
        depths.add(int(dataset_config.get("reasoning_depth", -1)))
        redundancies.add(int(dataset_config.get("support_redundancy", -1)))
        rounds.add(int(options.get("rounds", -1)))
        q_values.add(int(options.get("social_group_size", -1)))
        sensors.add(int(control_options.get("sensor_sample_size", -1)))
        budget = int(control_options.get("intervention_budget", -1))
        rho = float(options.get("epistemic_persistence", -1.0))
        budgets.add(budget)
        persistence.add(rho)
        q = int(options.get("social_group_size", -1))
        strategy = str(control_options.get("controller_evidence_strategy"))
        resolved_cells.add((q, strategy, rho, budget))
        task_id = str(options.get("task_id"))
        tasks.add(task_id)
        dispositions.add(str(options.get("receiver_epistemic_disposition")))
        strategies.add(strategy)
        modes.add(str(control_options.get("message_mode")))
        schedules.add(str(control_options.get("advocacy_schedule")))
        betas.add(float(control_options.get("beta")))
        thresholds.add(float(control_options.get("threshold")))
        repetitions.add(int(config.execution.repetitions))
        try:
            task = load_relational_task(dataset, task_id, population_size=12)
            raw_task = json.loads(Path(task.source_path).read_text(encoding="utf-8"))
            controller = create_control(config.control)
            target = controller.resolved_target_for_task(task, config.execution.seed)
            fact_id = controller.resolve_fact_id(task, config.execution.seed)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot validate task/controller: {exc}")
            continue
        generation = raw_task.get("generation", {})
        redundancies.add(int(generation.get("support_redundancy", -1)))
        targets.add(target)
        truths.add(task.correct_relation)
        _require(
            (target == task.correct_relation) is target_is_truth,
            "controller target truth alignment is incorrect",
            errors,
        )
        _require(
            fact_id in task.facts,
            "strategic evidence must be a frozen true fact",
            errors,
        )
        _require(
            len(task.supporting_fact_ids) == depth_value,
            f"task supporting-fact depth must equal {depth_value}",
            errors,
        )
        _require(
            all(
                sum(
                    fact_id_value in task.known_facts(agent_id)
                    for agent_id in task.agent_ids
                )
                == redundancy_value
                for fact_id_value in task.supporting_fact_ids
            ),
            f"every supporting fact must have redundancy {redundancy_value}",
            errors,
        )
        metadata = config.experiment.metadata
        _require(
            metadata.get("ground_truth") == task.correct_relation,
            "metadata truth is incorrect",
            errors,
        )
        _require(
            metadata.get("controller_target") == target,
            "metadata target is incorrect",
            errors,
        )
        _require(
            metadata.get("controller_target_is_truth") is target_is_truth,
            "metadata target truth alignment is incorrect",
            errors,
        )
        task_audit[strategy] = {
            "task_id": task_id,
            "fingerprint_sha256": _task_fingerprint(Path(task.source_path)),
            "ground_truth": task.correct_relation,
            "controller_target": target,
            "controller_target_is_truth": target_is_truth,
            "controller_evidence_strategy": strategy,
            "evidence_fact_id": fact_id,
            "evidence_fact_relation": task.fact(fact_id).relation,
            "evidence_fact_text": task.fact_text(fact_id),
        }

    _require(
        populations == {12},
        f"population size must be [12], got {sorted(populations)}",
        errors,
    )
    _require(rounds == {30}, f"rounds must be [30], got {sorted(rounds)}", errors)
    _require(
        q_values == set(expected_q),
        f"q values must be {expected_q}, got {sorted(q_values)}",
        errors,
    )
    _require(
        depths == {depth_value},
        f"L values must be [{depth_value}], got {sorted(depths)}",
        errors,
    )
    _require(
        redundancies == {redundancy_value},
        f"support redundancy must be [{redundancy_value}], got {sorted(redundancies)}",
        errors,
    )
    _require(sensors == {6}, f"sensor size must be [6], got {sorted(sensors)}", errors)
    _require(
        persistence == set(rho_values),
        f"rho values are incorrect: {sorted(persistence)}",
        errors,
    )
    _require(
        budgets == set(expected_budgets),
        f"b values are incorrect: {sorted(budgets)}",
        errors,
    )
    _require(
        tasks == {"task_0002"},
        f"task must be task_0002 only, got {sorted(tasks)}",
        errors,
    )
    _require(
        truths == {expected_truth},
        f"truth must be {expected_truth}, got {sorted(truths)}",
        errors,
    )
    target_value = expected_target or (
        expected_truth if target_is_truth else "NORTHWEST"
    )
    _require(
        targets == {target_value},
        f"controller target is incorrect: {sorted(targets)}",
        errors,
    )
    _require(
        dispositions == {"naive"},
        f"receiver must be naive, got {sorted(dispositions)}",
        errors,
    )
    _require(
        strategies == set(expected_strategies),
        f"evidence strategies must be {expected_strategies}, got {sorted(strategies)}",
        errors,
    )
    _require(
        modes == {"recommendation_plus_fact"},
        f"message mode is incorrect: {sorted(modes)}",
        errors,
    )
    _require(
        schedules == {"soft"}, f"schedule must be soft, got {sorted(schedules)}", errors
    )
    _require(betas == {4.0}, f"beta must be [4.0], got {sorted(betas)}", errors)
    _require(
        thresholds == {0.75}, f"theta must be [0.75], got {sorted(thresholds)}", errors
    )
    _require(
        repetitions == {repetitions_value},
        f"repetitions must be [{repetitions_value}], got {sorted(repetitions)}",
        errors,
    )
    _require(
        len(cells)
        == len(expected_q)
        * len(expected_strategies)
        * len(rho_values)
        * len(expected_budgets),
        "resolved cells must total "
        f"{len(expected_q) * len(expected_strategies) * len(rho_values) * len(expected_budgets)}, "
        f"got {len(cells)}",
        errors,
    )
    _require(
        len(resolved_cells)
        == len(expected_q)
        * len(expected_strategies)
        * len(rho_values)
        * len(expected_budgets),
        "q/evidence/rho/b cells must be unique and total "
        f"{len(expected_q) * len(expected_strategies) * len(rho_values) * len(expected_budgets)}, "
        f"got {len(resolved_cells)}",
        errors,
    )
    total_episodes = sum(config.execution.repetitions for config in cells)
    _require(
        total_episodes
        == len(expected_q)
        * len(expected_strategies)
        * len(rho_values)
        * len(expected_budgets)
        * repetitions_value,
        "total episodes must be "
        f"{len(expected_q) * len(expected_strategies) * len(rho_values) * len(expected_budgets) * repetitions_value}, "
        f"got {total_episodes}",
        errors,
    )

    report = {
        "contract": contract,
        "status": "failed" if errors else "permitted",
        "population_size": sorted(populations),
        "rounds": sorted(rounds),
        "q_values": sorted(q_values),
        "L_values": sorted(depths),
        "support_redundancy": sorted(redundancies),
        "sensor_size": sorted(sensors),
        "rho_values": sorted(persistence),
        "b_values": sorted(budgets),
        "target_semantics": ["truth only" if target_is_truth else "false only"]
        if targets
        else [],
        "receiver_dispositions": sorted(dispositions),
        "evidence_strategies": sorted(strategies),
        "message_modes": sorted(modes),
        "beta": sorted(betas),
        "theta": sorted(thresholds),
        "schedule": sorted(schedules),
        "number_of_frozen_tasks": len(tasks),
        "repetitions": sorted(repetitions),
        "structural_regimes": len(resolved_cells),
        "resolved_regimes": [list(value) for value in sorted(resolved_cells)],
        "total_cells": len(cells),
        "total_episodes": total_episodes,
        "matched_revised_theory_applicable": False,
        "tasks": [task_audit[key] for key in sorted(task_audit)],
        "errors": errors,
    }
    if errors:
        raise ValueError("Study preflight contract failed:\n- " + "\n- ".join(errors))
    return report


def _validate_persistence_exploratory_contract(spec: StudySpec) -> dict[str, Any]:
    """Validate Study 09c's exact 12-cell finite-persistence design."""

    return _validate_persistence_contract(
        spec,
        contract=PERSISTENCE_EXPLORATORY_CONTRACT,
        rho_values=[0.6, 0.8, 0.9],
        repetitions_value=1,
    )


def _validate_persistence_refinement_contract(spec: StudySpec) -> dict[str, Any]:
    """Validate Study 09d's exact 20-cell, 200-episode refinement design."""

    return _validate_persistence_contract(
        spec,
        contract=PERSISTENCE_REFINEMENT_CONTRACT,
        rho_values=[0.7, 0.75, 0.8, 0.85, 0.9],
        repetitions_value=10,
    )


def _validate_persistence_truth_refinement_contract(
    spec: StudySpec,
) -> dict[str, Any]:
    """Validate Study 09e's matched truth-aligned refinement design."""

    return _validate_persistence_contract(
        spec,
        contract=PERSISTENCE_TRUTH_REFINEMENT_CONTRACT,
        rho_values=[0.7, 0.75, 0.8, 0.85, 0.9],
        repetitions_value=10,
        target_is_truth=True,
    )


def _validate_persistence_q1_l2_contract(
    spec: StudySpec, *, truth_aligned: bool
) -> dict[str, Any]:
    """Validate the matched q=1, L=2 persistence reference design."""

    return _validate_persistence_contract(
        spec,
        contract=(
            PERSISTENCE_Q1_L2_TRUTH_CONTRACT
            if truth_aligned
            else PERSISTENCE_Q1_L2_FALSE_CONTRACT
        ),
        rho_values=[0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        repetitions_value=10,
        target_is_truth=truth_aligned,
        q_values_expected=[1],
        depth_value=2,
        redundancy_value=4,
        expected_truth="NORTHEAST",
        expected_target="NORTHEAST" if truth_aligned else "NORTH",
    )


def _validate_persistence_high_statistics_contract(
    spec: StudySpec, *, truth_aligned: bool
) -> dict[str, Any]:
    """Validate the focused 24-cell, 360-episode L=3 persistence family."""

    return _validate_persistence_contract(
        spec,
        contract=(
            PERSISTENCE_HIGH_STATISTICS_TRUTH_CONTRACT
            if truth_aligned
            else PERSISTENCE_HIGH_STATISTICS_FALSE_CONTRACT
        ),
        rho_values=[0.8, 0.85],
        repetitions_value=15,
        target_is_truth=truth_aligned,
        q_values_expected=[1, 2],
        evidence_strategies_expected=["strategic"],
        budget_values=[3, 4, 6, 8, 9, 12],
    )


def validate_study_preflight_contract(spec: StudySpec) -> dict[str, Any]:
    """Validate the optional cross-config contract without invoking an LLM."""

    contract = spec.preflight.get("contract")
    if contract is None:
        return {"contract": None, "status": "not_requested"}
    if contract == PERSISTENCE_EXPLORATORY_CONTRACT:
        return _validate_persistence_exploratory_contract(spec)
    if contract == PERSISTENCE_REFINEMENT_CONTRACT:
        return _validate_persistence_refinement_contract(spec)
    if contract == PERSISTENCE_TRUTH_REFINEMENT_CONTRACT:
        return _validate_persistence_truth_refinement_contract(spec)
    if contract == PERSISTENCE_Q1_L2_FALSE_CONTRACT:
        return _validate_persistence_q1_l2_contract(spec, truth_aligned=False)
    if contract == PERSISTENCE_Q1_L2_TRUTH_CONTRACT:
        return _validate_persistence_q1_l2_contract(spec, truth_aligned=True)
    if contract == PERSISTENCE_HIGH_STATISTICS_FALSE_CONTRACT:
        return _validate_persistence_high_statistics_contract(spec, truth_aligned=False)
    if contract == PERSISTENCE_HIGH_STATISTICS_TRUTH_CONTRACT:
        return _validate_persistence_high_statistics_contract(spec, truth_aligned=True)
    if contract != FALSE_TAKEOVER_CONTRACT:
        raise ValueError(f"unsupported study preflight contract {contract!r}")

    errors: list[str] = []
    cells: list[tuple[Path, Any]] = []
    for path in spec.configs:
        source = load_run_config_or_grid(path)
        _require(
            isinstance(source, GridSpec), f"{path.name} must be a grid config", errors
        )
        if isinstance(source, GridSpec):
            cells.extend((path, cell.config) for cell in source.cells)

    populations: set[int] = set()
    q_values: set[int] = set()
    depths: set[int] = set()
    redundancies: set[int] = set()
    sensor_sizes: set[int] = set()
    budgets: set[int] = set()
    targets: set[str] = set()
    truths: set[str] = set()
    dispositions: set[str] = set()
    strategies: set[str] = set()
    modes: set[str] = set()
    schedules: set[str] = set()
    betas: set[float] = set()
    thresholds: set[float] = set()
    task_keys: set[tuple[str, str]] = set()
    regimes: set[tuple[int, int]] = set()
    resolved_cells: set[tuple[str, int, int]] = set()
    repetitions: set[int] = set()
    dataset_manifests: dict[Path, Mapping[str, Any]] = {}
    task_audit: dict[tuple[str, str], dict[str, Any]] = {}

    for config_path, config in cells:
        game = config.game
        options = game.options
        control_options = config.control.options
        dataset = _dataset_path(spec, options.get("task_dataset_dir"))
        manifest_path = dataset / "manifest.json"
        if dataset not in dataset_manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid frozen dataset manifest {manifest_path}: {exc}")
                manifest = {}
            dataset_manifests[dataset] = manifest
            config_block = (
                manifest.get("config", {}) if isinstance(manifest, Mapping) else {}
            )
            _require(
                manifest.get("num_tasks") == 2,
                "frozen dataset must contain exactly 2 tasks",
                errors,
            )
            _require(
                config_block.get("population_size") == 12,
                "dataset N must equal 12",
                errors,
            )
            _require(
                config_block.get("reasoning_depth") == 3,
                "dataset L must equal 3",
                errors,
            )
            _require(
                config_block.get("support_redundancy") == 3,
                "dataset r must equal 3",
                errors,
            )
            _require(
                config_block.get("num_options") == 3, "dataset K must equal 3", errors
            )
            _require(
                config_block.get("no_single_agent_solution") is True,
                "dataset must forbid a single-agent solution",
                errors,
            )
            fingerprints = manifest.get("task_fingerprints_sha256", {})
            for filename in manifest.get("task_files", []):
                task_path = dataset / str(filename)
                _require(
                    task_path.is_file(), f"missing frozen task {task_path}", errors
                )
                if task_path.is_file():
                    _require(
                        fingerprints.get(filename) == _task_fingerprint(task_path),
                        f"fingerprint mismatch for {task_path}",
                        errors,
                    )

        task_id = str(options.get("task_id"))
        task_key = (str(dataset), task_id)
        task_keys.add(task_key)
        try:
            task = load_relational_task(
                dataset, task_id, population_size=game.population_size
            )
            raw_task = json.loads(Path(task.source_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot validate task {task_id}: {exc}")
            continue

        generation = raw_task.get("generation", {})
        support_ids = set(task.supporting_fact_ids)
        support_counts = {
            fact_id: sum(
                fact_id in task.known_facts(agent_id) for agent_id in task.agent_ids
            )
            for fact_id in support_ids
        }
        _require(
            set().union(
                *(set(task.known_facts(agent_id)) for agent_id in task.agent_ids)
            )
            >= support_ids,
            f"{task_id} population union does not contain the complete proof",
            errors,
        )
        _require(
            all(count == 3 for count in support_counts.values()),
            f"{task_id} supporting facts do not all have redundancy 3: {support_counts}",
            errors,
        )
        _require(
            all(
                not support_ids.issubset(task.known_facts(agent_id))
                for agent_id in task.agent_ids
            ),
            f"{task_id} gives an individual agent the complete proof",
            errors,
        )
        populations.add(int(game.population_size))
        q = int(options.get("social_group_size", -1))
        q_values.add(q)
        depths.add(int(task.reasoning_depth))
        redundancies.add(int(generation.get("support_redundancy", -1)))
        sensor = int(control_options.get("sensor_sample_size", -1))
        budget = int(control_options.get("intervention_budget", -1))
        sensor_sizes.add(sensor)
        budgets.add(budget)
        regimes.add((q, budget))
        resolved_cells.add((task_id, q, budget))
        repetitions.add(int(config.execution.repetitions))
        dispositions.add(str(options.get("receiver_epistemic_disposition")))
        strategies.add(str(control_options.get("controller_evidence_strategy")))
        modes.add(str(control_options.get("message_mode")))
        schedules.add(str(control_options.get("advocacy_schedule")))
        betas.add(float(control_options.get("beta")))
        thresholds.add(float(control_options.get("threshold")))

        target_value = control_options.get("target")
        _require(
            isinstance(target_value, str)
            and target_value not in {"correct", "random_incorrect"},
            f"{task_id} controller target must be a literal semantic relation",
            errors,
        )
        controller = create_control(config.control)
        try:
            target = controller.resolved_target_for_task(task, config.execution.seed)
            fact_id = controller.resolve_fact_id(task, config.execution.seed)
        except ValueError as exc:
            errors.append(f"{task_id} target/evidence validation failed: {exc}")
            continue
        targets.add(target)
        truths.add(task.correct_relation)
        _require(
            target in task.semantic_answers,
            f"{task_id} target {target} is not an answer option",
            errors,
        )
        _require(
            target != task.correct_relation,
            f"{task_id} controller target is truth",
            errors,
        )
        _require(
            fact_id in task.facts,
            f"{task_id} strategic evidence is not a frozen task fact",
            errors,
        )
        metadata = config.experiment.metadata
        _require(
            metadata.get("ground_truth") == task.correct_relation,
            f"{task_id} metadata ground_truth is incorrect",
            errors,
        )
        _require(
            metadata.get("controller_target") == target,
            f"{task_id} metadata controller_target is incorrect",
            errors,
        )
        _require(
            metadata.get("controller_target_is_truth") is False,
            f"{task_id} must record controller_target_is_truth=false",
            errors,
        )
        task_audit[task_key] = {
            "task_id": task_id,
            "fingerprint_sha256": _task_fingerprint(Path(task.source_path)),
            "ground_truth": task.correct_relation,
            "controller_target": target,
            "controller_target_is_truth": False,
            "strategic_fact_id": fact_id,
            "strategic_fact_relation": task.fact(fact_id).relation,
            "strategic_fact_text": task.fact_text(fact_id),
        }

    _require(
        populations == {12},
        f"population size must be [12], got {sorted(populations)}",
        errors,
    )
    _require(
        q_values == {2, 3}, f"q values must be [2, 3], got {sorted(q_values)}", errors
    )
    _require(depths == {3}, f"L values must be [3], got {sorted(depths)}", errors)
    _require(
        redundancies == {3},
        f"support redundancy must be [3], got {sorted(redundancies)}",
        errors,
    )
    _require(
        sensor_sizes == {6},
        f"sensor size must be [6], got {sorted(sensor_sizes)}",
        errors,
    )
    _require(
        budgets == {9, 12}, f"b values must be [9, 12], got {sorted(budgets)}", errors
    )
    _require(
        dispositions == {"naive"},
        f"receiver dispositions must be [naive], got {sorted(dispositions)}",
        errors,
    )
    _require(
        strategies == {"strategic"},
        f"evidence strategies must be [strategic], got {sorted(strategies)}",
        errors,
    )
    _require(
        modes == {"recommendation_plus_fact"},
        f"message modes must be [recommendation_plus_fact], got {sorted(modes)}",
        errors,
    )
    _require(
        schedules == {"soft"},
        f"controller schedule must be [soft], got {sorted(schedules)}",
        errors,
    )
    _require(betas == {4.0}, f"beta must be [4.0], got {sorted(betas)}", errors)
    _require(
        thresholds == {0.75}, f"theta must be [0.75], got {sorted(thresholds)}", errors
    )
    _require(
        len(task_keys) == 2,
        f"number of frozen tasks must be 2, got {len(task_keys)}",
        errors,
    )
    _require(
        repetitions == {1},
        f"repetitions must be [1], got {sorted(repetitions)}",
        errors,
    )
    _require(
        regimes == {(2, 9), (2, 12), (3, 9), (3, 12)},
        f"structural regimes are incorrect: {sorted(regimes)}",
        errors,
    )
    _require(len(cells) == 8, f"resolved cells must total 8, got {len(cells)}", errors)
    _require(
        len(resolved_cells) == 8,
        f"resolved task/q/b cells must be unique and total 8, got {len(resolved_cells)}",
        errors,
    )
    total_episodes = sum(config.execution.repetitions for _, config in cells)
    _require(
        total_episodes == 8, f"total episodes must be 8, got {total_episodes}", errors
    )

    report = {
        "contract": contract,
        "status": "failed" if errors else "permitted",
        "population_size": sorted(populations),
        "q_values": sorted(q_values),
        "L_values": sorted(depths),
        "support_redundancy": sorted(redundancies),
        "sensor_size": sorted(sensor_sizes),
        "b_values": sorted(budgets),
        "target_semantics": ["false only"] if not errors or targets else [],
        "receiver_dispositions": sorted(dispositions),
        "evidence_strategies": sorted(strategies),
        "message_modes": sorted(modes),
        "beta": sorted(betas),
        "theta": sorted(thresholds),
        "schedule": sorted(schedules),
        "number_of_frozen_tasks": len(task_keys),
        "repetitions": sorted(repetitions),
        "structural_regimes": len(regimes),
        "resolved_regimes": [list(value) for value in sorted(regimes)],
        "total_cells": len(cells),
        "total_episodes": total_episodes,
        "matched_revised_theory_applicable": False,
        "tasks": sorted(task_audit.values(), key=lambda row: row["task_id"]),
        "errors": errors,
    }
    if errors:
        raise ValueError("Study preflight contract failed:\n- " + "\n- ".join(errors))
    return report


@dataclass(frozen=True, slots=True)
class StudyPreflightResult:
    config_dir: Path
    output_dir: Path
    design: Mapping[str, Any]
    estimates: tuple[Mapping[str, Any], ...]


def run_study_preflight(
    config_dir: str | Path | StudySpec, output_dir: str | Path
) -> StudyPreflightResult:
    """Validate a whole study and write one combined no-completion-call report."""

    from mas_cc.cli.experiment import run_experiment_preflight

    spec = (
        config_dir if isinstance(config_dir, StudySpec) else discover_study(config_dir)
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    design = validate_study_preflight_contract(spec)
    estimates: list[Mapping[str, Any]] = []
    for index, path in enumerate(spec.configs):
        estimate = run_experiment_preflight(path, destination / f"config-{index:04d}")
        payload = estimate.to_dict()
        estimates.append(payload)
        if estimate.launch_status != "permitted":
            raise ValueError(f"experiment preflight denied {path.name}")

    calls = {
        key: sum(int(row["total_provider_requests"][key]) for row in estimates)
        for key in ("lower", "expected", "conservative")
    }
    design = {**design, "provider_calls": calls}
    (destination / "design_validation.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Study preflight",
        "",
        f"- Status: **{design['status'].upper()}**",
        f"- Population size: {design['population_size']}",
        f"- q values: {design['q_values']}",
        f"- L values: {design['L_values']}",
        f"- Support redundancy: {design['support_redundancy']}",
        f"- Sensor size q_c: {design['sensor_size']}",
        *([f"- rho values: {design['rho_values']}"] if "rho_values" in design else []),
        f"- b values: {design['b_values']}",
        f"- Target semantics: {design['target_semantics']}",
        f"- Receiver dispositions: {design['receiver_dispositions']}",
        f"- Evidence strategies: {design['evidence_strategies']}",
        f"- Message modes: {design['message_modes']}",
        f"- beta: {design['beta']}",
        f"- theta: {design['theta']}",
        f"- Schedule: {design['schedule']}",
        f"- Frozen tasks: {design.get('number_of_frozen_tasks', design.get('frozen_tasks'))}",
        f"- Repetitions: {design['repetitions']}",
        f"- Structural regimes: {design['structural_regimes']} {design['resolved_regimes']}",
        f"- Total episodes: {design['total_episodes']}",
        f"- Nominal provider calls: {calls['lower']}",
        f"- Expected provider calls: {calls['expected']}",
        f"- Conservative provider calls: {calls['conservative']}",
        "- Matched revised q=1 theory applicable: false",
        "",
        "## Frozen task audit",
        "",
    ]
    for task in design["tasks"]:
        fact_id = task.get("strategic_fact_id", task.get("evidence_fact_id"))
        fact_relation = task.get(
            "strategic_fact_relation", task.get("evidence_fact_relation")
        )
        lines.append(
            f"- `{task['task_id']}`: truth `{task['ground_truth']}`, false target "
            f"`{task['controller_target']}`, true strategic fact `{fact_id}` "
            f"(`{fact_relation}`), fingerprint `{task['fingerprint_sha256']}`"
        )
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return StudyPreflightResult(spec.config_dir, destination, design, tuple(estimates))


__all__ = [
    "FALSE_TAKEOVER_CONTRACT",
    "PERSISTENCE_EXPLORATORY_CONTRACT",
    "StudyPreflightResult",
    "run_study_preflight",
    "validate_study_preflight_contract",
]
