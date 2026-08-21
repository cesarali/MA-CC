#!/usr/bin/env python3
"""Preflight and minimal analysis for relational population Study 05."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mas_cc.config import GridSpec, load_run_config_or_grid
from mas_cc.control import create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP

DEFAULT_CONFIGS = (
    Path("configs/runs/relational_reasoning/population_study_05/relational_population_study05_state_matching_task0001.yaml"),
    Path("configs/runs/relational_reasoning/population_study_05/relational_population_study05_state_matching_task0002.yaml"),
)
EXPECTED_N_Z = (6, 9, 12)


def _target_and_votes(cell: Any) -> tuple[str, tuple[str, ...]]:
    game = create_game(cell.config.game)
    task = game.load_task(cell.config.game)
    raw_target = cell.config.control.options["target"]
    target = task.semantic_answers[raw_target] if isinstance(raw_target, int) else str(raw_target)
    votes = tuple(cell.config.game.options["initialization"]["initial_votes"])
    return target, votes


def _knowledge_classes(state: Any) -> tuple[int, ...]:
    supporting = set(state.supporting_fact_ids)
    return tuple(len(set(agent.initial_fact_ids) & supporting) for agent in state.agents)


def _state_signature(state: Any) -> dict[str, Any]:
    return {
        "task": state.task,
        "agent_ids": [str(agent.agent_id) for agent in state.agents],
        "initial_fact_ids": [list(agent.initial_fact_ids) for agent in state.agents],
        "initial_votes": list(state.initial_votes),
    }


def _stratified_order(task: Any) -> tuple[str, ...]:
    """Stable hash order within each class, then round-robin across classes."""

    supporting = set(task.supporting_fact_ids)
    strata: dict[int, list[str]] = defaultdict(list)
    for agent_id in task.agent_ids:
        k = len(set(task.known_facts(agent_id)) & supporting)
        strata[k].append(agent_id)
    for k, agents in strata.items():
        agents.sort(
            key=lambda agent_id: hashlib.sha256(
                f"{task.task_id}:{k}:{agent_id}".encode("utf-8")
            ).hexdigest()
        )
    active = sorted(k for k, agents in strata.items() if agents)
    result: list[str] = []
    while any(strata[k] for k in active):
        for k in active:
            if strata[k]:
                result.append(strata[k].pop(0))
    return tuple(result)


def preflight(config_paths: Iterable[Path], output: Path | None) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    total_episodes = 0
    for path in config_paths:
        loaded = load_run_config_or_grid(path)
        if not isinstance(loaded, GridSpec):
            raise ValueError(f"{path} is not a grid config")
        if loaded.base.game.population_size != 24:
            raise ValueError(f"{path}: N must be 24")
        if loaded.base.game.horizon != 1 or loaded.base.game.options.get("rounds") != 1:
            raise ValueError(f"{path}: run must contain exactly one population round")
        if not loaded.base.experiment.metadata.get("common_random_numbers_across_grid"):
            raise ValueError(f"{path}: common random numbers must be enabled")
        if len(loaded.cells) != 6:
            raise ValueError(f"{path}: expected exactly six grid cells")

        by_n: dict[int, dict[str, Any]] = defaultdict(dict)
        knowledge_signatures: set[str] = set()
        kappa_phi: set[tuple[float, float]] = set()
        nested_sets: dict[int, set[str]] = {}
        target_seen: str | None = None
        for cell in loaded.cells:
            game = create_game(cell.config.game)
            task = game.load_task(cell.config.game)
            target, votes = _target_and_votes(cell)
            target_seen = target
            if target not in task.semantic_answers or target in task.option_labels:
                raise ValueError(f"{path}: target is not a stable semantic relation")
            if target == task.correct_relation:
                raise ValueError(f"{path}: Study 05 target must be the pinned wrong target")
            n_z = votes.count(target)
            if n_z not in EXPECTED_N_Z:
                raise ValueError(f"{path} {cell.cell_id}: n_Z(0)={n_z}, expected 6/9/12")
            schedule = str(cell.config.control.options["advocacy_schedule"])
            control = create_control(cell.config.control)
            expected = (ADVOCATE_TARGET, 1.0) if schedule == "always" else (NO_OP, 0.0)
            for sensed_share in (0.0, 0.5, 1.0):
                if control.select_action(sensed_share, None) != expected:
                    raise ValueError(f"{path} {cell.cell_id}: deterministic action check failed")

            state = game.initialize(cell.config.game, loaded.base.execution.seed)
            classes = _knowledge_classes(state)
            facts = [list(agent.initial_fact_ids) for agent in state.agents]
            knowledge_signatures.add(json.dumps(facts, sort_keys=True))
            kappa = sum(classes) / (len(classes) * len(state.supporting_fact_ids))
            phi = sum(k == len(state.supporting_fact_ids) for k in classes) / len(classes)
            kappa_phi.add((kappa, phi))
            signature = _state_signature(state)
            if schedule in by_n[n_z]:
                raise ValueError(f"{path}: duplicate {n_z}/{schedule} cell")
            by_n[n_z][schedule] = signature

            supporters = {
                str(agent.agent_id)
                for agent, vote in zip(state.agents, state.initial_votes, strict=True)
                if vote == target
            }
            nested_sets[n_z] = supporters
            expected_supporters = set(_stratified_order(task)[:n_z])
            if supporters != expected_supporters:
                raise ValueError(f"{path} {cell.cell_id}: Z voters are not the declared stratified prefix")

        if set(by_n) != set(EXPECTED_N_Z):
            raise ValueError(f"{path}: missing n_Z conditions")
        if len(knowledge_signatures) != 1 or len(kappa_phi) != 1:
            raise ValueError(f"{path}: fact allocation, kappa_0, or phi_0 differs across cells")
        for n_z, arms in by_n.items():
            if set(arms) != {"always", "never"} or arms["always"] != arms["never"]:
                raise ValueError(f"{path}: always/never initial states differ at n_Z={n_z}")
        if not nested_sets[6] < nested_sets[9] < nested_sets[12]:
            raise ValueError(f"{path}: target-support sets are not strictly nested")

        episodes = len(loaded.cells) * loaded.base.execution.repetitions
        total_episodes += episodes
        reports.append(
            {
                "config": str(path),
                "task_id": loaded.base.game.options["task_id"],
                "target": target_seen,
                "cells": len(loaded.cells),
                "repetitions": loaded.base.execution.repetitions,
                "episodes": episodes,
                "logical_provider_calls": episodes * 24,
                "kappa_0": next(iter(kappa_phi))[0],
                "phi_0": next(iter(kappa_phi))[1],
                "nested_z_voters": {str(n): sorted(nested_sets[n]) for n in EXPECTED_N_Z},
            }
        )

    if total_episodes != 120:
        raise ValueError(f"combined pilot has {total_episodes} episodes, expected 120")
    result = {
        "status": "passed",
        "configs": reports,
        "total_cells": sum(item["cells"] for item in reports),
        "total_episodes": total_episodes,
        "total_logical_provider_calls": total_episodes * 24,
        "retry_bound_provider_calls": total_episodes * 24 * 2,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return result


def _round_payloads(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in run_dirs:
        paths = sorted(root.rglob("round_trajectory.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no round_trajectory.jsonl under {root}")
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                if payload.get("record_type") == "relational_imitation_round_feedback":
                    rows.append(payload)
    return rows


def analyze(
    run_dirs: Iterable[Path], output_dir: Path, bootstrap_resamples: int, confidence: float, seed: int
) -> list[dict[str, Any]]:
    rows = _round_payloads(run_dirs)
    pairs: dict[tuple[int, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["round_index"]) != 0:
            raise ValueError("Study 05 analysis accepts exactly one round per episode")
        target = str(row["controller_target"])
        before = tuple(row["population_state_before"])
        after = tuple(row["population_state_after"])
        n0 = before.count(target)
        n1 = after.count(target)
        if n0 not in EXPECTED_N_Z:
            raise ValueError(f"unexpected n_Z(0)={n0}")
        action = str(row["controller_action"])
        if action not in {ADVOCATE_TARGET, NO_OP}:
            raise ValueError(f"unexpected controller action {action!r}")
        if len(row.get("agent_ids", ())) != 24 or len(row.get("initial_knowledge_class_by_agent", ())) != 24:
            raise ValueError("agent-level initial state is missing from the trajectory")
        key = (n0, str(row["task_id"]), int(row["seed"]))
        if action in pairs[key]:
            raise ValueError(f"duplicate arm for pair {key}")
        pairs[key][action] = (n1 - n0) / 24.0

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    table: list[dict[str, Any]] = []
    for n0 in EXPECTED_N_Z:
        matched = [value for (n, _, _), value in pairs.items() if n == n0]
        if not matched or any(set(value) != {ADVOCATE_TARGET, NO_OP} for value in matched):
            raise ValueError(f"incomplete always/never pairs for n_Z(0)={n0}")
        advocate = np.array([value[ADVOCATE_TARGET] for value in matched], dtype=float)
        noop = np.array([value[NO_OP] for value in matched], dtype=float)
        boot = np.empty(bootstrap_resamples, dtype=float)
        for index in range(bootstrap_resamples):
            draw = rng.integers(0, len(matched), size=len(matched))
            boot[index] = float(np.mean(advocate[draw] - noop[draw]))
        ci = (
            (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha)))
            if bootstrap_resamples
            else (float("nan"), float("nan"))
        )
        table.append(
            {
                "n_Z_0": n0,
                "x_0": n0 / 24.0,
                "advocate_mean_delta_x": float(advocate.mean()),
                "noop_mean_delta_x": float(noop.mean()),
                "advocate_n": len(advocate),
                "noop_n": len(noop),
                "matched_pair_n": len(matched),
                "chi": float((advocate - noop).mean()),
                "chi_ci_low": ci[0],
                "chi_ci_high": ci[1],
                "bootstrap_unit": "matched_episode_pair",
                "bootstrap_resamples": bootstrap_resamples,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "state_matching_effects.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output_dir / "state_matching_effects.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    x = np.array([row["x_0"] for row in table])
    y = np.array([row["chi"] for row in table])
    low = np.array([row["chi_ci_low"] for row in table])
    high = np.array([row["chi_ci_high"] for row in table])
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker="o", capsize=4)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set(xlabel=r"$x_0=n_Z(0)/24$", ylabel=r"$\chi(x_0)$")
    fig.tight_layout()
    fig.savefig(output_dir / "state_matching_chi.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"status": "passed", "rows": table, "output_dir": str(output_dir)}, indent=2))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--config", type=Path, action="append", dest="configs")
    check.add_argument("--output", type=Path)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--run-dir", type=Path, action="append", required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--bootstrap-resamples", type=int, default=1000)
    analysis.add_argument("--confidence", type=float, default=0.95)
    analysis.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.configs or DEFAULT_CONFIGS, args.output)
    else:
        if args.bootstrap_resamples < 0 or not 0.0 < args.confidence < 1.0:
            parser.error("bootstrap resamples must be non-negative and confidence must be in (0,1)")
        analyze(args.run_dir, args.output_dir, args.bootstrap_resamples, args.confidence, args.seed)


if __name__ == "__main__":
    main()
