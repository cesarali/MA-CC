import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from mas_cc.analysis.epistemic_phase import (
    PRE_BOUNDARY,
    analyze_epistemic_phase_diagrams,
    build_epistemic_round_states,
    load_symbolic_tasks,
    reconstruct_active_inventories,
)
from mas_cc.studies import aggregate_study
from mas_cc.studies.submission import SubmissionEntry, write_submission_manifest
from mas_cc.studies.table_io import write_scientific_table


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "results/studies/musr_truthful_selective_task_calibration_01/tasks"
TASK_ID = "task_003"


def _facts() -> list[str]:
    raw = json.loads(
        (TASKS / TASK_ID / "facts/all_true_facts.json").read_text(encoding="utf-8")
    )
    return [str(row["fact_id"]) for row in raw]


def _inventory(agent_ids: list[str], facts: list[str]) -> dict[str, list[str]]:
    result = {agent: [] for agent in agent_ids}
    for index, fact in enumerate(facts):
        result[agent_ids[index % len(agent_ids)]].append(fact)
    return result


def _row(
    episode: str,
    round_index: int,
    action: int,
    before: int,
    after: int,
    initial: dict[str, list[str]],
    active_after: dict[str, list[str]],
    *,
    deactivated: list[dict[str, str]] | None = None,
) -> dict:
    agents = list(initial)
    return {
        "study_id": "study",
        "source_run_id": "run",
        "cell_id": "cell",
        "episode_id": episode,
        "round_index": round_index,
        "record_type": "relational_imitation_round_feedback",
        "task_id": TASK_ID,
        "initial_task_id": TASK_ID,
        "correct_answer": "ALLOCATION_0",
        "analysis_target": "ALLOCATION_2",
        "possible_answers": ["ALLOCATION_0", "ALLOCATION_1", "ALLOCATION_2"],
        "occupation_counts_before": [len(agents) - before, 0, before],
        "occupation_counts_after": [len(agents) - after, 0, after],
        "controller_target_share_before": before / len(agents),
        "controller_target_share": after / len(agents),
        "U_k": action,
        "P_U1_given_Y": 0.5,
        "protocol": "night_dawn_autonomous_day_v1",
        "epistemic_persistence": 0.85,
        "agent_ids": agents,
        "initial_active_fact_ids_by_agent": list(initial.values()),
        "persistence_deactivated_pairs": deactivated or [],
        "active_fact_ids_by_agent_after": active_after,
        "active_full_proof_agent_share_before": 0.0,
        "physical_initial_state_hash": f"block-{episode.rsplit('-', 1)[-1]}",
        "initialization_repetition": int(episode.rsplit("-", 1)[-1]),
    }


def test_dawn_inventory_reconstruction_subtracts_current_forgetting():
    initial = {"agent_001": ["f1", "f2"], "agent_002": ["f3"]}
    after0 = {"agent_001": ["f2"], "agent_002": ["f3"]}
    after1 = {"agent_001": ["f2"], "agent_002": []}
    frame = pd.DataFrame(
        [
            _row(
                "episode-0",
                0,
                0,
                0,
                0,
                initial,
                after0,
                deactivated=[{"agent_id": "agent_001", "fact_id": "f1"}],
            ),
            _row(
                "episode-0",
                1,
                0,
                0,
                0,
                initial,
                after1,
                deactivated=[{"agent_id": "agent_002", "fact_id": "f3"}],
            ),
        ]
    )
    reconstructed = reconstruct_active_inventories(frame)
    assert reconstructed.iloc[0]["active_fact_ids_by_agent_before"] == after0
    assert reconstructed.iloc[1]["active_fact_ids_by_agent_before"] == after1
    assert set(reconstructed["epistemic_boundary"]) == {PRE_BOUNDARY}


def test_symbolic_metrics_distinguish_collective_access_from_individual_access():
    facts = _facts()
    agents = [f"agent_{index:03d}" for index in range(1, len(facts) + 1)]
    distributed = _inventory(agents, facts)
    concentrated = {
        agent: ([] if index else facts) for index, agent in enumerate(agents)
    }
    row = _row("episode-0", 0, 1, 0, 1, distributed, concentrated)
    task = load_symbolic_tasks([TASK_ID], TASKS)
    states = build_epistemic_round_states(
        pd.DataFrame([row]),
        task,
        robustness_draws=4,
        reference_persistence=1.0,
        seed=9,
    )
    state = states.iloc[0]
    assert bool(state["collective_solvable"])
    assert state["symbolic_individual_solvability_share"] == 0.0
    assert state["fragmentation_gap"] == 1.0
    assert state["symbolic_individual_solvability_share_after"] == pytest.approx(
        1 / len(facts)
    )
    assert state["reference_robustness"] == 1.0
    assert state["active_union_fact_count"] == len(facts)
    assert state["evidence_scope"] == "union_of_participant_active_inventories"


def test_full_epistemic_analysis_emits_timeseries_drift_modulation_and_timing():
    facts = _facts()
    agents = [f"agent_{index:03d}" for index in range(1, len(facts) + 1)]
    distributed = _inventory(agents, facts)
    concentrated = {
        agent: ([] if index else facts) for index, agent in enumerate(agents)
    }
    rows = []
    for episode_index, action in enumerate((1, 0, 1, 0)):
        episode = f"episode-{episode_index}"
        state = distributed
        for round_index in range(3):
            after_state = concentrated if action else distributed
            rows.append(
                _row(
                    episode,
                    round_index,
                    action,
                    before=38,
                    after=40 if action else 38,
                    initial=distributed,
                    active_after=after_state,
                )
            )
            state = after_state
    cells = pd.DataFrame(
        [
            {
                "cell_id": "cell",
                "task_id": TASK_ID,
                "horizon": 3,
                "population_size": len(agents),
                "intervention_budget": 3,
                "epistemic_persistence": 0.85,
            }
        ]
    )
    outputs = analyze_epistemic_phase_diagrams(
        pd.DataFrame(rows),
        cells,
        task_dataset_dir=TASKS,
        robustness_draws=2,
        reference_persistence=1.0,
        bootstrap_resamples=4,
        confidence=0.9,
        seed=3,
    )
    assert set(outputs) == {
        "epistemic_round_timeseries",
        "epistemic_parameter_summary",
        "epistemic_state_occupancy",
        "epistemic_joint_drift",
        "epistemic_causal_susceptibility",
        "epistemic_modulation",
        "epistemic_capture_timing",
        "epistemic_capture_summary",
    }
    timeseries = outputs["epistemic_round_timeseries"]
    assert len(timeseries) == 12
    assert {
        "collective_solvable",
        "symbolic_individual_solvability_share",
        "fragmentation_gap",
        "configured_robustness",
        "active_union_fact_fraction",
        "active_fact_occurrence_count",
        "collective_gold_probability",
        "collective_normalized_entropy",
    }.issubset(timeseries)
    assert set(outputs["epistemic_joint_drift"]["branch"]) == {
        "silence",
        "activation",
        "contrast",
    }
    assert not outputs["epistemic_causal_susceptibility"].empty
    modulation = outputs["epistemic_modulation"].iloc[0]
    assert modulation["bootstrap_unit"] == "shared_initialization_block"
    timing = outputs["epistemic_capture_timing"]
    assert timing["capture_observed"].all()
    assert timing["capture_with_collective_solvability_throughout"].all()


def test_offline_aggregation_packages_epistemic_outputs(tmp_path):
    study = tmp_path / "offline-epistemic"
    analysis = study / "analysis"
    tables = analysis / "tables"
    recipe = study / "analysis.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        f"""theoretical_reference: none
estimators: []
resampling: {{bootstrap_resamples: 0, null_permutations: 0, confidence: 0.9, seed: 7}}
blackboard_epistemic_phase_outputs:
  task_dataset_dir: {TASKS}
  robustness_draws: 2
  reference_persistence: 1.0
  x_bins: 8
  phi_bands: 3
  capture_threshold: 0.75
  capture_consecutive_rounds: 2
""",
        encoding="utf-8",
    )
    (study / "study_manifest.json").write_text(
        json.dumps({"study_id": "offline-epistemic", "analysis_recipe": str(recipe)}),
        encoding="utf-8",
    )
    write_submission_manifest(
        study / "submission_manifest.csv",
        [
            SubmissionEntry(
                array_index=0,
                config_path=str(study / "missing-source.yaml"),
                config_hash="config",
                resolved_config_hash="resolved",
                output_dir=str(study / "missing-run"),
                expected_cell_count=1,
                expected_episode_count=2,
                execution_seed=1,
                git_commit="test",
            )
        ],
    )
    facts = _facts()
    agents = [f"agent_{index:03d}" for index in range(1, len(facts) + 1)]
    distributed = _inventory(agents, facts)
    rows = [
        _row(
            f"episode-{episode}", round_index, episode, 38, 40, distributed, distributed
        )
        for episode in (0, 1)
        for round_index in (0, 1)
    ]
    cell = pd.DataFrame(
        [
            {
                "study_id": "offline-epistemic",
                "source_run_id": "run",
                "cell_id": "cell",
                "task_id": TASK_ID,
                "horizon": 2,
                "population_size": len(agents),
                "intervention_budget": 3,
                "epistemic_persistence": 0.85,
            }
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "study_id": "offline-epistemic",
                "source_run_id": "run",
                "cell_id": "cell",
                "episode_id": f"episode-{episode}",
                "status": "completed",
            }
            for episode in (0, 1)
        ]
    )
    for name, frame in {
        "cells": cell,
        "episodes": episodes,
        "rounds": pd.DataFrame(rows),
        "micro_slots": pd.DataFrame(columns=["cell_id", "episode_id", "round_index"]),
    }.items():
        write_scientific_table(tables, name, frame)
    counts = {
        "expected_configs": 1,
        "found_configs": 1,
        "expected_cells": 1,
        "found_cells": 1,
        "sealed_cells": 1,
        "expected_episodes": 2,
        "completed_episodes": 2,
        "failed_episodes": 0,
        "aborted_episodes": 0,
        "duplicate_run_identities": 0,
        "duplicate_cell_identities": 0,
        "duplicate_episode_identities": 0,
        "round_rows": 4,
        "micro_slot_rows": 0,
    }
    (analysis / "validation.json").write_text(
        json.dumps(
            {
                "valid": True,
                "complete": True,
                "counts": counts,
                "errors": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (analysis / "analysis_manifest.json").write_text(
        json.dumps({"scientific_input_identity": "retained-input"}),
        encoding="utf-8",
    )

    result = aggregate_study(study)
    expected = {
        "epistemic_round_timeseries.parquet",
        "epistemic_parameter_summary.parquet",
        "epistemic_state_occupancy.parquet",
        "epistemic_joint_drift.parquet",
        "epistemic_causal_susceptibility.parquet",
        "epistemic_modulation.parquet",
        "epistemic_capture_timing.parquet",
        "epistemic_capture_summary.parquet",
    }
    assert expected.issubset({path.name for path in tables.glob("*.parquet")})
    validation = json.loads((analysis / "validation.json").read_text(encoding="utf-8"))
    assert validation["epistemic_phase"]["provider_calls"] == 0
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
    assert {f"tables/{name}" for name in expected}.issubset(names)
    assert "plots/epistemic_central_figure.png" in names
