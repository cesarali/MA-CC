import json
import math
import zipfile

import pandas as pd
import pytest

from mas_cc.analysis.causal_response import (
    analyze_causal_communication,
    build_causal_response_inputs,
    build_communication_funnel,
    estimate_causal_response,
)
from mas_cc.studies.aggregation import _blackboard_phase2_requested
from mas_cc.studies import aggregate_study
from mas_cc.studies.submission import SubmissionEntry, write_submission_manifest
from mas_cc.studies.table_io import write_scientific_table


def _round(
    cell: str,
    episode: str,
    round_index: int,
    action: int,
    before: float,
    after: float,
    *,
    target: str = "false",
    block: str | None = None,
) -> dict:
    return {
        "study_id": "study",
        "source_run_id": "run",
        "cell_id": cell,
        "episode_id": episode,
        "round_index": round_index,
        "U_k": action,
        "P_U1_given_Y": 0.5,
        "controller_target_share_before": before,
        "controller_target_share": after,
        "physical_initial_state_hash": block or episode,
        "initialization_repetition": int(episode.rsplit("-", 1)[-1]),
        "target_semantics": target,
        "chosen_message_mode": "REPORT" if action else None,
        "actual_controller_posts": action,
        "controller_message_exposures": action * 2,
        "controller_unique_readers": action * 2,
        "new_controller_facts": action,
        "reactivated_controller_fact_count": 0,
    }


def test_propensity_weighted_example_recovers_known_effect_and_weights():
    rounds = pd.DataFrame(
        [
            _round("c", "e-0", 0, 1, 0.2, 0.3),
            _round("c", "e-1", 0, 0, 0.2, 0.2),
            _round("c", "e-2", 0, 1, 0.2, 0.3),
            _round("c", "e-3", 0, 0, 0.2, 0.2),
        ]
    )
    inputs = build_causal_response_inputs(rounds)
    assert inputs.loc[inputs["U_t"] == 1, "ipw_contrast_weight"].tolist() == [2.0, 2.0]
    assert inputs.loc[inputs["U_t"] == 0, "ipw_contrast_weight"].tolist() == [-2.0, -2.0]
    effects, support, _ = estimate_causal_response(inputs, bootstrap_resamples=0)
    assert effects.loc[effects["lag"] == 1, "estimate"].item() == pytest.approx(0.1)
    assert support.loc[support["lag"] == 1, "support_status"].item() == "adequate"


def test_target_orientation_and_lags_are_episode_local_and_row_order_invariant():
    rows = [
        _round("truth", "truth-0", 0, 1, 0.2, 0.3, target="truth"),
        _round("truth", "truth-0", 1, 0, 0.3, 0.4, target="truth"),
        _round("false", "false-0", 0, 1, 0.7, 0.6, target="false"),
        _round("false", "false-0", 1, 0, 0.6, 0.5, target="false"),
    ]
    one = build_causal_response_inputs(pd.DataFrame(rows))
    two = build_causal_response_inputs(pd.DataFrame(reversed(rows)))
    columns = ["cell_id", "episode_id", "round_index", "delta_x_h1", "delta_x_h2"]
    pd.testing.assert_frame_equal(one[columns], two[columns])
    assert one.loc[
        (one["cell_id"] == "truth") & (one["round_index"] == 0), "delta_x_h2"
    ].item() == pytest.approx(0.2)
    assert one.loc[
        (one["cell_id"] == "false") & (one["round_index"] == 0), "delta_x_h2"
    ].item() == pytest.approx(-0.2)
    assert one.loc[one["round_index"] == 1, "delta_x_h2"].isna().all()


def test_missing_round_is_reported_and_shared_initialization_blocks_stay_intact():
    rounds = pd.DataFrame(
        [
            _round("a", "a-0", 0, 1, 0.2, 0.3, block="shared-0"),
            _round("a", "a-0", 1, 0, 0.3, 0.3, block="shared-0"),
            _round("a", "a-1", 0, 0, 0.2, 0.2, block="shared-1"),
            _round("b", "b-0", 0, 1, 0.2, 0.3, block="shared-0"),
            _round("b", "b-0", 1, 0, 0.3, 0.3, block="shared-0"),
            _round("b", "b-1", 0, 0, 0.2, 0.2, block="shared-1"),
        ]
    )
    cells = pd.DataFrame(
        [
            {"cell_id": "a", "horizon": 2},
            {"cell_id": "b", "horizon": 2},
        ]
    )
    inputs = build_causal_response_inputs(rounds, cells)
    effects, support, draws = estimate_causal_response(inputs, bootstrap_resamples=8, seed=7)
    assert set(support["incomplete_episode_count"]) == {1}
    assert set(effects.loc[effects["lag"].isin([1, 2]), "n_episodes"]) == {1}
    for draw in draws:
        counts = draw.groupby(["initialization_block_id", "cell_id"]).size().unstack(fill_value=0)
        assert (counts["a"] == counts["b"]).all()


def test_communication_funnel_audits_micro_counts_and_zero_cost_is_missing():
    rounds = pd.DataFrame(
        [
            _round("c", "e-0", 0, 1, 0.2, 0.3),
            _round("c", "e-1", 0, 0, 0.2, 0.2),
        ]
    )
    inputs = build_causal_response_inputs(rounds)
    micro = pd.DataFrame(
        [
            {
                "cell_id": "c",
                "episode_id": "e-0",
                "round_index": 0,
                "focal_agent_id": "0",
                "sampled_controller_message_ids": ["m"],
                "new_controller_fact_ids": ["f"],
                "reactivated_controller_fact_ids": [],
            },
            {
                "cell_id": "c",
                "episode_id": "e-0",
                "round_index": 0,
                "focal_agent_id": "1",
                "sampled_controller_message_ids": ["m"],
                "new_controller_fact_ids": [],
                "reactivated_controller_fact_ids": [],
            },
            {
                "cell_id": "c",
                "episode_id": "e-1",
                "round_index": 0,
                "focal_agent_id": "0",
                "sampled_controller_message_ids": [],
                "new_controller_fact_ids": [],
                "reactivated_controller_fact_ids": [],
            },
        ]
    )
    funnel = build_communication_funnel(inputs, micro)
    assert funnel["micro_slot_audit_available"].all()
    assert set(funnel["unique_reader_scope"]) == {"per_round_not_episode_wide"}
    outputs = analyze_causal_communication(
        rounds,
        pd.DataFrame([{"cell_id": "c", "target_semantics": "false"}]),
        micro,
        bootstrap_resamples=0,
    )
    reactivation = outputs["communication_efficiency"].query(
        "cost_metric == 'reactivated_controller_facts' and lag == 1"
    ).iloc[0]
    assert math.isnan(reactivation["response_per_expected_cost"])
    assert bool(reactivation["zero_denominator"])


def test_post_treatment_causal_strata_are_rejected():
    with pytest.raises(ValueError, match="post-treatment"):
        _blackboard_phase2_requested(
            {"blackboard_phase2_outputs": {"strata": ["chosen_message_mode"]}}
        )


def test_invalid_propensity_and_micro_mismatch_fail_loudly():
    bad = pd.DataFrame([_round("c", "e-0", 0, 1, 0.2, 0.3)])
    bad["P_U1_given_Y"] = 1.0
    with pytest.raises(ValueError, match="strictly between"):
        build_causal_response_inputs(bad)

    inputs = build_causal_response_inputs(
        pd.DataFrame([_round("c", "e-0", 0, 1, 0.2, 0.3)])
    )
    micro = pd.DataFrame(
        [
            {
                "cell_id": "c",
                "episode_id": "e-0",
                "round_index": 0,
                "focal_agent_id": "0",
                "sampled_controller_message_ids": [],
                "new_controller_fact_ids": [],
                "reactivated_controller_fact_ids": [],
            }
        ]
    )
    with pytest.raises(ValueError, match="disagrees"):
        build_communication_funnel(inputs, micro)


def test_retained_parquet_reaggregation_writes_phase2_handoff_without_run_trees(
    tmp_path,
):
    study = tmp_path / "offline-study"
    analysis = study / "analysis"
    tables = analysis / "tables"
    recipe = study / "analysis.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        """version: 1
theoretical_reference: none
estimators: [propensity_weighted_causal_response]
derived: [communication_response_efficiency]
blackboard_phase2_outputs: true
resampling: {bootstrap_resamples: 4, null_permutations: 0, confidence: 0.9, seed: 7}
""",
        encoding="utf-8",
    )
    (study / "study_manifest.json").write_text(
        json.dumps(
            {
                "study_id": "offline-study",
                "analysis_recipe": str(recipe.resolve()),
            }
        ),
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
                output_dir=str(study / "missing-run-tree"),
                expected_cell_count=1,
                expected_episode_count=4,
                execution_seed=1,
                git_commit="test",
            )
        ],
    )
    cell_id = "config-0000/cell-0000"
    cells = pd.DataFrame(
        [
            {
                "study_id": "offline-study",
                "source_run_id": "run",
                "cell_id": cell_id,
                "target_semantics": "false",
                "controller_communication_policy": "llm_structured_v1",
                "intervention_budget": 1,
                "horizon": 1,
            }
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "study_id": "offline-study",
                "source_run_id": "run",
                "cell_id": cell_id,
                "episode_id": f"episode-{index}",
                "status": "completed",
            }
            for index in range(4)
        ]
    )
    round_rows = []
    for index, action in enumerate((1, 0, 1, 0)):
        row = _round(
            cell_id,
            f"episode-{index}",
            0,
            action,
            0.25,
            0.5 if action else 0.25,
            block=f"block-{index}",
        )
        row.update(
            {
                "record_type": "relational_imitation_round_feedback",
                "possible_answers": ["A", "B"],
                "correct_answer": "A",
                "analysis_target": "B",
                "occupation_counts_before": [3, 1],
                "occupation_counts_after": [2, 2] if action else [3, 1],
            }
        )
        round_rows.append(row)
    rounds = pd.DataFrame(round_rows)
    micro = pd.DataFrame(columns=["cell_id", "episode_id", "round_index"])
    for name, frame in {
        "cells": cells,
        "episodes": episodes,
        "rounds": rounds,
        "micro_slots": micro,
    }.items():
        write_scientific_table(tables, name, frame)
    counts = {
        "expected_configs": 1,
        "found_configs": 1,
        "expected_cells": 1,
        "found_cells": 1,
        "sealed_cells": 1,
        "expected_episodes": 4,
        "completed_episodes": 4,
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
        "causal_response_round_inputs.parquet",
        "causal_response_effects.parquet",
        "causal_response_support.parquet",
        "communication_funnel.parquet",
        "communication_efficiency.parquet",
        "response_cost_frontier.parquet",
    }
    assert expected.issubset({path.name for path in tables.glob("*.parquet")})
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
    assert {f"tables/{name}" for name in expected}.issubset(names)
    assert "plots/causal_response_by_lag.png" in names
    validation = json.loads((analysis / "validation.json").read_text())
    assert validation["causal_response"]["provider_calls"] == 0
    stable_paths = [
        *(tables / name for name in sorted(expected)),
        analysis / "plots" / "causal_response_by_lag.png",
    ]
    first_bytes = {path.name: path.read_bytes() for path in stable_paths}
    aggregate_study(study)
    assert first_bytes == {path.name: path.read_bytes() for path in stable_paths}
