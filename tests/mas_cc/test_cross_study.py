"""Cross-study combination: pairing report, joint paired aggregation, descriptive concat, package output."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from mas_cc.analysis import cross_study

_fixtures = importlib.util.spec_from_file_location(
    "derived_fixtures", Path(__file__).with_name("test_derived_study_aggregation.py"))
derived_fixtures = importlib.util.module_from_spec(_fixtures)
_fixtures.loader.exec_module(derived_fixtures)


def _events_for(cell: str, strength: int):
    events = derived_fixtures._cell_events(cell, strength)
    return [replace(e, event={**e.event, "physical_initial_state_hash": e.episode_id.rsplit("/", 1)[-1]}) for e in events]


def _write_package(root: Path, name: str, cell_row: dict, events, budget: int, extra_table: bool = True) -> Path:
    tables = root / name / "analysis" / "tables"
    tables.mkdir(parents=True)
    cells = pd.DataFrame([{**cell_row, "intervention_budget": budget, "study_id": name}])
    cells.to_parquet(tables / "cells.parquet", index=False)
    rounds = pd.DataFrame([{
        "cell_id": e.cell_id, "episode_id": e.episode_id, "round_index": e.round_index,
        "initialization_artifact_hash": e.episode_id.rsplit("/", 1)[-1], "episode_complete": True,
    } for e in events])
    rounds.to_parquet(tables / "rounds.parquet", index=False)
    if extra_table:
        pd.DataFrame([{"metric": "eta_ir", "intervention_budget": budget, "estimate": 0.1 * budget,
                       "ci_low": 0.05 * budget, "ci_high": 0.15 * budget, "epistemic_persistence": cell_row["epistemic_persistence"]}]
                     ).to_parquet(tables / "study_aggregated_metrics.parquet", index=False)
    recipe = derived_fixtures._recipe()
    recipe["derived_study_aggregates"]["bootstrap"] = {"unit": "shared_initialization_block"}
    recipe["resampling"] = {"bootstrap_resamples": 6, "null_permutations": 3, "confidence": 0.95, "seed": 4}
    (root / name / "analysis" / "analysis_recipe.yaml").write_text(yaml.safe_dump(recipe))
    (root / name / "analysis" / "analysis_manifest.json").write_text(json.dumps({"analysis_hash": f"hash-{name}"}))
    return root / name / "analysis"


def _two_packages(tmp_path):
    cells = derived_fixtures._cells().to_dict("records")
    events_a = _events_for("truth-rho1", 1)
    events_b = _events_for("false-rho2", 1)
    a = _write_package(tmp_path, "study_a", cells[0], events_a, budget=3)
    b = _write_package(tmp_path, "study_b", cells[1], events_b, budget=6)
    return a, b, events_a + events_b


def test_pairing_report_sees_shared_initialization_blocks(tmp_path):
    a, b, _ = _two_packages(tmp_path)
    packages = [cross_study.load_package(f"A={a}"), cross_study.load_package(f"B={b}")]
    report = cross_study.pairing_report(packages)
    assert [p["blocks"] for p in report["packages"]] == [4, 4]
    assert report["pairs"][0]["shared_blocks"] == 4 and report["fully_paired"] is True


def test_combine_runs_joint_paired_aggregation_and_concatenates_descriptives(tmp_path):
    a, b, events = _two_packages(tmp_path)
    packages = [cross_study.load_package(f"A={a}"), cross_study.load_package(f"B={b}")]
    result = cross_study.combine(packages, events_builder=lambda rounds: events, workers=1)
    assert not result.study_metrics.empty
    assert set(result.study_metrics["intervention_budget"]) == {3, 6}
    assert set(result.study_metrics["bootstrap_unit"]) == {"shared_initialization_block"}
    assert (result.study_metrics["combined_from_packages"] == "A,B").all()
    assert set(result.descriptive["study_aggregated_metrics"]["source_package"]) == {"A", "B"}
    assert result.manifest["provider_calls"] == 0 and result.manifest["cells"] == 2
    pooled = cross_study.combine(packages, events_builder=lambda rounds: events, workers=2)
    pd.testing.assert_frame_equal(result.study_metrics, pooled.study_metrics)


def test_write_outputs_produces_tables_plots_and_manifest(tmp_path):
    a, b, events = _two_packages(tmp_path)
    packages = [cross_study.load_package(f"A={a}"), cross_study.load_package(f"B={b}")]
    result = cross_study.combine(packages, events_builder=lambda rounds: events)
    out = tmp_path / "combined"
    cross_study.write_outputs(result, out)
    manifest = json.loads((out / "cross_study_manifest.json").read_text())
    assert (out / "tables" / "combined_study_aggregated_metrics.parquet").is_file()
    assert (out / "tables" / "all_packages_study_aggregated_metrics.parquet").is_file()
    assert manifest["plots"] and all((out / "plots" / p).is_file() for p in manifest["plots"])
    assert [p["name"] for p in manifest["packages"]] == ["A", "B"]


def test_colliding_cell_ids_are_rejected(tmp_path):
    cells = derived_fixtures._cells().to_dict("records")
    events = _events_for("truth-rho1", 1)
    a = _write_package(tmp_path, "study_a", cells[0], events, budget=3)
    b = _write_package(tmp_path, "study_b", cells[0], events, budget=6)
    packages = [cross_study.load_package(f"A={a}"), cross_study.load_package(f"B={b}")]
    try:
        cross_study.combine(packages, events_builder=lambda rounds: events)
    except ValueError as exc:
        assert "collide" in str(exc)
    else:
        raise AssertionError("colliding cell ids must be rejected")
