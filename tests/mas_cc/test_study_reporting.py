from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mas_cc.cli.main import build_parser
from mas_cc.studies.reporting import build_study_report
from mas_cc.studies.table_io import write_scientific_table


METRICS = {
    "round_target_actuation_cmi": 0.2,
    "round_target_information_fraction": 0.4,
    "round_target_susceptibility": -0.05,
}


def _analysis_package(tmp_path: Path, *, bins: int = 8) -> Path:
    analysis = tmp_path / "study" / "analysis"
    tables = analysis / "tables"
    tables.mkdir(parents=True)
    cells = []
    primary = []
    derived = []
    occupancy = []
    rho_maps = []
    rho_occupancy = []
    for semantics in ("truth", "false", "none"):
        for rho in (0.7, 0.9):
            for q in (1, 3):
                for budget in (3, 6):
                    cell_id = f"{semantics}-{rho}-{q}-{budget}"
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "target_semantics": semantics,
                            "epistemic_persistence": rho,
                            "social_group_size": q,
                            "intervention_budget": budget,
                        }
                    )
                    for bin_index in range(bins):
                        common = {
                            "cell_id": cell_id,
                            "target_semantics": semantics,
                            "epistemic_persistence": rho,
                            "social_group_size": q,
                            "intervention_budget": budget,
                            "target_fraction_bin_index": bin_index,
                            "target_fraction_bin_lower": bin_index / bins,
                            "target_fraction_bin_upper": (bin_index + 1) / bins,
                            "target_fraction_bin_center": (bin_index + 0.5) / bins,
                            "target_fraction_bin_count": bins,
                            "n_observations": 10,
                            "n_episodes": 2,
                            "support_status": "unsupported"
                            if semantics == "none"
                            else "adequate",
                        }
                        for metric, value in METRICS.items():
                            primary.append(
                                {
                                    **common,
                                    "metric": metric,
                                    "estimate": value + bin_index / 100,
                                    "units": "bits"
                                    if metric == "round_target_actuation_cmi"
                                    else "fraction",
                                }
                            )
                        derived.append(
                            {
                                **common,
                                "metric": "eta_ir_state_local",
                                "estimate": 0.1 + bin_index / 100,
                                "units": "dimensionless",
                            }
                        )
                        occupancy.append(
                            {
                                **common,
                                "estimate": 10,
                                "phase_status": "visited",
                            }
                        )
    for semantics in ("truth", "false", "none"):
        for q in (1, 3):
            for budget in (3, 6):
                for bin_index in range(bins):
                    common = {
                        "target_semantics": semantics,
                        "social_group_size": q,
                        "intervention_budget": budget,
                        "target_fraction_bin_index": bin_index,
                        "target_fraction_bin_lower": bin_index / bins,
                        "target_fraction_bin_upper": (bin_index + 1) / bins,
                        "target_fraction_bin_center": (bin_index + 0.5) / bins,
                        "target_fraction_bin_count": bins,
                        "n_observations": 20,
                        "n_episodes": 4,
                        "phase_status": "unsupported"
                        if semantics == "none"
                        else "adequate",
                    }
                    for metric, value in {
                        "T_pi": 0.2,
                        "eta_IF": 0.4,
                        "eta_IR": 0.1,
                        "chi": -0.05,
                    }.items():
                        rho_maps.append({**common, "metric": metric, "estimate": value})
                    rho_occupancy.append({**common, "estimate": 20})
    for name, frame in {
        "cells": pd.DataFrame(cells),
        "episodes": pd.DataFrame(),
        "rounds": pd.DataFrame(),
        "micro_slots": pd.DataFrame(),
        "primary_estimates": pd.DataFrame(primary),
        "information_estimates": pd.DataFrame(primary),
        "support_diagnostics": pd.DataFrame(),
        "derived_observables": pd.DataFrame(derived),
        "state_occupancy_binned": pd.DataFrame(occupancy),
        "rho_aggregated_state_local_maps": pd.DataFrame(rho_maps),
        "rho_aggregated_state_occupancy": pd.DataFrame(rho_occupancy),
    }.items():
        write_scientific_table(tables, name, frame)
    counts = {
        "expected_cells": 24,
        "found_cells": 24,
        "expected_episodes": 48,
        "completed_episodes": 47,
        "failed_episodes": 0,
        "aborted_episodes": 0,
        "round_rows": 0,
        "micro_slot_rows": 0,
    }
    (analysis / "validation.json").write_text(
        json.dumps(
            {
                "valid": False,
                "complete": False,
                "counts": counts,
                "errors": ["one episode is missing"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (analysis / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "study_id": "report-fixture",
                "status": "incomplete",
                "scientific_input_identity": "input",
                "analysis_hash": "analysis",
                "tables": sorted(path.name for path in tables.glob("*.parquet")),
            }
        ),
        encoding="utf-8",
    )
    return analysis


def _config(tmp_path: Path, analysis: Path) -> Path:
    example = (
        Path(__file__).resolve().parents[2]
        / "configs/reports/state_budget_phase_report.example.yaml"
    )
    config = yaml.safe_load(example.read_text(encoding="utf-8"))
    config["report"]["id"] = "fixture-report"
    config["report"]["title"] = "Fixture Report"
    config["report"]["source_analysis"] = str(analysis)
    config["report"]["output_dir"] = str(tmp_path / "report")
    path = tmp_path / "report.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_cli_exposes_study_report_command():
    args = build_parser().parse_args(["study", "report", "--config", "report.yaml"])
    assert args.study_command == "report"
    assert args.config == Path("report.yaml")


def test_report_builds_all_formats_and_adapts_facets(tmp_path):
    result = build_study_report(_config(tmp_path, _analysis_package(tmp_path)))

    assert result.provisional
    assert result.markdown.is_file()
    assert result.latex.is_file()
    assert result.pdf.stat().st_size > 0
    assert result.manifest.is_file()
    assert len(result.figures) >= 14
    markdown = result.markdown.read_text(encoding="utf-8")
    assert "INCOMPLETE / PROVISIONAL" in markdown
    assert "gray does not mean zero" in markdown
    assert "eta_th" in markdown
    assert "state-local" in markdown
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["number_source_ledger_entries_validated"] > 0
    assert any(
        row["metric_id"] == "eta_th" and row["status"] == "unavailable"
        for row in manifest["phase_results"]
    )
    assert any(
        row["metric_id"] == "susceptibility_symlog" and row["status"] == "available"
        for row in manifest["phase_results"]
    )
    resolved_tpi = next(
        row
        for row in manifest["phase_results"]
        if row["metric_id"] == "T_pi" and row["mode"] == "resolved"
    )
    assert resolved_tpi["panel_count"] == 12


def test_report_refuses_fewer_than_eight_bins_without_rebinning(tmp_path):
    result = build_study_report(_config(tmp_path, _analysis_package(tmp_path, bins=4)))
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    state_results = [
        row
        for row in manifest["phase_results"]
        if row["metric_id"] in {"T_pi", "eta_IF", "eta_IR", "susceptibility"}
    ]
    assert state_results
    assert all(row["status"] == "unavailable" for row in state_results)
    assert all("at least 8" in row["reason"] for row in state_results)


def test_report_never_averages_duplicate_plot_coordinates(tmp_path):
    analysis = _analysis_package(tmp_path)
    primary_path = analysis / "tables/primary_estimates.parquet"
    primary = pd.read_parquet(primary_path)
    primary = pd.concat([primary, primary.iloc[[0]]], ignore_index=True)
    write_scientific_table(analysis / "tables", "primary_estimates", primary)

    result = build_study_report(_config(tmp_path, analysis))
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    tpi = [
        row
        for row in manifest["phase_results"]
        if row["metric_id"] == "T_pi" and row["mode"] == "resolved"
    ]
    assert tpi[0]["status"] == "unavailable"
    assert "share report coordinates" in tpi[0]["reason"]
