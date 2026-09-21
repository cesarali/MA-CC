"""The analysis catalog must read the canonical Parquet tables and offer plot-ready estimate series."""
from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest

from mas_cc.blackboard_dashboard.study_data import (
    BlackboardStudyReader,
    SERIES_POINT_LIMIT,
    _estimate_series,
)
from mas_cc.studies.table_io import write_scientific_table


def _estimates(rows):
    return pd.DataFrame(rows)


def _row(metric="m", budget=6.0, estimate=0.5, **extra):
    return {"metric": metric, "intervention_budget": budget, "estimate": estimate,
            "ci_low": estimate - 0.1, "ci_high": estimate + 0.1, "support_status": "adequate",
            "target_semantics": "false", "epistemic_persistence": 0.7, "social_group_size": 12, **extra}


def _catalog(tmp_path: Path, frames: dict[str, pd.DataFrame], *, valid=True):
    analysis = tmp_path / "study" / "analysis"
    (analysis / "tables").mkdir(parents=True)
    for name, frame in frames.items():
        write_scientific_table(analysis / "tables", name, frame)
    (analysis / "validation.json").write_text(json.dumps({"valid": valid}))
    (analysis / "analysis_manifest.json").write_text(json.dumps({"study_id": "s"}))
    reader = BlackboardStudyReader.__new__(BlackboardStudyReader)
    reader.source_kind, reader.study_dir, reader.manifest = "study", analysis.parent, {"study_id": "s"}
    return reader.analysis_catalog()


def test_parquet_tables_are_read_at_all(tmp_path: Path):
    """Regression: the catalog looked for .csv only, so every preview was silently empty."""
    catalog = _catalog(tmp_path, {"primary_estimates": _estimates([_row(), _row(budget=18.0)])})
    assert list(catalog["table_previews"]) == ["primary_estimates.parquet"]
    assert catalog["table_previews"]["primary_estimates.parquet"]["total_rows"] == 2
    assert list(catalog["estimate_series"]) == ["primary_estimates.parquet"]


def test_preview_columns_are_a_readable_subset(tmp_path: Path):
    wide = _row()
    wide.update({f"noise_{i}": i for i in range(40)})
    catalog = _catalog(tmp_path, {"primary_estimates": _estimates([wide])})
    columns = catalog["table_previews"]["primary_estimates.parquet"]["columns"]
    assert "estimate" in columns and "metric" in columns
    assert not any(column.startswith("noise_") for column in columns)


def test_series_carry_interval_and_only_varying_coordinates():
    frame = _estimates([
        _row(budget=6.0, estimate=0.4, epistemic_persistence=0.7),
        _row(budget=18.0, estimate=0.6, epistemic_persistence=0.7),
        _row(budget=6.0, estimate=0.2, epistemic_persistence=1.0),
    ])
    series = _estimate_series(frame)
    # social_group_size and target_semantics are constant here, so they are not part of the key
    assert series["series_by"] == ["epistemic_persistence"]
    assert series["x"] == "intervention_budget"
    points = series["metrics"]["m"]
    assert {point["series"] for point in points} == {"rho 0.7", "rho 1"}
    first = points[0]
    assert (first["x"], first["y"], first["lo"], first["hi"]) == (6.0, 0.4, 0.30000000000000004, 0.5)
    assert [point["x"] for point in points if point["series"] == "rho 0.7"] == [6.0, 18.0]


def test_rows_without_a_finite_estimate_become_gaps_not_zeros():
    frame = _estimates([_row(estimate=float("nan")), _row(budget=None), _row(budget=18.0, estimate=0.3)])
    points = _estimate_series(frame)["metrics"]["m"]
    assert [(point["x"], point["y"]) for point in points] == [(18.0, 0.3)]


def test_series_are_bounded():
    frame = _estimates([_row(budget=float(i)) for i in range(SERIES_POINT_LIMIT + 50)])
    series = _estimate_series(frame)
    assert series["points"] == SERIES_POINT_LIMIT and series["truncated"] is True


def test_a_table_without_the_needed_columns_yields_no_series():
    assert _estimate_series(pd.DataFrame({"metric": ["m"], "estimate": [1.0]}))["metrics"] == {}


def test_invalid_analysis_exposes_neither_previews_nor_series(tmp_path: Path):
    catalog = _catalog(tmp_path, {"primary_estimates": _estimates([_row()])}, valid=False)
    assert catalog["available"] is False
    assert catalog["table_previews"] == {} and catalog["estimate_series"] == {}


def test_front_end_charts_the_series_and_loads_an_already_open_panel():
    script = files("mas_cc.blackboard_dashboard.assets").joinpath("app.js").read_text(encoding="utf-8")
    # a <details> that starts open never fires 'toggle', so the loader must also run at startup
    assert "async function loadAnalysis()" in script
    assert "$('study-analysis').addEventListener('toggle', loadAnalysis);" in script
    assert "if ($('study-analysis').open) loadAnalysis();" in script
    assert "renderEstimateCharts(container, catalog.estimate_series)" in script
    assert "wireEstimateCharts(catalog.estimate_series || {})" in script
    # interval bands only while they stay legible; a tooltip always carries the interval
    assert re.search(r"names\.length <= BAND_SERIES_LIMIT", script)
    assert "HEADLINE_METRICS" in script and "round_target_actuation_cmi" in script
