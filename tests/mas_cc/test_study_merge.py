from dataclasses import replace
from pathlib import Path
import json
import shutil

import pandas as pd
import pytest
import yaml

from mas_cc.config import load_run_config, load_run_config_or_grid
from mas_cc.experiments import run_experiment_sync
from mas_cc.studies.aggregation import aggregate_study
from mas_cc.studies.manifest import discover_study
from mas_cc.studies.merge import merge_studies
from mas_cc.studies.submission import (
    build_submission_entries,
    write_submission_manifest,
    _study_manifest,
)


def make_study(tmp_path, name, budget):
    config = load_run_config(
        "configs/runs/relational_reasoning/misselaneous/relational_imitation_round_feedback_controlled_smoke.yaml",
        environment={},
    )
    config = replace(
        config,
        experiment=replace(config.experiment, name=name),
        control=replace(
            config.control,
            options={
                **config.control.options,
                "intervention_budget": budget[0]
                if isinstance(budget, list)
                else budget,
            },
        ),
        storage=replace(
            config.storage, artifact_profile="results_only", checkpoint_mode="episode"
        ),
        analysis=replace(config.analysis, enabled=False),
        metrics=replace(config.metrics, enabled=False),
    )
    directory = tmp_path / f"{name}-configs"
    directory.mkdir()
    payload = config.to_dict()
    if isinstance(budget, list):
        payload["grid"] = {"control.options.intervention_budget": budget}
    (directory / "run.yaml").write_text(yaml.safe_dump(payload))
    (directory / "study.yaml").write_text(
        f"study: {{name: {name}}}\nconfigs: [run.yaml]\n"
    )
    (directory / "analysis.yaml").write_text(
        yaml.safe_dump(
            {
                "theoretical_reference": "none",
                "estimators": ["round_target_actuation_cmi"],
                "resampling": {"bootstrap_resamples": 2, "null_permutations": 1},
            }
        )
    )
    spec = discover_study(directory)
    root = tmp_path / name
    entries = build_submission_entries(spec, root, git_commit="test")
    write_submission_manifest(root / "submission_manifest.csv", entries)
    (root / "study_manifest.json").write_text(
        json.dumps(_study_manifest(spec, entries))
    )
    if isinstance(budget, list):
        grid = load_run_config_or_grid(directory / "run.yaml")
        for cell in grid.cells:
            output = Path(entries[0].output_dir) / "shards" / cell.cell_id
            run_experiment_sync(cell.config, output, resume=False, show_progress=False)
            run_root = next(output.rglob("resolved_config.yaml")).parent
            (run_root / "overrides.json").write_text(
                json.dumps(
                    {
                        "cell_id": cell.cell_id,
                        "overrides": dict(cell.overrides),
                    }
                )
            )
    else:
        run_experiment_sync(
            config, Path(entries[0].output_dir), resume=False, show_progress=False
        )
    return root


def test_merge_raw_and_retained_then_reaggregate(tmp_path):
    a = make_study(tmp_path, "a", 1)
    b = make_study(tmp_path, "b", 2)
    before = (a / "study_manifest.json").read_bytes()
    aggregate_study(b, backend="local")
    shutil.rmtree(b / "runs")
    merged = tmp_path / "merged"
    result = merge_studies([a, b], merged)
    assert result["found_cells"] == 2
    assert (a / "study_manifest.json").read_bytes() == before
    cells = pd.read_parquet(merged / "analysis/tables/cells.parquet")
    assert set(cells.intervention_budget) == {1, 2}
    assert cells.cell_id.nunique() == 2
    episodes = pd.read_parquet(merged / "analysis/tables/episodes.parquet")
    assert set(episodes.cell_id) == set(cells.cell_id)
    assert not (merged / "analysis/tables/primary_estimates.parquet").exists()
    assert episodes.episode_key.nunique() == len(episodes)
    summary = aggregate_study(merged, backend="local")
    assert summary["complete"]
    estimates = pd.read_parquet(
        merged / "analysis/tables/information_estimates.parquet"
    )
    assert set(estimates.cell_id) == set(cells.cell_id)
    assert (merged / "analysis/provenance/merge_manifest.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        merge_studies([a, b], merged)


def test_overlap_is_explicit_and_recipes_must_match(tmp_path):
    a = make_study(tmp_path, "a", 1)
    b = make_study(tmp_path, "b", 1)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="overlapping scientific cell"):
        merge_studies([a, b], output)
    assert not output.exists()
    result = merge_studies([a, b], output, overlap="keep-first")
    assert result["found_cells"] == 1
    assert result["skipped_cells"] == 1
    (tmp_path / "b-configs/analysis.yaml").write_text(
        "theoretical_reference: none\nestimators: []\n"
    )
    with pytest.raises(ValueError, match="recipes differ"):
        merge_studies([a, b], tmp_path / "another")


def test_incomplete_and_duplicate_sources_rejected(tmp_path):
    a = make_study(tmp_path, "a", 1)
    b = make_study(tmp_path, "b", 2)
    with pytest.raises(ValueError, match="distinct"):
        merge_studies([a, a], tmp_path / "duplicate")
    aggregate_study(b, backend="local")
    shutil.rmtree(b / "runs")
    path = b / "analysis/validation.json"
    validation = json.loads(path.read_text())
    validation["complete"] = False
    path.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match="complete source"):
        merge_studies([a, b], tmp_path / "invalid")


def test_grid_union_from_extracted_analysis_and_run_tree(tmp_path):
    a = make_study(tmp_path, "a", [1, 2])
    b = make_study(tmp_path, "b", [3])
    aggregate_study(a, backend="local")
    extracted = tmp_path / "extracted-analysis"
    shutil.copytree(a / "analysis", extracted)
    # Confirm that the saved provenance configs suffice without the originals.
    shutil.rmtree(tmp_path / "a-configs")
    output = tmp_path / "combined"
    result = merge_studies([extracted, b], output)
    assert result["found_cells"] == 3
    summary = aggregate_study(output, backend="local")
    assert summary["complete"]
    cells = pd.read_parquet(output / "analysis/tables/cells.parquet")
    assert set(cells.intervention_budget) == {1, 2, 3}
    assert len(list((output / "configs").glob("*.yaml"))) == 3
