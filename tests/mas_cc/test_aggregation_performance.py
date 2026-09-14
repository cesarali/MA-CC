import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from mas_cc.analysis.epistemic_phase import SymbolicTask, _robustness, load_symbolic_tasks
from mas_cc.studies.aggregation import _package
from mas_cc.studies.performance import active_profile, measured_aggregation


TASKS = Path(__file__).resolve().parents[2] / "results/studies/musr_truthful_selective_task_calibration_01/tasks"


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.85, 1.0])
@pytest.mark.parametrize("seed", [1, 498])
def test_mask_lookup_matches_literal_survival_experiment(monkeypatch, rho, seed):
    task = load_symbolic_tasks(["task_003"], TASKS)["task_003"]
    facts = sorted(task.facts)
    inventory = {"a": facts, "b": facts[::2], "c": facts[::3]}
    actual = _robustness(task, inventory, rho, 137, seed)

    def literal(self, ids, retained):
        return np.asarray([
            self.solvable_from_masks([fact for fact, keep in zip(ids, draw) if keep])
            for draw in retained
        ])

    monkeypatch.setattr(SymbolicTask, "solvable_survival_draws", literal)
    assert actual == _robustness(task, inventory, rho, 137, seed)


def test_mask_lookup_preserves_large_integer_worlds_and_rejects_invalid_facts():
    task = SymbolicTask("test", "hash", "gold", 0, {}, None, {},
                        {"a": (1 << 130) | 1, "b": (1 << 130) | 2},
                        (1 << 130, 1, 2), (1 << 130) | 3)
    draws = np.array([[False, False], [True, False], [False, True], [True, True]])
    assert task.solvable_survival_draws(["a", "b"], draws).tolist() == [False, False, False, True]
    assert task.solvable_survival_draws([], np.empty((3, 0), dtype=bool)).tolist() == [False] * 3
    with pytest.raises(ValueError, match="unknown facts"):
        task.solvable_survival_draws(["missing"], np.ones((1, 1), dtype=bool))
    task.fact_world_masks = {"a": 1, "b": 2}
    task._mask_tables = None
    with pytest.raises(ValueError, match="no valid completions"):
        task.solvable_survival_draws(["a", "b"], np.ones((1, 2), dtype=bool))


def test_performance_provenance_and_streamed_package(tmp_path, monkeypatch):
    @measured_aggregation
    def aggregate(study_dir):
        root = Path(study_dir) / "analysis"
        (root / "tables").mkdir(parents=True)
        pd.DataFrame({"x": range(100)}).to_parquet(root / "tables/example.parquet")
        (root / "analysis_manifest.json").write_text("{}")
        profile = active_profile.get()
        profile.stage("packaging")
        return _package(root, "example")

    def no_read_bytes(*args):
        raise AssertionError("packaging must stream file contents")

    monkeypatch.setattr(Path, "read_bytes", no_read_bytes)
    archive = aggregate(tmp_path)
    assert not (tmp_path / "analysis/progress.json").exists()
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == {"analysis_manifest.json", "tables/example.parquet"}
        manifest = json.loads(package.read("analysis_manifest.json"))
        assert manifest == json.loads((tmp_path / "analysis/analysis_manifest.json").read_text())
        assert manifest["performance"]["stages"][-1]["stage"] == "packaging"
        assert manifest["performance"]["peak_rss_mib"] > 0


def test_failed_stage_remains_visible_and_profile_is_reset(tmp_path):
    mirror = tmp_path / "generation-progress.json"
    mirror.write_text(json.dumps({"generation_id": "generation",
                                 "job_states": {"finalizer": "PENDING"}}))

    @measured_aggregation
    def fail(study_dir, *, progress_path=None):
        active_profile.get().stage("robustness")
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        fail(tmp_path, progress_path=mirror)
    status = json.loads((tmp_path / "analysis/progress.json").read_text())
    assert status["stage"] == "robustness"
    assert status["status"] == "failed"
    assert json.loads(mirror.read_text()) == status
    assert status["generation_id"] == "generation"
    assert status["job_states"]["finalizer"] == "FAILED"
    assert active_profile.get() is None
