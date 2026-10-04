"""Behaviour introduced when main's site detection and report pools met this branch.

Two independent designs had to coexist:

* launcher choice: an explicit ``--execution-site`` still selects that site's
  launchers; with none, main's detection of the active Slurm cluster picks one,
  and ``preparation.json`` must record the site *that* launcher declares, or
  ``validate_study_execution_site`` refuses to start the worker;
* controller pools: ``controller_fact_pool_mode`` (task_004 suites) and
  ``controller_report_pool_mode`` (target-aligned truth arms) are separate
  options behind one resolver, and setting both is rejected.
"""

import json
import pathlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    RelationalRoundBudgetedControl,
)
from mas_cc.studies.runtime import validate_study_execution_site
from mas_cc.studies.submission import prepare_study
from tests.mas_cc.test_studies import _standalone_config


def _study(tmp_path, monkeypatch):
    _standalone_config(tmp_path / "config.yaml", name="merged")
    (tmp_path / "study.yaml").write_text(
        "study: {name: merged}\nconfigs: [config.yaml]\n", encoding="utf-8"
    )

    def fake_preflight(config, output):
        Path(output).mkdir(parents=True)
        return SimpleNamespace(launch_status="permitted")

    monkeypatch.setattr("mas_cc.cli.experiment.run_experiment_preflight", fake_preflight)


def _pretend_cluster(monkeypatch, name):
    monkeypatch.setattr("mas_cc.studies.site.active_cluster", lambda: name)
    monkeypatch.setattr("mas_cc.studies.submission.active_cluster", lambda: name)


@pytest.mark.parametrize(
    "cluster, launcher_dir, recorded, worker_env",
    [
        ("cygnus", "scripts/Cygnus/SLURM", "unspecified", None),
        (None, "scripts/Potsdam/SLURM", "potsdam", "potsdam"),
    ],
)
def test_no_site_uses_the_active_cluster_and_records_what_its_launcher_declares(
    tmp_path, monkeypatch, cluster, launcher_dir, recorded, worker_env
):
    _study(tmp_path, monkeypatch)
    _pretend_cluster(monkeypatch, cluster)
    result = prepare_study(tmp_path, tmp_path / "results", throttle=1)

    assert Path(result.command[-2]).parent.as_posix().endswith(launcher_dir)
    preparation = json.loads((result.study_dir / "preparation.json").read_text())
    assert preparation["execution_site"] == recorded
    # The worker launched by that script must accept the preparation.
    if worker_env is None:
        monkeypatch.delenv("MAS_CC_EXECUTION_SITE", raising=False)
    else:
        monkeypatch.setenv("MAS_CC_EXECUTION_SITE", worker_env)
    validate_study_execution_site(result.manifest_path)


def test_an_explicit_site_wins_over_cluster_detection(tmp_path, monkeypatch):
    _study(tmp_path, monkeypatch)
    _pretend_cluster(monkeypatch, "cygnus")
    monkeypatch.setenv("CESAR_RESULTS_ROOT", str(tmp_path))
    result = prepare_study(
        tmp_path, tmp_path / "results", throttle=1, execution_site="cesar"
    )

    assert Path(result.command[-2]).parent.as_posix().endswith("scripts/Cesar/SLURM")
    preparation = json.loads((result.study_dir / "preparation.json").read_text())
    assert preparation["execution_site"] == "cesar"


TASKS = pathlib.Path("results/studies/musr_truthful_selective_task_calibration_01/tasks")
OPTIONS = {
    "target": "ALLOCATION_2",
    "sensor_sample_size": 100,
    "sensing_mode": "board",
    "policy": "soft_target",
    "threshold": 0.5,
    "beta": 4.0,
    "intervention_budget": 90,
    "controller_budget_scope": "episode",
    "advocacy_schedule": "always",
    "message_mode": "recommendation_only",
    "controller_actuation_mode": "adaptive_communication",
    "controller_timing": "dawn_only",
    "controller_authoring": "llm_authored",
    "controller_report_cooldown_rounds": 0,
    "controller_report_max_posts_per_fact": 90,
    "controller_report_selection_strategy": "target_preserving_v1",
    "template_version": 3,
}


def test_both_pool_options_away_from_frozen_are_rejected():
    with pytest.raises(Exception, match="choose one pool option"):
        RelationalRoundBudgetedControl.from_options(
            {
                **OPTIONS,
                "controller_fact_pool_mode": "balanced",
                "controller_report_pool_mode": "target_aligned_v1",
            }
        )


@pytest.mark.skipif(
    not (TASKS / "task_004/controller/balanced_fact_pool.json").is_file(),
    reason="task_004 fixture not present",
)
@pytest.mark.parametrize("mode", ["frozen", "all_nondecisive", "balanced"])
def test_the_single_entry_point_serves_every_shared_pool_mode(mode):
    task = load_musr_team_allocation_task(TASKS, "task_004", population_size=15)
    control = RelationalRoundBudgetedControl.from_options(
        {**OPTIONS, "controller_fact_pool_mode": mode}
    )
    assert control.reportable_fact_ids_for_target(task, 1) == control.reportable_fact_ids(
        task
    )
