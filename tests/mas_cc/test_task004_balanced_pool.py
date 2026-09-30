import collections
import dataclasses
import json
import pathlib

import pytest

from mas_cc.games.relational_reasoning.data import load_musr_team_allocation_task
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    CONTROLLER_FACT_POOL_BALANCED,
    RelationalRoundBudgetedControl,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
    _initial_vote_task_projection,
)
from mas_cc.storage import canonical_hash

TASKS = pathlib.Path("results/studies/musr_truthful_selective_task_calibration_01/tasks")
POOL = TASKS / "task_004/controller/balanced_fact_pool.json"
# Initialization artifacts for the task_004 no-controller studies were frozen
# against this projection hash. The balanced pool is a controller-only field
# and must never move it, or those baselines stop pairing with new arms.
TASK004_INITIAL_VOTE_HASH = (
    "0093940c506f070d097c10820c225ccb0ee3e6ed019e511b8039e4f772e06310"
)

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

pytestmark = pytest.mark.skipif(not POOL.is_file(), reason="task_004 fixture not present")


@pytest.fixture(scope="module")
def task():
    return load_musr_team_allocation_task(TASKS, "task_004", population_size=15)


def _controller(mode):
    return RelationalRoundBudgetedControl.from_options(
        {**OPTIONS, "controller_fact_pool_mode": mode}
    )


def test_balanced_pool_leans_equally_toward_each_allocation(task):
    pool = json.loads(POOL.read_text())
    lean = {row["fact_id"]: row["lean"] for row in pool["facts"]}
    ids = _controller(CONTROLLER_FACT_POOL_BALANCED).reportable_fact_ids(task)
    assert list(ids) == pool["fact_ids"]
    assert collections.Counter(lean[i] for i in ids) == {
        "ALLOCATION_0": 9,
        "ALLOCATION_1": 9,
        "ALLOCATION_2": 9,
    }
    assert not set(ids) & set(task.decisive_fact_ids)


def test_balanced_pool_does_not_change_initialization_pairing(task):
    assert task.controller_balanced_fact_ids
    projection = _initial_vote_task_projection(task)
    assert "controller_balanced_fact_ids" not in projection
    assert canonical_hash(projection) == TASK004_INITIAL_VOTE_HASH


def test_other_pool_modes_are_unchanged(task):
    assert len(_controller("all_nondecisive").reportable_fact_ids(task)) == 43
    assert _controller("frozen").reportable_fact_ids(task) == (
        task.controller_reportable_fact_ids
    )


def test_balanced_mode_refuses_a_task_without_a_pool(task):
    bare = dataclasses.replace(task, controller_balanced_fact_ids=())
    with pytest.raises(ValueError, match="balanced_fact_pool.json"):
        _controller(CONTROLLER_FACT_POOL_BALANCED).reportable_fact_ids(bare)


def test_balanced_selections_stay_in_pool_and_say_so(task):
    controller = _controller(CONTROLLER_FACT_POOL_BALANCED)
    pool = set(controller.reportable_fact_ids(task))
    selections = controller.select_truthful_reports(
        task, episode_seed=1, round_index=3, live_fact_counts={}, selected_rounds={}
    )
    assert selections
    assert all(s.fact_id in pool for s in selections)
    assert {s.strategy_class for s in selections} == {"balanced"}


@pytest.mark.parametrize("target", ["ALLOCATION_2", "correct"])
def test_balanced_target_pool_keeps_only_the_targets_nine_facts(task, target):
    lean = {row["fact_id"]: row["lean"] for row in json.loads(POOL.read_text())["facts"]}
    controller = RelationalRoundBudgetedControl.from_options(
        {**OPTIONS, "target": target, "controller_fact_pool_mode": "balanced_target"}
    )
    resolved = controller.resolved_target_for_task(task, 1)
    ids = controller.reportable_fact_ids_for_target(task, 1)
    assert len(ids) == 9
    assert {lean[i] for i in ids} == {resolved}
    selections = controller.select_truthful_reports(
        task, episode_seed=1, round_index=3, live_fact_counts={}, selected_rounds={}
    )
    assert selections and all(s.fact_id in ids for s in selections)
    assert {s.strategy_class for s in selections} == {"balanced-target"}


def test_balanced_target_leans_do_not_change_initialization_pairing(task):
    assert task.controller_balanced_fact_leans
    assert canonical_hash(_initial_vote_task_projection(task)) == (
        TASK004_INITIAL_VOTE_HASH
    )
