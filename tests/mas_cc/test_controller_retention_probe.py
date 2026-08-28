"""Tests for the focused controller-retention local probe."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mas_cc.games.relational_reasoning.data import (
    list_relational_task_ids,
    load_relational_task,
)
from mas_cc.probes.controller_retention import vignette as vignette_module
from mas_cc.probes.controller_retention.analysis import (
    build_response_rows,
    cell_effects,
    effect_index,
    pair_outcomes,
    r_exposure,
)
from mas_cc.probes.controller_retention.config import (
    ProbeConfigError,
    build_probe_config,
    load_probe_config,
)
from mas_cc.probes.controller_retention.design import (
    ARMS_BY_Q,
    NO_OP,
    ONE_SLOT,
    TWO_SLOTS,
    DesignSpec,
    build_vignettes,
    controller_slot_count,
)
from mas_cc.probes.controller_retention.execution import build_call_specs
from mas_cc.probes.controller_retention.preflight import (
    preflight_payload,
    run_preflight,
)
from mas_cc.probes.controller_retention.runner import (
    PAIRED_EFFECTS_CSV,
    REPORT_MARKDOWN,
    analyze,
    prepare,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    REPO_ROOT
    / "src/mas_cc/relational_task_generator/relational_task_generator/datasets"
)
L1_DIR = DATASETS / "probe_n12_L1_k3"
L2_DIR = DATASETS / "probe_n12_L2_k3"
CONFIG_PATH = REPO_ROOT / "configs/probes/controller_retention.yaml"


@pytest.fixture(scope="module")
def tasks() -> dict[int, tuple]:
    return {
        depth: tuple(
            load_relational_task(directory, task_id)
            for task_id in list_relational_task_ids(directory)[:2]
        )
        for depth, directory in ((1, L1_DIR), (2, L2_DIR))
    }


def _grid(tasks) -> tuple:
    fingerprints = {
        (
            task.reasoning_depth,
            task.task_id,
        ): f"fp-{task.reasoning_depth}-{task.task_id}"
        for group in tasks.values()
        for task in group
    }
    return build_vignettes(DesignSpec(seed=7, tasks_per_depth=2), tasks, fingerprints)


def _task_for(tasks, item):
    return next(
        task for task in tasks[item.reasoning_depth] if task.task_id == item.task_id
    )


def _config_payload(model_count: int = 1):
    return {
        "models": [
            {
                "label": f"m{i}",
                "provider": "mock",
                "model": f"model-{i}",
                "max_retries": 0,
            }
            for i in range(model_count)
        ],
        "tasks": {
            "dataset_dirs": {1: str(L1_DIR), 2: str(L2_DIR)},
            "tasks_per_depth": 12,
        },
        "design": {
            "seed": 3,
            "reasoning_depths": [1, 2],
            "q_values": [2, 3],
            "receivers": ["naive"],
            "targets": ["truth", "false"],
            "replicates": 1,
        },
        "execution": {"workers": 2, "backend": "serial", "max_retries": 0},
        "storage": {"output_dir": "results/probes/test"},
    }


def test_shipped_config_is_one_model_and_240_calls():
    config = load_probe_config(CONFIG_PATH)
    assert [model.label for model in config.models] == ["gpt_oss_120b"]
    preflight = run_preflight(config)
    payload = preflight_payload(config, preflight)
    assert payload["passed"]
    assert len(preflight.vignettes) == 96
    assert payload["calls"]["calls_per_model"] == 240
    assert payload["calls"]["calls_total"] == 240
    assert payload["calls"]["calls_q2"] == 96
    assert payload["calls"]["calls_q3"] == 144


def test_config_accepts_any_nonempty_model_count():
    assert len(build_probe_config(_config_payload(1)).models) == 1
    assert len(build_probe_config(_config_payload(4)).models) == 4
    with pytest.raises(ProbeConfigError, match="at least one"):
        build_probe_config(_config_payload(0))


@pytest.mark.parametrize(
    "q,arm,expected",
    [
        (2, NO_OP, 0),
        (2, ONE_SLOT, 1),
        (3, NO_OP, 0),
        (3, ONE_SLOT, 1),
        (3, TWO_SLOTS, 2),
    ],
)
def test_controller_slot_counts(q, arm, expected):
    assert controller_slot_count(q, arm) == expected


def test_shared_vignette_has_exact_requested_arms(tasks):
    for item in _grid(tasks):
        task = _task_for(tasks, item)
        assert item.message_mode == "recommendation_only"
        assert item.receiver_epistemic_disposition == "naive"
        baseline = vignette_module.rendered_blocks(task, item, NO_OP)
        for arm in ARMS_BY_Q[item.q]:
            sources = vignette_module.social_sources(task, item, arm)
            assert len(sources) == item.q
            assert sum(
                source["source_type"] == "control" for source in sources
            ) == controller_slot_count(item.q, arm)
            rendered = vignette_module.rendered_blocks(task, item, arm)
            differing = {name for name in baseline if baseline[name] != rendered[name]}
            assert differing == (set() if arm == NO_OP else {"social_information"})


def test_q3_two_slots_adds_one_replacement(tasks):
    item = next(item for item in _grid(tasks) if item.q == 3)
    task = _task_for(tasks, item)
    one = vignette_module.social_sources(task, item, ONE_SLOT)
    two = vignette_module.social_sources(task, item, TWO_SLOTS)
    assert one[0]["source_type"] == two[0]["source_type"] == "control"
    assert one[1]["source_type"] == "ordinary"
    assert two[1]["source_type"] == "control"
    assert one[2] == two[2]


def test_call_specs_share_one_q3_noop_and_are_deterministic():
    config = build_probe_config(_config_payload())
    preflight = run_preflight(config)
    specs = build_call_specs(config, preflight.vignettes)
    assert len(specs) == 240
    assert len({spec.call_id for spec in specs}) == 240
    assert sum(spec.vignette.q == 3 and spec.arm == NO_OP for spec in specs) == 48
    assert [spec.call_id for spec in specs] == [
        spec.call_id
        for spec in build_call_specs(config, run_preflight(config).vignettes)
    ]


def test_response_contract_rejects_incomplete_json(tasks, monkeypatch):
    from mas_cc.llm_runtime.providers.capabilities import ProviderCapabilities
    from mas_cc.llm_runtime.providers.responses import CompletionResponse, ProviderUsage
    from mas_cc.probes.controller_retention import execution

    class Provider:
        name = "mock"
        capabilities = ProviderCapabilities()

        async def complete(self, request):
            return CompletionResponse(
                content='{"vote":"A"}',
                provider="mock",
                model="m",
                usage=ProviderUsage(input_tokens=1, output_tokens=1),
            )

    config = build_probe_config(_config_payload())
    preflight = run_preflight(config)
    spec = build_call_specs(config, preflight.vignettes)[0]
    monkeypatch.setattr(execution, "_worker_provider", lambda model: Provider())
    result = execution.execute_call(spec)
    assert not result.parse_ok
    assert result.provider_error is None
    assert result.validation_error


def test_analysis_uses_shared_noop_and_computes_rescue(tmp_path):
    config = build_probe_config(_config_payload())
    preflight = run_preflight(config)
    selected = [
        item
        for item in preflight.vignettes
        if item.reasoning_depth == 1 and item.target_semantics == "false"
    ][:12]
    raw = []
    model = config.models[0]
    for item in selected:
        for arm in ARMS_BY_Q[item.q]:
            hit = arm == TWO_SLOTS or (arm == ONE_SLOT and item.q == 2)
            raw.append(
                {
                    "call_id": item.call_id(model.call_identity, arm),
                    "model_label": model.label,
                    "arm": arm,
                    "vignette_id": item.vignette_id,
                    "final_vote_semantic": item.controller_target_semantic
                    if hit
                    else item.truth_semantic,
                    "parse_ok": True,
                    "latency": 0.1,
                }
            )
    rows = build_response_rows(config, preflight.vignettes, preflight.tasks, raw)
    effects = cell_effects(pair_outcomes(rows))
    index = effect_index(effects)
    q2 = index[(model.label, 1, "false", 2, ONE_SLOT)]
    q3_one = index[(model.label, 1, "false", 3, ONE_SLOT)]
    q3_two = index[(model.label, 1, "false", 3, TWO_SLOTS)]
    assert q2.delta_c == 1.0
    assert q3_one.delta_c == 0.0
    assert q3_two.delta_c == 1.0
    assert r_exposure(index, q3_two.key) == 1.0


def test_mock_end_to_end_report(tmp_path, monkeypatch):
    from mas_cc.probes.controller_retention import execution, runner

    config = build_probe_config(_config_payload())
    run = prepare(config, tmp_path)

    def fake_execute(spec, max_retries=0):
        vote = (
            spec.vignette.controller_target_semantic
            if spec.arm != NO_OP
            else spec.vignette.truth_semantic
        )
        return execution.CallResult(
            call_id=spec.call_id,
            model_label=spec.model.label,
            arm=spec.arm,
            vignette_id=spec.pair_id,
            response_text="{}",
            final_vote_semantic=vote,
            parse_ok=True,
            provider_error=None,
            validation_error=None,
            attempts=1,
            latency=0.01,
            input_tokens=10,
            output_tokens=5,
        )

    monkeypatch.setattr(
        execution, "_worker_entry", lambda payload: fake_execute(*payload)
    )
    runner.execute(run, stream=None)
    runner.analyze(run)
    assert run.completed_successfully
    report = (tmp_path / REPORT_MARKDOWN).read_text(encoding="utf-8")
    assert "Cross-model summary" in report
    assert "Model 1: `m0`" in report
    assert "Delta_C q=3 two-slots" in report
    assert "Exposure rescue" in report
    rows = list(csv.DictReader((tmp_path / PAIRED_EFFECTS_CSV).open(encoding="utf-8")))
    assert len(rows) == 12


def test_failed_rows_are_not_completed(tmp_path):
    from mas_cc.probes.controller_retention.execution import (
        RAW_CALLS_FILENAME,
        successful_call_ids,
    )

    path = tmp_path / RAW_CALLS_FILENAME
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "ok",
                        "parse_ok": True,
                        "provider_error": None,
                        "validation_error": None,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "provider",
                        "parse_ok": False,
                        "provider_error": "timeout",
                    }
                ),
                json.dumps(
                    {
                        "call_id": "invalid",
                        "parse_ok": False,
                        "validation_error": "bad JSON",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert successful_call_ids(path) == {"ok"}


def test_one_plot_per_model(tmp_path):
    pytest.importorskip("matplotlib")
    from mas_cc.probes.controller_retention.analysis import CellEffect
    from mas_cc.probes.controller_retention.plots import render_all

    effects = [
        CellEffect(("m0", depth, target, q, exposure), 12, 0.7, 0.2, 0.5)
        for depth in (1, 2)
        for target in ("truth", "false")
        for q, exposures in ((2, (ONE_SLOT,)), (3, (ONE_SLOT, TWO_SLOTS)))
        for exposure in exposures
    ]
    produced = render_all(effects, ["m0"], tmp_path)
    assert len(produced["m0"]) == 1
    assert produced["m0"][0].is_file()


def test_the_probe_does_not_modify_the_production_game():
    import inspect
    from mas_cc.games.relational_reasoning.imitation_round_feedback import (
        controller,
        runtime,
    )

    assert controller.MESSAGE_MODES == (
        "recommendation_only",
        "recommendation_plus_fact",
        "silent",
    )
    signature = inspect.signature(runtime.build_social_sources)
    assert "replaced_peer_slot" in signature.parameters
    assert "controller_slots" not in signature.parameters
