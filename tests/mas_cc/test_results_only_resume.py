from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import parse_run_config, resolved_config_yaml, load_run_config
from mas_cc.experiments import run_experiment_sync
from mas_cc.experiments.comet_monitor import MasterMonitor
from mas_cc.experiments.orchestrator import _pricing_identity
from mas_cc.games.hidden_bench.imitation.analysis import adapt_event, read_imitation_events
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers import (
    AtomicBudgetStateStore,
    BudgetLimits,
    MonetaryAmount,
    RuntimeBudgetGuard,
)
from mas_cc.metrics.aggregate import excluded_cell_episodes, read_cell_episodes
from mas_cc.storage import (
    ScientificIdentity,
    compact_run_directory,
    compact_imitation_event,
    discover_episode_artifact,
    episode_shard_path,
    write_completed_episode,
)


def _raw_config() -> dict:
    return load_run_config(
        "configs/runs/old/toy_game_smoke_test.yaml", environment={}
    ).to_dict()


def _results_only_config(*, repetitions: int = 3):
    config = load_run_config(
        "configs/runs/old/toy_game_smoke_test.yaml", environment={}
    )
    return replace(
        config,
        execution=replace(config.execution, repetitions=repetitions, parallelism=2),
        logging=replace(
            config.logging,
            comet=False,
            options={"prompt_examples": {"count": 0, "scope": "cell"}},
        ),
        storage=replace(
            config.storage,
            artifact_profile="results_only",
            checkpoint_mode="episode",
        ),
        budget=replace(
            config.budget,
            max_provider_requests=10_000,
            max_input_tokens=10_000_000,
            max_output_tokens=100_000,
            max_cost_per_run=100.0,
            system_max_cost_per_run=100.0,
        ),
    )


def test_storage_contract_aliases_and_resolved_prompt_scope():
    raw = _raw_config()
    raw["storage"] = {"artifact_profile": "results_only", "checkpoints": True}
    raw["logging"]["options"]["prompt_examples"] = {"count": 2}
    config = parse_run_config(raw)
    assert config.storage.artifact_profile == "results_only"
    assert config.storage.checkpoint_mode == "episode"
    assert config.logging.options["prompt_examples"] == {"count": 2, "scope": "cell"}
    rendered = resolved_config_yaml(config)
    assert "checkpoint_mode: episode" in rendered
    assert "checkpoints:" not in rendered

    raw["storage"]["checkpoint_mode"] = "episode"
    with pytest.raises(ConfigurationError, match="cannot both"):
        parse_run_config(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [("artifact_profile", "tiny"), ("checkpoint_mode", "round")],
)
def test_unknown_storage_modes_fail_at_the_precise_field(field, value):
    raw = _raw_config()
    raw["storage"][field] = value
    raw["storage"].pop("checkpoints", None)
    with pytest.raises(ConfigurationError, match=f"storage.{field}"):
        parse_run_config(raw)


def _event(before, after):
    options = ("A", "B", "C")
    return {
        "episode_id": "episode-0",
        "interaction_index": 1,
        "N": len(before),
        "K": len(options),
        "dynamics_mode": "classical",
        "possible_answers": list(options),
        "correct_answer": "C",
        "analysis_target": "C",
        "population_state_before": list(before),
        "occupation_counts_before": {item: before.count(item) for item in options},
        "population_state_after": list(after),
        "occupation_counts_after": {item: after.count(item) for item in options},
        "focal_opinion_before": before[0],
        "focal_opinion_after": after[0],
        "controller_action": "ADVOCATE_Z",
        "controller_policy": "threshold_target",
        "sensor_sample_size": len(before),
        "sensor_count_vector": {item: before.count(item) for item in options},
    }


def test_compact_imitation_adapter_preserves_the_exact_scientific_encoding(tmp_path):
    event = _event(["A", "B", "C"], ["C", "B", "C"])
    identity = ScientificIdentity(
        "run", "run", "episode-0", 7, "config", "prompts", "pricing",
        "hidden_bench_imitation", "classical", "threshold_target", "task",
    )
    row = compact_imitation_event(event, identity)
    row.update(
        population_metrics_json="{}", option_metrics_json="{}", final_metrics_json="{}"
    )
    write_completed_episode(
        tmp_path / "scientific_events.parquet",
        [row],
        identity,
        termination_reason="done",
        started_at="2026-08-12T00:00:00Z",
    )
    compact = read_imitation_events(tmp_path)[0]
    rich = adapt_event(event, episode_id="episode-0")
    for field in (
        "N_t", "N_t1", "Y_t", "U_t", "Z_t", "Z_t1", "Mtruth_t",
        "Mtruth_t1", "Morder_t", "Morder_t1", "Xf_t", "Xf_t1",
    ):
        assert getattr(compact, field) == getattr(rich, field)
    assert compact.event["m_ctrl_before"] == rich.event["m_ctrl_before"]
    assert compact.event["population_shares_after"] == rich.event["population_shares_after"]


def test_results_only_run_seals_without_verbose_files_and_resumes_before_provider(
    tmp_path: Path, monkeypatch
):
    config = _results_only_config(repetitions=3)
    first = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    assert first.completed == 3
    assert (first.output_dir / "scientific_events.parquet").is_file()
    assert (first.output_dir / "cell_complete.json").is_file()
    run_manifest = json.loads((first.output_dir / "manifest.json").read_text())
    cell_seal = json.loads((first.output_dir / "cell_complete.json").read_text())
    assert "scientific_events.parquet" in run_manifest["artifacts"]
    assert "scientific_events.parquet" in cell_seal["artifacts"]
    forbidden = {
        "trajectory.jsonl", "events.jsonl", "experiment.log", "budget_events.jsonl",
        "usage_cost.jsonl", "api_call_status.jsonl", "streaming.csv", "local_metrics.csv",
        "checkpoint_manifest.json",
    }
    assert not forbidden.intersection(
        path.name for path in first.output_dir.rglob("*") if path.is_file()
    )

    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator.create_llm_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be constructed for a sealed cell")
        ),
    )
    second = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    assert second.completed == 0
    assert second.skipped_resumed == 3


def test_budget_restart_charges_dead_reservations_and_never_restores_them(tmp_path):
    limits = BudgetLimits(
        max_requests=10, max_input_tokens=100, max_output_tokens=100,
        allow_unbounded_paid_requests=True,
    )
    first = RuntimeBudgetGuard(limits)
    store = AtomicBudgetStateStore(
        tmp_path / "budget_state.json",
        resolved_budget_hash="budget",
        pricing_snapshot_hash="pricing",
    )
    first.set_durable_state_sink(store.write)
    first.reserve(conservative_cost=None, input_tokens=25, output_tokens=30)

    resumed = RuntimeBudgetGuard(limits)
    assert store.restore(resumed)
    status = resumed.status()
    assert status["active_reservations"] == 0
    assert status["used_and_reserved"]["requests"] == 1
    assert status["used_and_reserved"]["input_tokens"] == 25
    assert status["used_and_reserved"]["output_tokens"] == 30


def test_budget_state_is_atomic_under_concurrent_reconciliation(tmp_path):
    limits = BudgetLimits(
        max_requests=100,
        max_input_tokens=10_000,
        max_output_tokens=10_000,
        allow_unbounded_paid_requests=True,
    )
    guard = RuntimeBudgetGuard(limits)
    store = AtomicBudgetStateStore(
        tmp_path / "budget_state.json",
        resolved_budget_hash="budget",
        pricing_snapshot_hash="pricing",
    )
    guard.set_durable_state_sink(store.write)

    def complete_one(_index):
        reservation = guard.reserve(
            conservative_cost=None, input_tokens=10, output_tokens=5
        )
        guard.reconcile(
            reservation, actual_cost=None, input_tokens=8, output_tokens=3
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(complete_one, range(32)))

    restored = RuntimeBudgetGuard(limits)
    assert store.restore(restored)
    status = restored.status()
    assert status["active_reservations"] == 0
    assert status["used_and_reserved"]["requests"] == 32
    assert status["used_and_reserved"]["input_tokens"] == 32 * 8
    assert status["used_and_reserved"]["output_tokens"] == 32 * 3


def test_authoritative_spend_cannot_reduce_the_conservative_restart_debit(tmp_path):
    def money(amount):
        return MonetaryAmount(
            amount,
            "USD",
            "test",
            "mock",
            "model",
            "test",
            "2026-08-12T00:00:00Z",
            "v1",
        )

    limits = BudgetLimits(max_cost=money(10), allow_unbounded_paid_requests=True)
    guard = RuntimeBudgetGuard(limits)
    store = AtomicBudgetStateStore(
        tmp_path / "budget_state.json",
        resolved_budget_hash="budget",
        pricing_snapshot_hash="pricing",
    )
    guard.set_durable_state_sink(store.write)
    guard.reserve(conservative_cost=money(2), input_tokens=1, output_tokens=1)

    lower = RuntimeBudgetGuard(limits)
    assert store.restore(lower, authoritative_cost=money(1))
    assert lower.status()["used_and_reserved"]["cost"]["amount"] == 2

    higher = RuntimeBudgetGuard(limits)
    assert store.restore(higher, authoritative_cost=money(3))
    assert higher.status()["used_and_reserved"]["cost"]["amount"] == 3


def test_pricing_identity_ignores_retrieval_time_but_not_rate_changes():
    quote = load_run_config(
        "configs/runs/old/toy_game_smoke_test.yaml", environment={}
    )
    from mas_cc.llm_runtime.providers import OfflinePricingSource

    first = OfflinePricingSource().fetch(
        quote.llm_provider.type, quote.llm_provider.model
    )
    later = replace(
        first,
        retrieved_at="2099-01-01T00:00:00Z",
        fresh_until="2099-01-02T00:00:00Z",
    )
    assert _pricing_identity(first) == _pricing_identity(later)
    assert first.pricing is not None
    changed = replace(
        later,
        pricing=replace(
            later.pricing,
            ordinary_input_per_million=(
                later.pricing.ordinary_input_per_million or 0
            )
            + 1,
        ),
    )
    assert _pricing_identity(first) != _pricing_identity(changed)


def test_temporary_episode_shard_is_not_a_resume_checkpoint(tmp_path, monkeypatch):
    identity = ScientificIdentity(
        "run", "run", "episode-0", 7, "config", "prompts", "pricing",
        "hidden_bench_imitation", "classical", "threshold_target", "task",
    )
    row = compact_imitation_event(_event(["A", "B", "C"], ["C", "B", "C"]), identity)
    destination = episode_shard_path(tmp_path, identity.episode_id)

    def interrupted_rename(_source, _destination):
        raise OSError("simulated interruption before publication")

    monkeypatch.setattr("mas_cc.storage.scientific.os.replace", interrupted_rename)
    with pytest.raises(OSError, match="simulated interruption"):
        write_completed_episode(
            destination,
            [row],
            identity,
            termination_reason="done",
            started_at="2026-08-12T00:00:00Z",
        )

    assert not destination.exists()
    assert destination.with_name(destination.name + ".tmp").is_file()
    assert discover_episode_artifact(tmp_path, identity) is None


def test_changed_config_is_rejected_before_provider_construction(tmp_path, monkeypatch):
    config = _results_only_config(repetitions=1)
    run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    incompatible = replace(
        config,
        game=replace(config.game, horizon=config.game.horizon + 1),
    )
    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator.create_llm_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be constructed for an incompatible checkpoint")
        ),
    )
    with pytest.raises(ValueError, match="resolved_config_hash"):
        run_experiment_sync(incompatible, tmp_path, resume=True, show_progress=False)


def test_failed_episode_is_excluded_and_does_not_seal_the_cell(tmp_path, monkeypatch):
    config = _results_only_config(repetitions=2)
    config = replace(
        config,
        execution=replace(config.execution, parallelism=1, fail_fast=False),
    )
    from mas_cc.games.runner import run_game as real_run_game

    calls = 0

    async def fail_once(game, episode_config, provider):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated infrastructure failure")
        return await real_run_game(game, episode_config, provider)

    monkeypatch.setattr("mas_cc.experiments.orchestrator.run_game", fail_once)
    result = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    assert [outcome.status for outcome in result.outcomes] == ["failed", "completed"]
    assert not (result.output_dir / "cell_complete.json").exists()
    assert len(read_cell_episodes(result.output_dir)) == 1
    assert excluded_cell_episodes(result.output_dir) == (
        (result.outcomes[0].episode_id, "failed"),
    )


def test_cell_prompt_sampling_writes_two_examples_in_one_file(tmp_path):
    config = load_run_config(
        "configs/runs/old/naming_convention_smoke_test.yaml", environment={}
    )
    config = replace(
        config,
        game=replace(config.game, horizon=4),
        execution=replace(config.execution, repetitions=5, parallelism=3),
        logging=replace(
            config.logging,
            comet=False,
            options={"prompt_examples": {"count": 2, "scope": "cell"}},
        ),
        storage=replace(
            config.storage,
            artifact_profile="results_only",
            checkpoint_mode="episode",
        ),
        budget=replace(
            config.budget,
            max_provider_requests=1_000,
            max_input_tokens=1_000_000,
            max_output_tokens=10_000,
        ),
    )
    result = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    prompt_path = result.output_dir / "prompt_examples.md"
    rendered = prompt_path.read_text(encoding="utf-8")
    assert rendered.count("## Example ") == 2
    assert result.outcomes[0].episode_id in rendered
    assert not list(result.output_dir.rglob("prompt_candidates.json.gz"))
    assert list(result.output_dir.rglob("*.md")) == [prompt_path]


def test_full_and_results_only_send_the_same_master_comet_calls(tmp_path, monkeypatch):
    class Sink:
        instances = []

        def __init__(self, enabled, *, project_name, run_name):
            self.enabled = enabled
            self.project_name = project_name
            self.run_name = run_name
            self.status = "active" if enabled else "disabled"
            self.reason = None
            self.reference = f"ref:{run_name}"
            self.url = f"https://comet.invalid/{run_name}"
            self.metrics = []
            self.parameters = {}
            self.tags = []
            self.figures = []
            self.images = []
            self.closed = False
            type(self).instances.append(self)

        def log_metrics(self, metrics, step):
            self.metrics.append((dict(metrics), step))

        def log_parameters(self, parameters):
            self.parameters.update(parameters)

        def add_tags(self, tags):
            self.tags.extend(tags)

        def log_figure(self, _figure, *, name, step):
            self.figures.append((name, step))

        def log_image(self, _path, *, name, step):
            self.images.append((name, step))

        def close(self):
            self.closed = True
            return {
                "status": self.status,
                "reference": self.reference,
                "url": self.url,
                "reason": None,
            }

    monkeypatch.setattr("mas_cc.experiments.comet_monitor.CometMetricSink", Sink)

    def deterministic_monitor(config, run_id, total_episodes, layout=None):
        return MasterMonitor(
            config.logging.comet,
            project_name="mas-cc",
            run_name=run_id,
            layout=layout,
            total_episodes=total_episodes,
            settings=config.observability.comet,
            now=lambda: 100.0,
        )

    monkeypatch.setattr(
        "mas_cc.experiments.orchestrator._master_monitor", deterministic_monitor
    )
    base = load_run_config(
        "configs/runs/old/naming_convention_smoke_test.yaml", environment={}
    )
    base = replace(
        base,
        game=replace(base.game, horizon=3),
        execution=replace(base.execution, repetitions=2, parallelism=2),
        logging=replace(
            base.logging,
            comet=True,
            options={"prompt_examples": {"count": 0, "scope": "cell"}},
        ),
    )

    def snapshot(profile):
        Sink.instances = []
        config = replace(
            base,
            storage=replace(
                base.storage,
                artifact_profile=profile,
                checkpoint_mode="episode",
            ),
        )
        run_experiment_sync(
            config, tmp_path / profile, resume=True, show_progress=False
        )
        return [
            {
                "run_name": sink.run_name,
                "metrics": sink.metrics,
                "parameters": sink.parameters,
                "tags": sink.tags,
                "figures": sink.figures,
                "images": sink.images,
                "closed": sink.closed,
            }
            for sink in Sink.instances
        ]

    assert snapshot("full") == snapshot("results_only")


def test_legacy_compactor_is_preview_safe_destructive_and_idempotent(tmp_path):
    config = load_run_config(
        "configs/runs/old/naming_convention_smoke_test.yaml", environment={}
    )
    config = replace(
        config,
        game=replace(config.game, horizon=3),
        execution=replace(config.execution, repetitions=2, parallelism=2),
        logging=replace(
            config.logging,
            comet=False,
            options={"prompt_examples": {"count": 0, "scope": "cell"}},
        ),
        storage=replace(
            config.storage, artifact_profile="full", checkpoint_mode="episode"
        ),
    )
    result = run_experiment_sync(config, tmp_path, resume=True, show_progress=False)
    scientific_before = read_cell_episodes(result.output_dir)
    unrelated_result = result.output_dir / "final.csv"
    unrelated_result.write_text("must,remain\n", encoding="utf-8")
    before = {
        str(path.relative_to(result.output_dir)): path.read_bytes()
        for path in result.output_dir.rglob("*")
        if path.is_file()
    }

    preview = compact_run_directory(result.output_dir)
    assert preview["dry_run"] is True
    assert {
        str(path.relative_to(result.output_dir)): path.read_bytes()
        for path in result.output_dir.rglob("*")
        if path.is_file()
    } == before

    compacted = compact_run_directory(result.output_dir, delete_raw=True)
    assert compacted["dry_run"] is False
    assert (result.output_dir / "scientific_events.parquet").is_file()
    assert (result.output_dir / "cell_complete.json").is_file()
    assert read_cell_episodes(result.output_dir) == scientific_before
    assert unrelated_result.read_text(encoding="utf-8") == "must,remain\n"
    count_after_first = compacted["after"]["files"]
    repeated = compact_run_directory(result.output_dir, delete_raw=True)
    assert repeated["after"]["files"] == count_after_first
    (result.output_dir / "scientific_events.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        compact_run_directory(result.output_dir, delete_raw=True)
