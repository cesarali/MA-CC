import asyncio
import csv
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import MetricsConfig, load_run_config
from mas_cc.core import AgentId
from mas_cc.games import create_game
from mas_cc.games.naming_convention import (
    METRICS,
    build_metrics,
    run_naming_convention_game_sync,
    to_round_view,
)
from mas_cc.games.naming_convention.metrics import RollingActionShare, RollingCoordinationRate
from mas_cc.llm_providers import CompletionResponse, ProviderCapabilities, ProviderUsage
from mas_cc.metrics import (
    AgentAbsoluteError,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    MeanAbsoluteError,
    RoundView,
    ValueShare,
)
from mas_cc.observability import DetailedAuditPolicy, RunRecorder


def _view(values: dict[str, str | None], targets: dict[str, float] | None = None) -> RoundView:
    agent_values = {AgentId(agent_id): value for agent_id, value in values.items()}
    agent_targets = None if targets is None else {AgentId(agent_id): value for agent_id, value in targets.items()}
    return RoundView(agent_values=agent_values, agent_targets=agent_targets)


# --- generic metric library, against synthetic round views -----------------


def test_value_share_ignores_unset_agents():
    metric = ValueShare("Q")
    view = _view({"a": "Q", "b": "Q", "c": "M", "d": None})
    assert metric.compute_round(view) == {None: pytest.approx(2 / 3)}


def test_value_share_with_no_known_values_is_zero():
    metric = ValueShare("Q")
    view = _view({"a": None, "b": None})
    assert metric.compute_round(view) == {None: 0.0}


def test_agent_current_value_passes_through_per_agent():
    metric = AgentCurrentValue()
    view = _view({"a": "Q", "b": "M"})
    assert metric.compute_round(view) == {AgentId("a"): "Q", AgentId("b"): "M"}


def test_dominant_value_share_picks_the_most_common_value():
    metric = DominantValueShare()
    view = _view({"a": "Q", "b": "Q", "c": "M"})
    assert metric.compute_round(view) == {None: pytest.approx(2 / 3)}


def test_first_consensus_time_returns_first_round_crossing_threshold():
    metric = FirstConsensusTime(threshold=0.75)
    views = [
        _view({"a": "Q", "b": "M", "c": "M", "d": "Q"}),  # 2/4 = 0.5
        _view({"a": "Q", "b": "Q", "c": "M", "d": "Q"}),  # 3/4 = 0.75 -> round 2
    ]
    assert metric.compute_final(tuple(views)) == 2


def test_first_consensus_time_is_none_when_never_reached():
    metric = FirstConsensusTime(threshold=0.99)
    views = [_view({"a": "Q", "b": "M"})]
    assert metric.compute_final(tuple(views)) is None


def test_agent_absolute_error_requires_targets():
    metric = AgentAbsoluteError()
    with pytest.raises(ValueError, match="agent_targets"):
        metric.compute_round(_view({"a": 1.0}))


def test_agent_and_mean_absolute_error_against_targets():
    view = _view({"a": 3.0, "b": 7.0}, targets={"a": 5.0, "b": 5.0})
    assert AgentAbsoluteError().compute_round(view) == {AgentId("a"): 2.0, AgentId("b"): 2.0}
    assert MeanAbsoluteError().compute_round(view) == {None: 2.0}


# --- naming-convention adapter, against a real game trajectory -------------


class ScriptedProvider:
    name = "mock"
    model = "scripted-metrics-v1"
    capabilities = ProviderCapabilities(supports_seed=True, reports_usage=True)

    def __init__(self, script=None):
        self.script = script or {}
        self.calls: dict[str, int] = {}

    async def complete(self, request):
        agent = str(request.metadata["agent_id"])
        attempt = self.calls.get(agent, 0)
        self.calls[agent] = attempt + 1
        values = self.script.get(agent, ('{"value":"Q","reason":"default"}',))
        content = values[min(attempt, len(values) - 1)]
        await asyncio.sleep(0)
        return CompletionResponse(
            content=content, provider=self.name, model=self.model,
            usage=ProviderUsage(10, 4, 14), finish_reason="stop", request_id=f"{agent}-{attempt + 1}",
        )

    def close(self):
        pass


def _config(*, population_size=4, horizon=4, memory_size=3):
    config = load_run_config("configs/runs/naming_convention_smoke_test_v3.yaml", environment={})
    game = replace(
        config.game, population_size=population_size, horizon=horizon,
        options={**dict(config.game.options), "memory_size": memory_size},
    )
    return replace(config, game=game)


def _run_all_q():
    config = _config()
    game = create_game(config.game)
    provider = ScriptedProvider()  # every agent always answers "Q"
    return run_naming_convention_game_sync(game, config, provider)


def test_to_round_view_reads_committed_action_per_agent():
    # population_size=2 guarantees every interaction picks the same pair, so
    # both agents have a recorded action by the end of the run.
    config = _config(population_size=2, horizon=3)
    game = create_game(config.game)
    result = run_naming_convention_game_sync(game, config, ScriptedProvider())
    view = to_round_view(result.final_state)
    assert set(view.agent_values) == {agent.agent_id for agent in result.final_state.agents}
    assert all(value == "Q" for value in view.agent_values.values())


def test_to_round_view_leaves_agents_that_have_not_played_as_none():
    result = _run_all_q()  # population_size=4, horizon=4: not everyone is guaranteed to play
    view = to_round_view(result.final_state)
    assert set(view.agent_values) == {agent.agent_id for agent in result.final_state.agents}
    assert all(value in ("Q", None) for value in view.agent_values.values())


def test_naming_convention_metrics_reach_full_consensus_when_every_agent_plays_q():
    result = _run_all_q()
    views = tuple(to_round_view(interaction.transition.next_state) for interaction in result.interactions)
    metrics = build_metrics()
    share_q = next(m for m in metrics if m.name == "population_action_share_q")
    consensus = next(m for m in metrics if isinstance(m, FirstConsensusTime))
    assert share_q.compute_round(views[-1]) == {None: 1.0}
    assert consensus.compute_final(views) == 1  # every agent starts and stays on Q


def test_default_metrics_export_matches_module_level_instance():
    assert [m.name for m in METRICS] == [m.name for m in build_metrics()]


def test_to_round_view_carries_recent_history_for_rolling_metrics():
    result = _run_all_q()
    view = to_round_view(result.final_state)
    assert len(view.recent_history) == len(result.interactions)
    assert all(entry["success"] for entry in view.recent_history)  # every agent stays on Q


# --- rolling metrics, against synthetic interaction history ----------------

_HISTORY = (
    {"interaction_index": 1, "actions": ["Q", "M"], "success": False},
    {"interaction_index": 2, "actions": ["Q", "Q"], "success": True},
    {"interaction_index": 3, "actions": ["M", "M"], "success": True},
    {"interaction_index": 4, "actions": ["Q", "M"], "success": False},
    {"interaction_index": 5, "actions": ["M", "M"], "success": True},
)


def test_rolling_coordination_rate_uses_only_the_window_tail():
    view = RoundView(agent_values={}, recent_history=_HISTORY)
    # last 4 entries (indices 2-5): success = T, T, F, T -> 3/4
    assert RollingCoordinationRate(window=4).compute_round(view) == {None: 0.75}
    # full history (5 entries): F, T, T, F, T -> 3/5
    assert RollingCoordinationRate(window=100).compute_round(view) == {None: 0.6}


def test_rolling_action_share_counts_played_actions_not_interactions():
    view = RoundView(agent_values={}, recent_history=_HISTORY)
    # last 4 entries' actions: Q,Q, M,M, Q,M, M,M -> 8 values, 3 Q / 5 M
    assert RollingActionShare("Q", window=4).compute_round(view) == {None: 3 / 8}
    assert RollingActionShare("M", window=4).compute_round(view) == {None: 5 / 8}


def test_rolling_metrics_are_zero_with_no_history_yet():
    empty_view = RoundView(agent_values={})
    assert RollingCoordinationRate(window=4).compute_round(empty_view) == {None: 0.0}
    assert RollingActionShare("Q", window=4).compute_round(empty_view) == {None: 0.0}


def test_rolling_metrics_reject_nonpositive_window():
    with pytest.raises(ValueError):
        RollingCoordinationRate(window=0)
    with pytest.raises(ValueError):
        RollingActionShare("Q", window=0)


# --- RunRecorder integration -----------------------------------------------


def _recorder(tmp_path: Path, *, with_metrics: bool) -> RunRecorder:
    return RunRecorder(
        tmp_path, run_id="episode-1", resolved_config={"schema_version": 1},
        policy=DetailedAuditPolicy(enabled=False), comet_enabled=False,
        checkpoint_enabled=False,
        metrics=build_metrics() if with_metrics else (),
        to_round_view=to_round_view if with_metrics else None,
        comet_metric_export=("population_action_share_q",),
    )


def test_recorder_without_metrics_creates_no_metrics_directory(tmp_path: Path):
    recorder = _recorder(tmp_path, with_metrics=False)
    result = _run_all_q()
    for index, interaction in enumerate(result.interactions, start=1):
        recorder.record_interaction(
            round_index=index, interaction=interaction, budget_status={}, state={}, prompt_definitions={},
        )
    recorder.finalize(status="completed", budget_status={})
    assert not (tmp_path / "metrics").exists()


def test_recorder_with_metrics_writes_streaming_and_final_csv(tmp_path: Path):
    recorder = _recorder(tmp_path, with_metrics=True)
    result = _run_all_q()
    for index, interaction in enumerate(result.interactions, start=1):
        recorder.record_interaction(
            round_index=index, interaction=interaction, budget_status={}, state={}, prompt_definitions={},
        )
    recorder.finalize(status="completed", budget_status={})

    streaming_rows = list(csv.DictReader((tmp_path / "metrics" / "streaming.csv").open()))
    assert streaming_rows
    assert {row["metric_name"] for row in streaming_rows} == {m.name for m in build_metrics() if hasattr(m, "compute_round")}
    population_rows = [row for row in streaming_rows if row["agent_id"] == "" and row["metric_name"] == "population_action_share_q"]
    assert population_rows[-1]["value"] == "1.0"

    final_rows = {row["metric_name"]: row["value"] for row in csv.DictReader((tmp_path / "metrics" / "final.csv").open())}
    assert final_rows["first_consensus_time"] == "1"


# --- config -----------------------------------------------------------------


def test_metrics_config_section_parses_from_yaml():
    config = load_run_config("configs/runs/naming_convention_smoke_test_v3.yaml", environment={})
    assert config.metrics.enabled is True
    assert "population_action_share_q" in config.metrics.comet_export


def test_metrics_config_defaults_when_section_omitted():
    # This legacy (pre-v3) config file has no `metrics:` section at all.
    config = load_run_config("configs/runs/naming_convention_smoke_test.yaml", environment={})
    assert config.metrics.enabled is True
    assert config.metrics.comet_export == ()


def test_metrics_available_section_parses_from_yaml_and_selects_comet_export():
    config = load_run_config(
        "configs/runs/naming_convention_tutorial_university_v3.yaml", environment={}
    )
    assert config.metrics.available["population_action_share_q"]["comet"] is True
    assert config.metrics.available["rolling_action_share_q"]["comet"] is False
    names = config.metrics.comet_export_names()
    assert "population_action_share_q" in names
    assert "rolling_coordination_rate" in names
    assert "rolling_action_share_q" not in names
    assert "agent_current_action" not in names


def test_comet_export_names_unions_legacy_list_and_per_metric_available():
    config = MetricsConfig(
        comet_export=("legacy_metric",),
        available={
            "new_metric": {"comet": True},
            "quiet_metric": {"comet": False},
            "untouched_metric": {},
        },
    )
    assert config.comet_export_names() == ("legacy_metric", "new_metric")


def test_comet_export_names_is_empty_by_default():
    assert MetricsConfig().comet_export_names() == ()
