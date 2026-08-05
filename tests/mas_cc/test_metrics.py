import asyncio
import csv
from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import ControlConfig, MetricsConfig, load_run_config
from mas_cc.core import AgentId
from mas_cc.control import create_control
from mas_cc.games import create_game, game_metrics
from mas_cc.games.naming_convention import (
    METRICS,
    build_metrics,
    run_naming_convention_game_sync,
    to_round_view,
)
from mas_cc.games.naming_convention.metrics import (
    RollingActionSharePerOption,
    RollingCoordinationRate,
)
from mas_cc.llm_runtime.providers import CompletionResponse, ProviderCapabilities, ProviderUsage
from mas_cc.metrics import (
    ActionSharePerOption,
    AgentAbsoluteError,
    AgentCurrentValue,
    DominantValueShare,
    FirstConsensusTime,
    MeanAbsoluteError,
    RoundView,
)
from mas_cc.metrics.interactions import InteractionOutcome, production_probabilities
from mas_cc.observability import DetailedAuditPolicy, RunRecorder


def _view(
    values: dict[str, str | None],
    targets: dict[str, float] | None = None,
    options: tuple[str, ...] = (),
) -> RoundView:
    agent_values = {AgentId(agent_id): value for agent_id, value in values.items()}
    agent_targets = None if targets is None else {AgentId(agent_id): value for agent_id, value in targets.items()}
    return RoundView(agent_values=agent_values, agent_targets=agent_targets, options=options)


# --- generic metric library, against synthetic round views -----------------


def test_action_share_per_option_ignores_unset_agents_and_sums_to_one():
    view = _view({"a": "Q", "b": "Q", "c": "M", "d": None}, options=("Q", "M"))
    shares = ActionSharePerOption().compute_round(view)
    assert shares == {"Q": pytest.approx(2 / 3), "M": pytest.approx(1 / 3)}
    assert sum(shares.values()) == pytest.approx(1.0)


def test_action_share_per_option_emits_every_declared_option_even_at_zero():
    view = _view({"a": "Q", "b": "Q"}, options=("Q", "M", "Z"))
    assert ActionSharePerOption().compute_round(view) == {"Q": 1.0, "M": 0.0, "Z": 0.0}


def test_action_share_per_option_with_no_known_values_is_zero():
    view = _view({"a": None, "b": None}, options=("Q", "M"))
    assert ActionSharePerOption().compute_round(view) == {"Q": 0.0, "M": 0.0}


def test_action_share_per_option_falls_back_to_observed_options():
    view = _view({"a": "Q", "b": "M"})  # no declared option set
    assert ActionSharePerOption().compute_round(view) == {"Q": 0.5, "M": 0.5}


def test_action_share_per_option_is_restricted_to_choice_games():
    assert ActionSharePerOption().requires_game_family == "choice"


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
    shares = next(m for m in metrics if m.name == "population_action_share_per_option")
    consensus = next(m for m in metrics if isinstance(m, FirstConsensusTime))
    assert shares.compute_round(views[-1]) == {"Q": 1.0, "M": 0.0}
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
    view = RoundView(agent_values={}, options=("Q", "M"), recent_history=_HISTORY)
    # last 4 entries' actions: Q,Q, M,M, Q,M, M,M -> 8 values, 3 Q / 5 M
    assert RollingActionSharePerOption(window=4).compute_round(view) == {"Q": 3 / 8, "M": 5 / 8}


def test_rolling_metrics_are_zero_with_no_history_yet():
    empty_view = RoundView(agent_values={}, options=("Q", "M"))
    assert RollingCoordinationRate(window=4).compute_round(empty_view) == {None: 0.0}
    assert RollingActionSharePerOption(window=4).compute_round(empty_view) == {"Q": 0.0, "M": 0.0}


def test_rolling_metrics_reject_nonpositive_window():
    with pytest.raises(ValueError):
        RollingCoordinationRate(window=0)
    with pytest.raises(ValueError):
        RollingActionSharePerOption(window=0)


# --- game family gating ------------------------------------------------------


def test_game_metrics_returns_the_naming_convention_set():
    config = _config()
    game = create_game(config.game)
    assert game.spec.game_family == "choice"
    metrics, adapter = game_metrics(game)
    assert [m.name for m in metrics] == [m.name for m in build_metrics()]
    assert adapter is to_round_view


def test_game_metrics_rejects_a_metric_from_another_family(monkeypatch):
    """A choice-only metric on a numeric game must fail at wiring, not at read time."""

    config = _config()
    game = create_game(config.game)
    monkeypatch.setattr(type(game), "spec", replace(game.spec, game_family="numeric"))
    with pytest.raises(ValueError, match="another family"):
        game_metrics(game)


# --- RunRecorder integration -----------------------------------------------


def _recorder(tmp_path: Path, *, with_metrics: bool) -> RunRecorder:
    return RunRecorder(
        tmp_path, run_id="episode-1", resolved_config={"schema_version": 1},
        policy=DetailedAuditPolicy(enabled=False), comet_enabled=False,
        checkpoint_enabled=False,
        metrics=build_metrics() if with_metrics else (),
        to_round_view=to_round_view if with_metrics else None,
        comet_metric_export=("population_action_share_per_option",),
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
    share_rows = [
        row for row in streaming_rows
        if row["metric_name"] == "population_action_share_per_option"
    ]
    # One row per option per round, with the option in the `series` column.
    assert {row["series"] for row in share_rows} == {"Q", "M"}
    assert {row["agent_id"] for row in share_rows} == {""}
    last_round = max(row["round_index"] for row in share_rows)
    final = {row["series"]: row["value"] for row in share_rows if row["round_index"] == last_round}
    assert final == {"Q": "1.0", "M": "0.0"}

    final_rows = {row["metric_name"]: row["value"] for row in csv.DictReader((tmp_path / "metrics" / "final.csv").open())}
    assert final_rows["first_consensus_time_by_action_share"] == "1"


def test_recorder_replaces_a_streaming_csv_left_by_an_older_metrics_schema(tmp_path: Path):
    """A rerun reuses its run directory; a pre-schema file must not be appended to.

    Appending 6-column rows under a 5-column header produced rows whose `value`
    field parsed as a metric name, which is what made plotting die with
    "could not convert string to float: 'population_action_share_per_option'".
    """

    stale = tmp_path / "metrics" / "streaming.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        "round_index,episode_id,agent_id,metric_name,value\n"
        "1,old-run,,population_action_share_q,0.5\n",
        encoding="utf-8",
    )

    recorder = _recorder(tmp_path, with_metrics=True)
    result = _run_all_q()
    for index, interaction in enumerate(result.interactions, start=1):
        recorder.record_interaction(
            round_index=index, interaction=interaction, budget_status={}, state={}, prompt_definitions={},
        )
    recorder.finalize(status="completed", budget_status={})

    rows = list(csv.DictReader(stale.open()))
    assert "population_action_share_q" not in {row["metric_name"] for row in rows}
    assert all(row["series"] is not None for row in rows)
    # Every value must still parse as the plotter parses it.
    for row in rows:
        if not row["agent_id"]:
            float(row["value"])


def test_recorder_appends_to_a_streaming_csv_of_the_current_schema(tmp_path: Path):
    """A matching header means a resumed episode, so its rows are kept."""

    existing = tmp_path / "metrics" / "streaming.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        "round_index,episode_id,agent_id,series,metric_name,value\n"
        "1,earlier,,Q,population_action_share_per_option,0.5\n",
        encoding="utf-8",
    )

    recorder = _recorder(tmp_path, with_metrics=True)
    result = _run_all_q()
    for index, interaction in enumerate(result.interactions, start=1):
        recorder.record_interaction(
            round_index=index, interaction=interaction, budget_status={}, state={}, prompt_definitions={},
        )
    recorder.finalize(status="completed", budget_status={})

    rows = list(csv.DictReader(existing.open()))
    assert rows[0]["episode_id"] == "earlier"  # preserved, not truncated
    assert len(rows) > 1


# --- binned trajectory tables, through the recorder --------------------------


def _forced_config(population_size=6, horizon=12, forced=("agent-000", "agent-001")):
    """A committed minority hard-wired to M, while every free agent answers Q."""

    config = _config(population_size=population_size, horizon=horizon)
    return replace(
        config,
        control=ControlConfig(
            mechanism="forced_action",
            options={"agent_ids": list(forced), "forced_value": "M"},
        ),
    )


def _run(config):
    game = create_game(config.game)
    return game, run_naming_convention_game_sync(
        game, config, ScriptedProvider(), control=create_control(config.control)
    )


def _record_all(recorder, result):
    for index, interaction in enumerate(result.interactions, start=1):
        recorder.record_interaction(
            round_index=index, interaction=interaction, budget_status={}, state={}, prompt_definitions={},
        )
    recorder.finalize(status="completed", budget_status={})


def test_interaction_records_carry_a_committed_flag_per_output():
    config = _forced_config()
    _, result = _run(config)
    view = to_round_view(result.final_state)
    forced = {"agent-000", "agent-001"}
    for entry in view.recent_history:
        expected = [str(agent) in forced for agent in entry["selected_agents"]]
        assert list(entry["committed"]) == expected
    # A mixed pair - one committed, one free - must be flagged per output.
    assert any(set(entry["committed"]) == {True, False} for entry in view.recent_history)


def test_recorder_writes_both_binned_trajectory_tables(tmp_path: Path):
    config = _config(population_size=6, horizon=12)
    _, result = _run(config)
    recorder = RunRecorder(
        tmp_path, run_id="ep-1", resolved_config={"schema_version": 1},
        policy=DetailedAuditPolicy(enabled=False), comet_enabled=False, checkpoint_enabled=False,
        metrics=build_metrics(), to_round_view=to_round_view,
        binning=config.metrics.binning_policy(config.game.population_size),
    )
    _record_all(recorder, result)

    success = list(csv.DictReader((tmp_path / "metrics" / "success_rate.csv").open()))
    production = list(csv.DictReader((tmp_path / "metrics" / "production_probability.csv").open()))
    # 12 interactions, bin size 6 (the population) -> two full bins.
    assert [row["bin_index"] for row in success] == ["0", "1"]
    assert all(row["num_pair_interactions"] == "6" for row in success)
    # Each bin contributes one row per option, over 2L = 12 individual outputs.
    assert {row["action"] for row in production} == {"Q", "M"}
    assert all(row["eligible_output_count"] == "12" for row in production)
    for row in production:
        assert int(row["action_count"]) / 12 == float(row["production_probability"])


def test_recorder_without_a_binning_policy_writes_no_trajectory_tables(tmp_path: Path):
    _, result = _run(_config())
    recorder = RunRecorder(
        tmp_path, run_id="ep-1", resolved_config={"schema_version": 1},
        policy=DetailedAuditPolicy(enabled=False), comet_enabled=False, checkpoint_enabled=False,
        metrics=build_metrics(), to_round_view=to_round_view,
    )
    _record_all(recorder, result)
    assert not (tmp_path / "metrics" / "success_rate.csv").exists()
    assert not (tmp_path / "metrics" / "production_probability.csv").exists()


def test_excluding_committed_outputs_reports_only_what_free_agents_chose(tmp_path: Path):
    """The forced agents always say M; the free agents always say Q."""

    config = _forced_config()
    _, result = _run(config)
    view = to_round_view(result.final_state)
    records = [InteractionOutcome.from_evaluator_entry(entry) for entry in view.recent_history]

    everyone = production_probabilities(records, action_space=("Q", "M"))
    free_only = production_probabilities(
        records, action_space=("Q", "M"), exclude_committed_outputs=True
    )
    assert everyone["M"] > 0.0  # the committed minority shows up
    assert free_only == {"Q": 1.0, "M": 0.0}  # no free agent ever adopted M


def test_binning_policy_defaults_its_bin_size_to_the_population():
    config = _config(population_size=7)
    assert config.metrics.binning_policy(config.game.population_size).bin_size == 7


def test_binning_policy_is_none_when_metrics_are_disabled():
    config = _config()
    disabled = replace(config.metrics, enabled=False)
    assert disabled.binning_policy(config.game.population_size) is None


# --- config -----------------------------------------------------------------


def test_metrics_config_section_parses_from_yaml():
    config = load_run_config("configs/runs/naming_convention_smoke_test_v3.yaml", environment={})
    assert config.metrics.enabled is True
    assert "population_action_share_per_option" in config.metrics.comet_export


def test_metrics_config_defaults_when_section_omitted():
    # This legacy (pre-v3) config file has no `metrics:` section at all.
    config = load_run_config("configs/runs/naming_convention_smoke_test.yaml", environment={})
    assert config.metrics.enabled is True
    assert config.metrics.comet_export == ()


def test_metrics_available_section_parses_from_yaml_and_selects_comet_export():
    config = load_run_config(
        "configs/runs/naming_convention_tutorial_university_v3.yaml", environment={}
    )
    assert config.metrics.available["population_action_share_per_option"]["comet"] is True
    assert config.metrics.available["rolling_action_share_per_option"]["comet"] is False
    names = config.metrics.comet_export_names()
    assert "population_action_share_per_option" in names
    assert "rolling_coordination_rate" in names
    assert "rolling_action_share_per_option" not in names
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
