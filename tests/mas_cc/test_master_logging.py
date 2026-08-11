"""Master-only Comet logging: who writes, on what step axis, and how often.

The properties under test are the ones that make the dashboard trustworthy:
one writer, a heartbeat that ticks on a timer rather than on completions, cell
curves stepped by round, and a remote outage that never reaches the run.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mas_cc.config import CometObservability, load_run_config_or_grid
from mas_cc.experiments.comet_monitor import (
    CellLayout,
    MasterMonitor,
    SweepLayout,
    sweep_parameters,
)
from mas_cc.metrics import AggregateResult, Curve


class FakeSink:
    """Stands in for `CometMetricSink`, recording exactly what was sent."""

    instances: list["FakeSink"] = []

    def __init__(self, enabled: bool, *, project_name: str, run_name: str) -> None:
        self.enabled = enabled
        self.project_name = project_name
        self.run_name = run_name
        self.status = "active" if enabled else "disabled"
        # The real sink sets both in `__init__`; `describe()` and the cell
        # summaries read them, so the stand-in has to carry them too.
        self.reference: str | None = f"key-{len(FakeSink.instances)}" if enabled else None
        self.url: str | None = f"https://comet.test/{run_name}" if enabled else None
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.parameters: dict[str, object] = {}
        self.figures: list[tuple[str, int]] = []
        self.images: list[tuple[Path, str, int]] = []
        self.tags: list[str] = []
        self.closed = False
        FakeSink.instances.append(self)

    def log_metrics(self, metrics, step):
        self.metrics.append((dict(metrics), step))

    def log_parameters(self, parameters):
        self.parameters.update(parameters)

    def log_figure(self, figure, *, name, step):
        self.figures.append((name, step))

    def log_image(self, path, *, name, step):
        self.images.append((Path(path), name, step))

    def add_tags(self, tags):
        self.tags.extend(tags)

    def close(self):
        self.closed = True
        return {"status": self.status, "reference": None, "reason": None}


@pytest.fixture
def sinks(monkeypatch):
    FakeSink.instances = []
    monkeypatch.setattr("mas_cc.experiments.comet_monitor.CometMetricSink", FakeSink)
    return FakeSink.instances


def _layout() -> SweepLayout:
    return SweepLayout(
        axes=(("game.options.control_value", (0, 1)),),
        cells={
            "cell-0000": CellLayout(coordinates=(0,), episodes=2),
            "cell-0001": CellLayout(coordinates=(1,), episodes=2),
        },
    )


def _monitor(**kwargs) -> MasterMonitor:
    settings = kwargs.pop("settings", CometObservability(heartbeat_seconds=3600.0))
    return MasterMonitor(
        True, project_name="mas-cc", run_name="sweep-1", layout=_layout(), settings=settings, **kwargs
    )


def test_the_sweep_experiment_steps_by_episodes_done(sinks):
    monitor = _monitor()
    monitor.start({"game": "toy"})

    monitor.episode_finished(status="completed", cell_id="cell-0000")
    monitor.episode_finished(status="failed", cell_id="cell-0000")

    steps = [step for _, step in sinks[0].metrics]
    assert steps == sorted(steps)
    last, _ = sinks[0].metrics[-1]
    assert last["episodes_done"] == 2.0
    assert last["episodes_completed"] == 1.0
    assert last["episodes_failed"] == 1.0
    assert last["progress_fraction"] == pytest.approx(0.5)


def test_progress_metrics_can_be_limited_to_episode_count(sinks):
    monitor = _monitor(
        settings=CometObservability(
            heartbeat_seconds=3600.0,
            progress_metrics=("episodes_done",),
        )
    )
    monitor.start()
    monitor.episode_finished(status="completed", cell_id="cell-0000")

    metrics, step = sinks[0].metrics[-1]
    assert metrics == {"episodes_done": 1.0}
    assert step == 1


def test_the_heartbeat_ticks_on_a_timer_even_when_nothing_completes(sinks):
    """Spec acceptance check 5, and the whole reason the heartbeat is a timer.

    A metric that only moves when an episode finishes cannot distinguish a dead
    master from a slow one; this one must keep publishing regardless.
    """

    monitor = _monitor(settings=CometObservability(heartbeat_seconds=0.02))
    monitor.start()
    deadline = time.monotonic() + 2.0
    while len(sinks[0].metrics) < 4 and time.monotonic() < deadline:
        time.sleep(0.02)
    published = len(sinks[0].metrics)
    monitor.close()

    assert published >= 4
    assert all(step == 0 for _, step in sinks[0].metrics[:published])


def test_the_heartbeat_stops_when_the_monitor_closes(sinks):
    monitor = _monitor(settings=CometObservability(heartbeat_seconds=0.02))
    monitor.start()
    time.sleep(0.1)
    summary = monitor.close()
    settled = len(sinks[0].metrics)
    time.sleep(0.1)

    assert len(sinks[0].metrics) == settled
    assert summary["writer"] == "master_only"
    assert sinks[0].closed


def test_eta_is_reported_as_unknown_rather_than_omitted_before_the_first_episode(sinks):
    monitor = _monitor()
    monitor.start()

    metrics, _ = sinks[0].metrics[-1]
    assert metrics["eta_minutes"] == -1.0

    monitor.episode_finished(status="completed", cell_id="cell-0000")
    metrics, _ = sinks[0].metrics[-1]
    assert metrics["eta_minutes"] > 0


def test_the_grid_image_is_throttled_to_every_n_completions(sinks):
    monitor = _monitor(settings=CometObservability(heartbeat_seconds=3600.0, grid_image_every_n_episodes=3))
    monitor.start()

    for _ in range(4):
        monitor.episode_finished(status="completed", cell_id="cell-0000")

    assert [name for name, _ in sinks[0].figures] == ["grid_progress"]
    assert sinks[0].figures[0][1] == 3  # logged at step = episodes_done


def test_a_completed_cell_gets_its_own_experiment_stepped_by_round(sinks):
    monitor = _monitor()
    monitor.start()
    result = AggregateResult(
        curves={
            "dominant_action_share": Curve(
                levels=("p10", "p50", "p90"), points={1: (0.4, 0.5, 0.6), 2: (0.8, 0.9, 1.0)}
            ),
            "active_fraction": Curve(levels=("value",), points={1: (1.0,), 2: (0.5,)}),
        },
        scalars={"converged_fraction": 0.75, "median_consensus_round": 12.0},
        episodes=2,
    )

    monitor.cell_completed("cell-0000", result)

    cell_sink = next(sink for sink in sinks if sink.run_name == "sweep-1/cell-0000")
    assert cell_sink.parameters["cell_id"] == "cell-0000"
    assert cell_sink.parameters["game.options.control_value"] == 0
    round_one = next(metrics for metrics, step in cell_sink.metrics if step == 1)
    assert round_one["dominant_action_share_p50"] == 0.5
    # A single-level curve keeps its bare name; only bands get a level suffix.
    assert round_one["active_fraction"] == 1.0
    assert cell_sink.closed


def test_metric_plots_are_uploaded_to_the_cell_experiment_when_requested(sinks, tmp_path):
    monitor = _monitor(
        settings=CometObservability(heartbeat_seconds=3600.0, metric_plots=True)
    )
    monitor.start()
    plot = tmp_path / "dominant_action_share.png"
    plot.write_bytes(b"png")
    result = AggregateResult(
        curves={
            "dominant_action_share": Curve(
                levels=("value",), points={1: (0.5,), 2: (0.9,)}
            )
        },
        episodes=2,
    )

    summary = monitor.cell_completed("cell-0000", result, metric_plots=(plot,))

    cell_sink = next(sink for sink in sinks if sink.run_name == "sweep-1/cell-0000")
    assert cell_sink.images == [(plot, "metric_plot_dominant_action_share", 2)]
    assert summary["metric_plots"] == 1


def test_metric_plots_are_not_uploaded_by_default(sinks, tmp_path):
    monitor = _monitor()
    monitor.start()
    plot = tmp_path / "dominant_action_share.png"
    plot.write_bytes(b"png")

    summary = monitor.cell_completed(
        "cell-0000", AggregateResult(episodes=2), metric_plots=(plot,)
    )

    cell_sink = next(sink for sink in sinks if sink.run_name == "sweep-1/cell-0000")
    assert cell_sink.images == []
    assert summary["metric_plots"] == 0


def test_a_completed_cells_headline_scalars_land_on_the_sweep_experiment(sinks):
    monitor = _monitor()
    monitor.start()
    monitor.episode_finished(status="completed", cell_id="cell-0000")
    result = AggregateResult(
        scalars={"converged_fraction": 0.75, "median_consensus_round": 12.0, "episode_rounds_max": 40.0},
        episodes=2,
    )

    monitor.cell_completed("cell-0000", result)

    published = {key: value for metrics, _ in sinks[0].metrics for key, value in metrics.items()}
    assert published["cell_cell-0000_converged_fraction"] == 0.75
    assert published["cell_cell-0000_median_consensus_round"] == 12.0
    # Bulk per-cell scalars belong on the cell experiment, not on the dashboard.
    assert "cell_cell-0000_episode_rounds_max" not in published


def test_a_comet_outage_never_reaches_the_run(sinks):
    """Comet is a view of a run whose real output is already on disk."""

    monitor = _monitor()
    monitor.start()

    def explode(*args, **kwargs):
        raise RuntimeError("comet is down")

    sinks[0].log_metrics = explode
    sinks[0].log_figure = explode

    monitor.episode_finished(status="completed", cell_id="cell-0000")
    monitor.cell_completed("cell-0000", AggregateResult(scalars={"converged_fraction": 1.0}))

    assert monitor.episodes_done == 1


def test_a_disabled_monitor_is_inert_but_still_safe_to_call(sinks):
    monitor = MasterMonitor(False, project_name="mas-cc", run_name="sweep-1", layout=_layout())
    monitor.start({"game": "toy"})
    monitor.episode_finished(status="completed", cell_id="cell-0000")
    summary = monitor.cell_completed("cell-0000", AggregateResult(scalars={"converged_fraction": 1.0}))

    assert monitor.status == "disabled"
    assert summary["status"] == "disabled"
    # No second experiment was created for the cell.
    assert len(sinks) == 1
    assert monitor.close()["episodes_finished"] == 1


def test_cell_experiments_can_be_switched_off_without_losing_the_sweep_dashboard(sinks):
    monitor = _monitor(
        settings=CometObservability(heartbeat_seconds=3600.0, cell_experiments=False)
    )
    monitor.start()

    monitor.cell_completed("cell-0000", AggregateResult(scalars={"converged_fraction": 1.0}))

    assert len(sinks) == 1
    assert any("cell_cell-0000_converged_fraction" in metrics for metrics, _ in sinks[0].metrics)


def test_master_aggregates_puts_a_single_cells_curves_and_plots_on_the_master(sinks, tmp_path):
    """One experiment for a one-cell run: nothing to overlay, nothing to split."""

    monitor = _monitor(
        settings=CometObservability(
            heartbeat_seconds=3600.0, metric_plots=True, master_aggregates=True
        )
    )
    monitor.start()
    plot = tmp_path / "m_order.png"
    plot.write_bytes(b"png")
    result = AggregateResult(
        curves={
            "m_order": Curve(levels=("p10", "p50", "p90"), points={1: (0.4, 0.5, 0.6)}),
            "m_ctrl": Curve(levels=("value",), points={1: (0.7,), 2: (0.8,)}),
        },
        scalars={"converged_fraction": 0.75},
        episodes=3,
    )

    summary = monitor.cell_completed("run", result, metric_plots=(plot,))

    # No child experiment was opened at all.
    assert len(sinks) == 1
    assert summary["published_to"] == "master"
    assert summary["metric_plots"] == 1
    assert sinks[0].images == [(plot, "metric_plot_m_order", 2)]
    published = {key: value for metrics, _ in sinks[0].metrics for key, value in metrics.items()}
    # A single cell keeps bare metric names, so the charts read as they would
    # have on a dedicated cell experiment.
    assert published["m_order_p50"] == 0.5
    assert published["m_ctrl"] == 0.8


def test_master_aggregates_prefixes_grid_cells_so_they_cannot_collide(sinks):
    monitor = _monitor(
        settings=CometObservability(heartbeat_seconds=3600.0, master_aggregates=True)
    )
    monitor.start()
    curve = {"m_order": Curve(levels=("value",), points={1: (0.5,)})}

    monitor.cell_completed("cell-0000", AggregateResult(curves=curve, episodes=1))
    monitor.cell_completed("cell-0001", AggregateResult(curves=curve, episodes=1))

    assert len(sinks) == 1
    published = {key for metrics, _ in sinks[0].metrics for key in metrics}
    assert "cell-0000_m_order" in published
    assert "cell-0001_m_order" in published


def test_analysis_sink_is_only_offered_under_master_aggregates(sinks):
    plain = _monitor()
    plain.start()
    assert plain.analysis_sink is None

    consolidated = _monitor(
        settings=CometObservability(heartbeat_seconds=3600.0, master_aggregates=True)
    )
    consolidated.start()
    # The master's own sink, so the analysis report joins the run it describes.
    assert consolidated.analysis_sink is sinks[-1]


def test_a_disabled_monitor_offers_no_analysis_sink():
    monitor = MasterMonitor(False, project_name="mas-cc", run_name="sweep-1", layout=_layout())
    monitor.start()
    assert monitor.analysis_sink is None


def test_sweep_parameters_describe_the_grid_and_the_aggregation_rules():
    grid = load_run_config_or_grid(
        "configs/runs/synthetic_games/synthetic_controlled_markov_empowerment.yaml", environment={}
    )
    axes = tuple((axis.path, tuple(axis.values)) for axis in grid.axes)

    parameters = sweep_parameters(grid.base, axes)

    assert parameters["grid.game.options.control_value"] == "0,1"
    assert parameters["repetitions_per_cell"] == grid.base.execution.repetitions
    assert parameters["forward_fill"] == "absorbing"
    assert parameters["percentiles"] == "10,50,90"
    assert all(isinstance(value, (str, int, float, bool)) for value in parameters.values())


def test_the_banner_line_reports_the_connection_not_the_config(sinks):
    """The silent case is the one that bites: `comet: true` and no API key.

    A run that intended to upload and did not looks exactly like a run that
    never meant to, unless something says otherwise on the console.
    """

    monitor = _monitor()
    monitor.start()
    sinks[0].url = "https://www.comet.com/acme/mas-cc-grids/abc123"

    described = monitor.describe()

    assert "master -> project 'mas-cc'" in described
    assert "https://www.comet.com/acme/mas-cc-grids/abc123" in described


def test_the_banner_line_distinguishes_switched_off_from_failed_to_connect(sinks):
    off = MasterMonitor(False, project_name="mas-cc", run_name="sweep-1", layout=_layout())
    off.start()
    assert "off" in off.describe()
    assert "logging.comet: true" in off.describe()

    broken = _monitor()
    broken.start()
    sinks[-1].status = "unavailable"
    sinks[-1].reason = "COMET_API_KEY is not set"
    assert "unavailable" in broken.describe()
    assert "COMET_API_KEY is not set" in broken.describe()


def test_sweep_parameters_name_the_run_and_its_game_knobs():
    """Opening a six-hour-old experiment must not require guessing what it was."""

    config = load_run_config_or_grid(
        "configs/runs/synthetic_games/synthetic_controlled_markov_repeated.yaml", environment={}
    )

    parameters = sweep_parameters(config)

    assert parameters["experiment_name"] == "synthetic-controlled-markov-repeated"
    assert parameters["experiment_tags"]
    assert parameters["game.options.control_value"] == 0
    assert parameters["game.options.control_alphabet"] == "Q,M"
    assert parameters["grid_axes"] == "(none - single cell)"
    assert parameters["cells"] == 1
    assert parameters["total_episodes"] == config.execution.repetitions
    assert all(isinstance(value, (str, int, float, bool)) for value in parameters.values())


def test_sweep_parameters_count_the_cells_of_a_real_grid():
    grid = load_run_config_or_grid(
        "configs/runs/synthetic_games/synthetic_controlled_markov_size_and_committee.yaml",
        environment={},
    )
    axes = tuple((axis.path, tuple(axis.values)) for axis in grid.axes)

    parameters = sweep_parameters(grid.base, axes)

    assert parameters["cells"] == 9
    assert parameters["total_episodes"] == 9 * grid.base.execution.repetitions
    assert "game.population_size" in parameters["grid_axes"]
