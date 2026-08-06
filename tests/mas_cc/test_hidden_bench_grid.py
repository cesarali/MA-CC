"""A real HiddenBench grid through the orchestrator (brief §9.7).

Checks the wiring the analysis pipeline depends on: that the games are driven by
an observer-aware runtime, that each episode therefore writes
`metrics/streaming.csv`, and that the finished grid directory is readable by
`analysis/reader.py::read_grid` without any modification to that reader.

Runs offline against the mock provider, with Comet off. Nothing here should ever
reach a network or a paid endpoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import load_run_config_or_grid
from mas_cc.config.grid import GridAxis, GridSpec
from mas_cc.experiments import run_experiment_grid_sync
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus not present; see docs/hidden_bench/data_provenance.md",
)

CONFIG = "configs/runs/hidden_bench_grid.yaml"
ENVIRONMENT = {"POTSDAM_API_KEY": "test-key", "BASE_POTSDAM_LLM_URL": "http://localhost"}


# The mock provider reached through `create_llm_provider` has no response
# factory, so it returns one fixed string for every request. A vote object is
# the right choice: it parses as a valid vote, and it is also a perfectly valid
# (if odd) non-empty discussion turn, so both phases get a usable response.
MOCK_RESPONSE = '{"vote": "West City", "rationale": "the bridge is still passable"}'


@pytest.fixture(scope="module")
def finished_grid(tmp_path_factory) -> Path:
    """The shipped grid config, shrunk to two cells and pointed at the mock.

    `pricing.mode: offline` and the mock provider together are what keep this
    free: the shipped config asks for live University pricing, which a test must
    never trigger.
    """

    grid = load_run_config_or_grid(CONFIG, environment=ENVIRONMENT)
    base = replace(
        grid.base,
        llm_provider=replace(
            grid.base.llm_provider,
            type="mock",
            options={**dict(grid.base.llm_provider.options), "response": MOCK_RESPONSE},
        ),
        game=replace(
            grid.base.game,
            population_size=4,
            options={**dict(grid.base.game.options), "rounds": 1},
        ),
        execution=replace(grid.base.execution, repetitions=2, parallelism=2),
        # The mock provider has no published price, and preflight correctly
        # refuses to launch what it cannot price (brief §8). `explicit_unknown_
        # _price_override` is the documented escape hatch, and taking it here is
        # safe precisely because the mock cannot spend anything.
        pricing=replace(
            grid.base.pricing,
            mode="offline",
            require_fresh_at_launch=False,
            explicit_unknown_price_override=True,
        ),
        # Same reason as the pricing override: the runtime guard also refuses
        # to send a request whose cost it cannot bound. Both refusals are the
        # budget machinery working, and both are safe to lift for a mock.
        budget=replace(grid.base.budget, allow_unbounded_paid_requests=True),
        logging=replace(grid.base.logging, comet=False),
    )
    # Two cells only: hidden vs full at one group size and one depth. That is
    # enough to prove the wiring, and it is the pair `gap_to_full` needs.
    spec = GridSpec(base=base, axes=(GridAxis("game.options.profile", ("hidden", "full")),))
    result = run_experiment_grid_sync(
        spec, tmp_path_factory.mktemp("hidden-bench-grid"), resume=False, show_progress=False
    )
    assert result.failed == 0, "grid cells failed"
    return result.output_dir


def test_the_grid_completes_both_cells(finished_grid: Path):
    cells = sorted(path.name for path in finished_grid.iterdir() if path.is_dir())
    assert len(cells) >= 2


def test_every_episode_wrote_streaming_metrics(finished_grid: Path):
    """`metrics/streaming.csv` is what `mas-cc analysis empowerment` reads back.

    A game without an observer-aware runtime still runs but writes none of
    this, and any grid over it is silently unanalysable - so this is the check
    that the orchestrator wiring actually took effect.
    """

    streaming = list(finished_grid.rglob("metrics/streaming.csv"))
    assert streaming, "no episode wrote streaming metrics"
    for path in streaming:
        header, *rows = path.read_text(encoding="utf-8").splitlines()
        assert "metric_name" in header and "value" in header
        assert rows, f"{path} has a header but no rows"


def test_the_papers_quantities_reach_the_metric_files(finished_grid: Path):
    names = set()
    for path in finished_grid.rglob("metrics/streaming.csv"):
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            names.add(line.split(",")[4])
    assert {"accuracy_average", "accuracy_majority", "unshared_disclosure_rate"} <= names


def test_the_finished_grid_is_readable_by_the_analysis_reader(finished_grid: Path):
    """§9.7: `read_grid` must consume this directory unmodified."""

    from mas_cc.analysis.reader import read_grid

    grid = read_grid(finished_grid)
    assert not grid.episodes.empty, "the reader found no episodes"
    assert not grid.rounds.empty, "the reader found no per-round macrostates"
    # The swept axis has to survive as a condition column - that column is what
    # `mas-cc analysis empowerment` conditions the estimate on.
    assert "game.options.profile" in grid.condition_columns
    assert set(grid.episodes["game.options.profile"]) == {"hidden", "full"}
    assert grid.episodes["cell_id"].nunique() == 2
