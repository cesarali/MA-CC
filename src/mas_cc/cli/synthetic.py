"""``mas-cc synthetic`` — run the games whose answers we already know.

Five commands, in the order the work actually goes:

``truth``     print the closed form for a config without running anything.
``sweep``     speed mode. The null distribution at eps = 0.5 (what MI the
              estimator reports when there is none) and the calibration curve
              across the grid. This is the cheapest thing that says something
              useful about the estimators we already have.
``episode``   fidelity mode. One episode through prompts, provider, parser,
              validator and recorder, with the estimate and the truth written
              side by side in the artifacts.
``parity``    both modes on the same seeds, demanding the same trajectory. This
              is what makes speed mode trustworthy enough to sweep in - and if
              it fails, it has found a pipeline bug, which is the point.
``empowerment`` the family-2 answer key: exact ``I(C;O)`` and
              ``I(C;S_t+h | S_t)`` for a *sweep* rather than a single config,
              because those depend on the grid you chose - which is the channel
              input distribution, and something we know exactly.

No preflight, no pricing, no budget guard: a synthetic run makes no billable
call, so gating it on a cost estimate would be theatre. Comet stays wherever
the config puts it; the shipped synthetic configs leave it off, because a
calibration rehearsal has no business uploading anything.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from mas_cc.config import RunConfig, load_run_config, resolved_config_yaml
from mas_cc.games import create_game, game_metrics
from mas_cc.games.synthetic import (
    SyntheticGame,
    SyntheticGameResult,
    calibration_curve,
    compare_modes,
    create_synthetic_provider_registry,
    episode_summary,
    metric_check,
    null_distribution,
    pairwise_estimates,
    plot_calibration,
    read_action_series,
    run_synthetic_game_sync,
    with_truth,
)
from mas_cc.games.synthetic.analysis import series_to_indices
from mas_cc.llm_runtime.prompts import PromptMarkdownLogger
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.metrics import plot_streaming_metrics
from mas_cc.observability import DetailedAuditPolicy, RunRecorder
from mas_cc.storage import results_run_dir

DEFAULT_EPSILON_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
"""Dense near zero noise, where 1 - H(q) changes fastest and bias shows first."""


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _load(config_path: str | Path) -> tuple[RunConfig, SyntheticGame]:
    config = load_run_config(Path(config_path).resolve())
    game = create_game(config.game)
    if not isinstance(game, SyntheticGame):
        raise ValueError(
            f"game.type {config.game.type!r} is not a synthetic game; "
            "`mas-cc synthetic` only runs games that can state their own ground truth"
        )
    return config, game


def _destination(config: RunConfig, output_dir: str | Path | None, *, run_id: str) -> Path:
    base = Path(output_dir) if output_dir is not None else Path(config.storage.output_dir)
    return results_run_dir(
        base, game=config.game.type, experiment=config.experiment.name, run_id=run_id
    )


def _seeds(config: RunConfig, count: int) -> tuple[int, ...]:
    """Consecutive seeds from the config's own base seed.

    Consecutive rather than random so a sweep is reproducible from two numbers
    in the config, and so a seed that misbehaves can be re-run on its own.
    """

    base = int(config.execution.seed)
    return tuple(range(base, base + count))


class _RecorderObserver:
    """Bridges the synthetic runtime onto the real `RunRecorder`.

    Deliberately thin. The recorder is one of the things under rehearsal, so it
    is used exactly as a real episode uses it rather than through a
    synthetic-specific shim. Budget status is empty because there is no spend
    to track.

    The optional Markdown prompt log is the one addition, and it earns its
    place: a rendered synthetic prompt sitting in `prompts/` next to a real
    one is how you confirm at a glance that the two are the same shape, which
    is the claim the whole fidelity mode rests on.
    """

    def __init__(
        self,
        recorder: RunRecorder,
        prompt_logger: PromptMarkdownLogger | None = None,
        prompt_example_rounds: int = 0,
    ) -> None:
        self.recorder = recorder
        self.prompt_logger = prompt_logger
        self.prompt_example_rounds = prompt_example_rounds
        self._logged_rounds: set[int] = set()

    def event(self, event_type: str, **payload: Any) -> None:
        self.recorder.event(event_type, **payload)

    def record_attempt(self, **payload: Any) -> None:
        self.recorder.record_attempt(**payload)
        round_index = payload.get("round_index")
        if (
            self.prompt_logger is None
            or round_index in self._logged_rounds
            or len(self._logged_rounds) >= self.prompt_example_rounds
        ):
            return
        agent_id = payload["request"].metadata.get("agent_id")
        self.prompt_logger.log(
            payload["prompt"],
            interaction_id=f"round_{round_index:03d}",
            title=f"Round {round_index} — agent {agent_id}",
            metadata={"round_index": round_index, "agent_id": str(agent_id)},
        )
        self._logged_rounds.add(round_index)

    def record_interaction(self, **payload: Any) -> None:
        self.recorder.record_interaction(**payload, budget_status={})


def _run_fidelity_episode(
    config: RunConfig, game: SyntheticGame, destination: Path
) -> tuple[SyntheticGameResult, Path]:
    """One full-pipeline episode, with the answer key written beside the output."""

    truth = game.ground_truth(config.game)
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))
    # Ground truth is a first-class artifact, written before the run rather
    # than after: pull the directory off the cluster and the answer is already
    # in the file, next to the estimate, whatever happened to the episode.
    _write(destination / "ground_truth.json", _json(truth.to_dict()))

    metrics, to_round_view = game_metrics(game)
    recorder = RunRecorder(
        destination,
        run_id=f"{config.experiment.name}-{config.execution.seed}",
        resolved_config=config.to_dict(),
        policy=DetailedAuditPolicy.from_mapping(
            config.logging.options.get("detailed_prompt_audit")
        ),
        comet_enabled=config.logging.comet,
        project_name=str(config.logging.options.get("comet_project", "mas-cc")),
        checkpoint_enabled=config.storage.checkpoints,
        metrics=metrics,
        to_round_view=to_round_view,
        comet_metric_export=config.metrics.comet_export_names() if config.metrics.enabled else (),
        binning=config.metrics.binning_policy(config.game.population_size),
    )
    provider = create_llm_provider(
        config.llm_provider, registry=create_synthetic_provider_registry()
    )
    example_count = int(
        dict(config.logging.options.get("prompt_examples", {}) or {}).get("count", 0)
    )
    observer = _RecorderObserver(
        recorder,
        PromptMarkdownLogger(destination / "prompts", overwrite=True) if example_count else None,
        example_count,
    )
    try:
        result = run_synthetic_game_sync(game, config, provider, observer=observer)
    except Exception as exc:
        recorder.event("run_failed", error_type=type(exc).__name__, error=str(exc))
        recorder.finalize(status="failed", budget_status={})
        raise
    finally:
        provider.close()
    recorder.finalize(status="completed", budget_status={})
    return result, destination


def _episode_estimates(
    game: SyntheticGame, config: RunConfig, destination: Path
) -> pd.DataFrame:
    """Estimate MI from what the recorder actually wrote, then join on truth."""

    truth = game.ground_truth(config.game)
    series = read_action_series(destination)
    labels = tuple(str(item) for item in config.game.options.get("actions", ("Q", "M")))
    actions = series_to_indices(series, labels)
    frame = with_truth(pairwise_estimates(actions, sorted(series), len(labels)), truth)
    frame.insert(0, "seed", int(config.execution.seed))
    return frame


# -- commands -------------------------------------------------------------


def run_synthetic_truth(config_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """Print (and optionally write) the closed form for a resolved config."""

    config, game = _load(config_path)
    payload = game.ground_truth(config.game).to_dict()
    if output_dir is not None:
        _write(Path(output_dir) / "ground_truth.json", _json(payload))
    print(_json(payload), end="")
    return payload


def run_synthetic_episode(
    config_path: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Fidelity mode: one episode through the whole pipeline."""

    config, game = _load(config_path)
    destination = _destination(
        config, output_dir, run_id=f"{config.experiment.name}-{config.execution.seed}"
    )
    result, destination = _run_fidelity_episode(config, game, destination)

    frame = _episode_estimates(game, config, destination)
    frame.to_csv(destination / "metrics" / "mi_estimates.csv", index=False)
    summary = episode_summary(frame)
    _write(destination / "metrics" / "mi_summary.json", _json(summary))

    checks = metric_check(destination, game.ground_truth(config.game))
    if not checks.empty:
        checks.to_csv(destination / "metrics" / "metric_check.csv", index=False)
    plotted = plot_streaming_metrics(
        destination / "metrics" / "streaming.csv", destination / "metrics" / "plots"
    )

    print(
        f"Episode complete: {len(result.interactions)} rounds, "
        f"{result.logical_decisions} decisions, termination={result.termination_reason!r}"
    )
    print(
        f"  mean I(A_i;A_j): estimate {summary['unsmoothed_mean']:.4f} bits "
        f"vs truth {summary['truth_mean']:.4f} bits "
        f"(gap {summary['gap_unsmoothed_mean']:+.4f})"
    )
    _print_metric_check(checks)
    print(f"  ground truth: {destination / 'ground_truth.json'}")
    print(f"  estimates:    {destination / 'metrics' / 'mi_estimates.csv'}")
    if plotted:
        print(f"  metric plots ({len(plotted)}): {destination / 'metrics' / 'plots'}")
    print(f"Comet: {recorder_comet_status(destination)}")
    print(f"Output: {destination}")
    return summary


def recorder_comet_status(destination: Path) -> str:
    """What the recorder's Comet sink actually did, read back from its own summary."""

    path = destination / "comet_summary.json"
    if not path.is_file():
        return "not written"
    payload = json.loads(path.read_text(encoding="utf-8"))
    status, reason = payload.get("status"), payload.get("reason")
    return status if not reason else f"{status} ({reason})"


def _print_metric_check(checks: pd.DataFrame) -> None:
    """The choice metrics next to their closed forms, as a readable block."""

    if checks.empty:
        return
    print("  metrics vs. ground truth:")
    for row in checks.to_dict("records"):
        label = row["metric"] + (f" [{row['series']}]" if row["series"] else "")
        observed = row["observed"]
        shown = f"{observed:.4f}" if isinstance(observed, float) else str(observed)
        if not row["has_expected"]:
            print(f"    {label:<44} {shown:>8}   (no closed form)")
            continue
        expected = row["expected"]
        target = f"{expected:.4f}" if isinstance(expected, float) else str(expected)
        # "--" is not a soft pass: it marks a row that was deliberately not
        # judged, so an unjudged row can never be mistaken for a passing one.
        mark = "-- " if row["agrees"] is None else ("ok " if row["agrees"] else "OFF")
        suffix = f"  ({row['note']})" if row["note"] else ""
        print(f"    {label:<44} {shown:>8}   expected {target:<8} {mark}{suffix}")


def run_synthetic_sweep(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    seeds: int = 200,
    epsilon_grid: Sequence[float] = DEFAULT_EPSILON_GRID,
) -> dict[str, Any]:
    """Speed mode: the null distribution and the calibration curve."""

    config, game = _load(config_path)
    destination = _destination(config, output_dir, run_id=f"{config.experiment.name}-sweep")
    seed_values = _seeds(config, seeds)
    _write(destination / "resolved_config.yaml", resolved_config_yaml(config))

    null_frame, null_summary = null_distribution(game, config.game, seed_values)
    null_frame.to_csv(destination / "metrics" / "null_distribution.csv", index=False)
    _write(destination / "metrics" / "null_summary.json", _json(null_summary))

    curve = calibration_curve(game, config.game, epsilon_grid, seed_values)
    curve.to_csv(destination / "metrics" / "calibration.csv", index=False)
    plot = plot_calibration(curve, destination / "metrics" / "plots" / "calibration.png")

    floor = null_summary.get("significance_floor_bits", float("nan"))
    print(
        f"Null distribution ({null_summary['seeds']} seeds x {null_summary['rounds']} rounds "
        f"x {null_summary['population_size']} agents, true MI = 0):"
    )
    for name in ("unsmoothed", "jeffreys", "miller_madow"):
        print(
            f"  {name:<13} mean {null_summary[f'{name}_mean']:+.5f}  "
            f"p95 {null_summary[f'{name}_p95']:.5f}  max {null_summary[f'{name}_max']:.5f} bits"
        )
    print(
        f"  significance floor: {floor:.5f} bits — a plug-in MI below this at "
        f"this population and round count is indistinguishable from zero."
    )
    print(f"Calibration curve ({len(curve)} epsilon points): {plot}")
    print(f"Output: {destination}")
    return {"null": null_summary, "calibration_points": int(len(curve)), "output_dir": str(destination)}


def run_synthetic_empowerment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    condition: str = "epsilon",
    values: Sequence[Any] = (),
    repetitions: int = 50,
    horizons: Sequence[int] = (1,),
    macrostate_bins: int | None = None,
) -> dict[str, Any]:
    """The exact family-2 quantities for a sweep, without running the sweep.

    This is the answer key for the empowerment statistics `analysis/pipeline.py`
    estimates. It needs the grid, not just a config, because ``p(c)`` is a
    property of what you swept and how many episodes each cell got.
    """

    from mas_cc.games.synthetic.empowerment import (
        AnalysisSpec,
        sweep_from_values,
        sweep_ground_truth,
    )

    config, game = _load(config_path)
    if not values:
        raise ValueError("--values must list at least one condition value to sweep")
    sweep = sweep_from_values(config.game, condition, values, repetitions=repetitions)
    analysis = AnalysisSpec(
        horizons=tuple(horizons), macrostate_bins=macrostate_bins
    )
    truth = sweep_ground_truth(game, sweep, analysis)

    destination = _destination(
        config, output_dir, run_id=f"{config.experiment.name}-empowerment"
    )
    payload = truth.to_dict()
    _write(destination / "empowerment_ground_truth.json", _json(payload))

    terminal = truth.value("terminal_mutual_information")
    terminal_bias = truth.value("terminal_plugin_bias")
    print(
        f"Sweep: {condition} over {list(values)}, {repetitions} episode(s) per cell "
        f"({truth.value('effective_sample_size'):.0f} total)"
    )
    print(
        f"  |C| = {truth.value('condition_alphabet_size'):.0f}, "
        f"|M| = {truth.value('macrostate_alphabet_size'):.0f} after binning, "
        f"lumpable = {bool(truth.value('macrostate_is_lumpable'))}"
    )
    print(f"  terminal I(C;O)      = {terminal:.6f} bits   (plug-in bias {terminal_bias:.4f})")
    print(f"  I(C;S)               = {truth.value('condition_state_mutual_information'):.6f} bits")
    print("  lagged I(C;S_t+h|S_t):")
    for horizon in horizons:
        value = truth.value("lagged_conditional_mutual_information", (str(horizon),))
        bias = truth.value("lagged_plugin_bias", (str(horizon),))
        # Bias above signal is the headline, not a footnote: an unbinned
        # macrostate can carry more degrees-of-freedom inflation than effect.
        flag = "  <-- bias exceeds the signal" if bias > abs(value) else ""
        print(f"    h={horizon:<3} {value:.6f} bits   (plug-in bias {bias:.4f}){flag}")
    print(f"Output: {destination / 'empowerment_ground_truth.json'}")
    return payload


def run_synthetic_parity(
    config_path: str | Path, output_dir: str | Path | None = None, *, seeds: int = 5
) -> dict[str, Any]:
    """Run both modes on the same seeds and demand the same trajectory."""

    config, game = _load(config_path)
    root = _destination(config, output_dir, run_id=f"{config.experiment.name}-parity")
    _write(root / "resolved_config.yaml", resolved_config_yaml(config))

    results: list[dict[str, Any]] = []
    for seed in _seeds(config, seeds):
        episode_config = replace(config, execution=replace(config.execution, seed=seed))
        destination = root / "episodes" / f"seed-{seed}"
        _run_fidelity_episode(episode_config, game, destination)
        recorded = read_action_series(destination)
        results.append(compare_modes(game, episode_config.game, seed, recorded).to_dict())

    frame = pd.DataFrame(results)
    frame.to_csv(root / "metrics" / "parity.csv", index=False)
    agreed = int(frame["identical"].sum())
    summary = {
        "seeds": len(results),
        "identical_trajectories": agreed,
        "all_identical": agreed == len(results),
        "total_mismatched_cells": int(frame["mismatched_cells"].sum()),
        "max_absolute_estimate_gap": float(frame["estimate_gap"].abs().max()),
        "output_dir": str(root),
    }
    _write(root / "metrics" / "parity_summary.json", _json(summary))

    print(f"Mode parity over {summary['seeds']} seed(s):")
    print(
        f"  bit-exact trajectories: {agreed}/{summary['seeds']}"
        + ("" if summary["all_identical"] else "  <-- the two modes disagree")
    )
    if not summary["all_identical"]:
        for row in results:
            if row["identical"]:
                continue
            print(
                f"    seed {row['seed']}: {row['mismatched_cells']} cell(s), first at "
                f"round {row['first_mismatch_round']}, agent index {row['first_mismatch_agent']}"
            )
    print(f"  max |estimate gap| between modes: {summary['max_absolute_estimate_gap']:.6f} bits")
    print(f"Output: {root}")
    return summary
