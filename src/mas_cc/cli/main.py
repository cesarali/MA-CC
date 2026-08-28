"""Command-line interface for the staged :mod:`mas_cc` implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from mas_cc import __version__
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers import ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mas-cc",
        description="Inspect and run the staged MAS-CC language-game framework.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="print the installed MAS-CC version")

    prompt = commands.add_parser(
        "prompt", help="compile and inspect prompts without an LLM"
    )
    prompt_commands = prompt.add_subparsers(dest="prompt_command", required=True)
    examples = prompt_commands.add_parser(
        "examples", help="write paper-grounded prompts as readable Markdown"
    )
    examples.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/inspection/paper_prompts"),
        help="artifact directory (default: %(default)s)",
    )
    examples.add_argument(
        "--hiddenbench-data",
        type=Path,
        default=DEFAULT_CORPUS_ROOT / "scaled" / "exact_replication" / "N_32.json",
        help="downloaded canonical or scaled HiddenBench JSON (default: %(default)s)",
    )
    examples.add_argument("--task-id", type=int, default=1)
    examples.add_argument("--agent-id", type=int, default=0)

    provider = commands.add_parser("provider", help="test one normalized LLM provider")
    provider_commands = provider.add_subparsers(dest="provider_command", required=True)
    provider_test = provider_commands.add_parser(
        "test", help="compile one prompt, run static preflight, and send one completion"
    )
    provider_test.add_argument(
        "--provider",
        choices=("mock", "openai", "university", "gemma_local"),
        required=True,
    )
    provider_test.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/components/prompts/basic_choice_v3.yaml"),
    )
    provider_test.add_argument("--provider-config", type=Path)
    provider_test.add_argument("--output-dir", type=Path, required=True)
    provider_test.add_argument("--logical-calls", type=int, default=1)
    provider_test.add_argument("--assumed-output-tokens", type=int, default=16)
    provider_test.add_argument("--max-output-tokens", type=int, default=16)
    provider_test.add_argument("--temperature", type=float, default=0.0)
    provider_test.add_argument("--seed", type=int, default=1026)
    provider_test.add_argument("--budget-usd", type=float)

    game = commands.add_parser("game", help="run a game through the generic interface")
    game_commands = game.add_subparsers(dest="game_command", required=True)
    game_run = game_commands.add_parser(
        "run", help="run one resolved game and write inspectable trajectory artifacts"
    )
    game_run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runs/old/toy_game_smoke_test.yaml"),
        help="run configuration (default: %(default)s)",
    )
    game_run.add_argument("--output-dir", type=Path, required=True)

    game_preflight = game_commands.add_parser(
        "preflight",
        help="price exactly one episode of this config; no provider calls sent",
    )
    game_preflight.add_argument("--config", type=Path, required=True)
    game_preflight.add_argument(
        "--output-dir",
        type=Path,
        help="artifact destination (default: this config's storage.output_dir)",
    )

    game_episode = game_commands.add_parser(
        "episode",
        help=(
            "run exactly one episode; progress, Comet, metrics, and prompt-example display "
            "are all read from the config's logging/metrics sections, not from flags"
        ),
    )
    game_episode.add_argument("--config", type=Path, required=True)
    game_episode.add_argument(
        "--output-dir",
        type=Path,
        help="artifact destination (default: this config's storage.output_dir)",
    )
    game_episode.add_argument(
        "--no-preflight",
        dest="skip_preflight",
        action="store_true",
        help="skip the cost/token estimate and its launch gate; runtime budget "
        "enforcement stays on regardless",
    )

    experiment = commands.add_parser(
        "experiment", help="price and run many concurrent episodes of one resolved game"
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    experiment_preflight = experiment_commands.add_parser(
        "preflight",
        help="estimate total calls/tokens/cost/runtime without provider I/O",
    )
    experiment_preflight.add_argument("--config", type=Path, required=True)
    experiment_preflight.add_argument(
        "--output-dir",
        type=Path,
        help="artifact destination (default: this config's storage.output_dir)",
    )
    experiment_run = experiment_commands.add_parser(
        "run", help="run config.execution.repetitions episodes concurrently"
    )
    experiment_run.add_argument("--config", type=Path, required=True)
    experiment_run.add_argument(
        "--output-dir",
        type=Path,
        help="artifact destination (default: this config's storage.output_dir)",
    )
    experiment_run.add_argument(
        "--approve-preflight",
        type=Path,
        help="path to a preflight_id.txt written by `experiment preflight`",
    )
    experiment_run.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip episodes already completed in --output-dir (default)",
    )
    experiment_run.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore any previously completed episodes and re-run everything",
    )
    experiment_run.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        default=True,
        help="disable tqdm progress bars; log one line per completed episode instead",
    )
    experiment_aggregate = experiment_commands.add_parser(
        "aggregate",
        help="recompute a finished run's cell aggregates from its episode files only",
    )
    experiment_aggregate.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="an `experiment run` output directory (a grid's, or a single experiment's)",
    )
    experiment_aggregate.add_argument(
        "--config",
        type=Path,
        help="take the aggregation: section from this config instead of the run's own; "
        "use to re-band, re-window, or change the fill rule without re-running episodes",
    )
    experiment_compact = experiment_commands.add_parser(
        "compact",
        help="preview or compact an existing full run into validated results-only artifacts",
    )
    experiment_compact.add_argument("--run-dir", type=Path, required=True)
    experiment_compact.add_argument(
        "--profile", choices=("results_only",), default="results_only"
    )
    experiment_compact.add_argument(
        "--delete-raw",
        action="store_true",
        help="after validated compact outputs exist, delete only the documented raw allowlist",
    )
    experiment_compact.add_argument(
        "--archive",
        action="store_true",
        help="also create a copy-friendly zip excluding :Zone.Identifier sidecars",
    )

    study = commands.add_parser(
        "study", help="submit and aggregate a folder of ordinary experiment configs"
    )
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_submit = study_commands.add_parser(
        "submit", help="preflight every config and submit one SLURM config array"
    )
    study_submit.add_argument("--config-dir", type=Path, required=True)
    study_submit.add_argument(
        "--results-dir",
        type=Path,
        help="common study result root (default: results/<study.name>)",
    )
    study_submit.add_argument(
        "--throttle",
        type=int,
        help="maximum number of simultaneously running SLURM array tasks",
    )
    study_submit.add_argument(
        "--job-script",
        type=Path,
        help="config-array job script (default: scripts/Potsdam/SLURM/run_config_array.job)",
    )
    study_aggregate = study_commands.add_parser(
        "aggregate",
        help="validate, normalize, analyze, plot, report, and package a study",
    )
    study_aggregate.add_argument("--study-dir", type=Path, required=True)
    study_aggregate.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="produce explicitly incomplete exploratory output despite validation failures",
    )

    synthetic = commands.add_parser(
        "synthetic",
        help="run the synthetic games whose closed-form answers we already know",
    )
    synthetic_commands = synthetic.add_subparsers(
        dest="synthetic_command", required=True
    )
    synthetic_truth = synthetic_commands.add_parser(
        "truth", help="print the closed form for a config without running anything"
    )
    synthetic_truth.add_argument("--config", type=Path, required=True)
    synthetic_truth.add_argument("--output-dir", type=Path)

    synthetic_episode = synthetic_commands.add_parser(
        "episode",
        help="fidelity mode: one episode through prompts, provider, parser and recorder",
    )
    synthetic_episode.add_argument("--config", type=Path, required=True)
    synthetic_episode.add_argument("--output-dir", type=Path)

    synthetic_sweep = synthetic_commands.add_parser(
        "sweep", help="speed mode: the null distribution and the calibration curve"
    )
    synthetic_sweep.add_argument("--config", type=Path, required=True)
    synthetic_sweep.add_argument("--output-dir", type=Path)
    synthetic_sweep.add_argument(
        "--seeds",
        type=int,
        default=200,
        help="episodes per grid point (default: %(default)s)",
    )
    synthetic_sweep.add_argument(
        "--epsilon-grid",
        nargs="+",
        type=float,
        help="noise levels to sweep (default: 0.0 to 0.5 in steps of 0.05)",
    )

    synthetic_empowerment = synthetic_commands.add_parser(
        "empowerment",
        help="exact I(C;O) and I(C;S_t+h|S_t) for a sweep, without running the sweep",
    )
    synthetic_empowerment.add_argument("--config", type=Path, required=True)
    synthetic_empowerment.add_argument("--output-dir", type=Path)
    synthetic_empowerment.add_argument(
        "--condition",
        default="epsilon",
        help="the swept axis: a game.options key, or population_size/horizon",
    )
    synthetic_empowerment.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="the condition values making up the grid",
    )
    synthetic_empowerment.add_argument(
        "--repetitions",
        type=int,
        default=50,
        help="episodes per cell (default: %(default)s)",
    )
    synthetic_empowerment.add_argument("--horizons", nargs="+", type=int, default=[1])
    synthetic_empowerment.add_argument(
        "--macrostate-bins",
        type=int,
        help="bin the macrostate onto this many common levels; required when sweeping "
        "population_size, and strongly advised otherwise (see the plug-in bias)",
    )

    synthetic_parity = synthetic_commands.add_parser(
        "parity", help="run both modes on the same seeds and demand the same trajectory"
    )
    synthetic_parity.add_argument("--config", type=Path, required=True)
    synthetic_parity.add_argument("--output-dir", type=Path)
    synthetic_parity.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="fidelity episodes to check (default: %(default)s)",
    )

    theory = commands.add_parser(
        "theory", help="run provider-free calculations for the single-affinity theory"
    )
    theory_commands = theory.add_subparsers(dest="theory_command", required=True)
    theory_simulate = theory_commands.add_parser(
        "simulate",
        help="simulate n -> Y -> U -> n' and compare trajectories with exact theory",
    )
    theory_simulate.add_argument(
        "--config", type=Path, required=True, help="theory simulation YAML"
    )
    theory_simulate.add_argument(
        "--output-dir", type=Path, required=True, help="artifact directory"
    )

    analysis = commands.add_parser(
        "analysis", help="offline analysis over completed run/grid output"
    )
    analysis_commands = analysis.add_subparsers(dest="analysis_command", required=True)
    empowerment = analysis_commands.add_parser(
        "empowerment",
        help="estimate mutual information between a swept control condition and the outcome",
    )
    empowerment.add_argument(
        "--grid-dir",
        type=Path,
        required=True,
        help="a completed `mas-cc experiment run --config <grid-config>` output directory",
    )
    empowerment.add_argument(
        "--output-dir",
        type=Path,
        help="analysis destination (default: <grid-dir>/analysis)",
    )
    empowerment.add_argument(
        "--condition-column",
        help="dotted grid-axis path to treat as the condition (default: the grid's only swept axis)",
    )
    empowerment.add_argument("--horizons", nargs="+", type=int, default=[1])
    empowerment.add_argument("--bootstrap-resamples", type=int, default=1000)
    empowerment.add_argument("--null-permutations", type=int, default=1000)
    empowerment.add_argument("--seed", type=int, default=1)

    imitation_analysis = analysis_commands.add_parser(
        "hidden-bench-imitation",
        help="derive the HiddenBench imitation pilot report and four discrete information estimates",
    )
    imitation_analysis.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="a completed hidden_bench_imitation run or grid directory",
    )
    imitation_analysis.add_argument(
        "--output-dir",
        type=Path,
        help="analysis destination (default: <run-dir>/hidden_bench_imitation_analysis)",
    )
    imitation_analysis.add_argument("--bootstrap-resamples", type=int, default=1000)
    imitation_analysis.add_argument("--null-permutations", type=int, default=1000)
    imitation_analysis.add_argument("--confidence", type=float, default=0.95)
    imitation_analysis.add_argument("--seed", type=int, default=1)

    round_feedback_analysis = analysis_commands.add_parser(
        "hidden-bench-round-feedback",
        help="derive round-level sensing, actuation, slot, and current estimates",
    )
    round_feedback_analysis.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="a completed hidden_bench_imitation_round_feedback run or grid directory",
    )
    round_feedback_analysis.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "analysis destination (default: "
            "<run-dir>/hidden_bench_imitation_round_feedback_analysis)"
        ),
    )
    round_feedback_analysis.add_argument(
        "--bootstrap-resamples", type=int, default=1000
    )
    round_feedback_analysis.add_argument("--null-permutations", type=int, default=1000)
    round_feedback_analysis.add_argument("--confidence", type=float, default=0.95)
    round_feedback_analysis.add_argument("--seed", type=int, default=1)

    relational_analysis = analysis_commands.add_parser(
        "relational-round-feedback",
        help="the same round-level estimators over a relational reasoning grid",
    )
    relational_analysis.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="a completed relational_imitation_round_feedback run or grid directory",
    )
    relational_analysis.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "analysis destination (default: "
            "<run-dir>/relational_imitation_round_feedback_analysis)"
        ),
    )
    relational_analysis.add_argument("--bootstrap-resamples", type=int, default=1000)
    relational_analysis.add_argument("--null-permutations", type=int, default=1000)
    relational_analysis.add_argument("--confidence", type=float, default=0.95)
    relational_analysis.add_argument("--seed", type=int, default=1)
    relational_analysis.add_argument(
        "--epistemic-bins",
        type=int,
        default=4,
        help="bins per axis for the coarse (kappa, phi) diagnostic conditioning",
    )

    benchmark = commands.add_parser(
        "benchmark",
        help="single-model task-validation benchmarks (no game, no agents)",
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", required=True
    )
    relational_support = benchmark_commands.add_parser(
        "relational-support",
        help="does holding the supporting facts decide whether a model solves the task?",
    )
    relational_support_commands = relational_support.add_subparsers(
        dest="relational_support_command", required=True
    )
    for name, description in (
        ("preflight", "generate tasks, render every prompt, validate; send nothing"),
        ("run", "preflight, then send one request per item to one model"),
    ):
        sub = relational_support_commands.add_parser(name, help=description)
        sub.add_argument("--config", type=Path, required=True, help="benchmark YAML")
        sub.add_argument(
            "--output-dir",
            type=Path,
            help="artifact directory (default: output.dir/<benchmark name> from the config)",
        )
        sub.add_argument(
            "--skip-reproducibility-check",
            action="store_true",
            help="skip regenerating each task from its stored seed (faster, weaker)",
        )
    relational_support_summarize = relational_support_commands.add_parser(
        "summarize",
        help="turn rows.jsonl into the A_k, per-subset and L=2 headline tables",
    )
    relational_support_summarize.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="a completed benchmark output directory",
    )
    relational_support_summarize.add_argument("--output-dir", type=Path)

    probe = commands.add_parser(
        "probe",
        help="provider-backed diagnostic probes; one prompt per call, no population loop",
    )
    probe_commands = probe.add_subparsers(dest="probe_command", required=True)
    for name, description in (
        ("preflight", "count calls and run every design check; send nothing"),
        ("run", "preflight, then send the outstanding calls and write the report"),
        ("analyze", "rebuild tables, plots, and the report from an existing run"),
    ):
        sub = probe_commands.add_parser(name, help=description)
        sub.add_argument(
            "--config",
            type=Path,
            default=Path("configs/probes/controller_retention.yaml"),
            help="probe configuration (default: %(default)s)",
        )
        sub.add_argument(
            "--output-dir",
            type=Path,
            help="artifact directory (default: storage.output_dir from the config)",
        )

    inspect = commands.add_parser(
        "inspect", help="produce stable phase inspection artifacts"
    )
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    phase = inspect_commands.add_parser("phase", help="inspect one implemented phase")
    phase.add_argument(
        "number",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7),
        help="implemented phase number",
    )
    phase.add_argument(
        "--config",
        type=Path,
        help="run config (defaults to the phase-specific smoke-test config)",
    )
    phase.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory (default: results/inspection/phase_NN)",
    )
    phase.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/components/prompts/basic_choice_v3.yaml"),
        help="prompt component for Phase 3 (default: %(default)s)",
    )
    phase.add_argument("--amendment", choices=("provider-economics-v2",))
    phase.add_argument(
        "--provider",
        choices=("mock", "openai", "university", "gemma_local"),
        default="university",
    )
    phase.add_argument(
        "--pricing-mode", choices=("live", "cached", "offline"), default="offline"
    )
    phase.add_argument("--pricing-cache", type=Path)
    phase.add_argument(
        "--live-completion",
        action="store_true",
        help="after immediate live revalidation, send one billable completion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "prompt" and args.prompt_command == "examples":
        from .prompt import generate_paper_prompt_examples

        try:
            destination = generate_paper_prompt_examples(
                args.output_dir,
                hiddenbench_data=args.hiddenbench_data,
                task_id=args.task_id,
                agent_id=args.agent_id,
            )
        except (OSError, ValueError, ConfigurationError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Paper prompt examples written: {destination}")
        return 0
    if args.command == "provider" and args.provider_command == "test":
        from .provider import run_provider_smoke_test

        try:
            passed = run_provider_smoke_test(
                args.provider,
                args.prompt,
                args.output_dir,
                provider_config_path=args.provider_config,
                logical_calls=args.logical_calls,
                assumed_output_tokens=args.assumed_output_tokens,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                seed=args.seed,
                budget_usd=args.budget_usd,
            )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Provider {args.provider} inspection {'passed' if passed else 'failed'}: "
            f"{args.output_dir}"
        )
        return 0 if passed else 1
    if args.command == "game" and args.game_command == "run":
        from .game import run_game_inspection

        try:
            passed = run_game_inspection(args.config, args.output_dir)
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Game inspection {'passed' if passed else 'failed'}: {args.output_dir}")
        return 0 if passed else 1
    if args.command == "game" and args.game_command == "preflight":
        from .episode import run_game_preflight

        try:
            estimate = run_game_preflight(args.config, args.output_dir)
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Game preflight {estimate.launch_status}: {args.config}")
        return 0 if estimate.launch_status == "permitted" else 1
    if args.command == "game" and args.game_command == "episode":
        from .episode import run_game_episode

        try:
            run_game_episode(
                args.config, args.output_dir, skip_preflight=args.skip_preflight
            )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if args.command == "experiment" and args.experiment_command == "preflight":
        from mas_cc.planning import GridPreflightEstimate

        from .experiment import resolve_output_dir, run_experiment_preflight

        try:
            estimate = run_experiment_preflight(args.config, args.output_dir)
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        destination = resolve_output_dir(args.config, args.output_dir)
        if isinstance(estimate, GridPreflightEstimate):
            print(
                f"Grid preflight {estimate.launch_status}: {estimate.cell_count} cell(s), "
                f"{estimate.total_episode_count} episode(s) total, {destination}"
            )
        else:
            print(
                f"Experiment preflight {estimate.launch_status}: {estimate.episode_count} "
                f"episode(s), {destination}"
            )
        return 0 if estimate.launch_status == "permitted" else 1
    if args.command == "experiment" and args.experiment_command == "run":
        from .experiment import run_experiment_command

        try:
            result = run_experiment_command(
                args.config,
                args.output_dir,
                resume=args.resume,
                show_progress=args.show_progress,
                approve_preflight=args.approve_preflight,
            )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Experiment {result.experiment_name}: {result.completed} completed, "
            f"{result.failed} failed, {result.skipped_resumed} skipped (resumed): "
            f"{result.output_dir}"
        )
        return 0 if result.failed == 0 else 1
    if args.command == "experiment" and args.experiment_command == "aggregate":
        from .experiment import run_aggregate_command

        try:
            summary = run_aggregate_command(args.run_dir, args.config)
        except (ConfigurationError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Aggregated {len(summary['cells_aggregated'])} cell(s) from {args.run_dir}"
        )
        for name, value in sorted(summary["sweep_metrics"].items()):
            print(f"  {name}: {value}")
        return 0 if summary["cells_aggregated"] else 1
    if args.command == "experiment" and args.experiment_command == "compact":
        from .experiment import run_compact_command

        try:
            summary = run_compact_command(
                args.run_dir,
                profile=args.profile,
                delete_raw=args.delete_raw,
                archive=args.archive,
            )
        except (ConfigurationError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        mode = "preview" if summary["dry_run"] else "completed"
        print(f"Compaction {mode}: {summary['run_dir']}")
        print(
            f"  before: {summary['before']['files']} files, {summary['before']['bytes']} bytes"
        )
        print(
            f"  after:  {summary['after']['files']} files, {summary['after']['bytes']} bytes"
        )
        if summary.get("archive"):
            print(f"  archive: {summary['archive']}")
        return 0
    if args.command == "study" and args.study_command == "submit":
        from mas_cc.studies import submit_study

        try:
            result = submit_study(
                args.config_dir,
                args.results_dir,
                throttle=args.throttle,
                job_script=args.job_script,
            )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Study {result.study_dir.name} submitted as SLURM job {result.job_id}: "
            f"{len(result.entries)} config(s), {result.study_dir}"
        )
        if result.execution_plan is not None:
            plan = result.execution_plan
            print(
                f"  Execution: {plan['shard_count']} cell shard(s), "
                f"throttle {plan['array_throttle']}, "
                f"{plan['total_episode_slots']} episode slot(s), "
                f"{plan['total_request_concurrency']} request slot(s), "
                f"~{plan['estimated_rpm']:.0f} RPM"
            )
            print(
                f"  Resources: {plan['cpus_per_task']} CPU(s), {plan['memory']}, "
                f"{plan['time_limit']} per active shard"
            )
        return 0
    if args.command == "study" and args.study_command == "aggregate":
        from mas_cc.studies import aggregate_study

        try:
            summary = aggregate_study(
                args.study_dir, allow_incomplete=args.allow_incomplete
            )
        except (ConfigurationError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Study {summary['study_id']} aggregated "
            f"({'complete' if summary['complete'] else 'incomplete'}): "
            f"{summary['archive']}"
        )
        return 0 if summary["complete"] else 1
    if args.command == "synthetic":
        from .synthetic import (
            DEFAULT_EPSILON_GRID,
            run_synthetic_empowerment,
            run_synthetic_episode,
            run_synthetic_parity,
            run_synthetic_sweep,
            run_synthetic_truth,
        )

        try:
            if args.synthetic_command == "truth":
                run_synthetic_truth(args.config, args.output_dir)
            elif args.synthetic_command == "episode":
                run_synthetic_episode(args.config, args.output_dir)
            elif args.synthetic_command == "empowerment":
                # Values arrive as strings; a numeric axis needs numbers, and a
                # non-numeric one must survive untouched.
                def _typed(raw: str):
                    try:
                        return int(raw)
                    except ValueError:
                        try:
                            return float(raw)
                        except ValueError:
                            return raw

                run_synthetic_empowerment(
                    args.config,
                    args.output_dir,
                    condition=args.condition,
                    values=[_typed(value) for value in args.values],
                    repetitions=args.repetitions,
                    horizons=tuple(args.horizons),
                    macrostate_bins=args.macrostate_bins,
                )
            elif args.synthetic_command == "sweep":
                run_synthetic_sweep(
                    args.config,
                    args.output_dir,
                    seeds=args.seeds,
                    epsilon_grid=tuple(args.epsilon_grid or DEFAULT_EPSILON_GRID),
                )
            else:
                summary = run_synthetic_parity(
                    args.config, args.output_dir, seeds=args.seeds
                )
                # A parity failure means the full pipeline and the bare sampler
                # disagree on the same seed, which is a real finding, not a
                # warning - so it leaves through a non-zero exit status.
                return 0 if summary["all_identical"] else 1
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if args.command == "theory" and args.theory_command == "simulate":
        from mas_cc.games.relational_reasoning.imitation_round_feedback.theory_simulation import (
            run_theory_simulation,
        )

        try:
            result = run_theory_simulation(args.config, args.output_dir)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Single-affinity theory simulation completed: "
            f"{result.config.episodes} episode(s), {result.config.rounds} cycle(s), "
            f"{args.output_dir}"
        )
        return 0
    if args.command == "analysis" and args.analysis_command == "empowerment":
        from .analysis import run_analysis_empowerment_command

        try:
            summary = run_analysis_empowerment_command(
                args.grid_dir,
                args.output_dir,
                condition_column=args.condition_column,
                horizons=tuple(args.horizons),
                bootstrap_resamples=args.bootstrap_resamples,
                null_permutations=args.null_permutations,
                seed=args.seed,
            )
        except (ConfigurationError, OSError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "analysis" and args.analysis_command == "hidden-bench-imitation":
        from .analysis import run_hidden_bench_imitation_analysis_command

        try:
            summary = run_hidden_bench_imitation_analysis_command(
                args.run_dir,
                args.output_dir,
                bootstrap_resamples=args.bootstrap_resamples,
                null_permutations=args.null_permutations,
                confidence=args.confidence,
                seed=args.seed,
            )
        except (ConfigurationError, OSError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if (
        args.command == "analysis"
        and args.analysis_command == "hidden-bench-round-feedback"
    ):
        from .analysis import run_hidden_bench_round_feedback_analysis_command

        try:
            summary = run_hidden_bench_round_feedback_analysis_command(
                args.run_dir,
                args.output_dir,
                bootstrap_resamples=args.bootstrap_resamples,
                null_permutations=args.null_permutations,
                confidence=args.confidence,
                seed=args.seed,
            )
        except (ConfigurationError, OSError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if (
        args.command == "analysis"
        and args.analysis_command == "relational-round-feedback"
    ):
        from .analysis import run_relational_round_feedback_analysis_command

        try:
            summary = run_relational_round_feedback_analysis_command(
                args.run_dir,
                args.output_dir,
                bootstrap_resamples=args.bootstrap_resamples,
                null_permutations=args.null_permutations,
                confidence=args.confidence,
                seed=args.seed,
                epistemic_bins=args.epistemic_bins,
            )
        except (ConfigurationError, OSError, ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark" and args.benchmark_command == "relational-support":
        from .benchmark import (
            run_relational_support_benchmark,
            run_relational_support_preflight,
            summarize_relational_support_benchmark,
        )

        try:
            if args.relational_support_command == "summarize":
                destination = summarize_relational_support_benchmark(
                    args.input_dir, args.output_dir
                )
                print(f"Benchmark summary written: {destination}")
                return 0
            handler = (
                run_relational_support_preflight
                if args.relational_support_command == "preflight"
                else run_relational_support_benchmark
            )
            ok, destination, message = handler(
                args.config,
                args.output_dir,
                verify_reproducibility=not args.skip_reproducibility_check,
            )
        except (
            ConfigurationError,
            ProviderError,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"{message}: {destination}")
        return 0 if ok else 1
    if args.command == "probe":
        from .probe import run_controller_retention_probe

        try:
            ok, destination, message = run_controller_retention_probe(
                args.config, args.output_dir, mode=args.probe_command
            )
        except (
            ConfigurationError,
            ProviderError,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"{message}: {destination}")
        return 0 if ok else 1
    if args.command == "inspect" and args.inspect_command == "phase":
        from .inspect import inspect_phase_1, inspect_phase_2, inspect_phase_3

        output_dir = args.output_dir or Path(
            f"results/inspection/phase_{args.number:02d}"
        )
        try:
            if args.number == 1:
                passed = inspect_phase_1(output_dir)
            elif args.number == 2:
                passed = inspect_phase_2(
                    args.config or Path("configs/runs/old/provider_smoke_test.yaml"),
                    output_dir,
                )
            elif args.number == 3:
                passed = inspect_phase_3(args.prompt, output_dir)
            elif args.number == 4:
                if args.amendment == "provider-economics-v2":
                    from .provider_economics import inspect_phase_4_amendment

                    passed = inspect_phase_4_amendment(
                        args.config
                        or Path("configs/runs/old/provider_smoke_test.yaml"),
                        output_dir,
                        provider=args.provider,
                        pricing_mode=args.pricing_mode,
                        cache_path=args.pricing_cache,
                        live_completion=args.live_completion,
                    )
                else:
                    from .provider import run_provider_smoke_test

                    passed = run_provider_smoke_test("mock", args.prompt, output_dir)
            elif args.number == 5:
                from .game import run_game_inspection

                passed = run_game_inspection(
                    args.config or Path("configs/runs/old/toy_game_smoke_test.yaml"),
                    output_dir,
                )
            elif args.number == 6:
                from .game import run_game_inspection

                passed = run_game_inspection(
                    args.config
                    or Path("configs/runs/old/naming_convention_smoke_test.yaml"),
                    output_dir,
                )
            else:
                from .phase7 import run_phase_7_inspection

                passed = run_phase_7_inspection(
                    args.config
                    or Path("configs/runs/old/naming_convention_smoke_test_v3.yaml"),
                    output_dir,
                )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Phase {args.number} inspection {'passed' if passed else 'failed'}: {output_dir}"
        )
        return 0 if passed else 1
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
