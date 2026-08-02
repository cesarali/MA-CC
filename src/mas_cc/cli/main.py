"""Command-line interface for the staged :mod:`mas_cc` implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from mas_cc import __version__
from mas_cc.core.exceptions import ConfigurationError, ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mas-cc",
        description="Inspect and run the staged MAS-CC language-game framework.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="print the installed MAS-CC version")

    prompt = commands.add_parser("prompt", help="compile and inspect prompts without an LLM")
    prompt_commands = prompt.add_subparsers(dest="prompt_command", required=True)
    examples = prompt_commands.add_parser(
        "examples", help="write paper-grounded prompts as readable Markdown"
    )
    examples.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inspection/paper_prompts"),
        help="artifact directory (default: %(default)s)",
    )
    examples.add_argument(
        "--hiddenbench-data",
        type=Path,
        default=Path(
            "scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/"
            "scaled/exact_replication/N_32.json"
        ),
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
        "--provider", choices=("mock", "openai", "university", "gemma_local"), required=True
    )
    provider_test.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/components/prompts/basic_binary_choice.yaml"),
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
        default=Path("configs/runs/toy_game_smoke_test.yaml"),
        help="run configuration (default: %(default)s)",
    )
    game_run.add_argument("--output-dir", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="produce stable phase inspection artifacts")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    phase = inspect_commands.add_parser("phase", help="inspect one implemented phase")
    phase.add_argument("number", type=int, choices=(1, 2, 3, 4, 5), help="implemented phase number")
    phase.add_argument(
        "--config",
        type=Path,
        help="run config (defaults to the phase-specific smoke-test config)",
    )
    phase.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory (default: inspection/phase_NN)",
    )
    phase.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/components/prompts/basic_binary_choice.yaml"),
        help="prompt component for Phase 3 (default: %(default)s)",
    )
    phase.add_argument("--amendment", choices=("provider-economics-v2",))
    phase.add_argument("--provider", choices=("mock", "openai", "university", "gemma_local"), default="university")
    phase.add_argument("--pricing-mode", choices=("live", "cached", "offline"), default="offline")
    phase.add_argument("--pricing-cache", type=Path)
    phase.add_argument(
        "--live-completion", action="store_true",
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
    if args.command == "inspect" and args.inspect_command == "phase":
        from .inspect import inspect_phase_1, inspect_phase_2, inspect_phase_3

        output_dir = args.output_dir or Path(f"inspection/phase_{args.number:02d}")
        try:
            if args.number == 1:
                passed = inspect_phase_1(output_dir)
            elif args.number == 2:
                passed = inspect_phase_2(
                    args.config or Path("configs/runs/provider_smoke_test.yaml"), output_dir
                )
            elif args.number == 3:
                passed = inspect_phase_3(args.prompt, output_dir)
            elif args.number == 4:
                if args.amendment == "provider-economics-v2":
                    from .provider_economics import inspect_phase_4_amendment

                    passed = inspect_phase_4_amendment(
                        args.config or Path("configs/runs/provider_smoke_test.yaml"),
                        output_dir, provider=args.provider,
                        pricing_mode=args.pricing_mode, cache_path=args.pricing_cache,
                        live_completion=args.live_completion,
                    )
                else:
                    from .provider import run_provider_smoke_test

                    passed = run_provider_smoke_test("mock", args.prompt, output_dir)
            else:
                from .game import run_game_inspection

                passed = run_game_inspection(
                    args.config or Path("configs/runs/toy_game_smoke_test.yaml"), output_dir
                )
        except (ConfigurationError, ProviderError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Phase {args.number} inspection {'passed' if passed else 'failed'}: {output_dir}")
        return 0 if passed else 1
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
