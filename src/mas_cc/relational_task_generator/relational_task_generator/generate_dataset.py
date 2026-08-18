#!/usr/bin/env python3
"""Command-line entry point for generating frozen relational reasoning tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from generator import write_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic multi-agent spatial reasoning tasks."
    )
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--reasoning-depth", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--support-redundancy", type=int, default=6)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--distractor-redundancy", type=int, default=1)
    parser.add_argument("--num-options", type=int, choices=range(2, 9), default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-single-agent-solution",
        dest="no_single_agent_solution",
        action="store_true",
        help="Guarantee that no individual agent initially has all supporting facts.",
    )
    parser.add_argument(
        "--allow-single-agent-solution",
        dest="no_single_agent_solution",
        action="store_false",
        help="Allow an individual agent to receive all supporting facts (default).",
    )
    parser.set_defaults(no_single_agent_solution=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace task/manifest files in a non-empty output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = write_dataset(
        output_dir=args.output,
        num_tasks=args.num_tasks,
        population_size=args.population_size,
        reasoning_depth=args.reasoning_depth,
        support_redundancy=args.support_redundancy,
        distractors=args.distractors,
        distractor_redundancy=args.distractor_redundancy,
        num_options=args.num_options,
        seed=args.seed,
        no_single_agent_solution=args.no_single_agent_solution,
        overwrite=args.overwrite,
    )
    print(
        f"Generated {manifest['num_tasks']} validated tasks in {args.output} "
        f"(fingerprint {manifest['dataset_fingerprint_sha256'][:12]}...)."
    )


if __name__ == "__main__":
    main()
