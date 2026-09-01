"""Command-line interface for native frozen Team Allocation generation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

from mas_cc.config.loader import load_component_config
from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.providers.registry import create_llm_provider

from .generate import GenerationConfig, generate_dataset
from .io_utils import sha256_file
from .provider_adapter import MuSRGenerationModel
from .validate import validate_frozen_task
from .validation_study import ValidationStudyConfig, run_validation_study
from .validation_comparison import add_validation_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate native MuSR-style Team Allocation tasks through MAS-CC providers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate a frozen dataset")
    generate.add_argument(
        "--provider", type=Path, required=True, help="MAS-CC provider YAML"
    )
    generate.add_argument("--model", help="override the provider component's model")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--num-tasks", type=int, default=20)
    generate.add_argument("--population-size", type=int, default=24)
    generate.add_argument("--branches-per-latent-fact", type=int, default=3)
    generate.add_argument("--statements-per-branch", type=int, default=2)
    generate.add_argument("--tree-depth", type=int, default=2)
    generate.add_argument("--evidence-redundancy", type=int, default=3)
    generate.add_argument("--min-margin", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--semantic-retries", type=int, default=3)
    generate.add_argument("--world-retries", type=int, default=3)
    generate.add_argument("--validation-attempts", type=int, default=3)
    generate.add_argument("--validation-required", type=int, default=2)
    generate.add_argument(
        "--skip-full-information-validation",
        action="store_true",
        help="development only; generated tasks still need full-information QA before use",
    )
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--max-output-tokens", type=int, default=2048)

    validate = subparsers.add_parser(
        "validate", help="validate a frozen dataset directory"
    )
    validate.add_argument("dataset_dir", type=Path)
    study = subparsers.add_parser(
        "validation-study", help="run the systematic three-world real-provider pilot"
    )
    study.add_argument(
        "--provider", type=Path, required=True, help="MAS-CC provider YAML"
    )
    study.add_argument("--model", default="microsoft/gpt-5.6-terra")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--seed", type=int, default=20260901)
    study.add_argument("--candidate-limit", type=int, default=12)
    study.add_argument("--branches-per-latent-fact", type=int, default=3)
    study.add_argument("--tree-depth", type=int, default=2)
    study.add_argument("--evidence-redundancy", type=int, default=3)
    study.add_argument("--generation-temperature", type=float, default=1.0)
    study.add_argument("--validation-temperature", type=float, default=1.0)
    compare = subparsers.add_parser(
        "add-validation-model",
        help="validate existing frozen study tasks with one additional model",
    )
    compare.add_argument(
        "--provider", type=Path, required=True, help="MAS-CC provider YAML"
    )
    compare.add_argument("--model", required=True)
    compare.add_argument("--study-dir", type=Path, required=True)
    compare.add_argument("--seed", type=int, default=20260902)
    compare.add_argument("--temperature", type=float, default=1.0)
    compare.add_argument("--max-output-tokens", type=int, default=1024)
    return parser


def _provider_config(path: Path, model_override: str | None) -> LLMProviderConfig:
    loaded = load_component_config(path, "llm_provider")
    if not isinstance(loaded, LLMProviderConfig):
        raise TypeError("provider component did not resolve to LLMProviderConfig")
    if model_override is None:
        return loaded
    values = loaded.to_dict()
    values["model"] = model_override
    return LLMProviderConfig(**values)


def validate_dataset_dir(path: Path) -> None:
    directory = path.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("manifest must be a JSON object")
    failures: list[str] = []
    task_hashes = manifest.get("task_hashes")
    if not isinstance(task_hashes, Mapping):
        raise RuntimeError("manifest task_hashes must be an object")
    if manifest.get("num_tasks") != len(task_hashes):
        failures.append("manifest num_tasks does not match task_hashes")
    expected_files = {str(filename) for filename in task_hashes}
    actual_files = {item.name for item in directory.glob("*.json")} - {"manifest.json"}
    if actual_files != expected_files:
        failures.append("dataset task files do not exactly match the manifest")
    for filename, expected_hash in task_hashes.items():
        task_path = directory / str(filename)
        if not task_path.exists():
            failures.append(f"missing {filename}")
            continue
        if sha256_file(task_path) != expected_hash:
            failures.append(f"hash mismatch: {filename}")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        failures.extend(f"{filename}: {error}" for error in validate_frozen_task(task))
    if failures:
        raise RuntimeError("dataset validation failed:\n- " + "\n- ".join(failures))


async def _generate(args: argparse.Namespace) -> dict[str, Any]:
    provider = create_llm_provider(_provider_config(args.provider, args.model))
    try:
        model = MuSRGenerationModel(
            provider,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        config = GenerationConfig(
            num_tasks=args.num_tasks,
            population_size=args.population_size,
            branches_per_latent_fact=args.branches_per_latent_fact,
            statements_per_branch=args.statements_per_branch,
            tree_depth=args.tree_depth,
            evidence_redundancy=args.evidence_redundancy,
            min_margin=args.min_margin,
            seed=args.seed,
            semantic_retries=args.semantic_retries,
            world_retries=args.world_retries,
            full_validation_attempts=args.validation_attempts,
            full_validation_required=args.validation_required,
            run_full_information_validation=not args.skip_full_information_validation,
        )
        return await generate_dataset(model, config, output=args.output)
    finally:
        provider.close()


async def _validation_study(args: argparse.Namespace) -> dict[str, Any]:
    provider = create_llm_provider(_provider_config(args.provider, args.model))
    try:
        config = ValidationStudyConfig(
            seed=args.seed,
            candidate_limit=args.candidate_limit,
            branches_per_latent_fact=args.branches_per_latent_fact,
            tree_depth=args.tree_depth,
            evidence_redundancy=args.evidence_redundancy,
            generation_temperature=args.generation_temperature,
            validation_temperature=args.validation_temperature,
        )
        repository_root = Path(__file__).resolve().parents[3]
        return await run_validation_study(
            provider,
            config,
            output=args.output,
            repository_root=repository_root,
        )
    finally:
        provider.close()


async def _add_validation_model(args: argparse.Namespace) -> dict[str, Any]:
    provider = create_llm_provider(_provider_config(args.provider, args.model))
    try:
        return await add_validation_model(
            provider,
            study_dir=args.study_dir,
            seed=args.seed,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    finally:
        provider.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate_dataset_dir(args.dataset_dir)
        print(f"Validated {args.dataset_dir}")
        return
    if args.command == "validation-study":
        result = asyncio.run(_validation_study(args))
        print(f"Completed validation study: {result['output']}")
        for row in result["behavioral"]:
            population = row.get("population_size")
            label = (
                row["condition"]
                if population is None
                else f"{row['condition']}_N{population}"
            )
            print(f"{label}: {row['accuracy']:.6f}")
        return
    if args.command == "add-validation-model":
        result = asyncio.run(_add_validation_model(args))
        print(f"Added validation model to: {result['output']}")
        for row in result["summary"]:
            population = row.get("population_size")
            label = (
                row["condition"]
                if population is None
                else f"{row['condition']}_N{population}"
            )
            print(f"{row['model']} {label}: {row['accuracy']:.6f}")
        return
    manifest = asyncio.run(_generate(args))
    print(f"Wrote {manifest['num_tasks']} tasks to {args.output}")
    print(f"Dataset fingerprint: {manifest['dataset_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
