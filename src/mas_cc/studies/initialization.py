"""Materialize shared relational initial conditions before cell-array execution."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from mas_cc.config import GridSpec, RunConfig, load_run_config_or_grid
from mas_cc.core import Seed
from mas_cc.games import create_game
from mas_cc.games.relational_reasoning.imitation_round_feedback.initialization import (
    artifact_from_actions,
    initialization_artifact_path,
    initialization_compatibility_key,
    paired_initialization_required,
    read_initialization_artifact,
    write_initialization_artifact,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    _execute_decision,
)
from mas_cc.llm_runtime.prompts import RegexTokenCounter
from mas_cc.llm_runtime.providers import (
    BudgetGuardedProvider,
    RuntimeBudgetGuard,
    create_llm_provider,
    resolve_budget_limits,
)
from mas_cc.planning import estimate_input_tokens


@dataclass(frozen=True, slots=True)
class InitializationPlanEntry:
    repetition_index: int
    episode_seed: int
    compatibility_key: str
    artifact_path: str


def _representative_configs(
    config_paths: Sequence[str | Path],
) -> tuple[RunConfig, ...]:
    result: list[RunConfig] = []
    for path in config_paths:
        source = load_run_config_or_grid(path)
        config = source.cells[0].config if isinstance(source, GridSpec) else source
        if not paired_initialization_required(config):
            continue
        result.append(config)
    if not result:
        raise ValueError("no paired_local_vote study config was found")
    return tuple(result)


def build_initialization_plan(
    config_paths: Sequence[str | Path], artifact_dir: str | Path
) -> tuple[InitializationPlanEntry, ...]:
    configs = _representative_configs(config_paths)
    repetitions = {config.execution.repetitions for config in configs}
    seeds = {config.execution.seed for config in configs}
    if len(repetitions) != 1 or len(seeds) != 1:
        raise ValueError(
            "paired study configs must use the same repetitions and root seed"
        )
    count, root_seed = repetitions.pop(), seeds.pop()
    directory = Path(artifact_dir).expanduser().resolve()
    entries: list[InitializationPlanEntry] = []
    for repetition in range(count):
        episode_seed = int(Seed(root_seed).derive(f"episode:{repetition}"))
        keys = {
            initialization_compatibility_key(
                create_game(config.game), config, episode_seed
            )
            for config in configs
        }
        if len(keys) != 1:
            raise ValueError(
                "false/truth configs do not have identical initialization compatibility"
            )
        entries.append(
            InitializationPlanEntry(
                repetition_index=repetition,
                episode_seed=episode_seed,
                compatibility_key=keys.pop(),
                artifact_path=str(directory / f"episode-seed-{episode_seed}.json"),
            )
        )
    return tuple(entries)


async def materialize_initializations(
    config_paths: Sequence[str | Path],
    artifact_dir: str | Path,
    provider_factory: Callable[[RunConfig], Any],
) -> tuple[InitializationPlanEntry, ...]:
    """Generate every repetition once; this command is a prerequisite, not a race."""

    configs = _representative_configs(config_paths)
    representative = configs[0]
    plan = build_initialization_plan(config_paths, artifact_dir)
    for entry in plan:
        episode_config = replace(
            representative,
            execution=replace(representative.execution, seed=entry.episode_seed),
        )
        game = create_game(episode_config.game)
        destination = Path(entry.artifact_path)
        if destination.is_file():
            read_initialization_artifact(
                destination, game, episode_config, entry.episode_seed
            )
            continue
        state = game.initialize(episode_config.game, entry.episode_seed)
        provider = provider_factory(episode_config)
        try:
            decisions = tuple(
                await asyncio.gather(
                    *(
                        _execute_decision(
                            game,
                            request,
                            state,
                            episode_config,
                            provider,
                            RegexTokenCounter(),
                            Seed(entry.episode_seed),
                            None,
                        )
                        for request in game.initial_vote_requests(
                            state, episode_config.game
                        )
                    )
                )
            )
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                close()
        artifact = artifact_from_actions(
            game,
            episode_config,
            entry.episode_seed,
            tuple(decision.action for decision in decisions),
            repetition_index=entry.repetition_index,
        )
        write_initialization_artifact(destination, artifact)
    return plan


def materialize_study_initializations(
    config_dirs: Sequence[str | Path], output_dir: str | Path
) -> tuple[InitializationPlanEntry, ...]:
    """Provider-backed prerequisite for a matched study pair; no dynamics run."""

    from mas_cc.cli.game import _budgets, _quote
    from mas_cc.studies.manifest import discover_study

    config_paths = [
        path for directory in config_dirs for path in discover_study(directory).configs
    ]
    representative = _representative_configs(config_paths)[0]
    quote = _quote(representative)
    system_budget, run_budget = _budgets(representative, quote)
    limits = resolve_budget_limits(system_budget, run_budget)

    def provider_factory(config: RunConfig) -> Any:
        return BudgetGuardedProvider(
            create_llm_provider(config.llm_provider),
            RuntimeBudgetGuard(limits),
            quote.pricing,
            input_token_estimator=estimate_input_tokens,
            input_token_multiplier=1.0,
        )

    plan = asyncio.run(
        materialize_initializations(config_paths, output_dir, provider_factory)
    )
    destination = Path(output_dir).expanduser().resolve()
    (destination / "initialization_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configs": [str(Path(path).resolve()) for path in config_paths],
                "artifacts": [asdict(entry) for entry in plan],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan


__all__ = [
    "InitializationPlanEntry",
    "build_initialization_plan",
    "materialize_initializations",
    "materialize_study_initializations",
]
