"""Banner and live progress reporting for experiment runs (Phase 9 console UX)."""

from __future__ import annotations

import logging
import sys
from typing import Any

LOGGER = logging.getLogger("mas_cc.experiment")


def format_money(amount: Any) -> str:
    if amount is None:
        return "unknown"
    return f"{amount.amount:.2f} {amount.unit}"


def format_banner(
    *,
    experiment_name: str,
    game_type: str,
    game_version: int,
    provider: str,
    model: str,
    episode_count: int,
    concurrency: int,
    prompt_family: str,
    prompt_version: int,
    prompt_definition_hash: str,
    budget_description: str,
    preflight_expected_cost: str,
    preflight_conservative_cost: str,
    preflight_status: str,
) -> str:
    definition_prefix = prompt_definition_hash[:8] if prompt_definition_hash else "n/a"
    lines = [
        f"Experiment: {experiment_name}",
        f"  Game:          {game_type} v{game_version}",
        f"  Provider:      {provider} / {model}",
        f"  Episodes:      {episode_count}  (parallelism: {concurrency})",
        f"  Prompt:        {prompt_family} v{prompt_version}  [def:{definition_prefix}...]",
        f"  Budget:        {budget_description}",
        f"  Preflight:     expected {preflight_expected_cost} / conservative "
        f"{preflight_conservative_cost} — {preflight_status}",
    ]
    return "\n".join(lines)


def format_grid_banner(
    *,
    experiment_name: str,
    game_type: str,
    game_version: int,
    provider: str,
    model: str,
    cell_count: int,
    total_episode_count: int,
    concurrency: int,
    axes: tuple[tuple[str, int], ...],
    preflight_expected_cost: str,
    preflight_conservative_cost: str,
    preflight_status: str,
) -> str:
    axes_description = ", ".join(f"{path} x{count}" for path, count in axes)
    lines = [
        f"Grid experiment: {experiment_name}",
        f"  Game:          {game_type} v{game_version}",
        f"  Provider:      {provider} / {model}",
        f"  Cells:         {cell_count}  ({axes_description})",
        f"  Episodes:      {total_episode_count} total  (parallelism: {concurrency}, shared across every cell)",
        f"  Preflight:     expected {preflight_expected_cost} / conservative "
        f"{preflight_conservative_cost} — {preflight_status}",
    ]
    return "\n".join(lines)


def format_episode_banner(
    *,
    experiment_name: str,
    game_type: str,
    game_version: int,
    provider: str,
    model: str,
    population_size: int,
    horizon: int,
    prompt_family: str,
    prompt_version: int,
    prompt_definition_hash: str,
    budget_description: str,
    preflight_expected_cost: str,
    preflight_conservative_cost: str,
    preflight_status: str,
    output_dir: str,
) -> str:
    definition_prefix = prompt_definition_hash[:8] if prompt_definition_hash else "n/a"
    lines = [
        f"Episode: {experiment_name}",
        f"  Game:          {game_type} v{game_version}  (population {population_size}, horizon {horizon})",
        f"  Provider:      {provider} / {model}",
        f"  Prompt:        {prompt_family} v{prompt_version}  [def:{definition_prefix}...]",
        f"  Budget:        {budget_description}",
        f"  Preflight:     expected {preflight_expected_cost} / conservative "
        f"{preflight_conservative_cost} — {preflight_status}",
        f"  Output:        {output_dir}",
    ]
    return "\n".join(lines)


def print_banner(text: str) -> None:
    print(text)
    for line in text.splitlines():
        LOGGER.info(line.strip())


class ExperimentProgress:
    """Dual tqdm bars (episodes, rounds) with a no-TTY logging fallback.

    Every method is a no-op-safe call regardless of whether progress bars are
    active, so callers never need to branch on ``show``.
    """

    def __init__(self, *, total_episodes: int, total_rounds: int, show: bool) -> None:
        self._show = show and sys.stdout.isatty()
        self._episode_bar = None
        self._round_bar = None
        if self._show:
            from tqdm.auto import tqdm

            self._episode_bar = tqdm(
                total=total_episodes, desc="Episodes", unit="episode",
                dynamic_ncols=True, position=0, mininterval=0.25,
            )
            self._round_bar = tqdm(
                total=total_rounds, desc="Steps", unit="step",
                dynamic_ncols=True, position=1, mininterval=0.25,
            )

    def round_tick(self, episode_id: str, round_index: int | None, count: int = 1) -> None:
        if self._round_bar is not None:
            if round_index is not None:
                self._round_bar.set_postfix_str(f"{episode_id} | step {round_index}", refresh=False)
            self._round_bar.update(count)

    def episode_done(self, episode_id: str, status: str) -> None:
        if self._episode_bar is not None:
            self._episode_bar.set_postfix_str(f"{episode_id}: {status}", refresh=True)
            self._episode_bar.update(1)
        else:
            LOGGER.info("episode %s: %s", episode_id, status)

    def close(self) -> None:
        if self._episode_bar is not None:
            self._episode_bar.close()
        if self._round_bar is not None:
            self._round_bar.close()
