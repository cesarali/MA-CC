"""Conservative, transparent storage estimate for semantic dashboard streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SemanticStorageEstimate:
    per_episode_bytes: int
    total_bytes: int
    files_per_episode: int
    episode_count: int
    population_size: int
    rounds: int
    social_group_size: int
    expected_message_characters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_semantic_storage(
    config: Mapping[str, Any], episode_count: int
) -> SemanticStorageEstimate | None:
    storage = config.get("storage", {})
    if (
        not isinstance(storage, Mapping)
        or storage.get("artifact_profile") != "dashboard_semantic"
    ):
        return None
    game = config.get("game", {})
    options = game.get("options", {}) if isinstance(game, Mapping) else {}
    storage_options = storage.get("options", {})
    if not isinstance(options, Mapping):
        options = {}
    if not isinstance(storage_options, Mapping):
        storage_options = {}
    population = int(game.get("population_size", 0))
    rounds = int(options.get("rounds", game.get("horizon", 0)))
    q = int(options.get("social_group_size", 1))
    message_chars = int(storage_options.get("expected_public_message_characters", 240))
    updates = population * rounds
    # Explicit accounting model: fixed semantic fields, sampled references,
    # public-message allowance, round snapshots, header/completion overhead.
    semantic_sidecar = (
        4096
        + updates * (1150 + q * 48 + message_chars)
        + rounds * (4096 + population * 96)
    )
    # Established compact scientific Parquet, round trajectory, microscopic
    # trajectory, and episode manifest/seal overhead remain alongside the two
    # semantic files. This is a planning bound, not a claimed measurement.
    compact_scientific = 32_768 + updates * (520 + q * 32) + rounds * 6_144
    per_episode = semantic_sidecar + compact_scientific
    return SemanticStorageEstimate(
        per_episode_bytes=per_episode,
        total_bytes=per_episode * episode_count,
        files_per_episode=6,
        episode_count=episode_count,
        population_size=population,
        rounds=rounds,
        social_group_size=q,
        expected_message_characters=message_chars,
    )


__all__ = ["SemanticStorageEstimate", "estimate_semantic_storage"]
