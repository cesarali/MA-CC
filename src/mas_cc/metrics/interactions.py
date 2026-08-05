"""Ashery-Baronchelli trajectory metrics over stored pairwise interactions.

Specified in
``docs/tdd/misselaneous/05082026_ashery_success_rate_and_production_probability.md``.
Two quantities, and the whole point is that they have *different denominators*:

- **Success rate** counts pair interactions: did the two agents say the same
  word? A bin of L interactions contributes L observations.
- **Production probability** counts individual outputs: how often was each word
  said at all? The same bin contributes 2L observations, because both agents
  speak every interaction.

Two trajectories can have identical production probabilities and opposite
success rates - ``(M,M),(Q,Q)`` and ``(M,Q),(Q,M)`` both give P(M)=0.5, but
success rates of 1 and 0. Production probability alone does not measure
coordination, which is why neither metric substitutes for the other.

Everything here is a pure function over persisted records: no provider, no game
engine, no LLM call. The game-facing adapter lives in
``games/naming_convention/metrics.py``.

Relation to the streaming metrics in this package: ``rolling_coordination_rate``
and ``rolling_action_share_per_option`` compute the *same two indicators* over a
trailing window, once per round. These compute them over non-overlapping
population-round bins, once per bin, and additionally keep the raw counts. Same
arithmetic, different windows and different output shape - see the spec's §7 on
why the two windows must not be conflated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

PARTIAL_BIN_POLICIES = ("drop", "include", "error")
"""What to do with a trailing bin that has fewer than ``bin_size`` interactions.

``drop`` omits it, ``include`` computes over whatever is there, ``error``
rejects the trajectory. Held fixed across the runs of one comparison, since a
partial bin is a smaller sample and its point is noisier than the rest.
"""


@dataclass(frozen=True, slots=True)
class BinnedTrajectoryPolicy:
    """How a run turns its interactions into binned trajectory metrics.

    Carried as one object so the three choices that change what a published
    number *means* - the bin width, what happens to a short final bin, and
    whether forced outputs count - always travel together and get recorded
    together.
    """

    bin_size: int
    partial_final_bin: str = "drop"
    exclude_committed_outputs: bool = False

    def __post_init__(self) -> None:
        if self.bin_size < 1:
            raise ValueError("BinnedTrajectoryPolicy.bin_size must be positive")
        if self.partial_final_bin not in PARTIAL_BIN_POLICIES:
            raise ValueError(
                f"partial_final_bin must be one of {PARTIAL_BIN_POLICIES}, "
                f"got {self.partial_final_bin!r}"
            )


@dataclass(frozen=True, slots=True)
class InteractionOutcome:
    """One recorded pair interaction: who said what, and which outputs were forced.

    ``actions`` is normally the two words produced in this interaction, in
    participant order. ``committed`` is the matching per-output flag - True
    where that output came from an agent hard-wired to a fixed word rather than
    a free decision. It is per *output*, not per interaction, so a pairing of
    one committed and one ordinary agent keeps the ordinary output and drops
    only the committed one.

    Stored as tuples rather than ``player_1``/``player_2`` fields so a game with
    more than two participants per interaction needs no new record type; for the
    pairwise case the two are equivalent.
    """

    interaction_index: int
    actions: tuple[str, ...]
    committed: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if len(actions) < 2:
            raise ValueError("InteractionOutcome.actions needs at least two participants")
        committed = tuple(bool(flag) for flag in self.committed) or (False,) * len(actions)
        if len(committed) != len(actions):
            raise ValueError(
                "InteractionOutcome.committed must carry one flag per action "
                f"({len(actions)} actions, {len(committed)} flags)"
            )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "committed", committed)

    @property
    def succeeded(self) -> bool:
        """True when every participant produced the same word.

        Which word does not matter: (M,M) and (Q,Q) are both successes.
        """

        return len(set(self.actions)) == 1

    @classmethod
    def from_evaluator_entry(cls, entry: Mapping[str, Any]) -> "InteractionOutcome":
        """Build one record from a naming-convention ``evaluator_history`` entry."""

        actions = tuple(str(value) for value in entry["actions"])
        committed = tuple(bool(flag) for flag in entry.get("committed", ()))
        return cls(
            interaction_index=int(entry["interaction_index"]),
            actions=actions,
            committed=committed,
        )


def success_rate(records: Sequence[InteractionOutcome]) -> float:
    """Fraction of pair interactions in which every participant matched."""

    if not records:
        raise ValueError("success_rate requires at least one interaction")
    return sum(record.succeeded for record in records) / len(records)


def production_counts(
    records: Sequence[InteractionOutcome],
    *,
    action_space: Iterable[str] | None = None,
    exclude_committed_outputs: bool = False,
) -> tuple[dict[str, int], int]:
    """Per-word output counts and the eligible-output total they are taken over.

    Returned alongside the normalized probabilities so a stored metric can be
    revalidated and recomputed later without the original trajectory.
    """

    counts: Counter[str] = Counter()
    eligible = 0
    for record in records:
        for action, is_committed in zip(record.actions, record.committed, strict=True):
            if exclude_committed_outputs and is_committed:
                continue
            counts[action] += 1
            eligible += 1
    words = tuple(sorted(counts)) if action_space is None else tuple(action_space)
    return {word: counts[word] for word in words}, eligible


def production_probabilities(
    records: Sequence[InteractionOutcome],
    *,
    action_space: Iterable[str] | None = None,
    exclude_committed_outputs: bool = False,
) -> dict[str, float]:
    """Empirical distribution of individual agent outputs over ``records``.

    Every ordinary interaction contributes one observation *per participant*, so
    the denominator for L pair interactions is 2L, not L. Supplying
    ``action_space`` gives unobserved-but-legal words an explicit 0.0 rather
    than omitting them, and makes the values sum to 1 over the full pool.

    This is an empirical action frequency. It is not a token probability, a
    softmax output, or any measure of model confidence.
    """

    counts, eligible = production_counts(
        records,
        action_space=action_space,
        exclude_committed_outputs=exclude_committed_outputs,
    )
    if eligible == 0:
        raise ValueError(
            "no eligible outputs for production probability"
            + (" (every output was committed and excluded)" if exclude_committed_outputs else "")
        )
    return {word: count / eligible for word, count in counts.items()}


def non_overlapping_bins(
    records: Sequence[InteractionOutcome],
    *,
    bin_size: int,
    partial_final_bin: str = "drop",
) -> list[Sequence[InteractionOutcome]]:
    """Split a trajectory into consecutive, non-overlapping bins of ``bin_size``.

    With ``bin_size`` = population size, one bin is a *population round*: N pair
    interactions. Note that N pair interactions are 2N participation slots, and
    random pairing does not guarantee each agent appears exactly once - a
    population round is a bookkeeping unit, not a guarantee about who spoke.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if partial_final_bin not in PARTIAL_BIN_POLICIES:
        raise ValueError(
            f"partial_final_bin must be one of {PARTIAL_BIN_POLICIES}, got {partial_final_bin!r}"
        )
    remainder = len(records) % bin_size
    if remainder and partial_final_bin == "error":
        raise ValueError(
            f"{len(records)} interactions do not divide into bins of {bin_size} "
            f"({remainder} left over); use partial_final_bin='drop' or 'include'"
        )
    bins: list[Sequence[InteractionOutcome]] = []
    for start in range(0, len(records), bin_size):
        current = records[start : start + bin_size]
        if len(current) < bin_size and partial_final_bin == "drop":
            break
        bins.append(current)
    return bins


def _bin_bounds(bin_records: Sequence[InteractionOutcome]) -> tuple[int, int]:
    return bin_records[0].interaction_index, bin_records[-1].interaction_index


def success_rate_rows(
    records: Sequence[InteractionOutcome],
    *,
    episode_id: str,
    bin_size: int,
    partial_final_bin: str = "drop",
) -> list[dict[str, Any]]:
    """One long-form row per bin, raw count kept beside the normalized rate."""

    return [
        {
            "episode_id": episode_id,
            "bin_index": index,
            "start_interaction": _bin_bounds(bin_records)[0],
            "end_interaction": _bin_bounds(bin_records)[1],
            "num_pair_interactions": len(bin_records),
            "success_count": sum(record.succeeded for record in bin_records),
            "success_rate": success_rate(bin_records),
        }
        for index, bin_records in enumerate(
            non_overlapping_bins(records, bin_size=bin_size, partial_final_bin=partial_final_bin)
        )
    ]


def production_probability_rows(
    records: Sequence[InteractionOutcome],
    *,
    episode_id: str,
    bin_size: int,
    action_space: Iterable[str] | None = None,
    partial_final_bin: str = "drop",
    exclude_committed_outputs: bool = False,
) -> list[dict[str, Any]]:
    """One long-form row per (bin, word), raw counts kept beside the probability."""

    words = None if action_space is None else tuple(action_space)
    rows: list[dict[str, Any]] = []
    for index, bin_records in enumerate(
        non_overlapping_bins(records, bin_size=bin_size, partial_final_bin=partial_final_bin)
    ):
        counts, eligible = production_counts(
            bin_records, action_space=words, exclude_committed_outputs=exclude_committed_outputs
        )
        if eligible == 0:
            raise ValueError(
                f"bin {index} has no eligible outputs for production probability"
            )
        start, end = _bin_bounds(bin_records)
        rows.extend(
            {
                "episode_id": episode_id,
                "bin_index": index,
                "start_interaction": start,
                "end_interaction": end,
                "action": word,
                "eligible_output_count": eligible,
                "action_count": count,
                "production_probability": count / eligible,
                "excluded_committed_outputs": exclude_committed_outputs,
            }
            for word, count in counts.items()
        )
    return rows


def consensus_flip(
    records: Sequence[InteractionOutcome],
    *,
    window: int,
    threshold: float = 0.95,
    exclude_committed_outputs: bool = False,
) -> tuple[int, str] | None:
    """First interaction where a trailing ``window`` is at least ``threshold`` successful.

    The paper's criterion, with ``window`` = 3N: a flip is called when at least
    95% of the most recent 3N pair interactions matched. Returns the
    1-based interaction index at which that first holds, together with the word
    that won - success rate alone says the population agreed but not on *what*,
    so the dominant word in the same window is read off with it. Returns None if
    the criterion never holds.

    This is a *rolling* window and is deliberately not the same thing as the
    non-overlapping bins used to plot a trajectory, even though both reduce the
    identical per-interaction success indicator.
    """

    if window < 1:
        raise ValueError("window must be positive")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    for end in range(window, len(records) + 1):
        recent = records[end - window : end]
        if success_rate(recent) < threshold:
            continue
        counts, eligible = production_counts(
            recent, exclude_committed_outputs=exclude_committed_outputs
        )
        if not eligible:
            continue
        winner = max(sorted(counts), key=lambda word: counts[word])
        return records[end - 1].interaction_index, winner
    return None


def binned_trajectory_tables(
    records: Sequence[InteractionOutcome],
    *,
    episode_id: str,
    policy: BinnedTrajectoryPolicy,
    action_space: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Both §9 tables for one episode: (success-rate rows, production-probability rows)."""

    return (
        success_rate_rows(
            records,
            episode_id=episode_id,
            bin_size=policy.bin_size,
            partial_final_bin=policy.partial_final_bin,
        ),
        production_probability_rows(
            records,
            episode_id=episode_id,
            bin_size=policy.bin_size,
            action_space=action_space,
            partial_final_bin=policy.partial_final_bin,
            exclude_committed_outputs=policy.exclude_committed_outputs,
        ),
    )


SUCCESS_RATE_FIELDS = [
    "episode_id", "bin_index", "start_interaction", "end_interaction",
    "num_pair_interactions", "success_count", "success_rate",
]

PRODUCTION_PROBABILITY_FIELDS = [
    "episode_id", "bin_index", "start_interaction", "end_interaction", "action",
    "eligible_output_count", "action_count", "production_probability",
    "excluded_committed_outputs",
]

__all__ = [
    "PARTIAL_BIN_POLICIES",
    "PRODUCTION_PROBABILITY_FIELDS",
    "SUCCESS_RATE_FIELDS",
    "BinnedTrajectoryPolicy",
    "InteractionOutcome",
    "binned_trajectory_tables",
    "consensus_flip",
    "non_overlapping_bins",
    "production_counts",
    "production_probabilities",
    "production_probability_rows",
    "success_rate",
    "success_rate_rows",
]
