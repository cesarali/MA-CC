"""Deterministic paired prompts and nested evidence-dose definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object

PROMPT_FAMILIES = ("validation", "game_init")


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    experiment: str
    prompt_family: str
    agent_number: int
    repetition: int
    evidence_ids: tuple[str, ...]
    option_mapping: Mapping[str, str]
    pair_id: str | None = None
    dose: int | None = None
    requested_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "experiment": self.experiment,
            "prompt_family": self.prompt_family,
            "agent_id": self.agent_number,
            "repetition": self.repetition,
            "evidence_ids": list(self.evidence_ids),
            "evidence_sha256": sha256_object(list(self.evidence_ids)),
            "semantic_option_mapping": dict(self.option_mapping),
            "pair_id": self.pair_id,
            "dose": self.dose,
            "requested_seed": self.requested_seed,
        }


def option_mapping(task: RelationalTask, seed: Seed) -> dict[str, str]:
    options = list(task.semantic_answers)
    seed.create_random().shuffle(options)
    return {letter: option for letter, option in zip("ABC", options, strict=True)}


def nested_card_order(task: RelationalTask, agent_number: int, root: Seed) -> tuple[str, ...]:
    groups = {key: list(values) for key, values in (task.supporting_fact_groups or {}).items()}
    if len(groups) != 9:
        raise ValueError("the local evidence probe requires exactly nine latent facts")
    natural = list(task.known_facts(f"agent_{agent_number:03d}"))
    card_group = {card: latent for latent, cards in groups.items() for card in cards}
    natural_groups = [card_group[card] for card in natural]
    if len(set(natural_groups)) != len(natural_groups):
        raise ValueError("the selected natural view must contain at most one branch per latent fact")
    rng = root.derive(f"dose-order:agent-{agent_number}").create_random()
    natural_group_order = list(dict.fromkeys(natural_groups))
    rng.shuffle(natural_group_order)
    natural_by_group = {card_group[card]: card for card in natural}
    order = [natural_by_group[group] for group in natural_group_order]
    unseen = [group for group in sorted(groups) if group not in natural_by_group]
    rng.shuffle(unseen)
    for group in unseen:
        candidates = list(groups[group])
        rng.shuffle(candidates)
        order.append(candidates[0])
    remaining = [card for card in task.fact_order if card not in set(order)]
    rng.shuffle(remaining)
    order.extend(remaining)
    if len(order) != 27 or set(order) != set(task.fact_order):
        raise ValueError("nested order must contain all 27 cards exactly once")
    return tuple(order)


def dose_definitions(
    task: RelationalTask, agents: Sequence[int], doses: Sequence[int], seed: int
) -> tuple[dict[str, Any], ...]:
    root = Seed(seed)
    card_group = {
        card: latent
        for latent, cards in (task.supporting_fact_groups or {}).items()
        for card in cards
    }
    rows: list[dict[str, Any]] = []
    for agent in agents:
        order = nested_card_order(task, agent, root)
        previous: tuple[str, ...] = ()
        natural = task.known_facts(f"agent_{agent:03d}")
        for dose in doses:
            selected = order[:dose]
            latent = tuple(dict.fromkeys(card_group[card] for card in selected))
            rows.append(
                {
                    "agent_id": agent,
                    "dose": dose,
                    "evidence_ids": list(selected),
                    "evidence_sha256": sha256_object(list(selected)),
                    "latent_fact_ids": list(latent),
                    "distinct_latent_fact_count": len(latent),
                    "added_since_previous_dose": list(selected[len(previous) :]),
                    "selection_seed": int(root.derive(f"dose-order:agent-{agent}")),
                    "natural_initial_card_count": len(natural),
                    "natural_initial_view_is_prefix": set(order[: len(natural)])
                    == set(natural),
                }
            )
            previous = selected
    return tuple(rows)


def build_call_plan(
    task: RelationalTask,
    agents: Sequence[int],
    pair_repetitions: int,
    doses: Sequence[int],
    dose_repetitions: int,
    seed: int,
) -> tuple[tuple[CallSpec, ...], tuple[dict[str, Any], ...]]:
    root = Seed(seed)
    specs: list[CallSpec] = []
    for agent in agents:
        evidence = task.known_facts(f"agent_{agent:03d}")
        for repetition in range(pair_repetitions):
            pair_id = f"agent-{agent:03d}-rep-{repetition:02d}"
            pair_seed = root.derive(f"prompt-equivalence:{pair_id}")
            mapping = option_mapping(task, pair_seed.derive("option-permutation"))
            order = list(PROMPT_FAMILIES)
            pair_seed.derive("dispatch-order").create_random().shuffle(order)
            for family in order:
                specs.append(
                    CallSpec(
                        call_id=f"equivalence:{pair_id}:{family}",
                        experiment="prompt_equivalence",
                        prompt_family=family,
                        agent_number=agent,
                        repetition=repetition,
                        evidence_ids=evidence,
                        option_mapping=mapping,
                        pair_id=pair_id,
                        requested_seed=int(pair_seed.derive("provider")),
                    )
                )
    definitions = dose_definitions(task, agents, doses, seed)
    by_key = {(row["agent_id"], row["dose"]): row for row in definitions}
    for agent in agents:
        for dose in doses:
            evidence = tuple(by_key[(agent, dose)]["evidence_ids"])
            for repetition in range(dose_repetitions):
                call_id = f"dose:agent-{agent:03d}:cards-{dose:02d}:rep-{repetition:02d}"
                call_seed = root.derive(call_id)
                specs.append(
                    CallSpec(
                        call_id=call_id,
                        experiment="evidence_dose",
                        prompt_family="game_init",
                        agent_number=agent,
                        repetition=repetition,
                        evidence_ids=evidence,
                        option_mapping=option_mapping(task, call_seed.derive("option-permutation")),
                        dose=dose,
                        requested_seed=int(call_seed.derive("provider")),
                    )
                )
    if len(specs) != 123 or len({spec.call_id for spec in specs}) != len(specs):
        raise ValueError("the local evidence call plan must contain 123 unique calls")
    return tuple(specs), definitions


__all__ = ["CallSpec", "build_call_plan", "dose_definitions", "nested_card_order", "option_mapping"]
