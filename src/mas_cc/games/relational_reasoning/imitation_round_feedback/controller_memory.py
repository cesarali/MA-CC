"""A bounded, public-only memory for the board-sensing controller.

The runtime retains the IDs sampled at each dawn.  This projection resolves
only those IDs, even though the game's append-only board contains other posts.
That distinction matters when a later experiment restores partial sensing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_public_observation_ledger(
    observations: Sequence[Mapping[str, Any]],
    messages_by_id: Mapping[str, Mapping[str, Any]],
    *,
    population_size: int,
    options: Sequence[str],
) -> dict[str, Any]:
    """Summarize observed posts, without inferring silent agents' votes or reads."""

    days: list[dict[str, Any]] = []
    agents: dict[str, dict[str, Any]] = {}
    facts: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    previous_day = 0
    option_order = tuple(str(option) for option in options)

    for observation in observations:
        day = int(observation["day"])
        if day <= previous_day:
            raise ValueError("controller board observations must be in day order")
        previous_day = day
        ids = tuple(str(item) for item in observation.get("message_ids", ()))
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError("controller board observations repeat a message ID")
        seen_ids.update(ids)
        selected: list[Mapping[str, Any]] = []
        for message_id in ids:
            message = messages_by_id.get(message_id)
            if message is None or int(message["round_created"]) != day - 1:
                raise ValueError("controller memory refers to a missing or wrong-day post")
            selected.append(message)
        selected.sort(
            key=lambda message: (
                int(message["micro_step_created"]),
                str(message["message_id"]),
            )
        )

        previously_seen_facts = set(facts)
        latest_votes: dict[str, str] = {}
        participant_fact_ids: set[str] = set()
        own_fact_ids: list[str] = []
        participant_messages = 0
        own_messages = 0
        for message in selected:
            fact_id = message.get("shared_fact_id")
            if str(message["author_kind"]) == "controller":
                own_messages += 1
                if fact_id is not None:
                    own_fact_ids.append(str(fact_id))
                continue
            author = str(message["author_id"])
            vote = str(message["vote"])
            if vote not in option_order:
                raise ValueError("controller memory contains a vote outside the task")
            participant_messages += 1
            latest_votes[author] = vote
            agent = agents.setdefault(
                author,
                {
                    "author_id": author,
                    "last_seen_day": day,
                    "last_posted_vote": vote,
                    "observed_post_count": 0,
                    "cited_fact_ids": set(),
                },
            )
            agent["last_seen_day"] = day
            agent["last_posted_vote"] = vote
            agent["observed_post_count"] += 1
            if message["message_type"] == "REPORT" and fact_id is not None:
                fact_id = str(fact_id)
                participant_fact_ids.add(fact_id)
                agent["cited_fact_ids"].add(fact_id)
                fact = facts.setdefault(
                    fact_id,
                    {
                        "fact_id": fact_id,
                        "first_seen_day": day,
                        "last_seen_day": day,
                        "observed_post_count": 0,
                        "participant_authors": set(),
                    },
                )
                fact["last_seen_day"] = day
                fact["observed_post_count"] += 1
                fact["participant_authors"].add(author)

        if len(latest_votes) > population_size:
            raise ValueError("more distinct participant authors than the population")
        days.append(
            {
                "day": day,
                "view_complete": bool(observation["view_complete"]),
                "messages_seen": len(selected),
                "participant_messages_seen": participant_messages,
                "distinct_participants_seen": len(latest_votes),
                "last_posted_vote_counts": {
                    option: sum(vote == option for vote in latest_votes.values())
                    for option in option_order
                },
                "new_participant_fact_ids": sorted(
                    participant_fact_ids - previously_seen_facts
                ),
                "controller_messages_seen": own_messages,
                "controller_fact_ids": own_fact_ids,
            }
        )

    return {
        "schema_version": 1,
        "population_size": population_size,
        "days": days,
        "participants": [
            {**agent, "cited_fact_ids": sorted(agent["cited_fact_ids"])}
            for _, agent in sorted(agents.items())
        ],
        "facts": [
            {
                **fact,
                "distinct_participant_authors": len(fact["participant_authors"]),
                "participant_authors": sorted(fact["participant_authors"]),
            }
            for _, fact in sorted(facts.items())
        ],
    }


__all__ = ["build_public_observation_ledger"]
