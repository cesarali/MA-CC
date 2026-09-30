"""Exact, provider-free decisions for symbolic MuSR blackboard agents.

The policy receives an agent's citation context and own posting history, never
the hidden world, gold answer, other agents' inventories, or unsampled board.
Canonical predicates are an interpretation dictionary, not extra evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.core import Seed
from mas_cc.games.protocols import DecisionRequest
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact

from .state import CitationContext, FOCAL_UPDATE

ALLOCATIONS = tuple(f"ALLOCATION_{index}" for index in range(3))


class BayesianAgentPolicy:
    """Posterior argmax with seeded ties and a simple evidence-sharing policy."""

    def __init__(self, facts: Sequence[CanonicalFact]) -> None:
        self._facts = {fact.fact_id: fact for fact in facts}
        if len(self._facts) != len(facts):
            raise ValueError("Bayesian fact dictionary contains duplicate IDs")
        self._index = TeamAllocationCompletionIndex()

    @classmethod
    def load(
        cls, dataset_dir: str, task_id: str, fact_ids: Sequence[str]
    ) -> BayesianAgentPolicy:
        path = Path(dataset_dir) / task_id / "facts/all_true_facts.json"
        if not path.is_file():
            raise ValueError(
                "agent_decision_mode 'bayesian' requires a symbolic MuSR task "
                f"with canonical predicates at {path}"
            )
        facts = tuple(
            CanonicalFact.from_dict(row)
            for row in json.loads(path.read_text(encoding="utf-8"))
        )
        if {fact.fact_id for fact in facts} != set(fact_ids):
            raise ValueError("Bayesian fact dictionary does not match the loaded task")
        return cls(facts)

    def ballot(
        self,
        request: DecisionRequest,
        context: CitationContext,
        memory: Sequence[Mapping[str, Any]],
        fact_texts: Mapping[str, str],
        *,
        seed: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return a normal ballot JSON plus auditable posterior metadata.

        Grounded reports are usable immediately, even when active-only citation
        rules prevent forwarding them until the next update. Historical facts
        that have faded are excluded unless observed again in this sample.
        """

        evidence = tuple(sorted(set(context.active_fact_ids) | set(context.observed_fact_ids)))
        posterior = self._index.metrics_for_facts(
            tuple(self._facts[fact_id] for fact_id in evidence)
        )
        probabilities = dict(zip(ALLOCATIONS, posterior.probabilities, strict=True))
        maxima = tuple(
            answer for answer, value in probabilities.items()
            if value == max(posterior.probabilities)
        )
        previous = request.observation.visible_state.get("current_vote")
        vote = (
            previous if previous in maxima
            else Seed(seed).derive("bayesian-vote-tie").create_random().choice(maxima)
        )

        # Novelty is relative to the sampled board, not the global board. Own
        # past reports break ties; repetition later permits memory reactivation.
        last_posted = {
            str(row["own_shared_fact_id"]): position
            for position, row in enumerate(memory)
            if row.get("own_shared_fact_id") is not None
        }
        observed = set(context.observed_fact_ids)
        candidates = context.citable_fact_ids
        shared = None
        if candidates:
            priorities = {
                fact_id: (fact_id in observed, last_posted.get(fact_id, -1))
                for fact_id in candidates
            }
            best = min(priorities.values())
            choices = sorted(fact_id for fact_id in candidates if priorities[fact_id] == best)
            shared = Seed(seed).derive("bayesian-report-tie").create_random().choice(choices)
        reason = "Exact posterior: " + ", ".join(
            f"{answer}={value:.6f}" for answer, value in probabilities.items()
        )
        letters = request.observation.visible_state["option_letters"]
        letter = next(key for key, answer in letters.items() if answer == vote)
        if request.stage == FOCAL_UPDATE:
            payload = {
                "vote": letter,
                "private_reason": reason,
                "public_message": {
                    "type": "NONE" if shared is None else "REPORT",
                    "text": None if shared is None else fact_texts[shared],
                    "shared_fact_id": shared,
                    "reply_to": None,
                },
            }
        else:
            payload = {"vote": letter, "reason": reason, "shared_fact_id": shared}
        return json.dumps(payload), {
            "agent_decision_mode": "bayesian",
            "bayesian": {
                "evidence_fact_ids": list(evidence),
                "posterior": probabilities,
                "valid_completion_count": posterior.valid_completion_count,
                "posting_policy": "sample_novelty_then_least_recent_self_report",
            },
        }
