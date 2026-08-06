"""Typed state and transition records shared by both HiddenBench games.

Both games subclass the generic `GameState`/`AgentState`/`Transition` from
`games/protocols.py` for the same reason `naming_convention` does: to add typed
readers over the same underlying `data`/`attributes`/`memory` fields, not to
introduce a parallel state model.

`HiddenBenchTransition.success`/`.payoff` exist because
`observability/recorder.py::record_interaction` reads exactly those two names
off every transition it is handed. Without them a HiddenBench episode runs but
writes no `metrics/streaming.csv`, and `mas-cc analysis empowerment --grid-dir`
reads that file - so these two properties are the whole reason a HiddenBench
grid is analysable at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mas_cc.core import AgentId
from mas_cc.games.protocols import AgentState, GameState, Transition, _thaw

PRE_VOTE = "pre_vote"
DISCUSS = "discuss"
POST_VOTE = "post_vote"
EXCHANGE = "exchange"
COMMIT = "commit"


@dataclass(frozen=True, slots=True)
class HiddenBenchAgentState(AgentState):
    """One agent's private information and its current standing vote."""

    @property
    def shared_information(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.attributes.get("shared_information", ()))

    @property
    def private_information(self) -> tuple[str, ...]:
        """`Iu_i` alone - never the shared facts, never anyone else's.

        Kept separate from `presented_information` so the privacy test has a
        single unambiguous thing to assert about.
        """

        return tuple(str(item) for item in self.attributes.get("private_information", ()))

    @property
    def presented_information(self) -> tuple[str, ...]:
        """`Is ∪ Iu_i` in this agent's own seeded shuffle order (§1.5)."""

        return tuple(str(item) for item in self.attributes.get("presented_information", ()))

    @property
    def evidence_types(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.attributes.get("evidence_types", ()))

    @property
    def committed_action(self) -> str | None:
        """The option this agent currently stands on; `None` before its first vote."""

        value = self.attributes.get("committed_action")
        return None if value is None else str(value)

    @property
    def pre_vote(self) -> str | None:
        value = self.attributes.get("pre_vote")
        return None if value is None else str(value)

    @property
    def post_vote(self) -> str | None:
        value = self.attributes.get("post_vote")
        return None if value is None else str(value)

    @property
    def rationales(self) -> tuple[Mapping[str, Any], ...]:
        """Every rationale this agent gave, in order. Rationales are data (§5)."""

        return tuple(self.attributes.get("rationales", ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "score": self.score,
            "private_information": list(self.private_information),
            "evidence_types": list(self.evidence_types),
            "pre_vote": self.pre_vote,
            "post_vote": self.post_vote,
            "committed_action": self.committed_action,
            "rationales": _thaw(self.rationales),
            "memory": _thaw(self.memory),
        }


@dataclass(frozen=True, slots=True)
class HiddenBenchGameState(GameState):
    """Whole-population state for one HiddenBench episode."""

    @property
    def phase(self) -> str:
        return str(self.data["phase"])

    @property
    def task(self) -> Mapping[str, Any]:
        return self.data["task"]

    @property
    def possible_answers(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.task["possible_answers"])

    @property
    def correct_answer(self) -> str:
        return str(self.task["correct_answer"])

    @property
    def hidden_information(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.task["hidden_information"])

    @property
    def transcript(self) -> tuple[Mapping[str, Any], ...]:
        """Every message spoken so far, in order. Public by construction."""

        return tuple(self.data.get("transcript", ()))

    @property
    def evaluator_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data.get("evaluator_history", ()))

    @property
    def termination_reason(self) -> str | None:
        value = self.data.get("termination_reason")
        return None if value is None else str(value)

    def hidden_bench_agent(self, agent_id: AgentId) -> HiddenBenchAgentState:
        agent = self.agent(agent_id)
        if not isinstance(agent, HiddenBenchAgentState):
            raise TypeError("hidden_bench state contains a non-hidden_bench agent")
        return agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "turn": self.turn,
            "phase": self.phase,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "task": _thaw(self.task),
            "transcript": _thaw(self.transcript),
            "agents": [agent.to_dict() for agent in self.agents],
            "evaluator_history": _thaw(self.evaluator_history),
            "assignment": _thaw(self.data.get("assignment", {})),
        }


@dataclass(frozen=True, slots=True)
class HiddenBenchTransition(Transition):
    """A transition that the shared recorder can read without a special case."""

    @property
    def success(self) -> bool:
        """Whether this step counted as a success for the run-level log.

        For a vote step: the majority-rule outcome. For a discussion or message
        step there is no notion of success, and `matched` is `None`, which reads
        as `False` here - a discussion turn is neither right nor wrong.
        """

        return bool(self.matched)

    @property
    def payoff(self) -> float:
        """Mean payoff across the agents that acted this step."""

        values = list(self.payoffs.values())
        return sum(values) / len(values) if values else 0.0
