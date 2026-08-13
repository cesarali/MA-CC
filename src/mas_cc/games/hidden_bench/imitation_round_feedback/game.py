"""HiddenBench imitation with one controller decision per population round."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mas_cc.config import GameConfig
from mas_cc.games.protocols import GameSpec

from ..imitation.game import HiddenBenchImitationGame
from ..imitation.state import ImitationGameState, ImitationTransition
from .state import GAME_TYPE, RoundFeedbackRules


class HiddenBenchImitationRoundFeedbackGame(HiddenBenchImitationGame):
    """The inherited HiddenBench state machinery under a two-clock runtime."""

    spec = GameSpec(
        game_type=GAME_TYPE,
        version=1,
        description=(
            "HiddenBench one-focal imitation with one sensed controller action "
            "and an exact randomized actuation budget per population round."
        ),
        game_family="choice",
        minimum_population=2,
        supported_topologies=("complete",),
    )

    def rules(self, config: GameConfig) -> RoundFeedbackRules:
        return RoundFeedbackRules.from_config(config)

    @staticmethod
    def _termination_reason(
        votes: Sequence[str], turn: int, rules: RoundFeedbackRules
    ) -> str | None:
        if turn >= rules.horizon:
            return "max_rounds_reached"
        # Consensus is checked only at a slow-clock boundary.  A round that
        # starts always contains exactly N microscopic opportunities.
        if (
            rules.stop_on_consensus
            and turn % rules.n_agents == 0
            and len(set(votes)) == 1
        ):
            return "consensus_reached"
        return None

    def apply_round_event_transition(
        self,
        state: ImitationGameState,
        *,
        round_fields: Mapping[str, Any],
        **transition_fields: Any,
    ) -> ImitationTransition:
        """Apply the inherited focal transition and enrich its persisted row."""

        transition = super().apply_event_transition(state, **transition_fields)
        event = {**dict(transition.event or {}), **dict(round_fields)}
        next_state = transition.next_state
        event_history = list(next_state.data.get("event_history", ()))
        if event_history:
            event_history[-1] = event
        evaluator_history = list(next_state.data.get("evaluator_history", ()))
        if evaluator_history:
            evaluator_history[-1] = {**dict(evaluator_history[-1]), **dict(round_fields)}
        enriched_state = ImitationGameState(
            game_type=next_state.game_type,
            turn=next_state.turn,
            agents=next_state.agents,
            terminated=next_state.terminated,
            data={
                **dict(next_state.data),
                "event_history": event_history,
                "evaluator_history": evaluator_history,
            },
        )
        return replace(transition, next_state=enriched_state, event=event)


__all__ = ["HiddenBenchImitationRoundFeedbackGame"]
