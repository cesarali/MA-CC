"""Provider-independent intervention contract.

A `Control` may override one agent's action for one interaction before the
decision loop ever asks the LLM for it — the generic hook every game's
runtime checks in `run_validated_decision` (`runtime/loop_runtime.py`). This
covers action-forcing control mechanisms; a mechanism that instead shapes
payoffs/transitions rather than actions would need a different hook, added
when a concrete use case needs it, not speculatively now.

`override` returns the forced action *value* (a plain string), not a full
`Action` object: a `Control` has no view of a game's own `Action.metadata`
conventions (e.g. `presented_actions`, `parser_mode`) - only the per-game
runtime that already builds `Action`s in its normal decision path knows that
shape, so it stays responsible for wrapping the forced value into one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Deferred: `mas_cc.games.naming_convention.runtime` imports this module,
    # and `games/protocols.py` is only needed here for annotations - mirrors
    # the same TYPE_CHECKING guard in `runtime/loop_runtime.py`.
    from mas_cc.core import AgentId
    from mas_cc.games.protocols import GameState


class Control(ABC):
    """One provider-independent intervention/control policy."""

    @abstractmethod
    def override(
        self, *, agent_id: "AgentId", interaction_index: int, state: "GameState"
    ) -> str | None:
        """Return the action value to force for this agent this interaction, or `None`.

        `None` means: let the normal LLM-backed decision proceed unchanged.
        """
