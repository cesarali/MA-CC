"""Provider-independent intervention contract.

A `Control` may override one agent's action for one interaction before the
decision loop ever asks the LLM for it — the generic hook every game's
runtime checks in `run_validated_decision` (`runtime/loop_runtime.py`). This
covers action-forcing control mechanisms. ``interaction_signal`` is the
backward-compatible companion hook for a measured intervention that shapes a
message or transition without overwriting the resulting action.

`override` returns the forced action *value* (a plain string), not a full
`Action` object: a `Control` has no view of a game's own `Action.metadata`
conventions (e.g. `presented_actions`, `parser_mode`) - only the per-game
runtime that already builds `Action`s in its normal decision path knows that
shape, so it stays responsible for wrapping the forced value into one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    # Deferred: `mas_cc.games.naming_convention.runtime` imports this module,
    # and `games/protocols.py` is only needed here for annotations - mirrors
    # the same TYPE_CHECKING guard in `runtime/loop_runtime.py`.
    from mas_cc.core import AgentId
    from mas_cc.games.protocols import GameState


@dataclass(frozen=True, slots=True)
class InteractionControlSignal:
    """A measured, auditable intervention that does not force an action.

    Games that support message- or transition-level control may consume this
    optional signal.  Existing action-forcing controls continue to use
    :meth:`Control.override` and need not implement the new hook.
    """

    action: str
    target: str | None = None
    message: str | None = None
    observation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("InteractionControlSignal.action must be non-empty")
        object.__setattr__(self, "observation", MappingProxyType(dict(self.observation)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RoundControlSignal:
    """One measured controller decision held fixed for a population round.

    Unlike :class:`InteractionControlSignal`, this signal is sampled at a
    round boundary and is not a request to actuate every microscopic update.
    The consuming game owns the within-round schedule and decides which
    update positions receive ``message`` as an influence slot.
    """

    action: str
    target: str | None = None
    message: str | None = None
    observation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("RoundControlSignal.action must be non-empty")
        object.__setattr__(self, "observation", MappingProxyType(dict(self.observation)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Control(ABC):
    """One provider-independent intervention/control policy."""

    @abstractmethod
    def override(
        self, *, agent_id: "AgentId", interaction_index: int, state: "GameState"
    ) -> str | None:
        """Return the action value to force for this agent this interaction, or `None`.

        `None` means: let the normal LLM-backed decision proceed unchanged.
        """

    def interaction_signal(
        self,
        *,
        agent_id: "AgentId",
        interaction_index: int,
        state: "GameState",
        rng: Any,
    ) -> InteractionControlSignal | None:
        """Optionally measure state and shape one local interaction.

        The default is deliberately inert, preserving every existing control
        and runtime.  ``rng`` is supplied by the game runtime so sensing is
        reproducible from the episode seed without controls owning global RNG.
        """

        return None

    def round_signal(
        self,
        *,
        round_index: int,
        state: "GameState",
        rng: Any,
    ) -> RoundControlSignal | None:
        """Optionally sense and choose once at a population-round boundary.

        The inert default is intentionally separate from
        :meth:`interaction_signal`: existing event-level controls keep their
        exact behavior, while round-aware games can opt into this slower
        control clock explicitly.
        """

        return None
