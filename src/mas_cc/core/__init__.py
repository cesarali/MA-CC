"""Small immutable identifiers and records shared by :mod:`mas_cc`.

Message/MessageRole, ValidationIssue/ValidationResult, and the
MasCCError/ValidationError/ConfigurationError exception hierarchy all live
in mas_cc.llm_runtime (messages.py, validation.py, exceptions.py) — moved
there so mas_cc.llm_runtime has zero dependency on this package. What
remains here (identifiers, Seed, Timestamp) is genuinely generic and not
needed by llm_runtime at all.
"""

from .ids import AgentId, ExperimentId, InteractionId, MessageId, RunId
from .random import Seed
from .records import Timestamp

__all__ = [
    "AgentId",
    "ExperimentId",
    "InteractionId",
    "MessageId",
    "RunId",
    "Seed",
    "Timestamp",
]
