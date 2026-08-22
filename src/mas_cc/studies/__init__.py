"""Study-level submission and aggregation over ordinary MA-CC runs."""

from .aggregation import aggregate_study
from .manifest import StudySpec, discover_study
from .submission import SubmissionResult, submit_study

__all__ = [
    "StudySpec",
    "SubmissionResult",
    "aggregate_study",
    "discover_study",
    "submit_study",
]
