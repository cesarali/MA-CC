"""Study-level submission and aggregation over ordinary MA-CC runs."""

from .aggregation import aggregate_study
from .manifest import StudySpec, discover_study
from .preflight import StudyPreflightResult, run_study_preflight
from .submission import SubmissionResult, prepare_study, submit_study

__all__ = [
    "StudySpec",
    "StudyPreflightResult",
    "SubmissionResult",
    "aggregate_study",
    "discover_study",
    "prepare_study",
    "run_study_preflight",
    "submit_study",
]
