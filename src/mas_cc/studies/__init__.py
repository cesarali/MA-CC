"""Study-level submission and aggregation over ordinary MA-CC runs."""

from .aggregation import aggregate_study
from .manifest import StudySpec, discover_study
from .initialization import materialize_study_initializations
from .preflight import StudyPreflightResult, run_study_preflight
from .submission import SubmissionResult, submit_study

__all__ = [
    "StudySpec",
    "StudyPreflightResult",
    "SubmissionResult",
    "aggregate_study",
    "discover_study",
    "materialize_study_initializations",
    "run_study_preflight",
    "submit_study",
]
