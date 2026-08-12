"""Frozen records for the HiddenBench corpus and its information assignments.

Implements §3.1 of `docs/tdd/architecture/05082026_HIDDENBENCH_BRIEF.md` against
the field names the repository's own preprocessing pipeline actually emits
(`scripts/local_llms/hiddenbench_population_pipeline/`), not the brief's prior
guess at them. The reconciliation is recorded in
`docs/hidden_bench/data_provenance.md`; the short version is that the corpus
carries `task_id`/`source_description`/`scenario_description` and a *structured*
`hidden_information` (one `{evidence_type, source_text}` object per latent
evidence type), where the brief guessed a flat tuple of strings and a `source`
field. Both views are available here: the structured one is what the corpus
stores, `hidden_information` is the flat view the game code wants.

Paper: Li, Naito & Shirado, *Systematic Failures in Collective Reasoning under
Distributed Information in Multi-Agent LLMs* (ICML 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class HiddenBenchDataError(ValueError):
    """A corpus record violated a §3.2 load-time assertion.

    Deliberately loud and deliberately not caught anywhere: a malformed Hidden
    Profile task silently degrades into a task that is merely *hard*, which is
    indistinguishable from a real result in the reported numbers.
    """


def _texts(values: Any, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HiddenBenchDataError(f"{field} must be a sequence of strings")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise HiddenBenchDataError(f"{field}[{index}] must be non-empty text")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class HiddenProfileTask:
    """One normalized Hidden Profile task.

    `hidden_information[i]` is the source text of evidence type `i`; the index
    *is* the evidence type, which is the convention the expansion script uses
    when it assigns types to agents, so the two never need a lookup table.

    Two descriptions, both from the corpus, because the scenario wording names
    the group size:

    - `source_description` is the authors' original text, which says "the other
      three community leaders" and is only correct at `n_agents_native`.
    - `scenario_description` is the pipeline's population-neutral rewrite ("the
      others"), correct at any N, paired with `population_instruction`.

    `description_for(n_agents)` picks between them, so a run at N = 4 is
    byte-identical to the paper and a run at N = 7 is not quietly telling every
    agent there are four of them.
    """

    task_id: int
    name: str
    source_description: str
    scenario_description: str
    shared_information: tuple[str, ...]
    hidden_information: tuple[str, ...]
    possible_answers: tuple[str, ...]
    correct_answer: str
    n_agents_native: int
    source: str
    rationale: str | None = None
    population_instruction: str | None = None
    validation_stats: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # §3.2 - every one of these is a "fail loudly", never a "drop the task".
        if self.correct_answer not in self.possible_answers:
            raise HiddenBenchDataError(
                f"task {self.name!r}: correct_answer {self.correct_answer!r} is not one of "
                f"possible_answers {list(self.possible_answers)}"
            )
        if len(self.possible_answers) < 3:
            raise HiddenBenchDataError(
                f"task {self.name!r}: the Hidden Profile paradigm requires K >= 3 options, "
                f"got {len(self.possible_answers)}"
            )
        if len(set(self.possible_answers)) != len(self.possible_answers):
            raise HiddenBenchDataError(f"task {self.name!r}: possible_answers contains duplicates")
        if len(self.hidden_information) != self.n_agents_native:
            raise HiddenBenchDataError(
                f"task {self.name!r}: {len(self.hidden_information)} hidden items but "
                f"n_agents_native is {self.n_agents_native}"
            )
        if len(set(self.hidden_information)) != len(self.hidden_information):
            raise HiddenBenchDataError(f"task {self.name!r}: duplicate hidden information items")
        if not self.hidden_information:
            raise HiddenBenchDataError(f"task {self.name!r}: no hidden information")
        overlap = set(self.shared_information) & set(self.hidden_information)
        if overlap:
            raise HiddenBenchDataError(
                f"task {self.name!r}: {len(overlap)} fact(s) appear in both shared_information "
                f"and hidden_information, which destroys the hidden/shared split"
            )

    def description_for(self, n_agents: int) -> str:
        """The scenario text to show every agent at this population size."""

        if n_agents == self.n_agents_native:
            return self.source_description
        if self.population_instruction:
            return f"{self.scenario_description}\n\n{self.population_instruction}"
        return self.scenario_description

    @property
    def unshared_information(self) -> tuple[str, ...]:
        """`Iu` - the full pool every scheme's assignment must reconstruct."""

        return self.hidden_information

    @property
    def wrong_answers(self) -> tuple[str, ...]:
        return tuple(item for item in self.possible_answers if item != self.correct_answer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "shared_information": list(self.shared_information),
            "hidden_information": list(self.hidden_information),
            "possible_answers": list(self.possible_answers),
            "correct_answer": self.correct_answer,
            "n_agents_native": self.n_agents_native,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class AgentInfoSet:
    """`I_i = Is ∪ Iu_i` for one agent, kept as two separate tuples.

    They stay separate here and are only merged (and shuffled together) at
    prompt-render time, because every privacy assertion in the test suite is a
    statement about `private` alone.

    `evidence_types` records *which* latent types this agent's private items
    realize, and `transformation` records how (`identity` for exact
    replication, `paraphrase`, `factor`). Both come straight from the
    pipeline's own allocation records so post-hoc analysis can condition on who
    knew what without re-deriving the assignment.
    """

    shared: tuple[str, ...]
    private: tuple[str, ...]
    evidence_types: tuple[int, ...] = ()
    transformation: str = "identity"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared", _texts(self.shared, "AgentInfoSet.shared"))
        # `private` may legitimately be empty: `padded` gives extra agents
        # shared information only, and that is the whole point of the scheme.
        if isinstance(self.private, (str, bytes)) or not isinstance(self.private, Sequence):
            raise HiddenBenchDataError("AgentInfoSet.private must be a sequence of strings")
        object.__setattr__(self, "private", tuple(str(item) for item in self.private))
        object.__setattr__(self, "evidence_types", tuple(int(item) for item in self.evidence_types))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared": list(self.shared),
            "private": list(self.private),
            "evidence_types": list(self.evidence_types),
            "transformation": self.transformation,
            "provenance": dict(self.provenance),
        }
