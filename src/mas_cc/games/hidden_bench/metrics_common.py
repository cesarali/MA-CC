"""HiddenBench-specific metrics, shared by both games (brief §7).

Everything here is written against `RoundView` alone, so the same metric objects
serve the plenary game and the dyadic one. The episode-invariant facts a
HiddenBench metric needs but `RoundView` has no field for - `correct_answer`,
the decoy, the phase of the current step - travel in `recent_history[-1]`, which
each game's `to_round_view` fills from its own round summary.

The reusable shelf (`ActionSharePerOption`, `DominantValueShare`,
`FirstConsensusTime`, …) is used unchanged; only the quantities the paper
reports and this repo cannot already compute live here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mas_cc.metrics.base import FinalMetric, MetricKey, StreamingMetric
from mas_cc.metrics.generic import RoundView

from .records import POST_VOTE, PRE_VOTE


def _summary(view: RoundView) -> Mapping[str, Any]:
    return view.recent_history[-1] if view.recent_history else {}


def _votes(view: RoundView) -> list[str]:
    return [value for value in view.agent_values.values() if value is not None]


def accuracy_average(view: RoundView) -> float:
    """The paper's default aggregation rule (§1.3): fraction of agents on `o*`."""

    votes = _votes(view)
    correct = _summary(view).get("correct_answer")
    if not votes or correct is None:
        return 0.0
    return sum(vote == correct for vote in votes) / len(votes)


def accuracy_majority(view: RoundView) -> float:
    """1 iff strictly more than half the agents are on `o*` (§1.3)."""

    votes = _votes(view)
    correct = _summary(view).get("correct_answer")
    if not votes or correct is None:
        return 0.0
    return float(sum(vote == correct for vote in votes) > len(votes) / 2)


class AccuracyAverage(StreamingMetric):
    """`Y` under the average rule, every round."""

    requires_game_family = "choice"

    def __init__(self, *, name: str = "accuracy_average") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        return {None: accuracy_average(view)}


class AccuracyMajority(StreamingMetric):
    """`Y` under the majority rule, every round."""

    requires_game_family = "choice"

    def __init__(self, *, name: str = "accuracy_majority") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        return {None: accuracy_majority(view)}


class DecoyShare(StreamingMetric):
    """Fraction of agents standing on the decoy option.

    The corpus carries no `decoy` field, so this reads `game.options.decoy` when
    one is configured and otherwise falls back to the **modal wrong option this
    round**. The fallback is a within-round statistic and is therefore *not* the
    same quantity as the paper's decoy: the principled derivation is the modal
    wrong option under `profile: hidden, rounds: 0`, which is a property of a
    whole grid cell and belongs in analysis, not in a per-round metric. Which of
    the two produced a given number is recorded by `decoy_share_is_derived`.
    """

    requires_game_family = "choice"

    def __init__(self, *, name: str = "decoy_share") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        votes = _votes(view)
        summary = _summary(view)
        correct = summary.get("correct_answer")
        if not votes or correct is None:
            return {None: 0.0}
        decoy = summary.get("decoy")
        if decoy is None:
            wrong = Counter(vote for vote in votes if vote != correct)
            if not wrong:
                return {None: 0.0}
            decoy, _ = wrong.most_common(1)[0]
        return {None: sum(vote == decoy for vote in votes) / len(votes)}


class UnsharedDisclosureRate(StreamingMetric):
    """Fraction of hidden facts surfaced in conversation by this round.

    **The paper's central diagnostic**: agents integrate pooled information
    perfectly well once they have it and fail to *surface* it in the first
    place, so this is the quantity that separates a reasoning failure from a
    disclosure failure.

    A lower bound. Detection is normalized keyword overlap
    (`data.py::disclosed_facts`); a faithful paraphrase that shares few content
    words is not counted. Read every value as "at least this fraction".
    """

    def __init__(self, *, name: str = "unshared_disclosure_rate") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        return {None: float(_summary(view).get("unshared_disclosure_rate", 0.0))}


class _PhaseAccuracy(FinalMetric):
    """`Y` measured at the one round whose phase matches, or `None`."""

    requires_game_family = "choice"

    def __init__(self, phase: str, *, name: str, rule: str = "average") -> None:
        super().__init__(name, scope="population")
        self.phase = phase
        self.rule = rule

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        measure = accuracy_average if self.rule == "average" else accuracy_majority
        for view in views:
            if _summary(view).get("phase") == self.phase:
                return measure(view)
        return None


class YPre(_PhaseAccuracy):
    """`Y_pre` - accuracy at the pre-discussion vote, before any pooling."""

    def __init__(self, *, name: str = "y_pre", rule: str = "average") -> None:
        super().__init__(PRE_VOTE, name=name, rule=rule)


class YPost(_PhaseAccuracy):
    """`Y_post` - accuracy at the post-discussion vote.

    Under `profile: full, rounds: 0` this *is* `Y_full`: same code path, no
    discussion, every agent holding all of `Iu`. `gap_to_full` is deliberately
    not computed here - it needs the paired `profile: full` cell of the grid and
    is resolved in analysis, where both cells are in hand.
    """

    def __init__(self, *, name: str = "y_post", rule: str = "average") -> None:
        super().__init__(POST_VOTE, name=name, rule=rule)


class Improvement(FinalMetric):
    """`Y_post - Y_pre` - what the discussion was worth.

    `None` when either endpoint is missing, never 0: "the discussion achieved
    nothing" and "one of the votes never happened" are different findings.
    """

    requires_game_family = "choice"

    def __init__(self, *, name: str = "improvement", rule: str = "average") -> None:
        super().__init__(name, scope="population")
        self.rule = rule

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        pre = YPre(rule=self.rule).compute_final(views)
        post = YPost(rule=self.rule).compute_final(views)
        return None if pre is None or post is None else post - pre


class FirstCommitmentAccuracy(FinalMetric):
    """The dyadic analogue of `Y_pre` - accuracy of each agent's *first* choice.

    `hidden_bench_naming` has no plenary pre-vote to measure, because there is
    no moment at which the whole population decides at once. "Before pooling" is
    therefore a per-agent notion: the choice an agent made the first time it was
    asked, on the least evidence it will ever hold. Not the same estimator as
    the paper's `Y_pre`, and named differently so the two are never averaged
    together by accident.
    """

    requires_game_family = "choice"

    def __init__(self, *, name: str = "accuracy_first_commitment") -> None:
        super().__init__(name, scope="population")

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        for view in reversed(views):
            value = _summary(view).get("accuracy_first_commitment")
            if value is not None:
                return float(value)
        return None


class FinalAccuracy(FinalMetric):
    """Accuracy of the standing population at the end of the episode.

    The dyadic analogue of `Y_post`. Pairs with `FirstCommitmentAccuracy`, and
    `NamingImprovement` is their difference.
    """

    requires_game_family = "choice"

    def __init__(self, *, name: str = "accuracy_final", rule: str = "average") -> None:
        super().__init__(name, scope="population")
        self.rule = rule

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        if not views:
            return None
        measure = accuracy_average if self.rule == "average" else accuracy_majority
        return measure(views[-1])


class NamingImprovement(FinalMetric):
    """`accuracy_final - accuracy_first_commitment`; `None` if either is missing."""

    requires_game_family = "choice"

    def __init__(self, *, name: str = "improvement") -> None:
        super().__init__(name, scope="population")

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        first = FirstCommitmentAccuracy().compute_final(views)
        final = FinalAccuracy().compute_final(views)
        return None if first is None or final is None else final - first


class FinalDisclosureRate(FinalMetric):
    """The disclosure rate reached by the end of the episode. A lower bound."""

    def __init__(self, *, name: str = "final_unshared_disclosure_rate") -> None:
        super().__init__(name, scope="population")

    def compute_final(self, views: tuple[RoundView, ...]) -> Any:
        if not views:
            return None
        return float(_summary(views[-1]).get("unshared_disclosure_rate", 0.0))


class DisclosureReach(StreamingMetric):
    """How many *distinct agents* each hidden fact has reached (naming only).

    In the plenary game every message is broadcast, so reach is trivially `N`
    for anything said at all. In the dyadic game a fact spreads only along the
    edges that happened to be sampled, so this is the diffusion curve - the
    quantity that makes an interaction graph worth having.

    Scoped `option`: one curve per hidden-fact index, which keeps it a single
    metric rather than one metric per fact.
    """

    def __init__(self, *, name: str = "disclosure_reach") -> None:
        super().__init__(name, scope="option")

    def compute_round(self, view: RoundView) -> Mapping[MetricKey, Any]:
        reach = _summary(view).get("disclosure_reach", ())
        return {f"fact_{index}": float(value) for index, value in enumerate(reach)}
