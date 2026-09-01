"""Exact, language-model-independent Team Allocation problem generation."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .schemas import CandidateAllocation, LatentFact, LatentProblem, TaskSpec

_LEVEL_WORDS = {1: "limited", 2: "moderate", 3: "strong"}
_COOPERATION_WORDS = {1: "poorly", 2: "adequately", 3: "very well"}

_PEOPLE = (
    ("Alice", "Bruno", "Chandra"),
    ("Diego", "Elena", "Farah"),
    ("Grace", "Hassan", "Iris"),
    ("Jonah", "Kira", "Luis"),
    ("Maya", "Noah", "Priya"),
    ("Rina", "Samir", "Talia"),
)

_TASK_PAIRS = (
    (
        TaskSpec("build the data pipeline", "software engineering"),
        TaskSpec("conduct stakeholder interviews", "interpersonal communication"),
    ),
    (
        TaskSpec("analyze the field measurements", "quantitative analysis"),
        TaskSpec("coordinate the community workshop", "group facilitation"),
    ),
    (
        TaskSpec("repair the monitoring equipment", "technical troubleshooting"),
        TaskSpec("prepare the public briefing", "public communication"),
    ),
    (
        TaskSpec("design the prototype", "product design"),
        TaskSpec("organize the deployment", "operational coordination"),
    ),
)


def cooperation_key(person_a: str, person_b: str) -> str:
    return "|".join(sorted((person_a, person_b)))


def enumerate_allocations(people: Sequence[str]) -> tuple[CandidateAllocation, ...]:
    if len(people) != 3 or len(set(people)) != 3:
        raise ValueError("Team Allocation requires exactly three distinct people")
    return tuple(
        CandidateAllocation(
            singleton=person,
            pair=tuple(other for other in people if other != person),  # type: ignore[arg-type]
        )
        for person in people
    )


def score_allocation(problem: LatentProblem, allocation: CandidateAllocation) -> int:
    first = problem.skill_matrix[allocation.singleton][0]
    second = sum(problem.skill_matrix[person][1] for person in allocation.pair)
    teamwork = problem.cooperation_matrix[cooperation_key(*allocation.pair)]
    return first + second + teamwork


def _draw_problem(rng: random.Random) -> LatentProblem:
    people = tuple(rng.choice(_PEOPLE))
    tasks = tuple(rng.choice(_TASK_PAIRS))
    skills = {person: (rng.randint(1, 3), rng.randint(1, 3)) for person in people}
    cooperation = {
        cooperation_key(people[i], people[j]): rng.randint(1, 3)
        for i in range(3)
        for j in range(i + 1, 3)
    }
    allocations = enumerate_allocations(people)
    temporary = LatentProblem(
        people=people,  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        skill_matrix=skills,
        cooperation_matrix=cooperation,
        candidate_allocations=allocations,  # type: ignore[arg-type]
        candidate_scores=(0, 0, 0),
        gold_index=0,
        margin_to_second_best=0,
    )
    scores = tuple(
        score_allocation(temporary, allocation) for allocation in allocations
    )
    ranked = sorted(scores, reverse=True)
    return LatentProblem(
        people=people,  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        skill_matrix=skills,
        cooperation_matrix=cooperation,
        candidate_allocations=allocations,  # type: ignore[arg-type]
        candidate_scores=scores,  # type: ignore[arg-type]
        gold_index=scores.index(ranked[0]),
        margin_to_second_best=ranked[0] - ranked[1],
    )


def generate_latent_problem(
    rng: random.Random,
    *,
    min_margin: int = 1,
    max_attempts: int = 10_000,
) -> LatentProblem:
    if min_margin < 1:
        raise ValueError("min_margin must be at least 1")
    for _ in range(max_attempts):
        problem = _draw_problem(rng)
        if problem.margin_to_second_best >= min_margin:
            return problem
    raise RuntimeError("could not generate a unique Team Allocation problem")


def latent_facts(problem: LatentProblem) -> tuple[LatentFact, ...]:
    facts: list[LatentFact] = []
    for person_index, person in enumerate(problem.people):
        for task_index, task in enumerate(problem.tasks):
            value = problem.skill_matrix[person][task_index]
            facts.append(
                LatentFact(
                    fact_id=f"skill_p{person_index}_t{task_index}",
                    kind="skill",
                    value=value,
                    people=(person,),
                    task_index=task_index,
                    hidden_claim=(
                        f"{person} has {_LEVEL_WORDS[value]} {task.skill} skill for "
                        f"the task to {task.name}."
                    ),
                )
            )
    for left in range(3):
        for right in range(left + 1, 3):
            first, second = problem.people[left], problem.people[right]
            value = problem.cooperation_matrix[cooperation_key(first, second)]
            facts.append(
                LatentFact(
                    fact_id=f"coop_p{left}_p{right}",
                    kind="cooperation",
                    value=value,
                    people=(first, second),
                    task_index=None,
                    hidden_claim=f"{first} and {second} work {_COOPERATION_WORDS[value]} together.",
                )
            )
    return tuple(facts)


def scenario_for(problem: LatentProblem) -> str:
    first, second = problem.tasks
    return (
        f"A project lead must allocate {', '.join(problem.people)} between two urgent tasks. "
        f"One person must {first.name}; the other two must jointly {second.name}. "
        "Each candidate allocation is judged by the relevant individual skills and by how "
        "well the two-person team cooperates. Use the evidence to choose the strongest allocation."
    )
