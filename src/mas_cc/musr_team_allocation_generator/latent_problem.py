"""Exact, language-model-independent Team Allocation problem generation."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .schemas import CandidateAllocation, LatentFact, LatentProblem, TaskSpec

_LEVEL_WORDS = {1: "limited", 2: "moderate", 3: "strong"}
_COOPERATION_WORDS = {1: "poorly", 2: "adequately", 3: "very well"}

# This is the authoritative latent prior used by both world sampling and the
# exact symbolic ambiguity analysis.  Keeping it explicit prevents the
# completion enumerator from silently drifting away from generation.
LATENT_VALUE_PRIOR = ((1, 1.0 / 3.0), (2, 1.0 / 3.0), (3, 1.0 / 3.0))
LATENT_VALUE_SUPPORT = tuple(value for value, _ in LATENT_VALUE_PRIOR)

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


def latent_values(problem: LatentProblem) -> tuple[int, ...]:
    """Return the nine latent values in the stable ``latent_facts`` order."""

    return tuple(fact.value for fact in latent_facts(problem))


def problem_from_latent_values(
    values: Sequence[int],
    *,
    people: Sequence[str] | None = None,
    tasks: Sequence[TaskSpec] | None = None,
) -> LatentProblem:
    """Construct and exactly score a world from its canonical latent vector."""

    if len(values) != 9:
        raise ValueError("Team Allocation requires exactly nine latent values")
    if any(value not in LATENT_VALUE_SUPPORT for value in values):
        raise ValueError(
            f"latent values must belong to {list(LATENT_VALUE_SUPPORT)}"
        )
    selected_people = tuple(people or _PEOPLE[0])
    selected_tasks = tuple(tasks or _TASK_PAIRS[0])
    if len(selected_people) != 3 or len(set(selected_people)) != 3:
        raise ValueError("Team Allocation requires exactly three distinct people")
    if len(selected_tasks) != 2:
        raise ValueError("Team Allocation requires exactly two tasks")
    skills = {
        selected_people[index]: (int(values[2 * index]), int(values[2 * index + 1]))
        for index in range(3)
    }
    cooperation = {
        cooperation_key(selected_people[0], selected_people[1]): int(values[6]),
        cooperation_key(selected_people[0], selected_people[2]): int(values[7]),
        cooperation_key(selected_people[1], selected_people[2]): int(values[8]),
    }
    allocations = enumerate_allocations(selected_people)
    temporary = LatentProblem(
        people=selected_people,  # type: ignore[arg-type]
        tasks=selected_tasks,  # type: ignore[arg-type]
        skill_matrix=skills,
        cooperation_matrix=cooperation,
        candidate_allocations=allocations,  # type: ignore[arg-type]
        candidate_scores=(0, 0, 0),
        gold_index=0,
        margin_to_second_best=0,
    )
    scores = tuple(score_allocation(temporary, item) for item in allocations)
    ranked = sorted(scores, reverse=True)
    return LatentProblem(
        people=selected_people,  # type: ignore[arg-type]
        tasks=selected_tasks,  # type: ignore[arg-type]
        skill_matrix=skills,
        cooperation_matrix=cooperation,
        candidate_allocations=allocations,  # type: ignore[arg-type]
        candidate_scores=scores,  # type: ignore[arg-type]
        gold_index=scores.index(ranked[0]),
        margin_to_second_best=ranked[0] - ranked[1],
    )


def _draw_problem(rng: random.Random) -> LatentProblem:
    people = tuple(rng.choice(_PEOPLE))
    tasks = tuple(rng.choice(_TASK_PAIRS))
    values = rng.choices(
        LATENT_VALUE_SUPPORT,
        weights=tuple(weight for _, weight in LATENT_VALUE_PRIOR),
        k=9,
    )
    return problem_from_latent_values(values, people=people, tasks=tasks)


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
