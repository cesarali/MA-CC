"""Versioned prompts for evidence generation and full-information QA."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .schemas import LatentFact, LatentProblem
from .symbolic_facts import CanonicalFact


def evidence_prompt(
    problem: LatentProblem,
    fact: LatentFact | CanonicalFact,
    *,
    branches: int,
    statements_per_branch: int,
    tree_depth: int,
    forbidden_phrases: Sequence[str],
) -> str:
    hidden_conclusion = (
        fact.hidden_claim if isinstance(fact, LatentFact) else fact.canonical_text
    )
    people = list(fact.people) if isinstance(fact, LatentFact) else list(problem.people)
    level = fact.value if isinstance(fact, LatentFact) else None
    target = {
        "hidden_fact_id": fact.fact_id,
        "hidden_conclusion": hidden_conclusion,
        "people": people,
        "kind": fact.kind,
        "level": level,
    }
    return f"""Create independent indirect evidence branches for a Team Allocation reasoning task.
The evidence must let a careful reader infer the hidden conclusion, but must never state it directly.
Use concrete past events, outcomes, or interactions. Each branch must describe a different event.
Do not recommend an assignment, name a correct/best option, disclose numeric scores, or repeat task labels as an ability verdict.
All explicit statements in one branch must form one coherent evidence card. Commonsense bridges are saved only as hidden provenance.

Scenario: three named people will be split between two tasks: {problem.tasks[0].name!r} and {problem.tasks[1].name!r}.
Hidden target (never copy into explicit statements): {json.dumps(target, ensure_ascii=False)}
Forbidden phrases: {json.dumps(list(forbidden_phrases), ensure_ascii=False)}

Return JSON only, with exactly {branches} branches and this shape:
{{
  "branches": [
    {{
      "intermediate_claims": ["indirect inference"],
      "statements": ["visible observation 1", "visible observation 2"],
      "commonsense_bridges": ["unstated rule connecting observations to the inference"]
    }}
  ]
}}
Each branch needs exactly {statements_per_branch} non-empty statements, one or more commonsense bridges, and exactly {max(0, tree_depth - 1)} intermediate_claims. Branches must not be paraphrases of each other.
"""


def full_information_prompt(task: dict[str, object]) -> str:
    evidence_lines: list[str] = []
    for card in task["evidence"]:  # type: ignore[index]
        joined = " ".join(card["text"])  # type: ignore[index]
        evidence_lines.append(f"- {card['evidence_id']}: {joined}")  # type: ignore[index]
    option_lines = [
        f"{option['index']}: {option['display_text']}"  # type: ignore[index]
        for option in task["options"]  # type: ignore[index]
    ]
    return f"""Solve this Team Allocation problem using all evidence.
One person does the first task and the other two jointly do the second. Compare each allocation using the person's relevant skill for the first task, both people's relevant skills for the second task, and the cooperation of that pair. Choose exactly one option.

Scenario:
{task["scenario"]}

Options:
{chr(10).join(option_lines)}

Evidence:
{chr(10).join(evidence_lines)}

Return JSON only: {{"option_index": 0, "rationale": "brief explanation grounded in the evidence"}}
"""
