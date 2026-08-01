from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence

from hiddenbench_common import (
    ValidationError,
    normalize_vote,
    source_hidden_texts,
    stable_seed,
    task_id,
)
from hiddenbench_llm_api import LLMClient


PAPER_SYSTEM_TEMPLATE = """{description}

You have received the following information, notice the order of these information
are randomly shuffle, the order of facts does not indicate importance or relationship,
please reason carefully:
{information}

Keep your response concise-just one or two sentences.{extra}"""


def experiment_agents(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(task.get("agents"), list):
        result = []
        for index, agent in enumerate(task["agents"]):
            result.append(
                {
                    "agent_id": int(agent.get("agent_id", index)),
                    "private_information": [
                        str(item) for item in agent.get("private_information", [])
                    ],
                    "evidence_type": agent.get("evidence_type"),
                    "evidence_types": agent.get("evidence_types"),
                    "component_ids": agent.get("component_ids"),
                }
            )
        if not result:
            raise ValidationError(f"Task {task_id(task)} contains no agents.")
        return result

    hidden = source_hidden_texts(task)
    return [
        {
            "agent_id": index,
            "private_information": [text],
            "evidence_type": index,
        }
        for index, text in enumerate(hidden)
    ]


def scenario_description(task: Mapping[str, Any]) -> str:
    return str(
        task.get(
            "scenario_description",
            task.get("description", task.get("source_description", "")),
        )
    )


def shuffled_information(
    shared: Sequence[str],
    private: Sequence[str],
    *,
    seed: int,
) -> list[str]:
    information = [*map(str, shared), *map(str, private)]
    rng = random.Random(seed)
    rng.shuffle(information)
    return information


def discussion_system_prompt(
    task: Mapping[str, Any],
    information: Sequence[str],
    *,
    extra: str = "",
) -> str:
    facts = "\n".join(f"- {fact}" for fact in information)
    return PAPER_SYSTEM_TEMPLATE.format(
        description=scenario_description(task),
        information=facts,
        extra=(f" {extra}" if extra else ""),
    )


def vote_user_prompt(
    possible_answers: Sequence[str],
    *,
    transcript: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    prefix = ""
    if transcript:
        messages = "\n".join(
            f"Agent {item['speaker_id']}: {item['message']}"
            for item in transcript
        )
        prefix = f"Previous messages from other people:\n{messages}\n\n"
    return (
        prefix
        + "Please decide and provide your rationale in the following JSON format:\n"
        + "{\n"
        + f'  "vote": <A string, {list(possible_answers)}>,\n'
        + '  "rationale": <A string, representing your rationale>\n'
        + "}"
    )


def call_vote(
    client: LLMClient,
    task: Mapping[str, Any],
    private_information: Sequence[str],
    *,
    seed: int,
    transcript: Sequence[Mapping[str, Any]] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    information = shuffled_information(
        task["shared_information"],
        private_information,
        seed=seed,
    )
    response, metadata = client.generate_json(
        system=discussion_system_prompt(task, information),
        user=vote_user_prompt(
            task["possible_answers"],
            transcript=transcript,
        ),
        temperature=temperature,
    )
    raw_vote = (
        response.get("vote")
        if isinstance(response, Mapping)
        else None
    )
    vote = normalize_vote(raw_vote, task["possible_answers"])
    return {
        "vote": vote,
        "raw_vote": raw_vote,
        "rationale": (
            response.get("rationale")
            if isinstance(response, Mapping)
            else None
        ),
        "correct": vote == task["correct_answer"],
        "information": information,
        "api": metadata,
    }


def call_message(
    client: LLMClient,
    task: Mapping[str, Any],
    private_information: Sequence[str],
    *,
    seed: int,
    transcript: Sequence[Mapping[str, Any]],
    first: bool,
    temperature: float | None = None,
    extra: str = "",
) -> dict[str, Any]:
    information = shuffled_information(
        task["shared_information"],
        private_information,
        seed=seed,
    )
    if first:
        user = "You are the first to speak."
    else:
        messages = "\n".join(
            f"Agent {item['speaker_id']}: {item['message']}"
            for item in transcript
        )
        user = (
            "Previous messages from other people:\n"
            f"{messages}\n\n"
            "It's your turn to speak."
        )
        if extra:
            user += f" {extra}"

    text, metadata = client.generate_text(
        system=discussion_system_prompt(
            task,
            information,
            extra=extra,
        ),
        user=user,
        temperature=temperature,
        max_output_tokens=500,
    )
    return {
        "message": text.strip(),
        "information": information,
        "api": metadata,
    }


def vote_metrics(
    votes: Sequence[Mapping[str, Any]],
    correct_answer: str,
) -> dict[str, Any]:
    normalized = [item.get("vote") for item in votes]
    correct_count = sum(vote == correct_answer for vote in normalized)
    counts = Counter(vote for vote in normalized if vote is not None)
    majority_vote = counts.most_common(1)[0][0] if counts else None
    majority_count = counts[majority_vote] if majority_vote is not None else 0
    unanimous = (
        len(normalized) > 0
        and normalized[0] is not None
        and all(vote == normalized[0] for vote in normalized)
    )
    return {
        "average_accuracy": correct_count / len(votes) if votes else 0.0,
        "correct_count": correct_count,
        "num_votes": len(votes),
        "majority_vote": majority_vote,
        "majority_fraction": majority_count / len(votes) if votes else 0.0,
        "majority_correct": majority_vote == correct_answer,
        "unanimous": unanimous,
        "unanimous_vote": normalized[0] if unanimous else None,
    }


def full_information(task: Mapping[str, Any]) -> list[str]:
    return source_hidden_texts(task)


def run_standard_session(
    client: LLMClient,
    task: Mapping[str, Any],
    *,
    session_index: int,
    communication_rounds: int,
    base_seed: int,
    temperature: float | None,
    early_stop: bool,
    speaker_order: str,
) -> dict[str, Any]:
    agents = experiment_agents(task)
    n = len(agents)
    session_seed = stable_seed(base_seed, task_id(task), session_index)

    pre_votes = []
    for agent in agents:
        vote = call_vote(
            client,
            task,
            agent["private_information"],
            seed=stable_seed(
                session_seed, "pre", agent["agent_id"]
            ),
            temperature=temperature,
        )
        vote["agent_id"] = agent["agent_id"]
        pre_votes.append(vote)

    rng = random.Random(stable_seed(session_seed, "speakers"))
    if speaker_order == "round_robin":
        offset = rng.randrange(n)
        speakers = [(offset + round_index) % n for round_index in range(
            communication_rounds
        )]
    elif speaker_order == "random":
        speakers = [rng.randrange(n) for _ in range(communication_rounds)]
    else:
        raise ValidationError(
            "speaker_order must be `round_robin` or `random`."
        )

    transcript: list[dict[str, Any]] = []
    stopped_early = False
    for round_index, speaker_index in enumerate(speakers):
        agent = agents[speaker_index]
        message = call_message(
            client,
            task,
            agent["private_information"],
            seed=stable_seed(
                session_seed,
                "discussion",
                round_index,
                agent["agent_id"],
            ),
            transcript=transcript,
            first=round_index == 0,
            temperature=temperature,
        )
        transcript.append(
            {
                "round": round_index + 1,
                "speaker_id": agent["agent_id"],
                **message,
            }
        )

        # Optional engineering convenience. Disabled by default to match the
        # paper's fixed-depth protocol.
        if early_stop and round_index + 1 >= n:
            interim = []
            for interim_agent in agents:
                decision = call_vote(
                    client,
                    task,
                    interim_agent["private_information"],
                    seed=stable_seed(
                        session_seed,
                        "interim",
                        round_index,
                        interim_agent["agent_id"],
                    ),
                    transcript=transcript,
                    temperature=temperature,
                )
                interim.append(decision)
            if vote_metrics(interim, task["correct_answer"])["unanimous"]:
                stopped_early = True
                break

    post_votes = []
    for agent in agents:
        vote = call_vote(
            client,
            task,
            agent["private_information"],
            seed=stable_seed(
                session_seed, "post", agent["agent_id"]
            ),
            transcript=transcript,
            temperature=temperature,
        )
        vote["agent_id"] = agent["agent_id"]
        post_votes.append(vote)

    all_hidden = full_information(task)
    full_votes = []
    for agent in agents:
        vote = call_vote(
            client,
            task,
            all_hidden,
            seed=stable_seed(
                session_seed, "full", agent["agent_id"]
            ),
            temperature=temperature,
        )
        vote["agent_id"] = agent["agent_id"]
        full_votes.append(vote)

    return {
        "session_index": session_index,
        "num_agents": n,
        "communication_rounds_requested": communication_rounds,
        "communication_rounds_completed": len(transcript),
        "speaker_order": speaker_order,
        "stopped_early": stopped_early,
        "pre_discussion_decisions": pre_votes,
        "discussion_history": transcript,
        "post_discussion_decisions": post_votes,
        "full_profile_decisions": full_votes,
        "metrics": {
            "pre": vote_metrics(pre_votes, task["correct_answer"]),
            "post": vote_metrics(post_votes, task["correct_answer"]),
            "full": vote_metrics(full_votes, task["correct_answer"]),
        },
    }
