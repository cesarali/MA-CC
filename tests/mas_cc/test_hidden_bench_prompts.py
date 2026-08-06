"""Golden-file prompt fidelity (brief §9.4).

The rendered prompts must match the paper's templates character for character,
*including the paper's own typos* ("randomly shuffle", "concise-just"). If a
well-meaning edit fixes the grammar, these tests fail - which is the point: a
reproduction whose prompts differ from the paper's is not a reproduction.

The golden strings below are taken from
`scripts/local_llms/hiddenbench_population_pipeline/scripts/hiddenbench_evaluation.py`,
this repository's already-run implementation of the same templates, rather than
from the brief's markdown-wrapped quotation of the PDF. See
`docs/hidden_bench/data_provenance.md` for the differences between the two.
"""

from __future__ import annotations

import pytest

from mas_cc.games.hidden_bench.vanilla.prompts import (
    bind_discussion_prompt,
    bind_vote_prompt,
    extract_json_object,
    normalize_vote,
    render_extra,
    shuffled_information,
)

SCENARIO = "You are a community leader deciding where to evacuate."
FACTS = ("The river level is still below the bridge.", "A mudslide covered the driveway.")
ANSWERS = ("West City", "East Town", "North Hill")

GOLDEN_SYSTEM = """You are a community leader deciding where to evacuate.

You have received the following information, notice the order of these information
are randomly shuffle, the order of facts does not indicate importance or relationship,
please reason carefully:
- The river level is still below the bridge.
- A mudslide covered the driveway.

Keep your response concise-just one or two sentences."""

GOLDEN_VOTE_INSTRUCTION = """Please decide and provide your rationale in the following JSON format:
{
  "vote": <A string, ['West City', 'East Town', 'North Hill']>,
  "rationale": <A string, representing your rationale>
}"""


def _messages(prompt) -> list[tuple[str, str]]:
    return [(message.role.value, message.content) for message in prompt.compile().messages]


def test_discussion_system_prompt_is_verbatim():
    prompt = bind_discussion_prompt(
        scenario=SCENARIO, information=FACTS, extra=None, transcript=(), is_first_speaker=True
    )
    role, content = _messages(prompt)[0]
    assert role == "system"
    assert content == GOLDEN_SYSTEM


def test_first_speaker_user_prompt_is_verbatim():
    prompt = bind_discussion_prompt(
        scenario=SCENARIO, information=FACTS, extra=None, transcript=(), is_first_speaker=True
    )
    assert _messages(prompt)[1] == ("user", "You are the first to speak.")


def test_subsequent_speaker_user_prompt_is_verbatim():
    transcript = ({"speaker_id": 0, "message": "Hello."}, {"speaker_id": 2, "message": "Agreed."})
    prompt = bind_discussion_prompt(
        scenario=SCENARIO, information=FACTS, extra=None, transcript=transcript,
        is_first_speaker=False,
    )
    assert _messages(prompt)[1] == (
        "user",
        "Previous messages from other people:\n"
        "Agent 0: Hello.\n"
        "Agent 2: Agreed.\n"
        "\n"
        "It's your turn to speak.",
    )


def test_pre_vote_prompt_has_no_transcript_and_the_verbatim_json_block():
    prompt = bind_vote_prompt(scenario=SCENARIO, information=FACTS, possible_answers=ANSWERS)
    messages = _messages(prompt)
    assert messages[0] == ("system", GOLDEN_SYSTEM)
    assert messages[1] == ("user", GOLDEN_VOTE_INSTRUCTION)
    assert len(messages) == 2


def test_post_vote_prompt_prefixes_the_group_discussion():
    prompt = bind_vote_prompt(
        scenario=SCENARIO,
        information=FACTS,
        possible_answers=ANSWERS,
        transcript=({"speaker_id": 1, "message": "The bridge is open."},),
    )
    messages = _messages(prompt)
    assert messages[1] == (
        "user",
        "Previous messages from other people:\n"
        "Agent 1: The bridge is open.\n"
        "\n" + GOLDEN_VOTE_INSTRUCTION,
    )


def test_the_papers_typos_are_reproduced_not_corrected():
    """Fixing these breaks the reproduction; the test exists to say so."""

    content = _messages(
        bind_discussion_prompt(
            scenario=SCENARIO, information=FACTS, extra=None, transcript=(), is_first_speaker=True
        )
    )[0][1]
    assert "are randomly shuffle," in content
    assert "concise-just one or two sentences" in content
    assert "are randomly shuffled" not in content
    assert "concise - just" not in content


# --------------------------------------------------------------------------
# The %extra% strategy hook
# --------------------------------------------------------------------------


def test_unset_extra_leaves_no_trailing_whitespace():
    """An unset hook must not change the prompt by even one character."""

    assert render_extra(None) == ""
    assert render_extra("") == ""
    assert render_extra("   ") == ""
    content = _messages(
        bind_discussion_prompt(
            scenario=SCENARIO, information=FACTS, extra=None, transcript=(), is_first_speaker=True
        )
    )[0][1]
    assert content.endswith("Keep your response concise-just one or two sentences.")


def test_extra_is_appended_with_exactly_one_leading_space():
    extra = "Share all the information you have."
    messages = _messages(
        bind_discussion_prompt(
            scenario=SCENARIO,
            information=FACTS,
            extra=extra,
            transcript=({"speaker_id": 0, "message": "Hi."},),
            is_first_speaker=False,
        )
    )
    assert messages[0][1].endswith(f"one or two sentences. {extra}")
    assert messages[1][1].endswith(f"It's your turn to speak. {extra}")


def test_vote_prompts_carry_no_extra():
    """A strategy ablation conditions the discussion, not the measurement."""

    plain = _messages(bind_vote_prompt(scenario=SCENARIO, information=FACTS, possible_answers=ANSWERS))
    assert plain[0][1] == GOLDEN_SYSTEM


# --------------------------------------------------------------------------
# Fact shuffling (§1.5)
# --------------------------------------------------------------------------


def test_shuffle_is_seeded_and_reproducible():
    shared, private = ("s1", "s2", "s3"), ("p1",)
    assert shuffled_information(shared, private, seed=42) == shuffled_information(
        shared, private, seed=42
    )
    assert sorted(shuffled_information(shared, private, seed=42)) == ["p1", "s1", "s2", "s3"]


def test_shared_and_private_are_shuffled_together():
    """If private facts always landed last, position would leak the asymmetry."""

    shared = tuple(f"s{index}" for index in range(8))
    private = ("PRIVATE",)
    positions = {
        shuffled_information(shared, private, seed=seed).index("PRIVATE") for seed in range(40)
    }
    assert len(positions) > 1, "the private fact always lands in the same position"
    assert positions != {len(shared)}


# --------------------------------------------------------------------------
# Tolerant vote parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        '{"vote": "West City", "rationale": "r"}',
        '```json\n{"vote": "West City", "rationale": "r"}\n```',
        'Sure! Here is my answer:\n{"vote": "West City", "rationale": "r"}\nHope that helps.',
        '  {"vote":"West City","rationale":"r"}  ',
    ],
    ids=["bare", "fenced", "prose-wrapped", "whitespace"],
)
def test_vote_parsing_tolerates_fences_and_stray_prose(response):
    """A retry burned on a well-formed vote is a retry that changes the numbers."""

    parsed = extract_json_object(response)
    assert parsed is not None
    assert normalize_vote(parsed["vote"], ANSWERS) == "West City"


def test_vote_normalization_is_case_and_whitespace_insensitive():
    assert normalize_vote("  west city ", ANSWERS) == "West City"
    assert normalize_vote("WEST CITY", ANSWERS) == "West City"


def test_ambiguous_vote_resolves_to_nothing_rather_than_the_first_match():
    """"not East Town, I say West City" must not silently become East Town."""

    assert normalize_vote("not East Town, I say West City", ANSWERS) is None
    assert normalize_vote("Somewhere else entirely", ANSWERS) is None
    assert normalize_vote(None, ANSWERS) is None


def test_unparseable_response_yields_no_object():
    assert extract_json_object("no json here at all") is None
    assert extract_json_object("{not valid json}") is None
