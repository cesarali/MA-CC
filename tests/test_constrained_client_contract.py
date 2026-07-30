import asyncio
import math
import random
import threading
import time

import pytest

from naming_game.gemma_local_client import (
    GemmaLocalAsyncLLMClient,
    GenerationResult,
    RuntimeDecisionResult,
    ScoringResult,
    _dispatch_candidate_scoring,
)
from naming_game.local_model_types import ConstrainedDecisionClient, ConstrainedLLMClient


class FakeRuntime:
    diagnostics = {"kind": "fake"}
    def generate(self, messages, **kwargs): return GenerationResult("answer", 4, 2)
    def score(self, messages, choices):
        return ScoringResult(4, tuple((i + 1,) * (i + 1) for i in range(len(choices))), tuple(-1.0 for _ in choices))


def test_structural_contract_order_tie_multitoken_and_usage():
    async def check():
        client = GemmaLocalAsyncLLMClient(runtime_factory=FakeRuntime)
        assert isinstance(client, ConstrainedLLMClient)
        response = await client.complete_constrained([{"role": "user", "content": "choose"}], choices=["B", "AA"])
        assert [score.choice for score in response.scores] == ["B", "AA"]
        assert response.selected_choice == "B"
        assert response.scores[1].token_ids == (2, 2)
        assert math.isclose(sum(score.probability for score in response.scores), 1)
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 3
    asyncio.run(check())


class CombinedRuntime:
    diagnostics = {"kind": "combined-fake"}

    def __init__(self, *, reason="coordinated", reason_valid=True, scores=(-1.0, -2.0)):
        self.reason = reason
        self.reason_valid = reason_valid
        self.log_likelihoods = tuple(scores)
        self.decision_calls = 0
        self.reason_calls = 0
        self.prefixes = []
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def generate(self, messages, **kwargs):
        raise AssertionError("combined decisions must not call public generation")

    def score(self, messages, choices):
        return ScoringResult(
            5,
            tuple((index + 10,) * (index + 1) for index in range(len(choices))),
            self.log_likelihoods[: len(choices)],
            5,
        )

    def decide(self, messages, choices, **kwargs):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(0.005)
            self.decision_calls += 1
            scoring = self.score(messages, choices)
            scaled = [value / kwargs["choice_temperature"] for value in scoring.log_likelihoods]
            peak = max(scaled)
            weights = [math.exp(value - peak) for value in scaled]
            probabilities = [weight / sum(weights) for weight in weights]
            if kwargs["selection_policy"] == "argmax":
                selected_index = max(range(len(choices)), key=probabilities.__getitem__)
            else:
                selected_index = random.Random(kwargs["seed"]).choices(
                    range(len(choices)), weights=probabilities, k=1
                )[0]
            if kwargs["output_format"] == "choice_only":
                return RuntimeDecisionResult(scoring, selected_index)
            self.reason_calls += 1
            prefix = f"{choices[selected_index]}\nReason: "
            self.prefixes.append(prefix)
            return RuntimeDecisionResult(
                scoring,
                selected_index,
                self.reason,
                self.reason_valid,
                reason_prompt_tokens=7,
                reason_tokens=2,
            )
        finally:
            with self.lock:
                self.active -= 1


def test_complete_decision_contract_choice_only_and_reason_accounting():
    async def check():
        runtime = CombinedRuntime(scores=(-1.0, -2.0))
        client = GemmaLocalAsyncLLMClient(runtime_factory=lambda: runtime)
        assert isinstance(client, ConstrainedDecisionClient)

        bare = await client.complete_decision(
            [{"role": "user", "content": "choose"}],
            choices=["B", "LONG"],
            output_format="choice_only",
            choice_temperature=0.5,
        )
        assert bare.selected_choice == "B"
        assert bare.content == "B"
        assert bare.reason is None and bare.reason_valid is None
        assert [score.choice for score in bare.scores] == ["B", "LONG"]
        assert [score.token_ids for score in bare.scores] == [(10,), (11, 11)]
        assert [score.log_likelihood for score in bare.scores] == [-1.0, -2.0]
        assert math.isclose(sum(score.probability for score in bare.scores), 1.0)
        assert bare.usage.prompt_tokens == 5
        assert bare.usage.completion_tokens == 3
        assert runtime.reason_calls == 0

        explained = await client.complete_decision(
            [{"role": "user", "content": "choose"}],
            choices=["B", "LONG"],
            output_format="choice_reason",
            choice_temperature=0.5,
            generation_temperature=0.25,
            max_reason_tokens=4,
        )
        assert explained.selected_choice == "B"
        assert explained.content == "B\nReason: coordinated"
        assert explained.reason == "coordinated" and explained.reason_valid is True
        assert runtime.prefixes == ["B\nReason: "]
        assert explained.usage.prompt_tokens == 12
        assert explained.usage.completion_tokens == 5
        assert runtime.decision_calls == 2
        assert client.stats["actual_calls"] == 2
        assert client.stats["successful_calls"] == 2
        assert client.stats["failed_calls"] == 0
        assert len(client.stats["latencies"]) == 2

    asyncio.run(check())


def test_argmax_ties_and_seeded_sampling_preserve_the_distribution():
    async def check():
        tied_runtime = CombinedRuntime(scores=(0.0, 0.0))
        tied = GemmaLocalAsyncLLMClient(runtime_factory=lambda: tied_runtime)
        tie = await tied.complete_decision(
            [{"role": "user", "content": "x"}],
            choices=["SECOND", "FIRST"],
            output_format="choice_only",
            selection_policy="argmax",
        )
        assert tie.selected_choice == "SECOND"

        runtime = CombinedRuntime(scores=(-0.1, -1.0))
        client = GemmaLocalAsyncLLMClient(runtime_factory=lambda: runtime)
        first = await client.complete_decision(
            [{"role": "user", "content": "x"}], choices=["A", "B"],
            output_format="choice_only", selection_policy="sample", seed=1,
        )
        repeated = await client.complete_decision(
            [{"role": "user", "content": "x"}], choices=["A", "B"],
            output_format="choice_only", selection_policy="sample", seed=1,
        )
        other = await client.complete_decision(
            [{"role": "user", "content": "x"}], choices=["A", "B"],
            output_format="choice_only", selection_policy="sample", seed=2,
        )
        assert first.selected_choice == repeated.selected_choice
        assert first.selected_choice != other.selected_choice
        assert [score.probability for score in first.scores] == [
            score.probability for score in other.scores
        ]

    asyncio.run(check())


def test_malformed_reason_never_changes_the_authoritative_choice():
    async def check():
        runtime = CombinedRuntime(reason="replacement B", reason_valid=False)
        response = await GemmaLocalAsyncLLMClient(
            runtime_factory=lambda: runtime
        ).complete_decision(
            [{"role": "user", "content": "x"}],
            choices=["A", "B"],
            output_format="choice_reason",
        )
        assert response.selected_choice == "A"
        assert response.content == "A\nReason: "
        assert response.reason is None and response.reason_valid is False

    asyncio.run(check())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"output_format": "json_reason"},
        {"output_format": "choice_only", "selection_policy": "last"},
        {"output_format": "choice_only", "choice_temperature": 0},
        {"output_format": "choice_only", "choice_temperature": float("inf")},
        {"output_format": "choice_only", "generation_temperature": -1},
        {"output_format": "choice_reason", "max_reason_tokens": 0},
    ),
)
def test_complete_decision_rejects_invalid_parameters_before_a_request(kwargs):
    client = GemmaLocalAsyncLLMClient(runtime_factory=CombinedRuntime)
    with pytest.raises(ValueError):
        asyncio.run(
            client.complete_decision(
                [{"role": "user", "content": "x"}], choices=["A", "B"], **kwargs
            )
        )
    assert client.stats["actual_calls"] == 0


@pytest.mark.parametrize(
    "scoring",
    (
        ScoringResult(1, ((1,),), (0.0,)),
        ScoringResult(1, ((1,), ()), (0.0, 1.0)),
        ScoringResult(1, ((1,), (2,)), (0.0, float("nan"))),
        ScoringResult(-1, ((1,), (2,)), (0.0, 1.0)),
    ),
)
def test_bad_runtime_scoring_fails_one_logical_request(scoring):
    class BadRuntime(CombinedRuntime):
        def score(self, messages, choices):
            return scoring

        def decide(self, messages, choices, **kwargs):
            return RuntimeDecisionResult(scoring, 0)

    async def check():
        client = GemmaLocalAsyncLLMClient(runtime_factory=BadRuntime)
        with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
            await client.complete_decision(
                [{"role": "user", "content": "x"}],
                choices=["A", "B"],
                output_format="choice_only",
            )
        assert client.stats["actual_calls"] == 1
        assert client.stats["successful_calls"] == 0
        assert client.stats["failed_calls"] == 1

    asyncio.run(check())


def test_combined_first_calls_initialize_once_and_inference_is_serialized():
    async def check():
        runtime = CombinedRuntime()
        initializations = 0

        def factory():
            nonlocal initializations
            initializations += 1
            return runtime

        client = GemmaLocalAsyncLLMClient(runtime_factory=factory)
        await asyncio.gather(
            *(
                client.complete_decision(
                    [{"role": "user", "content": "x"}],
                    choices=["A", "B"],
                    output_format="choice_only",
                )
                for _ in range(4)
            )
        )
        assert initializations == 1
        assert runtime.maximum == 1

    asyncio.run(check())


def test_cancellation_waits_for_worker_before_releasing_inference_slot():
    class SlowRuntime(CombinedRuntime):
        def decide(self, messages, choices, **kwargs):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.03)
                scoring = self.score(messages, choices)
                return RuntimeDecisionResult(scoring, 0)
            finally:
                with self.lock:
                    self.active -= 1

    async def check():
        runtime = SlowRuntime()
        client = GemmaLocalAsyncLLMClient(runtime_factory=lambda: runtime)
        cancelled = asyncio.create_task(
            client.complete_decision(
                [{"role": "user", "content": "x"}],
                choices=["A", "B"],
                output_format="choice_only",
            )
        )
        await asyncio.sleep(0.005)
        cancelled.cancel()
        second = asyncio.create_task(
            client.complete_decision(
                [{"role": "user", "content": "x"}],
                choices=["A", "B"],
                output_format="choice_only",
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert (await second).selected_choice == "A"
        assert runtime.maximum == 1
        assert client.stats["actual_calls"] == 2
        assert client.stats["successful_calls"] == 1
        assert client.stats["failed_calls"] == 1

    asyncio.run(check())


def test_scoring_dispatch_uses_one_batch_or_full_sequence_fallback():
    calls = {"single": 0, "sequence": 0}

    def single(ids):
        calls["single"] += 1
        return tuple(-float(value) for value in ids)

    def sequence(ids):
        calls["sequence"] += 1
        return -float(sum(ids))

    one_token, path = _dispatch_candidate_scoring(
        ((1,), (2,), (3,)),
        score_single_tokens=single,
        score_sequence=sequence,
    )
    assert path == "single_token" and one_token == (-1.0, -2.0, -3.0)
    assert calls == {"single": 1, "sequence": 0}

    calls = {"single": 0, "sequence": 0}
    multi_token, path = _dispatch_candidate_scoring(
        ((1,), (2, 3)),
        score_single_tokens=single,
        score_sequence=sequence,
    )
    assert path == "sequence" and multi_token == (-1.0, -5.0)
    assert calls == {"single": 0, "sequence": 2}


@pytest.mark.parametrize("choices,temp", [([], 1), ([""], 1), (["A", "A"], 1), (["A"], 0), (["A"], float("nan"))])
def test_invalid_choices(choices, temp):
    with pytest.raises(ValueError):
        asyncio.run(GemmaLocalAsyncLLMClient(runtime_factory=FakeRuntime).complete_constrained(
            [{"role": "user", "content": "x"}], choices=choices, temperature=temp))
