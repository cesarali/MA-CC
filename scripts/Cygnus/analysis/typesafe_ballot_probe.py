"""Live probe of the TypeSafe ballot provider against real ballot prompts.

Plays the relational blackboard game from the test fixture config with a
recording mock provider to obtain genuine compiled ballot prompts (system +
user messages exactly as an agent receives them), then replays the first N
through ``TypeSafeChoiceProvider`` against the live endpoint. Writes one JSON
record per ballot (letters offered, typed vote, probability vector,
confidence, latency, tokens) plus a summary. Credentials come from
``MA_CC_SYSTEMONE_KEY_FILE`` / ``LLM_KEY``; endpoint from ``LLM_BASE`` /
``MA_CC_SYSTEMONE_URL``.

    python scripts/Cygnus/analysis/typesafe_ballot_probe.py --calls 20 --output probe.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mas_cc.games import create_game  # noqa: E402
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (  # noqa: E402
    run_relational_imitation_round_feedback_game,
)
from mas_cc.llm_runtime.config import LLMProviderConfig  # noqa: E402
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider  # noqa: E402
from mas_cc.llm_runtime.providers.adapters.typesafe import TypeSafeChoiceProvider, presented_letters  # noqa: E402


def _fixtures():
    spec = importlib.util.spec_from_file_location("rb_fixtures", ROOT / "tests" / "mas_cc" / "test_relational_blackboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_ballots(rounds: int) -> list:
    fixtures = _fixtures()
    config = fixtures._config(rounds=rounds)
    requests = []

    def factory(request):
        requests.append(request)
        return json.dumps({"vote": "A", "private_reason": "recording pass",
                           "public_message": {"type": "NONE", "text": None, "shared_fact_id": None, "reply_to": None}})

    provider = MockLLMProvider(config.llm_provider, response_factory=factory)
    asyncio.run(run_relational_imitation_round_feedback_game(create_game(config.game), config, provider))
    return requests


async def replay(requests, calls: int, model: str):
    provider = TypeSafeChoiceProvider(LLMProviderConfig(type="typesafe", model=model))
    rows = []
    for index, request in enumerate(requests[:calls]):
        letters = presented_letters(request)
        started = time.perf_counter()
        response = await provider.complete(request)
        ballot = json.loads(response.content)
        info = response.raw_response["typesafe"]
        rows.append({
            "index": index, "letters": list(letters), "vote": ballot["vote"], "probabilities": info["probabilities"],
            "confidence": info["confidence"], "latency_seconds": round(time.perf_counter() - started, 3),
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            "cached": info["cached"], "model": response.model, "private_reason": ballot["private_reason"],
            "prompt_chars": sum(len(m.content) for m in request.messages),
        })
        print(f"[{index:02d}] {ballot['vote']} p={info['probabilities']} conf={info['confidence']} {rows[-1]['latency_seconds']}s",
              file=sys.stderr)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    requests = record_ballots(args.rounds)
    print(f"recorded {len(requests)} ballot prompts; replaying {min(args.calls, len(requests))}", file=sys.stderr)
    rows = asyncio.run(replay(requests, args.calls, args.model))
    live = [r for r in rows if not r["cached"]]
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    summary = {
        "calls": len(rows), "live": len(live), "model": sorted({r["model"] for r in rows}),
        "votes": dict(Counter(r["vote"] for r in rows)),
        "latency_seconds_median": statistics.median(r["latency_seconds"] for r in live) if live else None,
        "latency_seconds_max": max((r["latency_seconds"] for r in live), default=None),
        "input_tokens_total": sum(r["input_tokens"] or 0 for r in live),
        "usd_estimate": sum(r["input_tokens"] or 0 for r in live) * 42 / 1e9,
        "confidence_median": statistics.median(confidences) if confidences else None,
        "max_probability_median": statistics.median(max(r["probabilities"].values()) for r in rows if r["probabilities"]),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "ballots": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
