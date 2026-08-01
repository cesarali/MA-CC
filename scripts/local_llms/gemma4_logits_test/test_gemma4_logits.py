#!/usr/bin/env python3
"""Gemma 4 full-vocabulary and constrained-choice logits smoke test."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# This must happen before importing or initializing Hugging Face components.
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

MODEL_ID = "google/gemma-4-12B-it"
PROMPT = """You are participating in a small decision game.

Question:
Which number is larger: 7 or 3?

Choose exactly one option:
A. 7
B. 3
C. They are equal

Return only A, B, or C."""
MIN_FREE_BYTES = 30 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--choices", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Explicitly allow CPU fallback (very slow and memory intensive).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.choices or any(not choice or choice.isspace() for choice in args.choices):
        raise ValueError("--choices must contain one or more non-empty strings")
    if len(set(args.choices)) != len(args.choices):
        raise ValueError("--choices must not contain duplicates")
    if args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--top-k and --max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be a finite positive number")


def prepare_hf_home() -> Path:
    raw = os.environ.get("HF_HOME", "").strip().strip('"').strip("'")
    if not raw:
        raise RuntimeError("HF_HOME is absent. Set it in the repository .env file.")
    if raw.startswith("/replace/with/") or "high-capacity-storage" in raw:
        raise RuntimeError(f"HF_HOME still contains the placeholder path: {raw}")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    print(f"HF_HOME: {path}")
    print(f"HF_HOME free disk: {free / 1024**3:.1f} GiB")
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Insufficient free space in HF_HOME: need at least 30 GiB, "
            f"found {free / 1024**3:.1f} GiB"
        )
    return path


def memory_report(torch: Any, label: str) -> None:
    if not torch.cuda.is_available():
        return
    print(
        f"CUDA memory ({label}): allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB, "
        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GiB, "
        f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
    )


def model_input_device(model: Any, torch: Any) -> Any:
    device_map = getattr(model, "hf_device_map", None) or {}
    for device in device_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
            return torch.device(device)
    return next(model.parameters()).device


def apply_text_chat_template(processor: Any, messages: list[dict[str, str]]) -> Any:
    kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError) as exc:
        print(f"Chat template does not accept enable_thinking=False; retrying without it: {exc}")
        return processor.apply_chat_template(messages, **kwargs)


def tokenizer_for(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The loaded processor does not expose a tokenizer")
    return tokenizer


def candidate_token_ids(tokenizer: Any, choice: str) -> list[int]:
    ids = tokenizer.encode(choice, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Choice {choice!r} tokenized to an empty sequence")
    return ids


def score_choice_sequences(
    model: Any,
    processor: Any,
    prompt_inputs: Any,
    choices: list[str],
) -> dict[str, float]:
    """Teacher-force each complete choice and return summed conditional log p."""
    import torch

    tokenizer = tokenizer_for(processor)
    prompt_ids = prompt_inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    device = prompt_ids.device
    scores: dict[str, float] = {}

    with torch.inference_mode():
        for choice in choices:
            answer_ids = torch.tensor(
                [candidate_token_ids(tokenizer, choice)], dtype=prompt_ids.dtype, device=device
            )
            input_ids = torch.cat((prompt_ids, answer_ids), dim=1)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits[:, i] predicts input_ids[:, i + 1].
            answer_logits = outputs.logits[:, prompt_len - 1 : input_ids.shape[1] - 1, :]
            targets = answer_ids
            if answer_logits.shape[1] != targets.shape[1]:
                raise AssertionError("Answer logits and answer tokens have inconsistent lengths")
            token_log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
            score = token_log_probs.gather(-1, targets.unsqueeze(-1)).sum()
            if not torch.isfinite(score):
                raise FloatingPointError(f"Non-finite sequence score for {choice!r}")
            scores[choice] = score.item()
    return scores


def print_sequence_distribution(scores: dict[str, float], temperature: float) -> dict[str, float]:
    import torch

    choices = list(scores)
    values = torch.tensor([scores[c] for c in choices], dtype=torch.float32)
    probs = torch.softmax(values / temperature, dim=0)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    print("\nSequence-level constrained distribution:")
    for choice, score, prob in zip(choices, values.tolist(), probs.tolist()):
        print(f"  choice={choice!r} log_likelihood={score:.6f} probability={prob:.8f}")
    return dict(zip(choices, probs.tolist()))


def explain_hf_error(exc: BaseException) -> None:
    message = str(exc).lower()
    if any(marker in message for marker in ("401", "403", "gated", "unauthorized", "forbidden")):
        print(
            "Hugging Face access failed. Accept the Gemma model terms on the model page "
            "and authenticate with `huggingface-cli login` or an HF_TOKEN environment variable.",
            file=sys.stderr,
        )
    elif any(marker in message for marker in ("connection", "timed out", "incomplete", "xet")):
        print(
            "The model download was interrupted. Re-run the command; Hugging Face will resume "
            "from the cache in HF_HOME.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_hf_home()

    try:
        import torch
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Gemma 4 requires current torch, transformers (>=5.5), accelerate, "
            "huggingface-hub, safetensors, torchvision, and python-dotenv."
        ) from exc

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} (CUDA runtime {torch.version.cuda})")
    print(f"Transformers: {transformers.__version__}")
    print(f"Model: {args.model}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU total memory: {props.total_memory / 1024**3:.1f} GiB")
    elif not args.cpu:
        raise RuntimeError("CUDA is unavailable. Pass --cpu only if an explicit CPU fallback is intended.")
    else:
        print("WARNING: explicit CPU fallback enabled; loading may be extremely slow.")

    dtype = getattr(torch, args.dtype)
    load_kwargs: dict[str, Any] = {"device_map": "auto" if not args.cpu else "cpu"}
    transformers_major = int(transformers.__version__.split(".", 1)[0])
    load_kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = dtype

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
    except Exception as exc:
        explain_hf_error(exc)
        raise
    model.eval()
    print(f"Inferred device map: {getattr(model, 'hf_device_map', {'model': str(model.device)})}")
    memory_report(torch, "after load")

    messages = [
        {"role": "system", "content": "Answer using exactly one allowed option."},
        {"role": "user", "content": PROMPT},
    ]
    inputs = apply_text_chat_template(processor, messages)
    device = model_input_device(model, torch)
    inputs = inputs.to(device)
    tokenizer = tokenizer_for(processor)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits
    next_logits = logits[:, -1, :]
    next_probs = torch.softmax(next_logits.float(), dim=-1)
    vocab_size = len(tokenizer)
    print("\nFull-vocabulary logits:")
    print(f"  outputs.logits shape: {tuple(logits.shape)}")
    print(f"  next-token shape: {tuple(next_logits.shape)}")
    print(f"  vocabulary size: {vocab_size}")
    print(f"  dtype: {next_logits.dtype}; device: {next_logits.device}")
    print(f"  min/max: {next_logits.min().item():.6f} / {next_logits.max().item():.6f}")
    print(f"  all finite: {torch.isfinite(next_logits).all().item()}")
    assert logits.shape[0] == 1
    assert next_logits.shape[-1] == vocab_size
    assert torch.isfinite(next_logits).all() and torch.isfinite(next_probs).all()
    assert torch.allclose(next_probs.sum(), torch.tensor(1.0, device=device), atol=1e-5)

    k = min(args.top_k, vocab_size)
    top_probs, top_ids = torch.topk(next_probs[0], k)
    print(f"\nTop {k} next-token predictions:")
    for rank, (token_id, probability) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        token = tokenizer.decode([token_id])
        print(
            f"  {rank:>2} token_id={token_id:<7} token={token!r:<16} "
            f"logit={next_logits[0, token_id].item():.6f} probability={probability:.8f}"
        )
    memory_report(torch, "after forward")

    print("\nTokenizer choice inspection:")
    for choice in args.choices:
        for variant in (choice, " " + choice):
            print(f"  {variant!r}: {candidate_token_ids(tokenizer, variant)}")

    choice_ids = {choice: candidate_token_ids(tokenizer, choice) for choice in args.choices}
    if all(len(ids) == 1 for ids in choice_ids.values()):
        ids = torch.tensor([choice_ids[c][0] for c in args.choices], device=device)
        selected_logits = next_logits[0, ids]
        selected_probs = torch.softmax(selected_logits.float() / args.temperature, dim=0)
        assert torch.isfinite(selected_probs).all()
        assert torch.allclose(selected_probs.sum(), torch.tensor(1.0, device=device), atol=1e-5)
        print("\nSingle-token constrained distribution:")
        for choice, token_id, logit, probability in zip(
            args.choices, ids.tolist(), selected_logits.tolist(), selected_probs.tolist()
        ):
            print(
                f"  choice={choice!r} token_id={token_id} raw_logit={logit:.6f} "
                f"probability={probability:.8f}"
            )
    else:
        print("\nSingle-token scoring skipped: at least one choice contains multiple tokens.")

    sequence_scores = score_choice_sequences(model, processor, inputs, args.choices)
    sequence_probs = print_sequence_distribution(sequence_scores, args.temperature)
    winner = max(sequence_probs, key=sequence_probs.get)
    if "A" in args.choices and winner != "A":
        print(f"WARNING: expected diagnostic winner 'A', but sequence scoring selected {winner!r}.")

    if not args.no_generate:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        print(f"\nGenerated response: {processor.decode(new_tokens, skip_special_tokens=False)!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        explain_hf_error(error)
        raise
