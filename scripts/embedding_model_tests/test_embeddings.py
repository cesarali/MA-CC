"""Smoke-test loading supported sentence transformers and producing embeddings."""

from __future__ import annotations

import argparse

import torch

from _common import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    encode_normalized,
    load_frozen_encoder,
    set_deterministic_seed,
)


TRAJECTORIES = (
    "Agent A tells the group that the brass key is under the bridge.",
    "The team checks the bridge, finds the key, and opens the shared exit.",
    "Agent C withholds its evidence, so the group searches unrelated rooms.",
    "Conflicting messages prevent the agents from agreeing on a final answer.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load sentence-transformer checkpoints and validate their embeddings."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"checkpoint to test (default: {DEFAULT_MODEL})",
    )
    selection.add_argument(
        "--all", action="store_true", help="test every supported checkpoint"
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run_model_test(model_name: str, device: torch.device) -> None:
    encoder = load_frozen_encoder(model_name, device)
    embeddings = encode_normalized(
        encoder, TRAJECTORIES, model_name, device, e5_role="query"
    )

    if embeddings.ndim != 2:
        raise AssertionError(f"Expected a 2D embedding tensor, got {embeddings.shape}.")
    if embeddings.shape[0] != len(TRAJECTORIES):
        raise AssertionError("The number of embeddings does not match the inputs.")
    if not torch.isfinite(embeddings).all():
        raise AssertionError("The embeddings contain NaN or infinite values.")

    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise AssertionError(f"Embeddings are not unit-normalized; norms={norms}.")

    cosine_similarity = embeddings @ embeddings.T
    if not torch.isfinite(cosine_similarity).all():
        raise AssertionError("The cosine-similarity matrix contains invalid values.")

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Embedding tensor shape: {tuple(embeddings.shape)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("Cosine-similarity matrix:")
    print(cosine_similarity.cpu())


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)
    torch.set_printoptions(precision=4, sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = SUPPORTED_MODELS if args.all else (args.model,)
    for index, model_name in enumerate(models):
        if index:
            print()
        run_model_test(model_name, device)


if __name__ == "__main__":
    main()
