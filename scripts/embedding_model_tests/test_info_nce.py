"""Train small projection heads with a symmetric InfoNCE objective."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from _common import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    encode_normalized,
    load_frozen_encoder,
    set_deterministic_seed,
)


MATCHED_PAIRS = (
    (
        "The scout communicates that the brass key is beneath the bridge.",
        "The group searches beneath the bridge, finds the key, and reaches the exit.",
    ),
    (
        "An agent sends fabricated evidence claiming that the answer is red.",
        "The group trusts the false evidence and selects the wrong red answer.",
    ),
    (
        "The coordinator assigns one distinct room to each searcher.",
        "Every room is checked without duplication and the missing map is recovered.",
    ),
    (
        "A silent agent withholds its warning about the unstable path.",
        "The unaware group takes the unstable path and fails the crossing.",
    ),
    (
        "Two agents reconcile their maps and broadcast the verified north route.",
        "The whole team follows the north route and arrives at the rendezvous.",
    ),
    (
        "The leader neglects to distribute the limited protective equipment.",
        "Several unprotected agents cannot complete the hazardous mission.",
    ),
    (
        "A member asks the team to verify an uncertain code before submitting it.",
        "The verification catches an error and the team submits the correct code.",
    ),
    (
        "Agents talk over one another and never establish a shared plan.",
        "Their conflicting actions consume the remaining time and the task fails.",
    ),
)


class ProjectionMLP(nn.Module):
    """A small trainable projection head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(inputs), dim=-1)


def pairwise_similarity(
    phi: nn.Module,
    psi: nn.Module,
    contexts: torch.Tensor,
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Return the full cosine-similarity matrix between projected batches."""

    return phi(contexts) @ psi(outcomes).T


def symmetric_info_nce(similarities: torch.Tensor, temperature: float) -> torch.Tensor:
    """Use diagonal matches as positives in both retrieval directions."""

    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("InfoNCE expects a square pairwise-similarity matrix.")
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    logits = similarities / temperature
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )


def assert_finite_gradients(modules: Sequence[nn.Module]) -> None:
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"No gradient was produced for {name}.")
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Non-finite gradient detected for {name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit two projection heads with a synthetic symmetric InfoNCE loss."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"checkpoint to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")

    set_deterministic_seed(args.seed)
    torch.set_printoptions(precision=4, sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_frozen_encoder(args.model, device)

    context_texts = [context for context, _ in MATCHED_PAIRS]
    outcome_texts = [outcome for _, outcome in MATCHED_PAIRS]
    contexts = encode_normalized(
        encoder, context_texts, args.model, device, e5_role="query"
    )
    outcomes = encode_normalized(
        encoder, outcome_texts, args.model, device, e5_role="passage"
    )
    expected_shape = (len(MATCHED_PAIRS), contexts.shape[1])
    if contexts.shape != expected_shape or outcomes.shape != expected_shape:
        raise AssertionError(
            f"Unexpected encoder shapes: contexts={contexts.shape}, outcomes={outcomes.shape}."
        )
    if not torch.isfinite(contexts).all() or not torch.isfinite(outcomes).all():
        raise AssertionError("Encoder outputs contain non-finite values.")

    projection_dim = min(128, contexts.shape[1])
    hidden_dim = min(256, contexts.shape[1])
    phi = ProjectionMLP(contexts.shape[1], hidden_dim, projection_dim).to(device)
    psi = ProjectionMLP(outcomes.shape[1], hidden_dim, projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        [*phi.parameters(), *psi.parameters()], lr=args.learning_rate
    )

    with torch.no_grad():
        similarities_before = pairwise_similarity(phi, psi, contexts, outcomes)
        initial_loss = symmetric_info_nce(similarities_before, args.temperature)
    if similarities_before.shape != (len(MATCHED_PAIRS), len(MATCHED_PAIRS)):
        raise AssertionError("The initial pairwise-similarity matrix has the wrong shape.")
    if not torch.isfinite(initial_loss):
        raise AssertionError("The initial loss is not finite.")

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        similarities = pairwise_similarity(phi, psi, contexts, outcomes)
        loss = symmetric_info_nce(similarities, args.temperature)
        if not torch.isfinite(loss):
            raise AssertionError("A non-finite loss was encountered during training.")
        loss.backward()
        assert_finite_gradients((phi, psi))
        optimizer.step()

    with torch.no_grad():
        similarities_after = pairwise_similarity(phi, psi, contexts, outcomes)
        final_loss = symmetric_info_nce(similarities_after, args.temperature)
    if similarities_after.shape != similarities_before.shape:
        raise AssertionError("The final pairwise-similarity matrix has the wrong shape.")
    if not torch.isfinite(similarities_after).all() or not torch.isfinite(final_loss):
        raise AssertionError("Final similarities or loss contain non-finite values.")
    tolerance = 1e-5
    if final_loss.item() > initial_loss.item() - tolerance:
        raise AssertionError(
            f"InfoNCE did not decrease: initial={initial_loss.item():.6f}, "
            f"final={final_loss.item():.6f}."
        )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Initial loss: {initial_loss.item():.6f}")
    print(f"Final loss: {final_loss.item():.6f}")
    print("Similarity matrix before training:")
    print(similarities_before.cpu())
    print("Similarity matrix after training:")
    print(similarities_after.cpu())


if __name__ == "__main__":
    main()
