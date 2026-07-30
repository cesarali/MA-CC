"""Shared helpers for the embedding-model smoke tests."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Hugging Face reads its cache variables while its libraries are imported.
# Load the repository configuration first, while preserving variables exported
# by a cluster job or interactive shell.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPOSITORY_ROOT / ".env", override=False)


SUPPORTED_MODELS = (
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_MODEL = SUPPORTED_MODELS[0]
E5_TASK = "Represent this agent trajectory for behavioral analysis."


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable smoke tests."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_texts(
    texts: Sequence[str], model_name: str, *, e5_role: str = "query"
) -> list[str]:
    """Add E5's required retrieval-style prefix when appropriate."""

    if model_name.startswith("intfloat/e5-"):
        if e5_role == "query":
            return [f"query: {E5_TASK} {text}" for text in texts]
        if e5_role == "passage":
            return [f"passage: {text}" for text in texts]
        raise ValueError(f"Unsupported E5 role: {e5_role!r}")
    return list(texts)


def load_frozen_encoder(model_name: str, device: torch.device) -> SentenceTransformer:
    """Download/load a sentence transformer and freeze all of its parameters."""

    # Keep this import after the repository .env has been loaded so HF_HOME is
    # honored even when it is configured only in that file.
    from sentence_transformers import SentenceTransformer

    try:
        encoder = SentenceTransformer(model_name, device=str(device))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_name!r}. On first use, verify that Hugging Face "
            "is reachable and that its cache directory is writable."
        ) from exc

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def encode_normalized(
    encoder: SentenceTransformer,
    texts: Sequence[str],
    model_name: str,
    device: torch.device,
    *,
    e5_role: str = "query",
) -> torch.Tensor:
    """Encode text into detached, L2-normalized tensors on ``device``."""

    prepared = prepare_texts(texts, model_name, e5_role=e5_role)
    with torch.no_grad():
        embeddings = encoder.encode(
            prepared,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    # Some sentence-transformers versions internally use inference mode.
    # Cloning outside that context makes a normal, detached tensor that a
    # projection layer may save while computing its own weight gradients.
    return embeddings.detach().to(device).clone()
