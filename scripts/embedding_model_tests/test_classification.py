"""Classify synthetic agent trajectories with frozen sentence embeddings."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from _common import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    encode_normalized,
    load_frozen_encoder,
    set_deterministic_seed,
)


LABELED_TRAJECTORIES = (
    ("The scout shares the correct key location and everyone reaches the exit.", 1),
    ("Agents compare clues, agree on the blue door, and escape together.", 1),
    ("The team divides the search efficiently and recovers every missing tool.", 1),
    ("A clear warning lets all agents avoid the trap and complete the mission.", 1),
    ("Each agent reports evidence honestly, producing the correct group answer.", 1),
    ("The leader coordinates roles and the group finishes before the deadline.", 1),
    ("Two agents resolve their disagreement and jointly identify the safe route.", 1),
    ("The messenger relays the map accurately, allowing a successful rendezvous.", 1),
    ("The group verifies uncertain claims and unanimously selects the right key.", 1),
    ("Agents share supplies fairly and all members survive the final crossing.", 1),
    ("A concise status update synchronizes the team and secures the objective.", 1),
    ("The committee combines independent observations into a correct decision.", 1),
    ("The scout hides the key location and the group never finds the exit.", 0),
    ("Agents ignore conflicting clues and choose the locked red door.", 0),
    ("The team duplicates its search effort and leaves vital tools behind.", 0),
    ("A false warning sends everyone into the trap and the mission fails.", 0),
    ("Several agents invent evidence, producing the wrong group answer.", 0),
    ("Nobody coordinates roles, so the group misses the deadline.", 0),
    ("The agents continue arguing and take an unsafe route.", 0),
    ("The messenger corrupts the map and the agents fail to rendezvous.", 0),
    ("The group accepts an unchecked rumor and unanimously selects the wrong key.", 0),
    ("Agents hoard supplies and multiple members fail the final crossing.", 0),
    ("Missing status updates leave the team unsynchronized and defeated.", 0),
    ("The committee discards useful observations and reaches an incorrect decision.", 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic classification smoke test on frozen embeddings."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"checkpoint to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")

    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [text for text, _ in LABELED_TRAJECTORIES]
    labels = np.asarray([label for _, label in LABELED_TRAJECTORIES], dtype=np.int64)

    encoder = load_frozen_encoder(args.model, device)
    embeddings = encode_normalized(encoder, texts, args.model, device).cpu().numpy()
    if embeddings.shape[0] != labels.shape[0] or not np.isfinite(embeddings).all():
        raise AssertionError("Embedding output is incomplete or contains invalid values.")

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    classifier = LogisticRegression(max_iter=1_000, random_state=args.seed)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    if predictions.shape != y_test.shape:
        raise AssertionError(
            f"Prediction shape {predictions.shape} does not match {y_test.shape}."
        )
    if not np.isfinite(predictions).all() or not set(np.unique(predictions)) <= {0, 1}:
        raise AssertionError("Predictions contain invalid values.")

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Training size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["unsuccessful/uncoordinated", "successful/coordinated"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
