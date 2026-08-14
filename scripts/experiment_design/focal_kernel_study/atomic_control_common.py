"""Shared, dependency-light utilities for atomic-control calibration tooling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


BUCKETS: tuple[str, ...] = (
    "bucket_01_anonymous",
    "bucket_02_persistent_identity",
    "bucket_03_positive_reputation",
    "bucket_04_negative_reputation",
    "bucket_05_social_reputation",
    "bucket_06_strategic_uncertainty",
)

BUCKET_LABELS: dict[str, str] = {
    "bucket_01_anonymous": "Anonymous",
    "bucket_02_persistent_identity": "Identity",
    "bucket_03_positive_reputation": "+Reputation",
    "bucket_04_negative_reputation": "-Reputation",
    "bucket_05_social_reputation": "Social reputation",
    "bucket_06_strategic_uncertainty": "Strategic uncertainty",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shard(bucket: str, state_id: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    value = hashlib.sha256(f"{bucket}:{state_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") % num_shards


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_write_text(path, payload)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def verify_frozen_dataset(input_dir: Path) -> dict[str, Any]:
    """Verify both the canonical manifest digest and every frozen prompt digest."""

    dataset_path = input_dir / "DATASET_MANIFEST.json"
    prompt_manifest_path = input_dir / "PROMPT_MANIFEST.jsonl"
    if not dataset_path.is_file() or not prompt_manifest_path.is_file():
        raise ValueError(f"{input_dir} is not a frozen atomic-control dataset")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    prompt_manifest_bytes = prompt_manifest_path.read_bytes()
    observed = sha256_bytes(prompt_manifest_bytes)
    expected = dataset.get("dataset_hash")
    if observed != expected:
        raise ValueError(f"dataset hash mismatch: expected {expected}, observed {observed}")
    rows = read_jsonl(prompt_manifest_path)
    if len(rows) != dataset.get("number_of_prompts"):
        raise ValueError("PROMPT_MANIFEST row count does not match DATASET_MANIFEST")
    bucket_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket in BUCKETS:
        for metadata in read_jsonl(input_dir / bucket / "manifest.jsonl"):
            bucket_metadata[(bucket, str(metadata["state_id"]))] = metadata
    for row in rows:
        prompt_path = input_dir / str(row["prompt_path"])
        if not prompt_path.is_file():
            raise ValueError(f"frozen prompt is missing: {prompt_path}")
        if sha256_file(prompt_path) != row["prompt_sha256"]:
            raise ValueError(f"frozen prompt was modified: {prompt_path}")
        key = (str(row["bucket"]), str(row["state_id"]))
        metadata = bucket_metadata.get(key)
        if metadata is None:
            raise ValueError(f"frozen manifest metadata is missing: {key}")
        metadata_hash = sha256_bytes(canonical_json(metadata).encode("utf-8"))
        if metadata_hash != row["metadata_sha256"]:
            raise ValueError(f"frozen manifest metadata was modified: {key}")
    return dataset
