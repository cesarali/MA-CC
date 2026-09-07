"""Verify the isolated OSS transfer bundle without provider access."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "results/studies/musr_truthful_selective_isolated_oss_01"
    )
    manifest = json.loads(
        (root / "preparation/checksum_manifest.json").read_text(encoding="utf-8")
    )
    failures = []
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"hash mismatch: {relative}")
    rows = sum(
        1
        for line in (root / "preparation/evaluation_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if rows != 10_818:
        failures.append(f"manifest rows: {rows} != 10818")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"verified {len(manifest)} files and {rows} requests under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
