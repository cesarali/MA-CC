#!/usr/bin/env bash
# Regenerate tests/ci-deselect.txt from a full local run of the mas_cc suite.
# Run it on a machine WITHOUT the untracked study fixtures (a fresh clone) so
# the list captures exactly the fixture-dependent tests. Keeps the header.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m pytest -q --tb=no -rf -p no:cacheprovider tests/mas_cc > /tmp/ci-baseline.txt 2>&1 || true
header=$(sed -n '/^#/p' tests/ci-deselect.txt)
{ printf '%s\n' "$header"; grep -E '^FAILED ' /tmp/ci-baseline.txt | awk '{print $2}' | sort -u; } > tests/ci-deselect.txt
echo "wrote $(grep -cvE '^\s*(#|$)' tests/ci-deselect.txt) deselected tests"
