#!/usr/bin/env bash
# Daemonless image build for the hosted Blackboard Observatory: one application layer appended to a
# digest-pinned python base with crane - no Docker daemon, no Dockerfile (estate rule: daemonless builds).
#
#   deploy/observatory/build-image.sh <registry/repo> <tag> [--insecure]
#
# Prints the final digest. Requires: python3.12 (same minor as the base image; wheels are installed on the
# host and must match), pip, crane, and registry credentials already configured for crane.
set -euo pipefail

REPO=${1:?usage: build-image.sh <registry/repo> <tag> [--insecure]}
TAG=${2:?tag}
INSECURE=${3:-}
BASE=docker.io/library/python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "build host needs python 3.12 to match the base image"'
SITE=$WORK/layer/opt/observatory/site-packages
mkdir -p "$SITE"
python3 -m pip install --quiet --no-compile --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
  --python-version 3.12 --implementation cp --target "$SITE" -r "$HERE/requirements.txt"
python3 -m pip install --quiet --no-compile --no-deps --target "$SITE" "$ROOT"
find "$SITE" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Reproducible layer: sorted names, fixed mtime, root-owned, world-readable.
tar --sort=name --mtime='2026-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' \
  -C "$WORK/layer" -cf "$WORK/layer.tar" opt

crane append ${INSECURE:+--insecure} --base "$BASE" --new_layer "$WORK/layer.tar" --new_tag "$REPO:$TAG" >/dev/null
crane mutate ${INSECURE:+--insecure} "$REPO:$TAG" --tag "$REPO:$TAG" \
  --env PYTHONPATH=/opt/observatory/site-packages --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp --env MA_CC_DASHBOARD_CACHE=/cache \
  --user 1000:1000 --workdir /tmp \
  --entrypoint python3,-m,mas_cc.blackboard_dashboard >/dev/null
crane digest ${INSECURE:+--insecure} "$REPO:$TAG"
