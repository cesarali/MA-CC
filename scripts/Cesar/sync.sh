#!/usr/bin/env bash
#
# Push this working tree to the Cesar cluster.
#
# Only the repository is synced. The Python environment on Cesar
# (~/envs/MA-CC, ~/pythons, ~/bin/uv) is NOT touched and must never be copied
# from a laptop: those are compiled linux-x86_64 binaries and wheels, and the
# laptop is macOS/arm64. That environment is built on the cluster with uv and
# only needs rebuilding when dependencies change.
#
# Usage:
#   scripts/Cesar/sync.sh            # refuse to sync while jobs are running
#   scripts/Cesar/sync.sh --force    # sync anyway (see the warning below)
#   scripts/Cesar/sync.sh --dry-run  # show what would change

set -euo pipefail

CESAR_HOST="${CESAR_HOST:-darius@slurm-login.tail769bd2.ts.net}"
CESAR_REPO="${CESAR_REPO:-/shared/MA-CC}"
SSH_OPTS=(-o ConnectTimeout=20 -o BatchMode=yes)

force=0
dry=()
for arg in "$@"; do
  case "$arg" in
    --force)   force=1 ;;
    --dry-run) dry=(--dry-run --itemize-changes) ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${repo_root}"

# .env is deliberately excluded. It is a credential, it must not be clobbered
# by a laptop copy, and rsync leaves an excluded file alone even under
# --delete. Place it on the cluster once, by hand, at ${CESAR_REPO}/.env
# with mode 600.
excludes=(
  --exclude '.git'
  --exclude '.env'
  --exclude '.venv'
  --exclude 'pdfs'
  --exclude 'notebooks'
  --exclude 'docs/papers'
  --exclude 'docs/reports'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '*.egg-info'
  --exclude '.DS_Store'
)

if ! ssh "${SSH_OPTS[@]}" "${CESAR_HOST}" true 2>/dev/null; then
  echo "error: cannot reach ${CESAR_HOST} (tailscale down?)" >&2
  exit 1
fi

# Workers import from ${CESAR_REPO}/src at runtime. Replacing files underneath
# a running array is a good way to get a half-old, half-new interpreter state
# and results nobody can explain later.
active="$(ssh "${SSH_OPTS[@]}" "${CESAR_HOST}" 'squeue -a -h -o "%T" 2>/dev/null | wc -l' | tr -d '[:space:]')"
if [[ "${active}" != "0" && ${force} -eq 0 && ${#dry[@]} -eq 0 ]]; then
  echo "refusing to sync: ${active} SLURM task(s) are active on Cesar." >&2
  echo "Workers import from ${CESAR_REPO}/src; changing it mid-run corrupts the run." >&2
  echo "Wait for the queue to drain, or pass --force if you know the change is inert." >&2
  exit 1
fi

# Reproducibility note: Cesar has no git binary, so studies submitted there
# record an empty git_commit. Anything uncommitted here is therefore
# unreproducible from version control once it has run.
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "warning: working tree is dirty; Cesar records no git commit for runs." >&2
  git status --short | sed 's/^/  /' >&2
fi
sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

rsync -az --delete "${dry[@]}" "${excludes[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${repo_root}/" "${CESAR_HOST}:${CESAR_REPO}/"

if [[ ${#dry[@]} -eq 0 ]]; then
  # Leave a breadcrumb the cluster cannot derive on its own.
  ssh "${SSH_OPTS[@]}" "${CESAR_HOST}" \
    "printf '%s\n' '${sha}' > ${CESAR_REPO}/SYNCED_FROM_GIT_SHA"
  echo "synced ${repo_root} -> ${CESAR_HOST}:${CESAR_REPO} (git ${sha:0:12})"
fi
