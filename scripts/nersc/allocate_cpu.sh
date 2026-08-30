#!/usr/bin/env bash

# Allocate an interactive Perlmutter CPU shell. Run compute commands with srun.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
usage: allocate_cpu.sh --account PROJECT [--nodes 1-4] [--time HH:MM:SS]
                       [--immediate SECONDS] [--dry-run]

The QoS is intentionally fixed to interactive and the constraint to cpu.
EOF
}

account="${NERSC_CPU_ACCOUNT:-}"
nodes=1
walltime="01:00:00"
immediate=600
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) [[ $# -ge 2 ]] || nersc_die "--account requires a value"; account="$2"; shift 2 ;;
    --nodes) [[ $# -ge 2 ]] || nersc_die "--nodes requires a value"; nodes="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || nersc_die "--time requires a value"; walltime="$2"; shift 2 ;;
    --immediate) [[ $# -ge 2 ]] || nersc_die "--immediate requires a value"; immediate="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --qos|--qos=*|-q) nersc_die "QoS is fixed to interactive and cannot be overridden" ;;
    *) nersc_die "unknown argument: $1" ;;
  esac
done

[[ -n "${account}" ]] || nersc_die "--account or NERSC_CPU_ACCOUNT is required"
nersc_validate_account "${account}"
nersc_validate_nodes "${nodes}"
nersc_validate_walltime "${walltime}"
nersc_validate_immediate "${immediate}"
nersc_require_command salloc

command=(
  salloc
  --qos=interactive
  --constraint=cpu
  --account="${account}"
  --nodes="${nodes}"
  --time="${walltime}"
  --immediate="${immediate}"
  --job-name=mas-cc-interactive
)

if [[ "${dry_run}" == true ]]; then
  nersc_print_command "${command[@]}"
  exit 0
fi

exec "${command[@]}"
