#!/usr/bin/env bash

# Run one command on one interactive Perlmutter CPU node.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
usage: run_command.sh --account PROJECT [--time HH:MM:SS]
                      [--cpus-per-task 1-256] [--immediate SECONDS]
                      [--dry-run] -- COMMAND [ARG ...]

This creates a one-node allocation. For multi-node studies, use run_study.sh.
The QoS is intentionally fixed to interactive and the constraint to cpu.
EOF
}

account="${NERSC_CPU_ACCOUNT:-}"
walltime="00:10:00"
cpus_per_task="${NERSC_CPU_LOGICAL_CPUS}"
immediate=600
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) [[ $# -ge 2 ]] || nersc_die "--account requires a value"; account="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || nersc_die "--time requires a value"; walltime="$2"; shift 2 ;;
    --cpus-per-task) [[ $# -ge 2 ]] || nersc_die "--cpus-per-task requires a value"; cpus_per_task="$2"; shift 2 ;;
    --immediate) [[ $# -ge 2 ]] || nersc_die "--immediate requires a value"; immediate="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --qos|--qos=*|-q) nersc_die "QoS is fixed to interactive and cannot be overridden" ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) nersc_die "unknown argument before --: $1" ;;
  esac
done

[[ -n "${account}" ]] || nersc_die "--account or NERSC_CPU_ACCOUNT is required"
[[ $# -gt 0 ]] || nersc_die "a command is required after --"
nersc_validate_account "${account}"
nersc_validate_walltime "${walltime}"
nersc_validate_immediate "${immediate}"
nersc_validate_positive_integer "cpus per task" "${cpus_per_task}"
(( cpus_per_task <= NERSC_CPU_LOGICAL_CPUS )) || \
  nersc_die "cpus per task cannot exceed ${NERSC_CPU_LOGICAL_CPUS} logical CPUs"
command=(
  salloc
  --qos=interactive
  --constraint=cpu
  --account="${account}"
  --nodes=1
  --time="${walltime}"
  --immediate="${immediate}"
  --job-name=mas-cc-interactive
  srun
  --nodes=1
  --ntasks=1
  --cpus-per-task="${cpus_per_task}"
  --cpu-bind=cores
  "$@"
)

if [[ "${dry_run}" == true ]]; then
  nersc_print_command "${command[@]}"
  exit 0
fi

nersc_require_command salloc
nersc_require_command srun
exec "${command[@]}"
