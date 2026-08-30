#!/usr/bin/env bash

# Shared, policy-enforcing helpers for NERSC Perlmutter launchers.

readonly NERSC_MAX_INTERACTIVE_NODES=4
readonly NERSC_MAX_INTERACTIVE_SECONDS=14400
readonly NERSC_CPU_PHYSICAL_CORES=128
readonly NERSC_CPU_LOGICAL_CPUS=256
readonly NERSC_PYTHON_MODULE="python/3.11-24.1.0"
readonly NERSC_CONDA_ENV="MA-CC"
readonly NERSC_CONDA_ENV_PATH="/pscratch/sd/d/dfarough/conda_envs/MA-CC"
readonly NERSC_CONDA_PKGS="/pscratch/sd/d/dfarough/conda_pkgs"

NERSC_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly NERSC_SCRIPT_DIR
NERSC_REPO_ROOT="$(cd -- "${NERSC_SCRIPT_DIR}/../.." && pwd)"
readonly NERSC_REPO_ROOT

nersc_die() {
  echo "error: $*" >&2
  exit 2
}

nersc_require_command() {
  command -v "$1" >/dev/null 2>&1 || nersc_die "required command is unavailable: $1"
}

nersc_validate_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || nersc_die "${label} must be a positive integer"
}

nersc_validate_nodes() {
  nersc_validate_positive_integer "nodes" "$1"
  (( $1 <= NERSC_MAX_INTERACTIVE_NODES )) || \
    nersc_die "interactive QoS permits at most ${NERSC_MAX_INTERACTIVE_NODES} nodes"
}

nersc_validate_account() {
  [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || nersc_die "invalid NERSC account: $1"
}

nersc_walltime_seconds() {
  local walltime="$1"
  [[ "${walltime}" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]] || \
    nersc_die "wall time must use HH:MM:SS"
  local hours=$((10#${BASH_REMATCH[1]}))
  local minutes=$((10#${BASH_REMATCH[2]}))
  local seconds=$((10#${BASH_REMATCH[3]}))
  (( minutes < 60 && seconds < 60 )) || nersc_die "invalid wall time: ${walltime}"
  echo $((hours * 3600 + minutes * 60 + seconds))
}

nersc_validate_walltime() {
  local seconds
  seconds="$(nersc_walltime_seconds "$1")"
  (( seconds > 0 )) || nersc_die "wall time must be positive"
  (( seconds <= NERSC_MAX_INTERACTIVE_SECONDS )) || \
    nersc_die "interactive QoS wall time cannot exceed 04:00:00"
}

nersc_validate_immediate() {
  nersc_validate_positive_integer "immediate wait" "$1"
}

nersc_validate_results_dir() {
  local destination="$1"
  local allowed_root="${NERSC_RESULTS_ROOT:-/pscratch/sd/d/dfarough/MA-CC-results}"
  [[ "${destination}" == "${allowed_root}" || "${destination}" == "${allowed_root}/"* ]] || \
    nersc_die "results must be under ${allowed_root}, got ${destination}"
  [[ "${destination}" != "${NERSC_REPO_ROOT}" && "${destination}" != "${NERSC_REPO_ROOT}/"* ]] || \
    nersc_die "results must not be written inside the source repository"
}

nersc_load_ma_cc_environment() {
  # The module command is a shell function supplied by NERSC's login shell.
  type module >/dev/null 2>&1 || nersc_die "NERSC module command is unavailable"
  module load "${NERSC_PYTHON_MODULE}"
  export CONDA_PKGS_DIRS="${NERSC_CONDA_PKGS}"
  nersc_require_command conda
  [[ "$(readlink -f "${NERSC_CONDA_ENV_PATH}")" == "${NERSC_CONDA_ENV_PATH}" ]] || \
    nersc_die "MA-CC environment does not resolve to ${NERSC_CONDA_ENV_PATH}"
}

nersc_require_interactive_cpu_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || nersc_die "this command must run inside a SLURM allocation"
  [[ "${SLURM_JOB_QOS:-}" == "interactive" ]] || \
    nersc_die "refusing non-interactive NERSC QoS: ${SLURM_JOB_QOS:-unset}"
  local constraints="${SLURM_JOB_CONSTRAINTS:-}"
  if [[ "${constraints}" != *cpu* ]]; then
    nersc_require_command scontrol
    local job_record
    job_record="$(scontrol show job "${SLURM_JOB_ID}" --oneliner)" || \
      nersc_die "could not inspect SLURM allocation ${SLURM_JOB_ID}"
    [[ "${job_record}" =~ (Features|Constraints)=[^[:space:]]*cpu ]] || \
      nersc_die "refusing allocation without the Perlmutter cpu constraint"
  fi
}

nersc_print_command() {
  printf '%q ' "$@"
  printf '\n'
}
