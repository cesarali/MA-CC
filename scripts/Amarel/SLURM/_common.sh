#!/usr/bin/env bash

# Shared runtime checks for generic Rutgers Amarel MA-CC jobs.

readonly AMAREL_DEFAULT_SCRATCH_ROOT=/scratch/df630
readonly AMAREL_DEFAULT_MIN_SCRATCH_KB=1048576

if [[ -n "${AMAREL_REPO_ROOT:-}" ]]; then
  AMAREL_REPO_ROOT="$(cd -- "${AMAREL_REPO_ROOT}" && pwd -L)"
  AMAREL_SCRIPT_DIR="${AMAREL_REPO_ROOT}/scripts/Amarel/SLURM"
else
  AMAREL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
  AMAREL_REPO_ROOT="$(cd -- "${AMAREL_SCRIPT_DIR}/../../.." && pwd -L)"
fi
readonly AMAREL_SCRIPT_DIR AMAREL_REPO_ROOT

amarel_die() {
  echo "error: $*" >&2
  exit 2
}

amarel_resolve_conda() {
  if [[ -n "${AMAREL_CONDA_EXE:-}" ]]; then
    [[ -x "${AMAREL_CONDA_EXE}" ]] || \
      amarel_die "AMAREL_CONDA_EXE is not executable: ${AMAREL_CONDA_EXE}"
    printf '%s\n' "${AMAREL_CONDA_EXE}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  local candidate
  for candidate in \
    /home/df630/miniforge3/bin/conda \
    /home/df630/.local/share/miniforge3/bin/conda \
    /home/df630/miniconda3/bin/conda
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  amarel_die "Conda is unavailable; build the MA-CC environment from environment.yml"
}

amarel_prepare_runtime() {
  local scratch_root="${AMAREL_SCRATCH_ROOT:-${AMAREL_DEFAULT_SCRATCH_ROOT}}"
  local minimum_kb="${AMAREL_MIN_SCRATCH_KB:-${AMAREL_DEFAULT_MIN_SCRATCH_KB}}"
  [[ "${scratch_root}" == /scratch/df630 || "${scratch_root}" == /scratch/df630/* ]] || \
    amarel_die "Amarel caches must stay under /scratch/df630"
  [[ "${minimum_kb}" =~ ^[1-9][0-9]*$ ]] || \
    amarel_die "AMAREL_MIN_SCRATCH_KB must be a positive integer"
  local available_kb
  available_kb="$(df -Pk "${scratch_root}" | awk 'NR == 2 {print $4}')"
  [[ "${available_kb}" =~ ^[0-9]+$ ]] || \
    amarel_die "could not determine free space on ${scratch_root}"
  (( available_kb >= minimum_kb )) || \
    amarel_die "${scratch_root} has less than ${minimum_kb} KiB available"

  export HF_HOME="${scratch_root}/cache/huggingface"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export COMET_CACHE_DIR="${scratch_root}/cache/comet"
  export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
  export CONDA_ENVS_PATH="${AMAREL_CONDA_ENVS_PATH:-${scratch_root}/conda_envs}"
  export CONDA_PKGS_DIRS="${AMAREL_CONDA_PKGS_DIRS:-${scratch_root}/conda_pkgs}"
  mkdir -p \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${COMET_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${CONDA_ENVS_PATH}" \
    "${CONDA_PKGS_DIRS}"
  cd "${AMAREL_REPO_ROOT}"
}
