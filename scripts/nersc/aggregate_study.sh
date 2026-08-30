#!/usr/bin/env bash

# Strictly aggregate one completed study inside an interactive CPU allocation.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  echo "usage: aggregate_study.sh --study-dir DIR" >&2
}

study_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --study-dir) [[ $# -ge 2 ]] || nersc_die "--study-dir requires a value"; study_dir="$2"; shift 2 ;;
    --qos|--qos=*|-q) nersc_die "QoS is fixed to interactive and cannot be overridden" ;;
    --help|-h) usage; exit 0 ;;
    *) nersc_die "unknown argument: $1" ;;
  esac
done

[[ -n "${study_dir}" ]] || nersc_die "--study-dir is required"
study_dir="$(realpath -m "${study_dir}")"
nersc_validate_results_dir "${study_dir}"
[[ -f "${study_dir}/execution_manifest.csv" ]] || \
  nersc_die "prepared execution manifest is missing: ${study_dir}"

nersc_require_interactive_cpu_allocation
nersc_load_ma_cc_environment
cd "${NERSC_REPO_ROOT}"
exec conda run --no-capture-output -p "${NERSC_CONDA_ENV_PATH}" \
  mas-cc study aggregate --study-dir "${study_dir}"
