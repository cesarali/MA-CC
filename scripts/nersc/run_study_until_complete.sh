#!/usr/bin/env bash

# Resume a prepared study through successive four-hour interactive allocations.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
usage: run_study_until_complete.sh --account PROJECT --nodes 1-4 --study-dir DIR
                                   [--wait-for-job JOB_ID] [--immediate SECONDS]
                                   [--retry-delay SECONDS] [--aggregate]

The supervisor requests only qos=interactive CPU allocations. It resumes after
each four-hour rollover, stops on a recorded scientific failure, and optionally
retries strict aggregation allocations until its manifest and ZIP are complete.
EOF
}

account="${NERSC_CPU_ACCOUNT:-}"
nodes=""
study_dir=""
wait_for_job=""
immediate=600
retry_delay=30
aggregate=false
ready_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) [[ $# -ge 2 ]] || nersc_die "--account requires a value"; account="$2"; shift 2 ;;
    --nodes) [[ $# -ge 2 ]] || nersc_die "--nodes requires a value"; nodes="$2"; shift 2 ;;
    --study-dir) [[ $# -ge 2 ]] || nersc_die "--study-dir requires a value"; study_dir="$2"; shift 2 ;;
    --wait-for-job) [[ $# -ge 2 ]] || nersc_die "--wait-for-job requires a value"; wait_for_job="$2"; shift 2 ;;
    --immediate) [[ $# -ge 2 ]] || nersc_die "--immediate requires a value"; immediate="$2"; shift 2 ;;
    --retry-delay) [[ $# -ge 2 ]] || nersc_die "--retry-delay requires a value"; retry_delay="$2"; shift 2 ;;
    --aggregate) aggregate=true; shift ;;
    --ready-file) [[ $# -ge 2 ]] || nersc_die "--ready-file requires a value"; ready_file="$2"; shift 2 ;;
    --qos|--qos=*|-q) nersc_die "QoS is fixed to interactive and cannot be overridden" ;;
    --help|-h) usage; exit 0 ;;
    *) nersc_die "unknown argument: $1" ;;
  esac
done

[[ -n "${account}" ]] || nersc_die "--account or NERSC_CPU_ACCOUNT is required"
[[ -n "${nodes}" ]] || nersc_die "--nodes is required"
[[ -n "${study_dir}" ]] || nersc_die "--study-dir is required"
nersc_validate_account "${account}"
nersc_validate_nodes "${nodes}"
nersc_validate_immediate "${immediate}"
[[ "${retry_delay}" =~ ^[0-9]+$ ]] || nersc_die "--retry-delay must be a non-negative integer"
if [[ -n "${wait_for_job}" && ! "${wait_for_job}" =~ ^[0-9]+$ ]]; then
  nersc_die "--wait-for-job must be a numeric SLURM job id"
fi
study_dir="$(realpath -m "${study_dir}")"
nersc_validate_results_dir "${study_dir}"
[[ -f "${study_dir}/execution_manifest.csv" ]] || \
  nersc_die "prepared execution manifest is missing: ${study_dir}"
nersc_require_command flock
nersc_require_command squeue
nersc_load_ma_cc_environment

mkdir -p "${study_dir}/runtime"
exec 9>"${study_dir}/runtime/nersc-rollover.lock"
flock -n 9 || nersc_die "another rollover supervisor already owns this study"
if [[ -n "${ready_file}" ]]; then
  ready_file="$(realpath -m "${ready_file}")"
  [[ "${ready_file}" == "${study_dir}/runtime/nersc-rollover.ready" ]] || \
    nersc_die "ready file must be the study runtime readiness marker"
  temporary_ready="${ready_file}.$$.tmp"
  printf '%s\n' "$$" >"${temporary_ready}"
  mv "${temporary_ready}" "${ready_file}"
fi

# The detached launcher may be invoked from an environment that carries stale
# Slurm metadata. Never inherit it into a fresh interactive allocation request.
inherited_slurm_variables=("${!SLURM_@}")
if (( ${#inherited_slurm_variables[@]} )); then
  unset "${inherited_slurm_variables[@]}"
fi

if [[ -n "${wait_for_job}" ]]; then
  echo "[nersc-rollover] waiting for allocation ${wait_for_job} to leave the queue"
  while squeue --noheader --jobs "${wait_for_job}" | grep -q .; do
    sleep 30
  done
fi

progress() {
  "${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/study_progress.py" \
    --study-dir "${study_dir}" --format tsv
}

allocation=0
while true; do
  IFS=$'\t' read -r expected sealed episodes failed in_progress complete <<<"$(progress)"
  echo "[nersc-rollover] cells=${sealed}/${expected} episodes=${episodes} failed=${failed} in_progress=${in_progress}"
  (( failed == 0 )) || nersc_die "recorded failed episodes require inspection; refusing automatic resubmission"
  if [[ "${complete}" == true ]]; then
    break
  fi

  allocation=$((allocation + 1))
  echo "[nersc-rollover] requesting interactive CPU allocation ${allocation}"
  set +e
  "${NERSC_SCRIPT_DIR}/run_study.sh" \
    --account "${account}" \
    --nodes "${nodes}" \
    --time 04:00:00 \
    --immediate "${immediate}" \
    --study-dir "${study_dir}"
  allocation_status=$?
  set -e

  IFS=$'\t' read -r expected sealed episodes failed in_progress complete <<<"$(progress)"
  echo "[nersc-rollover] allocation=${allocation} exit=${allocation_status} cells=${sealed}/${expected} episodes=${episodes} failed=${failed}"
  (( failed == 0 )) || nersc_die "recorded failed episodes require inspection; refusing automatic resubmission"
  if [[ "${complete}" != true ]]; then
    echo "[nersc-rollover] clean incomplete state; requesting the next interactive allocation"
    if (( retry_delay > 0 )); then
      sleep "${retry_delay}"
    fi
  fi
done

echo "[nersc-rollover] all ${expected} cell seals are complete"
if [[ "${aggregate}" == true ]]; then
  aggregation_attempt=0
  while true; do
    IFS=$'\t' read -r aggregation_status aggregation_detail <<<"$(
      "${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/aggregation_progress.py" \
        --study-dir "${study_dir}" --format tsv
    )"
    if [[ "${aggregation_status}" == complete ]]; then
      echo "[nersc-rollover] strict aggregation complete: ${aggregation_detail}"
      break
    fi
    [[ "${aggregation_status}" != failed ]] || \
      nersc_die "strict aggregation requires inspection: ${aggregation_detail}"

    aggregation_attempt=$((aggregation_attempt + 1))
    echo "[nersc-rollover] requesting interactive CPU allocation for strict aggregation attempt ${aggregation_attempt}"
    set +e
    "${NERSC_SCRIPT_DIR}/run_command.sh" \
      --account "${account}" \
      --time 04:00:00 \
      --immediate "${immediate}" \
      -- "${NERSC_SCRIPT_DIR}/aggregate_study.sh" --study-dir "${study_dir}"
    aggregation_exit=$?
    set -e

    IFS=$'\t' read -r aggregation_status aggregation_detail <<<"$(
      "${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/aggregation_progress.py" \
        --study-dir "${study_dir}" --format tsv
    )"
    echo "[nersc-rollover] aggregation attempt=${aggregation_attempt} exit=${aggregation_exit} status=${aggregation_status}"
    if [[ "${aggregation_status}" == complete ]]; then
      echo "[nersc-rollover] strict aggregation complete: ${aggregation_detail}"
      break
    fi
    [[ "${aggregation_status}" != failed ]] || \
      nersc_die "strict aggregation requires inspection: ${aggregation_detail}"
    echo "[nersc-rollover] aggregation remains clean and incomplete; retrying in a fresh interactive allocation"
    if (( retry_delay > 0 )); then
      sleep "${retry_delay}"
    fi
  done
fi
