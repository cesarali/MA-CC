#!/usr/bin/env bash

# Start or ensure one detached rollover supervisor for a prepared NERSC study.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
usage: start_study_supervisor.sh --account PROJECT --nodes 1-4 --study-dir DIR
                                 [--wait-for-job JOB_ID] [--immediate SECONDS]
                                 [--retry-delay SECONDS] [--aggregate]
                                 [--ensure] [--dry-run]

Starts run_study_until_complete.sh under nohup+setsid. --ensure is idempotent:
it exits successfully when the requested study/aggregation is complete or a
supervisor owns its rollover lock, making this suitable for a watchdog.
All compute and aggregation allocations remain qos=interactive CPU jobs.
EOF
}

account="${NERSC_CPU_ACCOUNT:-}"
nodes=""
study_dir=""
wait_for_job=""
immediate=600
retry_delay=30
aggregate=false
ensure=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) [[ $# -ge 2 ]] || nersc_die "--account requires a value"; account="$2"; shift 2 ;;
    --nodes) [[ $# -ge 2 ]] || nersc_die "--nodes requires a value"; nodes="$2"; shift 2 ;;
    --study-dir) [[ $# -ge 2 ]] || nersc_die "--study-dir requires a value"; study_dir="$2"; shift 2 ;;
    --wait-for-job) [[ $# -ge 2 ]] || nersc_die "--wait-for-job requires a value"; wait_for_job="$2"; shift 2 ;;
    --immediate) [[ $# -ge 2 ]] || nersc_die "--immediate requires a value"; immediate="$2"; shift 2 ;;
    --retry-delay) [[ $# -ge 2 ]] || nersc_die "--retry-delay requires a value"; retry_delay="$2"; shift 2 ;;
    --aggregate) aggregate=true; shift ;;
    --ensure) ensure=true; shift ;;
    --dry-run) dry_run=true; shift ;;
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

"${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/study_plan.py" \
  --study-dir "${study_dir}" --nodes "${nodes}" >/dev/null

runtime_dir="${study_dir}/runtime"
mkdir -p "${runtime_dir}"
rollover_lock="${runtime_dir}/nersc-rollover.lock"
start_lock="${runtime_dir}/nersc-rollover-start.lock"
pid_file="${runtime_dir}/nersc-rollover.pid"
log_file="${runtime_dir}/nersc-rollover.log"
ready_file="${runtime_dir}/nersc-rollover.ready"

nersc_require_command flock
exec 8>"${start_lock}"
if [[ "${ensure}" == true ]]; then
  flock 8
else
  flock -n 8 || nersc_die "another process is starting this study supervisor"
fi

if ! flock -n "${rollover_lock}" -c true; then
  if [[ "${ensure}" == true ]]; then
    echo "[nersc-rollover] supervisor already active for ${study_dir}"
    exit 0
  fi
  nersc_die "a rollover supervisor already owns this study"
fi

IFS=$'\t' read -r expected sealed episodes failed in_progress complete <<<"$(
  "${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/study_progress.py" \
    --study-dir "${study_dir}" --format tsv
)"
if [[ "${complete}" == true ]]; then
  if [[ "${aggregate}" != true ]]; then
    echo "[nersc-rollover] study already complete: cells=${sealed}/${expected}"
    exit 0
  fi
  IFS=$'\t' read -r aggregation_status aggregation_detail <<<"$(
    "${NERSC_CONDA_ENV_PATH}/bin/python" "${NERSC_SCRIPT_DIR}/aggregation_progress.py" \
      --study-dir "${study_dir}" --format tsv
  )"
  if [[ "${aggregation_status}" == complete ]]; then
    echo "[nersc-rollover] study already complete and aggregated: ${aggregation_detail}"
    exit 0
  fi
  [[ "${aggregation_status}" != failed ]] || \
    nersc_die "strict aggregation requires inspection: ${aggregation_detail}"
fi
(( failed == 0 )) || nersc_die "recorded failed episodes require inspection"

supervisor=(
  "${NERSC_SCRIPT_DIR}/run_study_until_complete.sh"
  --account "${account}"
  --nodes "${nodes}"
  --study-dir "${study_dir}"
  --immediate "${immediate}"
  --retry-delay "${retry_delay}"
  --ready-file "${ready_file}"
)
if [[ -n "${wait_for_job}" ]]; then
  supervisor+=(--wait-for-job "${wait_for_job}")
fi
if [[ "${aggregate}" == true ]]; then
  supervisor+=(--aggregate)
fi

if [[ "${dry_run}" == true ]]; then
  echo -n "nohup setsid "
  nersc_print_command "${supervisor[@]}"
  echo "[nersc-rollover] log=${log_file} pid=${pid_file}"
  exit 0
fi

nersc_require_command nohup
nersc_require_command setsid
: >>"${log_file}"
: >"${ready_file}"
# Do not leak the one-shot startup lock into the long-lived child. The child
# acquires the separate rollover lock before this launcher reports success.
nohup setsid "${supervisor[@]}" 8>&- >>"${log_file}" 2>&1 </dev/null &
supervisor_pid=$!
temporary_pid="${pid_file}.${supervisor_pid}.tmp"
printf '%s\n' "${supervisor_pid}" >"${temporary_pid}"
mv "${temporary_pid}" "${pid_file}"

ready=false
for _attempt in $(seq 1 50); do
  if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
    nersc_die "rollover supervisor exited during startup; inspect ${log_file}"
  fi
  if [[ "$(<"${ready_file}")" == "${supervisor_pid}" ]]; then
    ready=true
    break
  fi
  sleep 0.2
done
[[ "${ready}" == true ]] || \
  nersc_die "rollover supervisor did not acquire its study lock; inspect ${log_file}"

echo "[nersc-rollover] started pid=${supervisor_pid} study=${study_dir}"
echo "[nersc-rollover] log=${log_file}"
