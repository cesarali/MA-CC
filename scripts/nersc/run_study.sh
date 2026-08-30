#!/usr/bin/env bash

# Prepare and run a generic MA-CC study across interactive Perlmutter CPU nodes.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
  cat <<'EOF'
usage: run_study.sh --account PROJECT --nodes 1-4
                    (--config-dir DIR --results-dir DIR | --study-dir DIR)
                    [--time HH:MM:SS] [--throttle N]
                    [--immediate SECONDS] [--allow-shorter-than-plan]
                    [--dry-run]

--config-dir prepares the study before allocating. --study-dir resumes an
already prepared study. The QoS is fixed to interactive; batch submission is disabled.
EOF
}

account="${NERSC_CPU_ACCOUNT:-}"
nodes=""
config_dir=""
results_dir=""
study_dir=""
walltime=""
throttle=""
immediate=600
dry_run=false
allow_shorter_than_plan=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) [[ $# -ge 2 ]] || nersc_die "--account requires a value"; account="$2"; shift 2 ;;
    --nodes) [[ $# -ge 2 ]] || nersc_die "--nodes requires a value"; nodes="$2"; shift 2 ;;
    --config-dir) [[ $# -ge 2 ]] || nersc_die "--config-dir requires a value"; config_dir="$2"; shift 2 ;;
    --results-dir) [[ $# -ge 2 ]] || nersc_die "--results-dir requires a value"; results_dir="$2"; shift 2 ;;
    --study-dir) [[ $# -ge 2 ]] || nersc_die "--study-dir requires a value"; study_dir="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || nersc_die "--time requires a value"; walltime="$2"; shift 2 ;;
    --throttle) [[ $# -ge 2 ]] || nersc_die "--throttle requires a value"; throttle="$2"; shift 2 ;;
    --immediate) [[ $# -ge 2 ]] || nersc_die "--immediate requires a value"; immediate="$2"; shift 2 ;;
    --allow-shorter-than-plan) allow_shorter_than_plan=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --qos|--qos=*|-q) nersc_die "QoS is fixed to interactive and cannot be overridden" ;;
    --help|-h) usage; exit 0 ;;
    *) nersc_die "unknown argument: $1" ;;
  esac
done

[[ -n "${account}" ]] || nersc_die "--account or NERSC_CPU_ACCOUNT is required"
[[ -n "${nodes}" ]] || nersc_die "--nodes is required"
nersc_validate_account "${account}"
nersc_validate_nodes "${nodes}"
nersc_validate_immediate "${immediate}"
if [[ -n "${throttle}" ]]; then
  nersc_validate_positive_integer "throttle" "${throttle}"
fi

if [[ -n "${config_dir}" ]]; then
  [[ -z "${study_dir}" ]] || nersc_die "use --config-dir or --study-dir, not both"
  [[ -n "${results_dir}" ]] || nersc_die "--results-dir is required with --config-dir"
  config_dir="$(realpath -m "${config_dir}")"
  study_dir="$(realpath -m "${results_dir}")"
  [[ -d "${config_dir}" ]] || nersc_die "config directory does not exist: ${config_dir}"
else
  [[ -n "${study_dir}" ]] || nersc_die "--config-dir or --study-dir is required"
  [[ -z "${results_dir}" ]] || nersc_die "--results-dir is only valid with --config-dir"
  [[ -z "${throttle}" ]] || nersc_die "--throttle is only valid while preparing a study"
  study_dir="$(realpath -m "${study_dir}")"
fi
nersc_validate_results_dir "${study_dir}"

cd "${NERSC_REPO_ROOT}"
nersc_load_ma_cc_environment
if [[ -n "${config_dir}" ]]; then
  site_results_root="${NERSC_RESULTS_ROOT:-/pscratch/sd/d/dfarough/MA-CC-results}"
  prepare=(
    conda run --no-capture-output -n "${NERSC_CONDA_ENV}"
    mas-cc study prepare --config-dir "${config_dir}" --results-dir "${study_dir}"
    --require-results-under "${site_results_root}"
    --execution-site nersc
  )
  if [[ -n "${throttle}" ]]; then
    prepare+=(--throttle "${throttle}")
  fi
  "${prepare[@]}"
fi

# Only one allocation may own a prepared study.  Besides preventing duplicate
# workers, this lock makes it safe to clear provider leases left behind when
# the preceding interactive allocation was killed at its four-hour wall.
if [[ "${dry_run}" == false ]]; then
  nersc_require_command flock
  mkdir -p "${study_dir}/runtime"
  exec 8>"${study_dir}/runtime/nersc-allocation.lock"
  flock -n 8 || nersc_die "another NERSC allocation already owns this study"
  "${NERSC_CONDA_ENV_PATH}/bin/python" \
    "${NERSC_SCRIPT_DIR}/reset_provider_control_leases.py" \
    --study-dir "${study_dir}"
fi

plan_line="$({
  conda run -n "${NERSC_CONDA_ENV}" python "${NERSC_SCRIPT_DIR}/study_plan.py" \
    --study-dir "${study_dir}" --nodes "${nodes}" --format tsv
} 2>&1)" || nersc_die "could not build NERSC launch plan: ${plan_line}"
IFS=$'\t' read -r manifest worker_kind total_workers physical_cpus planned_time shard_count <<<"${plan_line}"

[[ -n "${walltime}" ]] || walltime="${planned_time}"
nersc_validate_walltime "${walltime}"
planned_seconds="$(nersc_walltime_seconds "${planned_time}")"
requested_seconds="$(nersc_walltime_seconds "${walltime}")"
if (( requested_seconds < planned_seconds )); then
  [[ "${allow_shorter_than_plan}" == true ]] || \
    nersc_die "requested wall time ${walltime} is shorter than planned shard time ${planned_time}"
  echo "[nersc] resumable allocation time ${walltime} is shorter than planned shard time ${planned_time}" >&2
fi

echo "[nersc] study=${study_dir}"
echo "[nersc] shards=${shard_count} workers=${total_workers} nodes=${nodes} physical_cpus_per_worker=${physical_cpus}"
echo "[nersc] scheduler=qos:interactive,constraint:cpu,time:${walltime},account:${account}"

command=(
  salloc
  --qos=interactive
  --constraint=cpu
  --account="${account}"
  --nodes="${nodes}"
  --time="${walltime}"
  --immediate="${immediate}"
  --job-name=mas-cc-study-interactive
  srun
  --nodes="${nodes}"
  --ntasks="${nodes}"
  --ntasks-per-node=1
  --cpus-per-task="${NERSC_CPU_LOGICAL_CPUS}"
  --cpu-bind=none
  "${NERSC_CONDA_ENV_PATH}/bin/python"
  "${NERSC_SCRIPT_DIR}/run_study_rank.py"
  --study-dir "${study_dir}"
  --manifest "${manifest}"
  --worker-kind "${worker_kind}"
  --total-workers "${total_workers}"
)

if [[ "${dry_run}" == true ]]; then
  nersc_print_command "${command[@]}"
  exit 0
fi

exec "${command[@]}"
