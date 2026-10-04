#!/usr/bin/env bash

# Credential-free compute-node smoke for the NERSC interactive launch path.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: smoke_compute.sh OUTPUT_DIR" >&2
  exit 2
fi

output_dir="$(realpath -m "$1")"
nersc_validate_results_dir "${output_dir}"
nersc_require_interactive_cpu_allocation
[[ ! -e "${output_dir}" ]] || nersc_die "smoke output already exists: ${output_dir}"

physical_cores="$({ lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l; })"
(( physical_cores == NERSC_CPU_PHYSICAL_CORES )) || \
  nersc_die "expected ${NERSC_CPU_PHYSICAL_CORES} physical CPU cores, saw ${physical_cores}"

cd "${NERSC_REPO_ROOT}"
nersc_load_ma_cc_environment
conda run -n "${NERSC_CONDA_ENV}" python -c \
  'import mas_cc, pandas, pyarrow; print("[smoke] imports=mas_cc,pandas,pyarrow")'

config="configs/runs/relational_reasoning/misselaneous/relational_imitation_round_feedback_controlled_smoke.yaml"
conda run --no-capture-output -n "${NERSC_CONDA_ENV}" mas-cc experiment preflight \
  --config "${config}" --output-dir "${output_dir}-preflight"
conda run --no-capture-output -n "${NERSC_CONDA_ENV}" mas-cc experiment run \
  --config "${config}" --output-dir "${output_dir}" --no-progress

find "${output_dir}" -name manifest.json -type f -print -quit | grep -q . || \
  nersc_die "mock experiment did not write a run manifest"
echo "[smoke] job=${SLURM_JOB_ID} host=$(hostname) qos=${SLURM_JOB_QOS} physical_cores=${physical_cores}"
echo "[smoke] output=${output_dir}"
