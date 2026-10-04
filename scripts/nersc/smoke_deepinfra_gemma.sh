#!/usr/bin/env bash

# Live DeepInfra/Gemma validation inside a separate NERSC interactive CPU allocation.

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: smoke_deepinfra_gemma.sh OUTPUT_ROOT" >&2
  exit 2
fi

output_root="$(realpath -m "$1")"
nersc_validate_results_dir "${output_root}"
nersc_require_interactive_cpu_allocation
[[ ! -e "${output_root}" ]] || nersc_die "smoke output already exists: ${output_root}"

physical_cores="$({ lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l; })"
(( physical_cores == NERSC_CPU_PHYSICAL_CORES )) || \
  nersc_die "expected ${NERSC_CPU_PHYSICAL_CORES} physical CPU cores, saw ${physical_cores}"

cd "${NERSC_REPO_ROOT}"
nersc_load_ma_cc_environment
mkdir -p "${output_root}"

readonly model="google/gemma-4-E4B-it"
MAS_CC_RUN_DEEPINFRA_SMOKE=1 \
MAS_CC_DEEPINFRA_SMOKE_MODEL="${model}" \
  conda run --no-capture-output -n "${NERSC_CONDA_ENV}" \
  python -m pytest tests/mas_cc/test_deepinfra_provider.py \
  -k 'live_account_limit_smoke or live_chat_completion_smoke' -q -s

run_smoke() {
  local label="$1"
  local config="$2"
  local preflight="${output_root}/preflight-${label}"
  local output="${output_root}/${label}"

  conda run --no-capture-output -n "${NERSC_CONDA_ENV}" \
    mas-cc experiment preflight \
    --config "${config}" \
    --output-dir "${preflight}"
  conda run --no-capture-output -n "${NERSC_CONDA_ENV}" \
    mas-cc experiment run \
    --config "${config}" \
    --output-dir "${output}" \
    --approve-preflight "${preflight}/preflight_id.txt" \
    --no-progress
}

run_smoke \
  n6-r3 \
  configs/runs/relational_reasoning/relational_imitation_round_feedback_deepinfra_gemma_N6_R3_smoke.yaml
run_smoke \
  study08-one-episode \
  configs/runs/relational_reasoning/population_study_08_deepinfra_gemma_one_episode_smoke.yaml

"${NERSC_CONDA_ENV_PATH}/bin/python" - "${output_root}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
expected = {"n6-r3": (3, 18), "study08-one-episode": (10, 240)}
for label, (round_count, micro_count) in expected.items():
    run_root = root / label
    manifests = tuple(run_root.rglob("manifest.json"))
    rounds = tuple(run_root.rglob("round_trajectory.jsonl"))
    micros = tuple(run_root.rglob("micro_slot_trajectory.jsonl"))
    scientific = tuple(run_root.rglob("scientific_events.parquet"))
    failures = tuple(run_root.rglob("failure.json"))
    if len(manifests) != 1 or len(rounds) != 1 or len(micros) != 1:
        raise SystemExit(f"{label}: incomplete artifact set")
    actual_rounds = sum(1 for line in rounds[0].open() if line.strip())
    records = [json.loads(line) for line in micros[0].open() if line.strip()]
    unique = {(row.get("round_index"), row.get("within_round_index")) for row in records}
    if (actual_rounds, len(records), len(unique)) != (
        round_count,
        micro_count,
        micro_count,
    ):
        raise SystemExit(
            f"{label}: rounds={actual_rounds} micro={len(records)} unique={len(unique)}"
        )
    if len(scientific) != 1 or failures:
        raise SystemExit(
            f"{label}: scientific={len(scientific)} failures={len(failures)}"
        )
    print(
        f"[smoke] {label} rounds={actual_rounds} micro={len(records)} "
        f"unique={len(unique)} scientific=1 failures=0"
    )
PY

echo "[smoke] job=${SLURM_JOB_ID} host=$(hostname) qos=${SLURM_JOB_QOS} physical_cores=${physical_cores}"
echo "[smoke] output_root=${output_root}"
