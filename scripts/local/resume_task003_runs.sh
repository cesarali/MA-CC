#!/usr/bin/env bash
# Resume the task-003 trace runs. Safe to re-run: completed episodes are
# skipped via their episode checkpoints.
#
# DO NOT EDIT THE CONFIGS between runs. Each checkpoint stores a
# resolved_config_hash and resume is refused if it changes, which restarts
# every episode from zero.
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
V=.venv/bin/mas-cc
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# Stream every agent call to the local Phoenix UI (http://localhost:6006) as it
# happens. Purely additive: unset these two variables and the run behaves
# exactly as before. They are not part of any config, so they do not change the
# resolved config hash and therefore do not invalidate a single checkpoint.
export MAS_CC_TRACE_ENDPOINT="${MAS_CC_TRACE_ENDPOINT:-http://localhost:6006/v1/traces}"
export MAS_CC_TRACE_PROJECT="${MAS_CC_TRACE_PROJECT:-mas-cc}"

log "1/3 v0 deepseek controlled (16 cells, 80 episodes)"
$V experiment run --config configs/runs/smoke/task003_v0_deepseek/controlled.yaml \
   --output-dir results/studies/task003_v0_deepseek/controlled --no-progress 2>&1 | tail -3

# The v4.1 reasoning arm is HELD (operator decision, 2026-09-23): it costs
# ~$7 and 5-7 hours because deepseek-v4.1-flash spends ~4,000 output tokens of
# reasoning per call. Opt in explicitly so re-running this script to resume the
# v0 arms cannot start it by accident.
if [[ "${RUN_V41:-0}" != "1" ]]; then
  log "v4.1 reasoning arm held; re-run with RUN_V41=1 to include it"
  log "ALL DONE (v0 only)"
  exit 0
fi

log "2/3 initialize v4.1 reasoning arm"
$V study initialize --config-dir configs/runs/smoke/task003_reasoning_v41 \
   --output-dir results/studies/task003_reasoning_v41_initializations 2>&1 | tail -2
ls results/studies/task003_reasoning_v41_initializations/*.json >/dev/null 2>&1 || {
  log "STOP: v4.1 initialize produced no artifacts"; exit 1; }

log "3/3 v4.1 reasoning arm (2 cells, 10 episodes)"
$V experiment run --config configs/runs/smoke/task003_reasoning_v41/controlled.yaml \
   --output-dir results/studies/task003_reasoning_v41/controlled --no-progress 2>&1 | tail -3
$V experiment run --config configs/runs/smoke/task003_reasoning_v41/no_control.yaml \
   --output-dir results/studies/task003_reasoning_v41/no_control --no-progress 2>&1 | tail -3
log "ALL DONE"
