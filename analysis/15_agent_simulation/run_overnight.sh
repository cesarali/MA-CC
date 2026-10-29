#!/bin/bash
# Run the 15-agent campaign: task003 first (about 1 hour), then Object Placements (about 4-7 hours). 12 workers.
# Keep the laptop plugged in and the lid open; caffeinate prevents idle and system sleep on AC power.
# Usage, from the repository root:   bash analysis/15_agent_simulation/run_overnight.sh
set -euo pipefail
cd "$(dirname "$0")/../.."                       # repository root
LOG=analysis/15_agent_simulation/logs
mkdir -p "$LOG"
RUNNER=analysis/task003_llm_free/simulation_2/simulator/run_simulation_2.py
STAMP=$(date +%Y-%m-%d_%H%M)
for T in task003 op; do
  echo "=== $T started $(date)" | tee -a "$LOG/run_$STAMP.log"
  caffeinate -is .venv/bin/python "$RUNNER" --config "analysis/15_agent_simulation/configs/sim2_15agents_$T.yaml" \
      --workers 12 >> "$LOG/run_$STAMP.log" 2>&1
  echo "=== $T finished $(date)" | tee -a "$LOG/run_$STAMP.log"
done
