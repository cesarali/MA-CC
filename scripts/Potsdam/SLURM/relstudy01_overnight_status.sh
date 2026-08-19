#!/bin/bash
# Status of the relational population study 01 overnight run.
#
#   scripts/Potsdam/SLURM/relstudy01_overnight_status.sh
#
# Reports real completion, not SLURM state. A cell whose episodes all failed
# still exits 0 and is recorded COMPLETED by SLURM (execution.fail_fast is
# false), so the count that matters is how many cells wrote cell_complete.json.
set -uo pipefail

JOBS=${JOBS:-1829474,1829492}
LOGS=/work/ojedamarin/Projects/LanguageGames/MA-CC/logs
RES=/work/ojedamarin/Projects/LanguageGames/MA-CC/results/relational_population_study01_overnight

ok=$(find "$RES" -name cell_complete.json 2>/dev/null | wc -l)
bad=$(grep -hE "aggregated [01] episode\(s\)" "$LOGS"/relstudy01_overnight_*.err 2>/dev/null | wc -l)
http=$(grep -hE "HTTP (429|500|502|503|504)" "$LOGS"/relstudy01_overnight_*.err 2>/dev/null | wc -l)
run=$(squeue -u "$USER" -h -j "$JOBS" -t RUNNING 2>/dev/null | wc -l)
pend=$(squeue -u "$USER" -h -j "$JOBS" -t PENDING 2>/dev/null | wc -l)

echo "relational population study 01 -- overnight   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  cells complete : ${ok}/120"
echo "  running        : ${run}      pending array entries: ${pend}"
echo "  degraded cells : ${bad}   (episodes lost; these need a rerun)"
echo "  provider errors: ${http}"

sacct -j "$JOBS" -X --noheader --format=State%16 2>/dev/null | sort | uniq -c | sed 's/^/  slurm: /'

# Mean wall of genuinely-completed cells -> ETA. >600s filters cells that
# died fast without producing data.
sacct -j "$JOBS" -X --noheader --format=State%12,ElapsedRaw%12 2>/dev/null \
 | awk -v ok="$ok" '$1=="COMPLETED" && $2>600 {s+=$2;n++}
     END{ if(n>0){ m=s/n/60; rem=120-ok;
       printf "  mean cell wall : %.1f min (n=%d)\n", m, n;
       printf "  ETA            : %.1f h for %d remaining cells at 4 parallel\n", rem*m/4/60, rem } }'

if [ "$bad" -gt 0 ]; then
  echo
  echo "  degraded cells (no usable episodes):"
  grep -lE "aggregated [01] episode\(s\)" "$LOGS"/relstudy01_overnight_*.err 2>/dev/null \
    | sed 's/.*_\([0-9]*\)\.err/    array task \1/'
fi
