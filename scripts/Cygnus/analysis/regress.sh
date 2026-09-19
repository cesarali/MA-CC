#!/usr/bin/env bash
# Byte-identity regression gate for finalizer changes on Cygnus.
#
#   scripts/Cygnus/analysis/regress.sh <candidate-repo-root> [b9b15|potsdam|checkpoint|all]
#
# For each named reference study it submits one Slurm job that finalizes the
# study with the CANDIDATE tree into a side directory and compares every
# published table against the REFERENCE package (the one the unmodified code
# produced). A change counts as landed only when every job prints
# "RESULT: byte-equivalent tables". See REGRESSION.md for the references.
set -euo pipefail
CANDIDATE=$(realpath "${1:?candidate repo root}")
WHICH="${2:-all}"
PY=${MA_CC_PYTHON:-/shared/home/cesar/.local/share/mamba/envs/MA-CC/bin/python}
OUT=${MA_CC_REGRESS_OUT:-$HOME/agg/e2e}
mkdir -p "$OUT"

declare -A STUDY REF SBATCH EXTRA MODE BUNDLE STUDY_ID
STUDY[b9b15]=$HOME/MA-CC/results/studies/recomm_only_q12_chatoss_false_control_b9_b15_cygnus
REF[b9b15]=${STUDY[b9b15]}/analysis/tables
SBATCH[b9b15]="--cpus-per-task=16 --mem=44G --time=03:00:00"
EXTRA[b9b15]=""

# potsdam: a FROZEN GENERATION bundle, not a study with runs. It carries the Slurm
# graph's per-cell information fragments (computed on Potsdam) and the finalizer's
# inputs; the reference (job 42) was produced by the graph's finalize role on it.
# The gate therefore relocates the bundle afresh and re-runs the finalize role
# with the candidate tree: every finalizer stage is exercised, the information
# stage is NOT (its fragments come from the bundle).
# The code comes from the candidate (PYTHONPATH); MA_CC_REPOSITORY_ROOT stays the
# ~/MA-CC snapshot because the epistemic-phase stage reads the MuSR task
# artifacts from <root>/results/studies/... (job 82 failed without them).
STUDY[potsdam]=$HOME/agg/iclr_false_arm_potsdam/recomm_only_q12_chatoss_false_control
REF[potsdam]=${STUDY[potsdam]}/analysis/tables
SBATCH[potsdam]="--cpus-per-task=8 --mem=44G --time=03:00:00"
EXTRA[potsdam]=""
MODE[potsdam]=frozen
BUNDLE[potsdam]=$HOME/agg/recomm_only_q12_chatoss_false_control_frozen_aggregation_inputs_20260918.zip
STUDY_ID[potsdam]=recomm_only_q12_chatoss_false_control

# checkpoint family: relocated bundle, lineage file set aside, incomplete inputs allowed;
# the package publishes under analysis-runs/<run>/output (see relocate + runbook §4)
STUDY[checkpoint]=$HOME/agg/checkpoint_ensemble_01/blackboard_checkpoint_ensemble_01
REF[checkpoint]=${STUDY[checkpoint]}/analysis-runs/parallel-tracked-20260918/output/tables
# The reference ran on big-0 with 32 CPUs / 110 G, but its manifest shows a 6.8 GiB
# peak RSS: a standard node (16 CPUs, 44 G) runs it, slower but without queueing
# behind whatever occupies the big node.
SBATCH[checkpoint]="--cpus-per-task=16 --mem=44G --time=08:00:00"
EXTRA[checkpoint]="--allow-incomplete"

names=(b9b15 potsdam checkpoint)
[ "$WHICH" = all ] || names=("$WHICH")
for name in "${names[@]}"; do
  [ -d "${REF[$name]}" ] || { echo "reference tables missing for $name: ${REF[$name]}" >&2; exit 2; }
  if [ "${MODE[$name]:-local}" = frozen ]; then
    # relocate the bundle into a fresh root, run the finalize role from the candidate tree
    wrap="W=$OUT/$name-\$SLURM_JOB_ID && mkdir -p \$W && cd \$W && export PYTHONPATH=$CANDIDATE/src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
      $PY -m zipfile -e ${BUNDLE[$name]} bundle/ && \
      $PY -m mas_cc.studies.relocate --bundle bundle --study-root \$W/${STUDY_ID[$name]} | tee relocate.log && \
      M=\$(grep -o '\"manifest\": \"[^\"]*\"' relocate.log | cut -d'\"' -f4) && \
      MA_CC_REPOSITORY_ROOT=$HOME/MA-CC MA_CC_PYTHON=$PY bash $CANDIDATE/scripts/Cygnus/SLURM/run_study_analysis.job finalize \$M && \
      $PY $CANDIDATE/scripts/Cygnus/analysis/compare_dirs.py ${REF[$name]} \$W/${STUDY_ID[$name]}/analysis/tables"
  else
    wrap="cd $CANDIDATE && export PYTHONPATH=$CANDIDATE/src MA_CC_REPOSITORY_ROOT=$HOME/MA-CC MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
      $PY $CANDIDATE/scripts/Cygnus/analysis/finalize_to.py ${STUDY[$name]} $OUT/$name-\$SLURM_JOB_ID ${EXTRA[$name]} && \
      $PY $CANDIDATE/scripts/Cygnus/analysis/compare_dirs.py ${REF[$name]} $OUT/$name-\$SLURM_JOB_ID/tables"
  fi
  job=$(sbatch --parsable --job-name="regress-$name" ${SBATCH[$name]} \
    --output="$OUT/regress-$name-%j.out" --error="$OUT/regress-$name-%j.err" --wrap "$wrap")
  echo "$name -> job $job (log $OUT/regress-$name-$job.out)"
done
echo "when done: grep -h RESULT $OUT/regress-*-*.out"
