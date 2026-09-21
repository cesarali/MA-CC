# ICLR false arm — Potsdam aggregation handoff

**Last checked:** 2026-09-18 19:30 CEST  
**Study:** `recomm_only_q12_chatoss_false_control`  
**Study root:** `/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control`

## Scientific run status

This is the completed ICLR q=12 **false-control arm** run, not a separate
experiment. It used DeepInfra `openai/gpt-oss-120b`, Task-003, 24 agents,
30 rounds, and 60 repetitions for each of nine `(rho, intervention_budget)`
cells.

- All **9/9 cells** have durable `cell_complete.json` seals.
- Each seal contains **60 episodes**, for **540/540 completed episodes**.
- Science array `1882419` completed successfully: all nine tasks exited `0:0`.
- The recorded run ledger is **$55.2575**, 399,769 requests, 815,816,988 input
  tokens, and 147,483,766 output tokens. This is the internal run ledger,
  excluding initialization calls whose cost is not persisted in these files.
- No more provider calls are made during aggregation or finalization.

The provider coordinator used Redis during simulation only. Its dedicated
Potsdam job `1882318` was cancelled at 19:04 CEST after all episodes sealed.
Redis is not required by the saved-data analysis pipeline.

## Current aggregation state

The first dependent aggregation trigger (`1882428`) was allocated 1 CPU / 8
GB and was OOM-killed before it could freeze canonical inputs or schedule any
analysis work. It did not alter the sealed scientific outputs.

The strict retry is active and does **not** use `--allow-incomplete`:

| Role | Slurm job | State at handoff | Resources |
|---|---:|---|---|
| Canonical-input/submit trigger | `1883826` | Completed (`0:0`) | 4 CPU, 32 GB |
| Prepare validation | `1883829` | Completed (`0:0`) | 1 CPU, 32 GB |
| Information-group array | `1883830` | All 9 tasks completed (`0:0`) | 1 CPU, 16 GB/task; throttle 9 |
| Study finalizer | `1883831` | Pending (`Priority`) | 8 CPU, 64 GB, 12 h |

The information-array tasks correspond to the nine physical scientific cells,
not arbitrary scheduler shards. Their independent MI/CMI, bootstrap, and null
calculations have all completed successfully. The remaining finalizer combines
those fragments, computes the configured derived/causal/epistemic analyses,
renders plots and reports, validates the package, and publishes it atomically.

## Monitoring

Frozen analysis generation:

```text
e079de9f3e394d3991b3
```

Authoritative live-progress file until publication:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control/analysis/.work/e079de9f3e394d3991b3/progress.json
```

At handoff it reported `completed_groups: 9`, `failed_groups: []`, and a
pending finalizer. Monitor both this file and Slurm because scheduler success
alone is not proof that the analysis package was published:

```bash
squeue -j 1883831
sacct -X -j 1883830,1883831 \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem
jq . /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/recomm_only_q12_chatoss_false_control/analysis/.work/e079de9f3e394d3991b3/progress.json
```

Finalizer stdout/stderr:

```text
<study-root>/logs/analysis-e079de9f3e394d3991b3/finalize-1883831.{out,err}
```

On success, the atomic published package is expected at:

```text
<study-root>/analysis/recomm_only_q12_chatoss_false_control_analysis.zip
```

`analysis/progress.json` is transient and is removed after a successful final
publication. The final `analysis_manifest.json` records stage timings,
measurements, resource policy, validation, and the completed group fragments.

## Analysis and parallelism policy

The study's `analysis.yaml` now declares the relevant scheduler policy:

- nine concurrent one-CPU / 16-GB information groups;
- a 32-GB canonical preparation allocation; and
- an eight-CPU / 64-GB finalizer with a 12-hour limit.

The generic launchers remain in use:
`scripts/Potsdam/SLURM/run_study_aggregate.job` and
`scripts/Potsdam/SLURM/run_study_analysis.job`. No study-specific analysis
implementation or replacement estimator was added. The existing MI/CMI,
bootstrap, null, causal-response, blackboard calibration, and epistemic-phase
engines specified in `analysis.yaml` are authoritative.

## If the finalizer fails

1. Preserve the generation and its nine completed group fragments; do not
   rerun the 540 episodes or delete the study root.
2. Inspect `finalize-1883831.err`, `sacct`, and the generation `progress.json`
   to identify whether it was memory, wall-time, or scientific validation.
3. Reuse the frozen generation and completed fragments for a finalizer retry;
   do not submit another information array unless a fragment is invalid.
4. Do not reduce bootstrap/permutation counts or alter the scientific recipe
   merely to make the scheduler run fit. Any scientific change requires an
   explicit decision and a new analysis generation.
