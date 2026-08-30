# Study 09h DeepInfra/DeepSeek NERSC production preflight

Preflight date: 2026-08-30. This report is specific to the NERSC Perlmutter
deployment of `deepseek-ai/DeepSeek-V4-Flash-0731`; it does not replace or
reinterpret the Potsdam preflight in the source Study 09h directory.

## Decision and scientific contract

- Status: **PERMITTED**, subject to the prepared-manifest checks below.
- Strict contract: `relational_persistence_high_statistics_false_v1`.
- Frozen task: `n12_L3_r03_k3/task_0002`; truth `NORTH`; false controller
  target `NORTHWEST`; selected strategic fact `f1` is a real frozen-task fact.
- Fixed: `N=12`, `L=3`, 30 rounds, `q_c=6`, naive receivers, strategic
  evidence, `recommendation_plus_fact`, soft schedule, `beta=4.0`,
  `theta=0.75`, temperature `0.0`, and a 4,096-token output cap.
- Exact axes: `q={1,2}` x `rho={0.80,0.85}` x
  `b={3,4,6,8,9,12}`.
- Structural cells: 24; repetitions per cell: 15; total episodes: 360.
- Study 09d reuse: **zero**. Its provider/model condition differs, so its
  episodes are not observations of this DeepInfra/DeepSeek system.
- Analysis recipe is byte-identical to source Study 09h and uses the existing
  per-cell MI/CMI, bootstrap, null, support, current, and affinity machinery.

The authoritative generated static report is outside the repository at:

```text
/pscratch/sd/d/dfarough/MA-CC-results/inspection/study09h-deepinfra-deepseek-v4-flash-0731-nersc-preflight-20260830T2045Z
```

## Provider and live smoke evidence

- Authenticated model lookup: HTTP 200; public text-generation model; JSON and
  function support; 1,048,576-token context.
- Base rates refreshed from authenticated/public DeepInfra metadata:
  $0.08/M ordinary input tokens, $0.016/M cached input tokens, and $0.18/M
  output tokens. The versioned MA-CC snapshot is
  `2026-08-30-deepinfra-v4-flash-0731-live-v1`.
- Authenticated account limits: 200 concurrent requests and 1,100,000 TPM.
- NERSC interactive CPU job `57753115`: account-limit lookup, one 4,096-token-
  cap chat smoke, and a 64-request concurrent JSON-object burst all completed
  with HTTP-successful responses. Failed setup job `57753110` sent no provider
  request; it only demonstrated that bare system `python` is unavailable.
- NERSC interactive CPU job `57753268`: a production-shaped one-episode run
  completed all three smoke rounds. All 48 provider calls validated on their
  first attempt, all 36 decisions resolved to an available option, and the
  episode retained three round records and 36 interaction records. Measured
  usage was 23,578 input tokens and 3,582 output tokens (about $0.0025 at the
  pinned base rates). The smoke result is external to the checkout at
  `/pscratch/sd/d/dfarough/MA-CC-results/smoke/study09h-deepinfra-deepseek-v4-flash-0731-one-episode-20260830T2050Z`.

## Calls, tokens, and cost

All token counts are deterministic planning estimates, not provider-tokenizer
measurements. Expected and conservative output figures deliberately charge the
full 4,096-token cap on every expected/conservative attempt; they are bounds,
not a prediction of ordinary ballot length.

| Quantity | Nominal/lower | Expected | Conservative |
|---|---:|---:|---:|
| Provider calls | 133,920 | 141,120 | 267,840 |
| Input tokens | 64,510,560 | 67,969,800 | 129,021,120 |
| Output tokens | 133,920 | 578,027,520 | 1,097,072,640 |
| Cost (USD) | $5.18 | $109.48 | $207.79 |

The configured per-run and system launch ceiling is $300, with finite request
and token limits above every conservative estimate.

## NERSC execution and rollover

- Preparation only: `mas-cc study prepare --execution-site nersc` writes the
  deterministic manifests beneath an external `/pscratch` study root.
- Execution: 24 generic scientific-cell shards; planned throttle 20; 10
  episode slots and 10 local request permits per active shard.
- Effective planned ceiling: 200 simultaneous episodes/request attempts.
  Shared provider control starts at 100, may recover to the authenticated 200
  maximum, and smoothly paces all attempts to 1,200 RPM.
- Four Perlmutter CPU nodes are requested through the generic NERSC supervisor.
  The plan runs 20 workers at 10 physical CPUs and 12 GB each; the provider,
  not the 512 allocated physical cores, is the binding limit.
- Nominal call time at 1,200 RPM is 1.86 hours; expected call time is 1.96
  hours; the all-retry conservative bound is 3.72 hours before overhead and
  outages. These are service-rate projections, not wall-time guarantees.
- Every allocation explicitly uses `--qos=interactive --constraint=cpu` and a
  four-hour wall. The detached supervisor reuses the same manifests and result
  root after a clean timeout, stops on scientific failures, and performs strict
  aggregation in an interactive CPU allocation only after all 24 cells seal.

No study-specific SLURM job or NERSC launcher is introduced. Source scheduler
fields remain Potsdam-safe; NERSC result rebinding and scheduler enforcement
occur only in generated artifacts and `scripts/nersc/`.
