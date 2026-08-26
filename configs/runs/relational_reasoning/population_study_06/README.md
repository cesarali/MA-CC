# Population Study 06 — high-resolution control efficiency

Study 06 maps actuation strength and feedback-policy switching at fixed sensing:

```text
N=24, q=1, q_c=12, beta=4, rounds=10
b=[4,8,12,16,20,24]
theta=[0.20,0.35,0.50,0.65,0.80]
```

The main atlas has 30 policy conditions over four matched worlds. The targeted
ablation adds `b=[8,16,24] x beta=[2,4,8]` at `theta=0.50`. Both use ten
repetitions per task and condition. Submission expands these into 156 original
scientific-cell shards. At the declared 18-node/900-RPM planning bound, ideal
workload division is about 2 h 40 m before queueing and runtime variance.

## Scientific inheritance and changes

The primary templates are Study 04's
`relational_population_study04_qc06.yaml`,
`relational_population_study04_qc12.yaml`, and
`relational_population_study04_qc18.yaml`, with q_c=12 as the direct base.
The round-feedback estimator, relational current implementation, and Study 05
effective-affinity implementation are reused by `study aggregate`.

Held identical to Study 04: GPT-OSS provider/model settings, N=24, L=2, r=6,
q=1, q_c=12, ten rounds, local-vote initialization, complete topology, public
votes, social distrust, no consensus stopping, semantic option shuffling,
soft stochastic target control, fixed incorrect target index 2,
`recommendation_only`, beta=4 in the main atlas, and `results_only` retention.

New axes are intervention budget `b` and threshold `theta`; beta varies only in
the sparse ablation. The matched task set is task_0001 through task_0004. Their
correct option positions are B, B, A, A respectively, so target index 2 (option
C before per-call semantic shuffling) is incorrect in every selected world.

The current seeding mechanism derives episode streams from grid-cell indices.
It has no clean cross-cell common-random-number switch, so the configs do not
claim CRN pairing. They do preserve the same task set and study seed.

## Files

- `study.yaml`: automatic cell-array policy, capped at 18 active nodes and an
  estimated 864 RPM.
- `study06_main_b_theta.yaml`: 120 cells, 1,200 episodes.
- `study06_beta_ablation.yaml`: 36 cells, 360 episodes.
- `analysis.yaml`: pooled canonical analysis, bootstrap/nulls/support, derived
  eta_IR, currents, effective affinity/compliance, and atlas/ablation plots.
- `study06_second_model_validation.yaml`: gated Qwen single-model benchmark.
- `study06_second_model_anchor.yaml`: optional 36-cell Qwen anchor.
- `PREFLIGHT.md`: calls, token bounds, cost, timing, and concurrency report.

No study-specific SLURM job exists. Submission uses the generic
`scripts/Potsdam/SLURM/run_study_cell_array.job`.

## Run the default study

From the repository root, submission (this is the provider-consuming step) is:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main study submit \
  --config-dir configs/runs/relational_reasoning/population_study_06 \
  --results-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_06
```

`study.yaml` supplies throttle 1. Do not pass a larger `--throttle` without a
new provider-load check.

After all cells seal, aggregate offline with:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_06
```

Strict aggregation is intentional. Use `--allow-incomplete` only for visibly
partial exploratory output, never for final CMI reporting.

## Optional Qwen robustness gate

The supported alternative is `gwdg/qwen3-30b-a3b-instruct-2507`. First run:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main \
  benchmark relational-support preflight \
  --config configs/runs/relational_reasoning/population_study_06/study06_second_model_validation.yaml

conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main \
  benchmark relational-support run \
  --config configs/runs/relational_reasoning/population_study_06/study06_second_model_validation.yaml
```

Predeclared eligibility criteria for the anchor are: full-information accuracy
at least 0.90; a full-minus-partial accuracy gap at least 0.40; zero-information
performance broadly near the 1/3 chance floor; provider/parser error rate at
most 0.05; and no correct-position accuracy range larger than 0.15. Inspect the
benchmark summary rather than inferring these from overall accuracy. If any
criterion fails, do not launch or interpret the anchor.

If it passes, preflight and run `study06_second_model_anchor.yaml` through the
ordinary experiment CLI. It is intentionally absent from the default manifest
because the current study workflow cannot express a data-dependent scheduler
gate and the full optional path would exceed 48 hours.

## Analysis scope and repository limitations

Information estimates retain whole-episode bootstrap, policy-conditional
action randomization, sensing permutations, estimator variants, and support
diagnostics. Unsupported CMI slices are masked by plotting. Pooled estimates
are recomputed from pooled canonical observations and are never averages of
execution-shard CMIs.

The current standardized plotter directly produces heatmaps for cell-level
metrics exposed by the established engines: signed response, CMI variants,
eta_IR, action entropy, sensing MI, h_eff, gamma_eff, and J_c. Canonical round
and micro-slot tables retain final shares, kappa, phi, and state-local records,
but the generic plotter does not yet promote those raw fields/conditioning bins
to declarative plot coordinates. Likewise, the current engine does not expose
Delta S_sys or Sigma. They are therefore retained for downstream analysis but
not fabricated as automatic estimates. No estimator machinery was rewritten.
