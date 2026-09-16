# Blackboard checkpoint small test 01

This is an inspection-only version of the checkpoint experiment. It is not
powered for scientific inference.

## Fixed design

- Population: `N = 6`
- Social sample: `q = 3`
- Controller sensor sample: `q_C = 3`
- Posting budget: `b = 2`
- Shared preparation: `L = 2` uncontrolled rounds
- Branch continuation: `M = 2` rounds
- Independent parents/repetitions: `K = 3`
- Branches per parent: `none`, `always_truth`, `always_false`,
  `sensing_truth`, and `sensing_false`
- Runtime concurrency: three parent bundles, provider ceiling 60, one active
  SLURM node

The hash-pinned six-agent distribution in `artifacts/` is deterministic,
covers all 27 evidence cards once, and has zero agents that can structurally
certify the answer alone. It is an inspection fixture derived from validated
`task_002`; it is not an additional behavioral validation of that task at
`N = 6`.

The destination must also contain the already-established validated base task
at `results/studies/musr_team_allocation_validation_01/tasks/task_002/base_task.json`.
That generated study asset is intentionally outside Git; the new `N = 6`
distribution and controller design are hash-pinned repository artifacts.

With five branches for each of three parent checkpoints, a complete run starts
three scheduler-visible parent-bundle episodes and produces 15 continuation
trajectories. The parent prefix is generated once per parent, then all five
branches resume from the exact same checkpoint after round 2. Counting stored
trajectory segments gives three shared preparation prefixes plus 15 branch
continuations, but the preparation prefix is not rerun for every branch.
Branches within each parent bundle execute sequentially for resume-safe sealing.
Consequently, the natural steady-state request concurrency is approximately
three, with a possible initialization burst of up to 18 concurrent requests;
the configured ceiling of 60 no longer imposes an artificial bottleneck.

## Expected analysis artifacts

Aggregation is configured to emit checkpoint contrasts and branch-specific
round metrics, including the `T_pi`, `eta_IF`, `eta_IR`, and `chi` aliases plus
sensor MAE/MSE, controller-action entropy, support diagnostics, and null
results. Assigned-policy information includes the unsmoothed, Jeffreys,
Miller-Madow, and uniform row-smoothing (`lambda = 1` and `12.5`) variants.
The optional sensing-only randomized-gate response `tau_hat_1` is also emitted
separately and is never applied to deterministic always/none branches.
With only three parents and two continuation rounds, these values are useful
for checking schemas, grouping, and estimator support—not for interpreting
effect sizes or uncertainty.

In particular, the always-act branches have constant controller action, so
action-dependent information metrics should be reported as unsupported or
degenerate. The sensing branches can also lack action variation by chance in
such a tiny sample; that outcome is diagnostic, not a failed run.

## Commands (not executed while preparing this fixture)

On Potsdam, use the repository's dedicated environment:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc study preflight configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_small_test_01/study.yaml
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc study submit configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_small_test_01/study.yaml
```

Submission writes results and SLURM logs below the `/work` root declared in
`study.yaml`. Aggregation uses the same generic study workflow and the local
`analysis.yaml` recipe.
