# NERSC production-like CMI smoke

This credential-free study validates the production execution and analysis
path without spending provider quota:

- 12 distinct scientific cells;
- 30 episodes per cell, 360 episodes total;
- N=24 and ten rounds per episode;
- two sensing levels, three actuation budgets, and two matched worlds;
- one generic cell worker per original cell;
- `results_only` round and micro-slot retention;
- strict study aggregation with established MI/CMI, whole-episode bootstrap,
  null permutations, support diagnostics, derived observables, and heatmaps.

It must run through `scripts/nersc/run_study.sh` with the interactive QoS. The
authoritative result root is outside the repository:

```text
/pscratch/sd/d/dfarough/MA-CC-results/studies/nersc-production-like-cmi-smoke-20260828
```

After every cell seals, aggregate on an interactive CPU node with:

```bash
scripts/nersc/run_command.sh --account m4539 --time 00:30:00 -- \
  bash -lc 'cd /pscratch/sd/d/dfarough/MA-CC && \
    module load python/3.11-24.1.0 && \
    conda run --no-capture-output -n MA-CC mas-cc study aggregate \
      --study-dir /pscratch/sd/d/dfarough/MA-CC-results/studies/nersc-production-like-cmi-smoke-20260828'
```

## Validated run (2026-08-28)

The production-like smoke completed through the generic NERSC launchers:

- execution allocation `57680097`: two CPU nodes, `qos=interactive`, 12
  workers, `COMPLETED` with exit code `0:0`;
- aggregation allocation `57680225`: one CPU node, `qos=interactive`, strict
  aggregation, `COMPLETED` with exit code `0:0`;
- 12/12 sealed cells and 360/360 completed episodes, with no failed or
  aborted episodes;
- 95,040 mock completion requests (264 per episode), exercising 86,400 agent
  interactions without provider spend;
- 3,600 round rows and 86,400 micro-slot rows, with no duplicate cell or
  episode identities and no missing scientific events;
- 84/84 configured CMI estimates finite and support-qualified as `adequate`
  across seven CMI metrics, using 200 whole-episode bootstrap resamples and
  200 null permutations;
- 720 primary estimates, 132 derived observables, five requested plots, and
  a readable 23-entry analysis archive.

Both controller actions were observed (`ADVOCATE_Z=2348`, `NO_OP=1252`). All
CMI metrics and `eta_ir` were supported. Some higher-order thermodynamic
outputs were explicitly marked unsupported (`eta_th`: 9/12 cells;
affinity-weighted current and its expenditure: 5/12 cells) where their
sign/denominator requirements were not identified. This is a support result,
not an aggregation failure. Strict `validation.json` reports `valid=true`,
`complete=true`, with no errors or warnings. Its duplicate-run count of one is
the expected shared logical run identity reconstructed from the 12 cell
shards; cell and episode duplicate counts are both zero.
