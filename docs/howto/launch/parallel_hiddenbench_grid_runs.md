# Running HiddenBench grids in parallel

This note records what we learned while launching the round-feedback classical
and GPT-OSS grids on GWDG/Potsdam SLURM.

## Why `execution.parallelism` was not enough

`execution.parallelism` schedules episodes concurrently inside one Python
process. This works well when episodes mostly wait for LLM responses, but it
does not spread CPU-bound classical dynamics across CPU cores. In the
classical test, a 20-CPU allocation consumed approximately one core because
Python threads still share the GIL. Raising `parallelism` or requesting more
CPUs therefore did not materially accelerate it.

The effective solution was **process-level cell sharding**: run each original
grid cell as an independent SLURM array task. Each task gets its own Python
process and CPU core. A 24-cell classical grid that was projected to take many
hours finished in roughly 20–40 minutes per shard.

## Files defined for this workflow

- `scripts/experiment_design/run_classical_grid_cell.py`
  - Generic seed-preserving grid-cell runner (the historical filename says
    `classical`, but it also supports provider-backed grids).
  - Selects an existing `GridCell` instead of reconstructing a one-cell grid.
    This preserves the original cell index, cell ID, and episode seeds.
- `scripts/Potsdam/SLURM/hidden_bench_classical_grid_cell_array.job`
  - One CPU-bound classical cell per SLURM array task.
  - Uses one CPU and 4 GB per shard; scaling comes from array processes.
- `scripts/Potsdam/SLURM/hidden_bench_llm_grid_cell_array.job`
  - Provider-backed variant with proxy and environment checks.
  - Must always be submitted with an array throttle.
- `scripts/Potsdam/SLURM/hidden_bench_imitation_round_feedback_grid.job`
  - Original single-process grid launcher. Keep using it when one shared Comet
    run, one shared budget guard, or simple resume behavior matters more than
    throughput.

## Classical launch

For a 24-cell provider-free grid:

```bash
sbatch --array=0-23 \
  --job-name=hb_classical_shard \
  --output=/work/ojedamarin/Projects/LanguageGames/MA-CC/logs/hb_classical_%A_%a.out \
  --error=/work/ojedamarin/Projects/LanguageGames/MA-CC/logs/hb_classical_%A_%a.err \
  scripts/Potsdam/SLURM/hidden_bench_classical_grid_cell_array.job \
  path/to/config.yaml experiment_label
```

Use the exact zero-based range `0-(cell_count-1)`. Each shard writes to a
separate directory below:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/
  imitation_round_feedback_sharded_<label>/shard_NNNN/
```

Never point multiple processes at the same result directory. Concurrent
checkpoint, budget, aggregate, and Comet writes are not safe.

## LLM launch and rate limiting

Every shard creates its own provider client, so
`llm_provider.request_concurrency` applies **per shard**, not globally. Bound
the maximum simultaneous requests as:

```text
active array shards × request_concurrency per shard
```

For the GPT-OSS steerability study we used:

- controlled array: `--array=0-29%4`;
- no-control array: `--array=0-4%1`;
- provider concurrency: 4 per process;
- combined maximum: `(4 + 1) × 4 = 20` simultaneous requests.

At the observed 10-second latency, that is roughly 120 requests/minute. Even at
one-second latency it is at most roughly 1,200 requests/minute, below the
reported GPT-OSS allowance of 2,000 requests/minute. This margin also reduces
HTTP 500 errors and retry bursts.

Example:

```bash
sbatch --array=0-29%4 \
  --job-name=gptoss_grid \
  --output=/work/ojedamarin/Projects/LanguageGames/MA-CC/logs/gptoss_%A_%a.out \
  --error=/work/ojedamarin/Projects/LanguageGames/MA-CC/logs/gptoss_%A_%a.err \
  scripts/Potsdam/SLURM/hidden_bench_llm_grid_cell_array.job \
  configs/runs/imitation_round_feedback/gpt-oss-steer-test.yaml \
  gpt_oss_steer_test
```

Do not infer safety from the advertised RPM alone. Keep a concurrency margin,
retain retries and timeouts, inspect the first budget-state updates, and check
logs for HTTP errors before leaving the array unattended.

## Preflight and population preparation

Before submission:

1. Read task IDs from `grid.game.options.task_id` in the actual config.
2. Prepare/validate each task in
   `data/hidden_bench/scaled/paraphrased_replication/N_24.json` using
   `ensure_paraphrased_population.py`.
3. Run `mas-cc experiment preflight` on the complete source grid.
4. Confirm cell count, episode count, pricing, token ceilings, and launch
   permission.
5. Submit the array and verify running-task counts, Comet initialization,
   successful request accounting, and the first trajectory files.

Do not maintain a separate hard-coded task list for population preparation;
derive it from the configs so it cannot drift.

## Trade-offs and result collection

Sharding improves throughput and isolates failures to one cell, but it changes
the operational shape of the run:

- each cell has its own result root, budget state, and Comet experiment;
- there is no automatic single-grid aggregate across shard directories;
- final packaging or postprocessing must inventory the completed shards and
  combine their per-cell outputs;
- an analysis-now snapshot must copy only finalized shards, never active ones.

The scientific behavior is unchanged: configs, prompts, model calls, cell IDs,
and episode seeds are preserved. The change is scheduling, not simulation
logic.
