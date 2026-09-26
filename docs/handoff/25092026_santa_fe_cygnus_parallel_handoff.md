# Santa Fe targeted beta/CMI study: Cygnus handoff

Prepared locally on 2026-09-25. **No Cygnus job was submitted.** The synthetic
model makes no LLM/provider calls. This package includes repository-relative
source, configs, docs and tests; it excludes unrelated local edits and results.

The generic game mechanics are explained in
`docs/documentation/santa_fe_game_mechanics.md`.

## Scientific recipes

`configs/santa_fe/beta_cmi_sweep.yaml` is the final sweep:

| beta regime | beta evidence | beta social |
| --- | ---: | ---: |
| competition low | 0.75 | 1.0 |
| competition baseline | 1.0 | 1.3 |
| competition high | 1.25 | 1.6 |
| evidence dominated | 2.0 | 0.5 |
| social dominated | 0.5 | 2.0 |

Each pair is run at `rho=0.75,1.0` and `b=0,3,6,...,24` with `N=24`, `F=10`,
`q=3`, 30 rounds and 100 episodes per cell: **90 cells, 9,000 episodes and
279,000 expected canonical round rows**. Pair names are preserved in saved
rows. Four CMI variants condition on current target count alone, plus binned
mean coverage, plus binned population coverage, or plus both. The shared MA-CC
round estimator handles all four and retains the prior round-by-round sensing,
sensor/action, population and truth information statistics, with 200
whole-episode bootstrap draws and 999 null draws per pooled and per-round
estimate. The
population-only coordinate uses the estimator's `phi` key as a synthetic alias.
This does not give it HiddenBench phi semantics. Zero-budget CMI remains
undefined because no controller action is posted.

`configs/santa_fe/sample_size_cmi.yaml` is a separate calibration at beta
`(1.0,1.3)`, `b=6`, both rho values. It simulates 400 reference episodes per
rho, then uses 200 subsampling repetitions at `n=10,20,30,50,75,100,200`.
Each repetition requests 100 episode-bootstrap and 199 policy-null draws for
plain CMI. This is a 2-reference-job, 14-calibration-task, 1-finalizer Slurm
workflow. The sample-size result can be produced independently of the sweep.

## Parallel execution and results

The sweep launcher has four `afterok` stages: a 90-task simulation array,
cell aggregation, a 90-task information array, and finalization. The arrays
request four CPUs per task and throttle at eight; Slurm resources are in the
YAML. Cell trajectories and per-cell information outputs carry SHA-256 seals.
Aggregation refuses missing or changed cells. Raw null draws stay in each
cell; compact estimates are joined centrally. The targeted results are under
`results/studies/santa_fe_beta_cmi/`, including:

```text
config.yaml
execution_plan.json
cells/cell-0000/{rounds.parquet,cell_complete.json,information/}
trajectories/round_trajectories.parquet
summaries/{final_summary.csv,susceptibility.csv}
information/llm_parallel/round_information_estimates.parquet
information/beta_cmi/{detectability.csv,detectability.parquet,beta_cmi_report.md,analysis_recipe.yaml}
plots/beta_cmi/*.png
```

The calibration root is `results/studies/santa_fe_sample_size_cmi/`, with
`sample_size_cmi/sample_size_cmi_repetitions.parquet`,
`sample_size_cmi_summary.csv`, the copied recipe and detection plots. The
report reads those completed results; it does not claim a scientific answer
before they exist.

## Cygnus agent checklist

1. Verify the host identity and the site's current checkout, Python and
   shared result path. Read `docs/handoff/cygnus-end-to-end.md`, the
   `ma-cc-study-workflow` skill and the `ma-cc-cygnus-study-workflow` skill.
   Do not use Potsdam Python paths. This local checkout could not verify the
   Cygnus environment because read-only SSH stopped at host-key verification;
   do not bypass host-key checking.
2. Extract `results/santa_fe_cygnus_handoff_20260925.tar.gz` into the
   established Cygnus checkout. Verify its SHA-256 checksum from the transfer.
   Confirm the selected Python imports `santa_fe`, `mas_cc`, `pandas`,
   `pyarrow`, `matplotlib` and `yaml`.
3. Run focused tests, dry plans and shell syntax checks. Use the actual
   resolved Python path and checkout path; the example below shows the
   expected layout:

   ```bash
   export ROOT="$HOME/MA-CC-cygnus"
   export PYTHONPATH="$ROOT/src"
   export PY="$HOME/.local/share/mamba/envs/MA-CC/bin/python"
   "$PY" -m pytest "$ROOT/tests/mas_cc/test_santa_fe.py" -q
   "$PY" -m santa_fe.cli --config "$ROOT/configs/santa_fe/beta_cmi_sweep.yaml" --dry-run
   "$PY" -m santa_fe.cli --config "$ROOT/configs/santa_fe/sample_size_cmi.yaml" --dry-run
   "$PY" -m santa_fe.cluster plan-cygnus --config "$ROOT/configs/santa_fe/beta_cmi_sweep.yaml" --python "$PY" --repository-root "$ROOT"
   "$PY" -m santa_fe.sample_size_cmi plan-cygnus --config "$ROOT/configs/santa_fe/sample_size_cmi.yaml" --python "$PY" --repository-root "$ROOT"
   bash -n "$ROOT/results/studies/santa_fe_beta_cmi/submit_cygnus.sh"
   bash -n "$ROOT/results/studies/santa_fe_sample_size_cmi/sample_size_cmi/submit_sample_size_cygnus.sh"
   ```

4. Inspect both generated launchers and recipes for paths, arrays, throttle,
   resource requests, logs and dependencies. Planning writes files only.
   A scheduler smoke should use a tiny separate config/result root.
5. Once the Cygnus operator authorizes full execution, run both generated
   scripts and monitor `squeue`/`sacct` and root-local logs. Do not use
   `sbatch --export`; launchers carry the resolved Python/checkout paths.
6. Verify 90 simulation seals, 90 information seals, 279,000 canonical round
   rows, 360 beta/CMI detectability rows, 14 completed sample-size tasks and
   2,800 calibration repetitions. Inspect action support and null-adjusted CMI
   before interpreting any apparent maximum.
7. After calibration completes, refresh the beta report:

   ```bash
   "$PY" -m santa_fe.beta_cmi --config "$ROOT/configs/santa_fe/beta_cmi_sweep.yaml" --sample-size-summary "$ROOT/results/studies/santa_fe_sample_size_cmi/sample_size_cmi/sample_size_cmi_summary.csv"
   ```

The generic MA-CC `mas-cc study` launcher cannot consume this synthetic
config or its cell files. These dedicated array commands use the existing
Santa Fe cell runner and the existing MA-CC information engine; there is no
study-specific `.job` file.
