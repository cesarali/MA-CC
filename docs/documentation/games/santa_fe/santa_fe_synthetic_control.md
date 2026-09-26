# Santa Fe synthetic control experiment

This is a synthetic reference model. It makes no LLM or provider calls. The simulator
is in `src/santa_fe/game.py`; it generates trajectories only. Analysis modules read
the saved round trajectories after generation.

For a step-by-step explanation of the generic game, read
[How the Santa Fe synthetic-control game works](santa_fe_game_mechanics.md).

## Run

Use the local `MA-CC` Conda environment:

```bash
conda run -n MA-CC python -m santa_fe.cli --config configs/santa_fe/exploratory.yaml --dry-run
conda run --live-stream -n MA-CC python -m santa_fe.cli --config configs/santa_fe/exploratory.yaml
conda run --live-stream -n MA-CC python -m santa_fe.cli --config configs/santa_fe/study.yaml
```

The exploratory config has 3 budget cells and 20 episodes per cell. It produces
trajectories, final summaries, evolution and budget plots, and a descriptive
`report.md`. Statistical analysis is disabled. The study config has 144 cells,
100 episodes per cell, and enables permutation and bootstrap analysis. It can be
expensive in CPU time. Always inspect `--dry-run` output first.

Results go exactly to `output.results_dir`, resolved relative to the current
working directory. A nonempty result directory is refused to prevent accidental
overwrite. Each run copies its YAML to `<results_dir>/config.yaml` and records
cell and seed metadata in `metadata.json`.

## Output

```text
<results_dir>/
  config.yaml
  metadata.json
  trajectories/round_trajectories.parquet
  trajectories/micro_trajectories.parquet    # when enabled
  summaries/final_summary.csv
  summaries/information_summary.csv         # when enabled
  summaries/susceptibility.csv               # when enabled
  summaries/susceptibility_strata.csv        # supported state-local rows
  nulls/null_summary.csv                     # when enabled
  nulls/null_samples.parquet                 # when enabled
  bootstrap/bootstrap_summary.csv           # when enabled
  sample_size/reference_cell_*.parquet      # when enabled
  sample_size/no_control_cell_*.parquet      # when enabled
  sample_size/sample_size_repetitions.parquet
  sample_size/sample_size_summary.csv
  plots/
  report.md
```

Round and micro rows have `cell_id` and `seed`. Both `budget_fraction` and the
integer `budget` are retained. `controller_U` is the policy decision; at budget
zero it can be 1 although no message is posted. `controller_effective_U` is 1
only when a controller message is posted. Analysis uses the effective action.
`budget_used` is the count of controller posts for the next round.

For information analysis, `X_t` is the binned target share after round `t`,
`Y_t` is the controller's observation of that closed round, and `U_t` is the
effective action posted for the next round. `X_{t+1}` is the next round's target
share. Round 0 and the final round are excluded from transition estimates.
The selected coarse epistemic coordinate is mean fact coverage, population
fact coverage, or both. These are finite-sample plugin estimators and may be
biased or undefined when conditioning strata lack action overlap.

The MI null shuffles observations across transitions. Conditional nulls shuffle
actions **within** discrete `X_t` or `(X_t, kappa_t)` strata. The conditional
null cannot be configured to use global action shuffling. P-values use the
finite-draw `(1 + exceedances)/(1 + draws)` convention. The bootstrap resamples
whole episodes; missing susceptibility intervals remain missing when action
overlap is absent. These statistics describe association, not causal effects.

To enable sample-size calibration, set `sample_size_study.enabled: true` in a
copy of the study YAML. It generates independent reference and no-control
trajectories for every cell, then repeatedly subsamples episodes. The
`sample_size_summary.csv` reports bias against the synthetic reference,
variance, CI width and reference coverage. Action-information false-positive
rates use *sham policy decisions* from runs with zero posts; natural MI between
sensor and state is not called a false-positive control effect. With very
sparse cells these rates are missing because action overlap is unavailable.

## Semantics and limitations

Each microscopic update samples the previous round's front-page board. Facts
survive with probability `rho`; sampled board facts can be acquired. The
controller senses the closed current board and posts target-vote recommendations
for the next round. It never injects facts. A per-round budget bounds the number
of controller posts. The exploratory report only describes realized simulation
outcomes. No model claim follows from its plots alone.

## LLM-parallel information analysis by round

The post-hoc adapter applies the established MA-CC
`round_information_analysis` engine to the saved synthetic trajectories. It
runs independently of simulation and can be repeated without generating new
episodes:

```bash
conda run --live-stream -n MA-CC python -m santa_fe.llm_parallel \
  --config configs/santa_fe/exploratory.yaml
```

`analysis.llm_parallel.processes` sets the number of analysis workers.
Independent round groups run in separate processes; each group has a stable
seed, so worker scheduling does not change results. The exploratory recipe uses
100 bootstrap draws and 100 null draws; the full-study recipe requests 1000 of
each. The `--config` YAML is copied to the analysis output, along with a SHA-256
hash of the source trajectory file.

For each physical cell, the adapter estimates sensing MI at rounds
`t=1,...,R` and transfer-style actuation CMI at transition rounds
`t=1,...,R-1`, then estimates the whole cell once from **pooled
round observations**. The pooled estimate is calculated from those observations;
it is not an average of per-round estimates. Its output is under
`<results_dir>/information/llm_parallel/`:

- `round_information_estimates.parquet` and `.csv`: observed values, Jeffreys
  and Miller–Madow variants, episode bootstrap intervals, null summaries,
  p-values, and support diagnostics by cell, statistic, and round;
- `round_information_nulls.parquet`: individual sensing-permutation or
  policy-resampling draws;
- `plots/`: one time-series figure per available cell/statistic;
- `round_information_report.md`, `analysis_config.json`, and
  `analysis_recipe.yaml`: readable results and provenance.

The synthetic mapping is exact at its own clock: `X_t` is the population vote
count after round `t`; `Y_t` is the sensed closed-board vote count; `U_t` is the
recommendation posted for the next board; and `X_{t+1}` is the next population
vote count. The transfer-style statistic is
`I(U_t; X_{t+1} | X_t)`, with optional binned fact-coverage conditioning.
The extra coverage coordinates are synthetic proxies; they are not identical
to the LLM game's epistemic variables. Zero-budget cells have no physical
controller action and therefore publish sensing MI but no actuation CMI.
The null for actuation redraws each action from its recorded policy probability,
while sensing MI uses a sensor permutation. These are the same null procedures
used by the LLM round engine. A time-resolved CMI with sparse action support
is descriptive and should not be read as a causal effect. With only 20 exploratory
episodes, per-round direct-counting estimates can have substantial finite-sample
bias; compare them with the null mean and support counts.

## Cygnus Slurm arrays

`configs/santa_fe/study_cygnus.yaml` has the full 144-cell design and separate
Slurm policies for simulation, information analysis, and finalization. The
resumable array implementation is `santa_fe.cluster`. On Cygnus, run
`python -m santa_fe.cluster plan-cygnus` after verifying the checkout, Python,
and shared result root. It writes a reviewable four-stage `submit_cygnus.sh`;
planning itself does not submit a job. The detailed site handoff is
`docs/handoff/25092026_santa_fe_cygnus_parallel_handoff.md`. A transfer bundle
of the Santa Fe files is at `results/santa_fe_cygnus_handoff_20260925.tar.gz`.

## Targeted beta/CMI study

`configs/santa_fe/beta_cmi_sweep.yaml` defines exactly five named beta pairs,
`rho={0.75,1.0}`, and nine budgets: **90 cells × 100 episodes = 9,000
simulation episodes**. The beta pairs remain paired rather than expanding into
a Cartesian grid. `santa_fe.cluster` executes each cell separately in a Slurm
array and seals its trajectory and information outputs. The analysis uses the
same `round_information_analysis` engine as the LLM study for plain target CMI
and three extra binned coverage conditionings: mean, population, and both.
The prior sensing and other round-by-round information statistics remain in
the targeted recipe.
The synthetic adapter uses the engine's `phi` key for population coverage;
this is an alias only, with no HiddenBench phi interpretation. The shared
engine's actuation null resamples actions from the recorded policy probability.

After aggregation, `information/beta_cmi/detectability.csv` has one row per
cell and conditioning variant with observed CMI, null mean/SD/95th percentile,
empirical p-value, null-corrected CMI, z-score, action counts/support and rate,
susceptibility, and final shares. Zero-budget CMI is undefined because there is
no physical action. `plots/beta_cmi/` contains budget curves for the seven
requested metrics and rho-faceted beta-regime heatmaps. The generated report
uses measured cells and marks the 50-episode question as pending until the
separate calibration is available.

`configs/santa_fe/sample_size_cmi.yaml` fixes the baseline beta pair and
budget `b=6` at both rho values. Its separate Slurm workflow simulates 400
reference episodes per rho, then runs 200 independent subsampling repetitions
at each episode count `[10,20,30,50,75,100,200]`. Each repetition calls the
same round estimator with episode bootstrap and policy-resampling null. The
summary gives estimate and corrected-estimate variance, p-value quantiles,
confidence interval width, and detection probability. These are repeated
subsamples of one reference population, so their probabilities describe that
synthetic reference, not universal LLM power.

On Cygnus, generate both launchers after validating site paths:

```bash
python -m santa_fe.cluster plan-cygnus --config configs/santa_fe/beta_cmi_sweep.yaml --python "$PY" --repository-root "$ROOT"
python -m santa_fe.sample_size_cmi plan-cygnus --config configs/santa_fe/sample_size_cmi.yaml --python "$PY" --repository-root "$ROOT"
```

Planning writes scripts and copies recipes but does not submit jobs. After both
jobs finish, refresh the beta report with:

```bash
python -m santa_fe.beta_cmi --config configs/santa_fe/beta_cmi_sweep.yaml --sample-size-summary "$ROOT/results/studies/santa_fe_sample_size_cmi/sample_size_cmi/sample_size_cmi_summary.csv"
```

## Versioned v3 microscopic model

The earlier sections document the historical `santa_fe_legacy_v2` studies.
The opt-in v3 model changes the persistence clock and makes peer and controller
facts vote-aligned. Read the [v3 game mechanics](santa_fe_game_mechanics.md)
and the [v3 implementation handoff](../../../handoff/santa_fe_v3_implementation.md)
before interpreting its retained trajectories or designing a new sweep. The
small `configs/santa_fe/v3_pilot.yaml` recipe has a local validation report;
the earlier 90-cell Cygnus beta/CMI recipe remains a legacy-v2 recipe.
