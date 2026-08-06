# Running the empowerment phase diagram on Slurm

A short note on
[`scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job`](../../../scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job) —
how to submit it, not what it computes. For the concepts underneath (`grid:`, cells, the MI
block, Comet), see [`what_to_run.md`](what_to_run.md) and
[`running_an_experiment.md`](running_an_experiment.md); this doc only covers the Slurm wrapper.

## What it runs

One arm at a time of the naming-convention empowerment phase diagram — population size
`{10, 20}` swept via `grid:`, against the live University proxy — through the same three
steps every grid here goes through: `mas-cc experiment preflight` → `mas-cc experiment run` →
`mas-cc analysis empowerment`.

| `RUN_MODE` | Config | Response contract |
| --- | --- | --- |
| `reasoning` (default) | [`configs/runs/empowerment_runs/naming_convention_empowerment_reasoning.yaml`](../../../configs/runs/empowerment_runs/naming_convention_empowerment_reasoning.yaml) | `json_reason` — agents return a short explanation with their choice |
| `choice-only` | [`configs/runs/empowerment_runs/naming_convention_empowerment_choice_only.yaml`](../../../configs/runs/empowerment_runs/naming_convention_empowerment_choice_only.yaml) | `choice_only` — bare `Q`/`M`, no explanation |

Submit both to get the full 2×2 (size × reasoning) picture — see either config's header for
why that split is two files rather than a second grid axis on one.

## Prerequisites

- Repo checked out on the cluster, with a `.env` at its root carrying `POTSDAM_API_KEY` and
  `BASE_POTSDAM_LLM_URL`. The job checks for this file and fails fast if it's missing.
- A `MA-CC` conda env with `mas-cc` installed (`conda run -n MA-CC mas-cc version` should
  print something, not an import error).

## Submitting

```bash
sbatch --export=ALL,RUN_MODE=reasoning    scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job
sbatch --export=ALL,RUN_MODE=choice-only  scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job
```

## Overriding paths and analysis knobs

The job derives `REPO_ROOT` from its own file location, so it works under any account's
checkout without editing the script. `RESULTS_ROOT` and `CONDA_BIN` default from that but can
be overridden at submit time — do this if your account writes large output to a `/work`
scratch mount instead of your home directory:

```bash
sbatch --export=ALL,RUN_MODE=reasoning,RESULTS_ROOT=/work/<you>/mas-cc-results \
  scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job
```

The offline analysis pass (step 3) takes the same knobs `empowerment_grid.job` does:

```bash
sbatch --export=ALL,RUN_MODE=reasoning,NULL_PERMUTATIONS=2000,HORIZONS="1 2" \
  scripts/Potsdam/SLURM/naming_convention_empowerment_phase_diagram.job
```

## What you get, and where

Output lands at `${RESULTS_ROOT}/naming_convention_empowerment_<arm>/naming_convention/<experiment-name>/<experiment-name>-20260806/`
— printed as `grid_dir` in the job's own log. Read it the same way as any grid run
([`what_to_run.md` §8](what_to_run.md#8-reading-the-output-directory)): `cells/cell-000{0,1}/aggregate.json`
for the per-size scalars, `analysis/mi_estimates.csv` for the bootstrap-CI mutual information.

Every step prints its own duration (`[time] preflight: 00h:02m:14s`, etc.), and the last line
of the log is the total:

```
[done] total_wall_clock: 00h:41m:03s (2463s)
```

## Resuming

The job does **not** pass `--no-resume`. If it hits the wall-clock limit or gets killed,
re-submitting the same `RUN_MODE` picks up wherever `manifest.json` says each episode left off
— delete the run's `GRID_DIR` first if you actually want a clean re-run instead.

## Do not pass `--approve-preflight`

With `pricing.mode: live` (both configs use it), the preflight ID hashes provenance timestamps
that change on every re-fetch, so an approval taken from step 1 can never match step 2's
re-fetch. The job deliberately omits the flag; `experiment run` re-validates pricing and
re-checks the budget internally regardless, so nothing is skipped by leaving it out.
