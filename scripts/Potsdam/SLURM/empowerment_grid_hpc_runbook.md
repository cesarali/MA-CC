# Empowerment grid on the Potsdam HPC — runbook

Companion to `empowerment_grid.job`. Written for an agent that **has** access to
the university cluster, by one that **does not**: everything below about the
code and the science was verified by running it locally; everything about the
cluster is unverified and marked as such.

---

## 1. What this job actually computes

The quantity is **mutual information between an intervention and an outcome**:

```
I(C ; O)      C = control.options.forced_value  (which word a committed agent is held to)
              O = the episode's terminal macrostate (which word won)
```

plus a lagged conditional form `I(C ; macrostate_{t+h} | macrostate_t)`.

**The single most important structural fact:** MI is *not* an episode-level
quantity and *not* a run-level metric. It does not appear in any
`metrics/streaming.csv`. Each episode contributes **one cell** to a contingency
table, and the estimate only exists **across the grid**. That is why the job has
three steps, not one:

| step | command | produces |
|---|---|---|
| 1 | `mas-cc experiment preflight` | cost/token estimate, no network I/O |
| 2 | `mas-cc experiment run` | N episodes across 2 control cells |
| 3 | `mas-cc analysis empowerment` | `analysis/mi_estimates.csv` ← the MI |

Skipping step 3 produces a directory full of episodes and **no mutual
information at all**. This is the most likely way to think the job succeeded
when it did not.

---

## 2. Before you submit — things I could not verify

The job's header was inherited from the old `empowerment_experiment.job` and
carries paths I have no way to check. **Verify each of these before the first
submit:**

- `#SBATCH --chdir=/home/ojedamarin/Projects/LanguageGames/MA-CC` — does this
  repo path exist for the submitting user? The local checkout is at
  `/home/cesarali/LanguageGames/MA-CC`, so the cluster user may differ.
- `RESULTS_ROOT=/work/ojedamarin/Projects/LanguageGames/MA-CC/results` — writable?
- `CONDA_BIN=/home/ojedamarin/.local/share/miniforge3/bin/conda` — exists, and
  does `conda run -n MA-CC mas-cc version` work?
- `#SBATCH --output=/work/ojedamarin/Projects/LanguageGames/slurm-%j.out` — the
  directory must already exist or the job fails before running anything.
- Proxy variables (`proxy2.uni-potsdam.de:3128`) — inherited unchanged from the
  old job; confirm still correct, since the University LLM proxy *and* Comet
  both need egress.
- `--time=1-00:00:00` — I set this down from the old job's 4 days. The pilot as
  configured is ~10 minutes of wall clock. Raise it if you scale up.

Also confirm the package is installed in the `MA-CC` env such that the `mas-cc`
console script resolves (`conda run -n MA-CC mas-cc version`). The old job
called `naming-game experiment`, a **different, now-superseded CLI**. If
`mas-cc` is missing, the env predates this work and needs `pip install -e .`.

---

## 3. Submit sequence

### Step A — free rehearsal (do this first, always)

```bash
sbatch --export=ALL,RUN_MODE=mock scripts/Potsdam/SLURM/empowerment_grid.job
```

Mock provider, offline pricing, no network calls, Comet forced off. Exercises
grid expansion → control layer → parallel episodes → metrics → MI analysis.
Takes well under a minute of compute.

**Expected result: terminal MI ≈ 0.0.** That is correct and is the point — the
mock answers "Q" unconditionally, so the forced word cannot influence the
outcome. A large MI here means something is broken. Note this is a *degenerate*
zero (the outcome only ever takes one level), so it proves plumbing, not
estimation quality.

### Step B — live pilot

```bash
sbatch --export=ALL,RUN_MODE=pilot scripts/Potsdam/SLURM/empowerment_grid.job
```

20 episodes (2 cells × 10), N=3 agents, University proxy. Resumable: re-submitting
after an interruption skips episodes whose `manifest.json` says `completed`.

Tunable at submit time without editing files:

```bash
sbatch --export=ALL,RUN_MODE=pilot,NULL_PERMUTATIONS=2000,HORIZONS="1 2 3 5" \
  scripts/Potsdam/SLURM/empowerment_grid.job
```

---

## 4. Monitoring while it runs

Two independent channels, and they are **deliberately at different levels**:

**Run-level Comet (new).** One remote experiment for the *whole grid*, named
`<experiment.name>-<seed>`, e.g. `naming-convention-empowerment-pilot-20260805`.
Controlled by `logging.comet` in the config. It receives, each time an episode
finishes:

- `episodes_completed`, `episodes_failed`, `episodes_skipped_resumed`,
  `episodes_skipped_aborted`, `episodes_finished`, `progress_fraction`
- `cell_<cell_id>_completed` — per-cell progress, so you can see one arm lagging
- `budget_requests`, `budget_input_tokens`, `budget_output_tokens`, `budget_cost`

The x-axis (`step`) is *episodes finished*, which stays monotonic even though
episodes complete out of order under `execution.parallelism`.

**Episode-level Comet is OFF and stays off** on this path — `_execute_episode`
hard-codes `comet_enabled=False` so a 20-episode grid does not fan out into 20
remote experiments. Turning on run-level monitoring does not turn that on. If
you want per-round curves for one episode, run it through
`mas-cc game episode`, which is a different code path.

**Slurm log.** The job passes `--no-progress` because a Slurm log is not a TTY;
tqdm bars would render as noise. You get one line per finished episode instead.

**After the run**, `<GRID_DIR>/comet_run_summary.json` records what actually
happened remotely:

| `status` | meaning |
|---|---|
| `active` | published successfully; `reference` is the Comet experiment key |
| `disabled` | `logging.comet: false` — expected for mock mode |
| `unavailable` | enabled but no `COMET_API_KEY` reachable; **run still succeeded** |
| `error` | connected but failed to close cleanly |

A Comet failure never aborts the run — verified locally for both the disabled
and the no-key paths. This is deliberate: an unattended cluster job must not die
because a metrics service is unreachable.

⚠️ **Comet credential footgun.** The sink falls back to reading `COMET_API_KEY`
out of `${REPO_ROOT}/.env` when the environment variable is unset. The repo's
`.env` **does** contain a real key. So any run started from the repo root with
`logging.comet: true` publishes for real. The job blanks the variable explicitly
in mock mode; keep that guard if you refactor. (There is prior history here — a
mock-provider validation once triggered a real Comet upload.)

---

## 5. Known issues — read before debugging

### 5.1 Do not use `--approve-preflight` with live pricing

It **cannot** succeed. `compute_grid_preflight_id`
(`src/mas_cc/cli/experiment.py`) hashes the entire pricing snapshot including
`retrieved_at` / `fresh_until`. With `pricing.mode: live` the quote is re-fetched
at launch, those timestamps change, and the ID never matches:

```
approved preflight ID does not match the current resolved config/pricing
```

Meanwhile the orchestrator's own revalidation (`_pricing_terms`) deliberately
*strips* `source`/`retrieved_at`/`version` before comparing — the correct
behaviour. The two checks disagree about what "the same price" means.

The job therefore omits the flag. **Safety is not reduced:** `run_experiment_grid`
runs `static_grid_preflight` internally and refuses to launch unless it returns
`permitted`, and the runtime budget guard stays active throughout.

*Fix, if someone wants it:* have `compute_preflight_id` /
`compute_grid_preflight_id` hash the same provenance-stripped view that
`_pricing_terms` already builds.

### 5.2 `label_swap_invariance.csv` can report `False` on valid data

`swap_condition_and_outcome_labels` (`src/mas_cc/analysis/surrogates.py`)
relabels a **random half** of episodes rather than all of them. That changes the
marginals, which mechanically changes the MI ceiling — locally, a perfectly
determined table went from balanced 3/3 to 2/4, dropping the maximum achievable
MI from 1.000 to H(⅓,⅔) = 0.918 bits, and the check flagged non-invariance on
data that was completely clean. **Do not treat `invariant_within_tolerance: False`
as evidence of a problem** without checking the marginals first.

### 5.3 Grid path must match the config

The job derives `GRID_DIR` from hardcoded `EXPERIMENT_NAME` and `SEED`. If
anyone edits `experiment.name` or `execution.seed` in the YAML without updating
the job, step 3 fails with `No cells/ under ...`. The job checks for this
explicitly rather than letting the analysis produce an empty result.

---

## 6. Reading the results

```
<GRID_DIR>/
├── grid_summary.csv / .json          per-cell completed/failed counts
├── comet_run_summary.json            remote monitoring status
├── resolved_base_config.yaml         exactly what ran
├── cells/cell-0000/
│   ├── overrides.json                ← this cell's condition value
│   └── data/episodes/<id>/metrics/streaming.csv   ← the MI's raw input
└── analysis/
    ├── mi_estimates.csv              ← THE RESULT
    ├── null_results.csv              permutation / circular-shift nulls
    └── label_swap_invariance.csv     see 5.2
```

**Read `mi_estimates.csv`, not the command's printed summary.** The CLI prints
only `terminal_mi_jeffreys`, which is the most heavily shrunk of three
estimators. The CSV carries all three:

- `unsmoothed` — plugin/MLE, no prior
- `jeffreys` — Jeffreys prior, what gets printed
- `miller_madow` — bias-corrected; **can exceed the theoretical ceiling** at
  small n (observed 1.12 bits on a binary pair, where 1.0 is the maximum)

**How far apart the three are is your sample-size diagnostic.** At n=6 locally
they read 1.000 / 0.456 / 1.120 — useless. At n=20 they read 0.214 / 0.170 /
0.178 — usable. Convergence between them is the signal that n is adequate.

Compute the empirical p-value yourself from `null_results.csv`; the CLI does not
print one:

```python
import pandas as pd
e = pd.read_csv("analysis/mi_estimates.csv")
n = pd.read_csv("analysis/null_results.csv")
row = e[e.statistic == "terminal"].iloc[0]
g = n[n.statistic == "terminal"]
print("p =", (g.jeffreys >= row.jeffreys).mean())
```

Be aware the null is **quantized**: with 20 episodes and a binary condition,
label permutation yields only a handful of distinct MI values, so p-values sit
on a lattice. Locally the null's 95th percentile landed *exactly* on the observed
value. Do not read p = 0.054 as meaningfully different from 0.05 at this n.

---

## 7. ⚠️ Open decision — the pilot config is knowingly under-sized

**Resolve this before spending cluster time on the live pilot.**

`naming_convention_empowerment_university_v3.yaml` currently has
`game.horizon: 15` (5 population rounds at N=3). The local 20-episode run showed
this is **too short**:

- Only **9 of 20** episodes reached success-rate consensus within the horizon.
- Those that did converged at interactions **12, 12, 12, 12, 13, 14, 14, 15, 15**
  — every one of them at the buzzer.
- Among the 9 converged episodes, the committed agent determined the outcome
  **9 times out of 9** (perfect, MI = 1 bit).
- Among the 11 unconverged, there was **no relationship at all**.

The reported 0.17–0.21 bits is therefore perfect empowerment diluted by episodes
that had not finished — it measures the horizon, not the system. The permutation
test came back p = 0.054, i.e. not clearing the null, for the same reason.

**Recommended change before the live pilot:**

```yaml
game:
  horizon: 45      # 15 population rounds at N=3, was 15
```

Budget scales with it — `max_provider_requests` should go 2000 → 6000, and
`max_output_tokens` 300000 → 900000. Cost on the University proxy measured 0.00
accounting units, so the real constraint is wall clock (~10 min → ~30 min).

I did not make this change, because it is a scientific-design decision for the
project owner, not a mechanical fix. **Confirm with them before submitting the
pilot**, or you will spend cluster time reproducing a known artifact.

---

## 8. Scientific caveats worth carrying forward

- **Committed fraction is large.** One forced agent out of three is a 33%
  minority, well above the Ashery tipping point. Expect near-ceiling empowerment
  once the horizon is adequate. Good as a plumbing signal; not yet an
  interesting measurement. The interesting sweep is `control.options.agent_ids`
  (minority size), to find where empowerment collapses.
- **Asymmetry observed.** Locally, forcing "M" succeeded 9/10 while forcing "Q"
  succeeded 6/10 — the model has a baseline pull toward M. At n=10 per cell this
  is suggestive only, but it is a real candidate finding worth powering properly.
- **The lagged conditional MI showed nothing** (p = 0.14–0.56 at h=1,2,3).
  Expected: conditioning on the current macrostate absorbs most of the control's
  influence. Do not read its absence as failure.
