# Running an experiment in `mas_cc` — a manual

This is the companion to [`building_a_game.md`](building_a_game.md), one layer up. That manual is
about *one* game run; this one is about turning that same resolved config into an **experiment** —
many episodes of the same game, run concurrently, priced before you spend anything, and watchable
on the console while they run. As in the companion manual, each concept gets two passes: a
**Technical** explanation and, right after it, an **In plain terms** explanation using the same
lab-notebook analogy.

**Accompanying config:**
[`configs/runs/old/naming_convention_experiment_tutorial_university_v3.yaml`](../../../configs/runs/old/naming_convention_experiment_tutorial_university_v3.yaml) —
the exact same small game as the single-run tutorial config (4 agents, 6 interactions), turned into
a 5-episode experiment. Section 4 below shows the CLI commands that run it.

**There is a CLI for this already** — `mas-cc experiment preflight` and `mas-cc experiment run`.
Unlike the companion manual, this one *is* the CLI walkthrough; there's no separate "explicit
Python underneath" section, because the CLI here isn't hiding anything you'd want to see instead —
the interesting object (the priced, resolved experiment plan) is exactly what `preflight` prints.

## Table of contents

1. [The big picture: what an experiment adds](#1-the-big-picture-what-an-experiment-adds)
2. [The two commands](#2-the-two-commands)
3. [Configuration: what's different from a single run](#3-configuration-whats-different-from-a-single-run)
4. [Running the example](#4-running-the-example)
5. [Reading the output](#5-reading-the-output)
6. [Interrupting and resuming](#6-interrupting-and-resuming)
7. [What's different from a single `mas-cc game run`](#7-whats-different-from-a-single-mas-cc-game-run)
8. [Grid sweeps: many cells, not just many repetitions](#8-grid-sweeps-many-cells-not-just-many-repetitions)
9. [Where to go next](#9-where-to-go-next)

---

## 1. The big picture: what an experiment adds

| Layer | What it is | Where it lives |
| --- | --- | --- |
| Experiment preflight | Prices N episodes (calls/tokens/cost/runtime) before any provider call | `src/mas_cc/planning/experiment_preflight.py` |
| Orchestrator | Runs N episodes concurrently, one shared budget, one seed per episode, resumable | `src/mas_cc/experiments/orchestrator.py` |
| Console UX | The banner and the live episode/round progress bars | `src/mas_cc/experiments/console.py` |
| CLI | `mas-cc experiment preflight` / `mas-cc experiment run` | `src/mas_cc/cli/experiment.py` |

**In plain terms:** the companion manual's lab analogy had one sample run through the procedure
once. An **experiment** is the same procedure run on **many samples** (here, "sample" = one
independent playthrough of the population game, called an **episode**), with three things a single
run doesn't need:

- a **cost estimate for the whole batch**, not just one sample, checked against your budget before
  you spend anything;
- **several samples running side by side**, not one at a time, up to a concurrency limit you set;
- a **lab log you can walk away from and come back to** — if the batch is interrupted, finished
  samples stay finished; only the unfinished ones re-run.

An experiment does not change what a game *is* or how one episode plays out — everything from the
companion manual (the `Game` contract, prompts, the decision loop, metrics) is exactly the same
underneath, called once per episode. This layer only adds the batching, the pricing, the
concurrency, and the console feedback around it.

---

## 2. The two commands

**Technical:**

```bash
mas-cc experiment preflight --config <run-config.yaml> [--output-dir <dir>]
mas-cc experiment run       --config <run-config.yaml> [--output-dir <dir>] [--approve-preflight <path>] [--resume|--no-resume] [--no-progress]
```

`--output-dir` is optional on both commands: pass it to override where artifacts land, or omit it to
use the config's own `storage.output_dir` (which itself defaults to `results/`, already covered by
`.gitignore`).

`preflight` performs **no provider I/O at all** — no model call, no client, nothing billable. It
reads the config, builds the game's provider-neutral call plan, multiplies it by
`execution.repetitions`, and reports the result. `run` performs an immediate pricing revalidation
(the same live-vs-offline check the companion manual's single-run path does), then launches.

**In plain terms:** `preflight` is "how much would this experiment cost and take, on paper, before
we touch the instrument at all." `run` is "actually do it." Keeping them as two separate commands
means you can look at the bill before it exists — and, with `--approve-preflight`, prove that the
bill you looked at is the one you're actually paying, not a different config that quietly changed
in between.

---

## 3. Configuration: what's different from a single run

**Technical:** an experiment config is an ordinary `RunConfig` — the same shape the companion
manual walks through section by section — with exactly two fields doing the new work, both
already in `ExecutionConfig` (`src/mas_cc/config/models.py`):

```yaml
execution:
  seed: 20260803
  repetitions: 5      # this is what makes it an experiment instead of one run
  parallelism: 3       # at most 3 episodes in flight at once
  fail_fast: true
```

Everything else in the file is unchanged in *shape* from a single-run config, but two sections
need different *values* because they now describe the whole batch, not one sample:

| Section | Single run | Experiment | Why |
| --- | --- | --- | --- |
| `budget` | Sized for 1 episode's worth of calls/tokens/cost | Sized for `repetitions` episodes, with headroom | The budget guard is **shared across every concurrent episode** — it does not reset per episode. Five episodes really do spend five episodes' worth of the same ceiling. |
| `logging.comet` | Can be `true` | Left `false` here | `mas-cc experiment run` does not wire Comet per episode (see [section 7](#7-whats-different-from-a-single-mas-cc-game-run)) — leaving it `true` would just be a config value with no effect. |

The accompanying config's budget section, with the arithmetic spelled out:

```yaml
budget:
  accounting_unit: proxy_accounting_unit
  system_max_cost_per_run: 1.5     # was 0.25 for 1 episode; ~5x with headroom
  max_cost_per_run: 1.5
  max_provider_requests: 200        # 5 episodes x 36 conservative-per-episode = 180, +headroom
  max_input_tokens: 1000000
  max_output_tokens: 100000
  allow_unbounded_paid_requests: false
```

`36` conservative requests per episode (population 4, horizon 6, one retry budget) comes straight
out of the game's own `call_plan` — you don't have to compute it by hand; `experiment preflight`
reports it for you (section 4). The `200` above is that number times 5, rounded up with room to
spare, so a genuinely well-behaved run isn't denied by an overly tight ceiling.

**In plain terms:** think of the budget as one shared petty-cash envelope for the whole batch of
samples, not one envelope per sample. If you only refill it for one sample's worth of spending and
then run five samples against it, the fifth sample (or third, or second — whichever happens to draw
the last dollar) fails when the envelope runs dry, exactly as it should: the alternative would be
silently overspending your approved budget.

---

## 4. Running the example

The accompanying config points at the real University provider and needs `POTSDAM_API_KEY` and
`BASE_POTSDAM_LLM_URL` (same as the companion manual). Run these from the repository root.

### 4.1 Price it first — no provider call happens here

```bash
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/old/naming_convention_experiment_tutorial_university_v3.yaml \
  --output-dir inspection/experiment_tutorial/preflight
```

This writes `inspection/experiment_tutorial/preflight/report.md`, which for this config reads:

```text
- Status: PASS (permitted)
- Experiment: naming-convention-experiment-tutorial; game naming_convention; provider ... .
- Episodes: 5 (concurrency 3).
- Expected total cost: ...; conservative: ... .
- Rough total runtime: ...s.
- Preflight ID: <64-character hash> — pass this to `mas-cc experiment run --approve-preflight`...
```

Also written: `per_episode_estimate.json` (one episode's demand), `experiment_estimate.json` (all
5, multiplied), `pricing_snapshot.json`, `budget_status.json`, and `preflight_id.txt` — just the
ID, so you can hand its path straight to the next command.

### 4.2 Approve and run

```bash
conda run --live-stream -n MA-CC mas-cc experiment run \
  --config configs/runs/old/naming_convention_experiment_tutorial_university_v3.yaml \
  --output-dir results \
  --approve-preflight inspection/experiment_tutorial/preflight/preflight_id.txt
```

`--approve-preflight` is optional — `run` always does its own internal preflight check regardless —
but when you pass it, the CLI additionally recomputes the same ID from the *current* resolved
config and pricing and refuses to launch if they've drifted since you looked at the estimate:

```text
approved preflight ID does not match the current resolved config/pricing; re-run
`mas-cc experiment preflight` and approve the new estimate
```

That's the safety property, not a bug you'd hit by accident: it means the estimate you read is
provably the one being executed.

### 4.3 What you'll see while it runs

```text
Experiment: naming-convention-experiment-tutorial
  Game:          naming_convention v1
  Provider:      university / gwdg/qwen3-30b-a3b-instruct-2507
  Episodes:      5  (parallelism: 3)
  Prompt:        naming_convention_decision v1  [def:33232f05...]
  Budget:        1.50 proxy_accounting_unit
  Preflight:     expected ... / conservative ... — permitted
Episodes:  60%|██████      | 3/5 [00:42<00:28, episode]
Rounds:    58%|█████▊      | 17/30 [round 4/6 | episode naming-convention-experiment-tutorial-20260803-0002]
```

Two independent progress bars: **Episodes** counts finished episodes out of 5; **Rounds** counts
completed interactions across *all* episodes out of the total (`repetitions × horizon`), with the
postfix showing which specific episode is currently ticking. Pass `--no-progress` to replace both
bars with one log line per completed episode instead — useful in CI or anywhere stdout isn't a
terminal (this happens automatically if stdout isn't a TTY, even without the flag).

---

## 5. Reading the output

**Technical:** results land under the same convention the companion manual uses for a single run —
`results_run_dir` (`src/mas_cc/storage/results.py`) — with one new subtree:

```text
results/naming_convention/naming-convention-experiment-tutorial/naming-convention-experiment-tutorial-20260803/
├── manifest.json
├── resolved_config.yaml
├── experiment_summary.json       # completed/failed/skipped counts, preflight vs. actual budget
├── experiment_summary.csv        # one row per episode: seed, status, interactions, termination_reason
└── data/episodes/
    ├── naming-convention-experiment-tutorial-20260803-0000/
    │   ├── manifest.json          # this episode's own outcome - resume reads this
    │   ├── events.jsonl, api_call_status.jsonl, usage_cost.jsonl, budget_events.jsonl
    │   ├── audit_traces.jsonl, prompt_block_traces.jsonl
    │   ├── checkpoint_manifest.json
    │   ├── metrics/{streaming.csv, final.csv}
    │   └── comet_summary.json     # always "disabled" here - see section 7
    ├── .../0001/  ... 0004/        # same shape, one directory per episode
```

Every episode subdirectory is the **exact same file set** Phase 7 produces for one run — that's
deliberate reuse, not a parallel implementation. `experiment_summary.json`'s
`actual_budget_status` is the shared `RuntimeBudgetGuard`'s final tally across all 5 episodes
combined, directly comparable to `experiment_estimate.json` from the preflight step.

**In plain terms:** each episode gets its own lab notebook page, identical in shape to what you'd
get from running it alone — nothing about a single episode's record changes just because it was
part of a batch. The `experiment_summary` is the cover sheet stapled on top: how many samples
finished, how many failed, and whether the batch as a whole came in on budget.

---

## 6. Interrupting and resuming

**Technical:** an episode counts as done purely by the existence of its
`data/episodes/<episode_id>/manifest.json` with `"status": "completed"`. If you interrupt a run
(Ctrl-C, a crash, a machine restart) and re-run the identical command, `run` (default `--resume`)
skips every episode whose manifest already says `completed` and only executes the rest — including
retrying any episode that had failed. Pass `--no-resume` to ignore prior progress and re-run every
episode from scratch.

```bash
# interrupted partway through episode 3 of 5 - re-running the same command:
conda run --live-stream -n MA-CC mas-cc experiment run \
  --config configs/runs/old/naming_convention_experiment_tutorial_university_v3.yaml \
  --output-dir results
# -> "Episodes: 2 skipped (resumed)" for the two already-completed episodes,
#    then continues with episode 3 onward
```

**In plain terms:** think of each episode's manifest as a stamp on that lab notebook page reading
"experiment complete." Coming back to a stack of pages, you don't redo the ones already stamped —
you only pick up where the stamps stop. A page that was started but never stamped (a crash mid-episode,
or an earlier failure) is treated as not done, and gets redone.

---

## 7. What's different from a single `mas-cc game run`

Two things this layer deliberately does **not** do yet, so you're not surprised by their absence:

- **Comet is off, always, per episode.** `mas-cc game run`/the Phase 7 path can upload one run's
  metrics to Comet (`logging.comet: true`). `mas-cc experiment run` hardcodes
  `comet_enabled=False` for every episode's recorder — turning `logging.comet` on in an experiment
  config has no effect. The reason is explicit, not an oversight: naively wiring it through would
  mean N episodes become N separate remote Comet experiments for one `mas-cc experiment run`
  invocation, which nobody asked for. Aggregating properly into one Comet experiment per
  *experiment* (not per episode) is future work, not yet built.
- **`execution.repetitions` alone only replicates, it doesn't sweep.** It reruns the *same*
  resolved configuration N times with N different seeds, for statistics — not N different
  configurations. For that, see [section 8](#8-grid-sweeps-many-cells-not-just-many-repetitions).

Everything else — the game, the prompts, the decision loop, the metrics — is identical to a single
run, because it *is* the same code, called once per episode.

---

## 8. Grid sweeps: many cells, not just many repetitions

**Technical:** a `grid:` section at the top level of the same config file names one or more fields
by dotted path and lists the values to sweep. Every combination (the **cartesian product** across
all listed fields) becomes one **cell** — an ordinary experiment in its own right, with its own
resolved config and its own `execution.repetitions` episodes. `mas-cc experiment preflight`/`run`
auto-detect the `grid:` section — there is no separate grid command.

```yaml
grid:
  game.horizon: [6, 12]              # 2 values
  # llm_provider.temperature: [0.0, 0.7]   # add a second axis -> 2 x 2 = 4 cells
```

Every cell **shares one provider client, one pricing quote, and one budget guard** — the whole
grid is priced and budgeted as a single unit, not cell-by-cell. That's why a grid axis cannot
target `llm_provider.type`, `llm_provider.model`, `game.type`, `budget.*`, or `pricing.*`: sweeping
any of those would mean a different client/quote/guard per cell, which is a different, unbuilt
feature — attempting it raises a clear `ConfigurationError` naming the field. Concurrency is
shared too: `execution.parallelism` from the base config bounds *every* episode of *every* cell
combined, not parallelism-per-cell — a 2-cell, 3-parallelism grid never runs more than 3 episodes
at once, whether those 3 come from one cell or split across both. `execution.fail_fast` is
likewise grid-wide: one episode's failure aborts every not-yet-started episode in every cell, not
just the cell it happened in.

**In plain terms:** `execution.repetitions` is "run this exact recipe N times to see how much the
outcome varies." A grid is "try N *different* recipes" — vary the horizon, vary the temperature —
while still cooking out of the same shared pantry (one provider, one price list) and the same
shared oven with a fixed number of racks (one concurrency limit, one budget), so the whole batch
of recipe variations is priced and capped together, not recipe by recipe.

**Example**, sweeping the companion single-cell config's `game.horizon` over two values:
[`configs/runs/old/naming_convention_grid_tutorial_university_v3.yaml`](../../../configs/runs/old/naming_convention_grid_tutorial_university_v3.yaml) —
identical to `naming_convention_experiment_tutorial_university_v3.yaml` in every other field, plus
the `grid:` section above and a budget scaled for the combined demand of both cells.

```bash
mas-cc experiment preflight \
  --config configs/runs/old/naming_convention_grid_tutorial_university_v3.yaml \
  --output-dir inspection/grid_tutorial/preflight
```

For this config, that reports 2 cells, 6 total episodes (3 repetitions per cell), 324 conservative
provider requests (108 for the `horizon: 6` cell, 216 for `horizon: 12`, summed — not each checked
against the full budget separately, then multiplied). The launch/run commands are identical in
shape to a single-cell experiment:

```bash
mas-cc experiment run \
  --config configs/runs/old/naming_convention_grid_tutorial_university_v3.yaml \
  --output-dir results \
  --approve-preflight inspection/grid_tutorial/preflight/preflight_id.txt
```

```text
Grid experiment: naming-convention-grid-tutorial
  Game:          naming_convention v1
  Provider:      university / gwdg/qwen3-30b-a3b-instruct-2507
  Cells:         2  (game.horizon x2)
  Episodes:      6 total  (parallelism: 3, shared across every cell)
  Preflight:     expected ... / conservative ... — permitted
```

**Output layout** adds one nesting level under the same `results_run_dir` convention:

```text
results/naming_convention/naming-convention-grid-tutorial/naming-convention-grid-tutorial-20260803/
├── manifest.json
├── resolved_base_config.yaml
├── grid_summary.json                 # completed/failed/skipped per cell, plus grid-wide totals
├── grid_summary.csv                  # one row per cell: cell_id, counts, overrides
└── cells/
    ├── cell-0000/                    # game.horizon: 6
    │   ├── resolved_config.yaml       # this cell's fully resolved RunConfig
    │   ├── overrides.json             # {"game.horizon": 6}
    │   ├── cell_summary.json
    │   └── data/episodes/<episode_id>/...   # exactly the single-experiment episode shape
    └── cell-0001/                    # game.horizon: 12
        └── ... (same shape)
```

**Resuming** works exactly as in [section 6](#6-interrupting-and-resuming), just per episode
across every cell: re-running the same `run` command skips every episode whose
`cells/<cell_id>/data/episodes/<episode_id>/manifest.json` already says `completed`, cell by cell,
and only re-runs the rest.

---

## 9. Where to go next

- **The single-run companion manual:** [`building_a_game.md`](building_a_game.md) — everything an
  episode is actually doing underneath this batching layer.
- **The design doc this implements:**
  [`tdd/architecture/03082026_MAS_CC_Phase9_Experiment_Orchestration_Plan_v1.md`](../../../tdd/architecture/03082026_MAS_CC_Phase9_Experiment_Orchestration_Plan_v1.md) —
  scope, what's deferred (control policies, offline analysis), and why. Grid sweeps were added
  after this doc was written and are not yet reflected in it; this manual is the current source of
  truth for them.
  - **The real files**, for comparison: `src/mas_cc/config/grid.py`,
  `src/mas_cc/planning/{experiment_preflight,grid_preflight}.py`,
  `src/mas_cc/experiments/{orchestrator,console}.py`, `src/mas_cc/cli/experiment.py`.
- **Tests as executable examples:** `tests/mas_cc/test_experiments.py` — preflight multiplication,
  concurrent seed determinism, resume semantics, fail-fast behavior, shared-budget enforcement, all
  for a single (non-grid) experiment. `tests/mas_cc/test_grid.py` — cartesian expansion, forbidden
  axes, per-cell resume, cross-cell fail-fast abort, and (directly, by tracking concurrent calls) the
  combined-concurrency-pool property described in section 8.

**What I actually validated:** I ran the exact `preflight` → `run` sequence in section 4, with a
mock provider standing in for University (so no live network call or spend happened while writing
this manual), confirming 5 episodes complete, the shared budget accounting is correct across them,
the banner and directory layout are exactly as shown, and that re-running with `--resume` skips
completed episodes. I separately ran the grid example in section 8 the same way — with live
University pricing for the preflight numbers quoted there (108 conservative requests for the
`horizon: 6` cell, 216 for `horizon: 12`, 324 summed — genuinely free for this proxy model) and a
mock-provider substitution for the actual `run`/resume cycle — confirming all 6 episodes across
both cells complete, `grid_summary.csv` shows 3 completed / 0 failed / 0 skipped for each cell, and
re-running skips all 6 as resumed. Running the commands yourself, with your own University
credentials, is how you get a real model-driven experiment or grid.
