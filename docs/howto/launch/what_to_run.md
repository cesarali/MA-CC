# What to run, and why — the map

Every other doc in this folder explains one thing well. This one answers the question
you actually have when you sit down: **which command, on which config, and what comes
out.** It duplicates nothing; it points at the others.

If you read only one section, read [§1](#1-the-single-fact-that-removes-most-of-the-confusion).

1. [The single fact that removes most of the confusion](#1-the-single-fact-that-removes-most-of-the-confusion)
2. [What makes a config "one episode" vs "many"](#2-what-makes-a-config-one-episode-vs-many)
3. [The command map](#3-the-command-map)
4. [Why `synthetic` has its own commands, and what `empowerment` really is](#4-why-synthetic-has-its-own-commands-and-what-empowerment-really-is)
5. [Three configs you can run right now](#5-three-configs-you-can-run-right-now)
6. [Turning Comet on, and checking it worked](#6-turning-comet-on-and-checking-it-worked)
7. [More than two axes: what the grid picture can and cannot do](#7-more-than-two-axes-what-the-grid-picture-can-and-cannot-do)
8. [Reading the output directory](#8-reading-the-output-directory)

---

## 1. The single fact that removes most of the confusion

**How many episodes run is decided by the command, not by the config.** The same file can
be run three ways:

| You run | Episodes | What it uses |
| --- | --- | --- |
| `mas-cc game episode --config X` | exactly **1** | ignores `execution.repetitions` |
| `mas-cc experiment run --config X` | `execution.repetitions` | one cell |
| `mas-cc experiment run --config X` where X has a `grid:` section | cells × `execution.repetitions` | the cartesian product of the axes |

Two ready-to-paste examples for the first two rows, both on
[`configs/runs/old/naming_convention_smoke_test_v3.yaml`](../../../configs/runs/old/naming_convention_smoke_test_v3.yaml) —
a mock-provider config, so these cost nothing, need no API key, and touch no network:

```console
# one episode
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main game episode \
    --config configs/runs/old/naming_convention_smoke_test_v3.yaml --output-dir results
```

```console
# execution.repetitions episodes (one cell)
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
    --config configs/runs/old/naming_convention_smoke_test_v3.yaml --output-dir results
```

Swap in any other config's path and both still work unchanged — `--output-dir` is optional on both
(see [`config_reference.md` §6](config_reference.md#6-storage--where-files-land)) and defaults to
that config's own `storage.output_dir`, so it's shown here just to make the two commands land in
predictable, comparable places on a first run.

So there is no such thing as an "MI config" or a "grid config" as a separate species. There
is one config format. Adding a top-level `grid:` section to it turns `experiment run` into a
sweep, and that is the *only* structural difference.

One caveat that trips people up, and it is enforced rather than documented-and-hoped:

```console
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main synthetic episode --config configs/runs/synthetic_games/synthetic_controlled_markov_empowerment_lagged.yaml
configuration validation failed: grid: unknown field
```

Every command below follows this same shape: `conda run -n MA-CC --no-capture-output python -m
mas_cc.cli.main <subcommand> ...`. `-n MA-CC` runs it in that conda environment without activating
it first; `--no-capture-output` is the part that's easy to miss — `conda run` buffers stdout/stderr
and only flushes it at the end by default, so without this flag a long `experiment run` looks frozen
until it exits. With it, output streams live exactly as if the environment were activated.

A config with a `grid:` section can **only** go through `mas-cc experiment run`. The
single-episode commands reject it outright rather than silently running cell zero, because
"I ran the grid config and got one episode" is a much worse failure than an error message.

**`repetitions` is where statistical power comes from, not `horizon`.** For the
across-episode mutual information, the sampling unit is the *episode* — one row each.
Doubling `game.horizon` gives each episode more rounds and buys you nothing for that
estimate; doubling `execution.repetitions` is what tightens it.

---

## 2. What makes a config "one episode" vs "many"

Three fields, and they do not overlap:

```yaml
game:
  horizon: 40            # rounds WITHIN one episode
execution:
  repetitions: 40        # episodes per cell     <- only `experiment run` reads this
  parallelism: 12        # episodes in flight at once, shared across the whole grid
grid:                    # presence of this section = sweep. absence = single cell.
  game.options.control_value: [0, 1, 2]
```

`grid:` keys are **dotted paths into the resolved config**, so anything in the config is
sweepable — `game.population_size`, `game.options.epsilon`, `control.options.forced_value`,
`prompt.prompt_version`. Four things are deliberately *not* sweepable
(`llm_provider.type`, `llm_provider.model`, `game.type`, and anything under `budget.`/
`pricing.`): every cell shares one provider client, one pricing quote and one budget guard,
so sweeping those would need a different, unbuilt feature. You get a clear error, not a
surprise.

Values can be lists, which is how "committee size" becomes an axis — see the third config in
[§5](#5-three-configs-you-can-run-right-now).

---

## 3. The command map

```
                   one episode          many episodes         many cells
                 ------------------   ------------------   ------------------
 real game       mas-cc game episode  mas-cc experiment run   + a grid: section
 synthetic game  mas-cc synthetic     mas-cc experiment run   + a grid: section
                   episode
```

| Command | Runs anything? | Use it when |
| --- | --- | --- |
| `mas-cc game preflight` | no | you want the cost/token estimate before spending |
| `mas-cc game episode` | 1 episode | debugging a game, a prompt, or a provider |
| `mas-cc experiment preflight` | no | pricing a whole batch or grid before launch |
| `mas-cc experiment run` | N or cells×N | **this is the one that produces data** |
| `mas-cc experiment aggregate` | no | recompute cell curves from a finished run's files |
| `mas-cc analysis empowerment` | no | offline MI with bootstrap CIs over a finished grid |
| `mas-cc synthetic truth` | no | closed form for one config |
| `mas-cc synthetic empowerment` | **no** | closed form for a whole *sweep* — the answer key |
| `mas-cc synthetic episode` | 1 episode | one synthetic episode through the real machinery |
| `mas-cc synthetic sweep` | simulation | the null distribution and calibration curve |
| `mas-cc synthetic parity` | both modes | proves fast mode and full mode agree |

Full detail per command: [`running_an_episode.md`](running_an_episode.md),
[`running_an_experiment.md`](running_an_experiment.md),
[`running_synthetic_games.md`](running_synthetic_games.md). Every config field:
[`config_reference.md`](config_reference.md).

---

## 4. Why `synthetic` has its own commands, and what `empowerment` really is

The synthetic games are **not a different way to run experiments**. They are ordinary games
— same `Game` contract, same decision loop, same recorder — whose agents happen to be lookup
tables plus coins instead of LLMs, so we can derive the right answer on paper. They run
through `mas-cc experiment run` exactly like the naming-convention game does. All three configs
in §5 do precisely that.

The `mas-cc synthetic ...` subcommands exist for the things you can *only* do when you know
the answer, and this is the bit that confuses everyone:

> **`mas-cc synthetic empowerment` does not run a sweep. It computes what the sweep's
> answer would be, in closed form, without running anything.**

Three different objects, all called "empowerment", in dependency order:

| | What it is | How you get it |
| --- | --- | --- |
| **the answer key** | exact `I(C;O)` for a resolved sweep | `mas-cc synthetic empowerment` (runs nothing) |
| **the live estimate** | `I(C;O)` from the episodes as they land | written to `sweep_metrics.json` by `experiment run` |
| **the offline estimate** | same, plus bootstrap CIs and permutation nulls | `mas-cc analysis empowerment --grid-dir` |

The whole point of a synthetic game is to put all three side by side and check they agree.
Grid A in §5 does that automatically: it requests the `mi_ground_truth_gap` sweep metric, so
the run computes its own answer key and reports `estimate − truth` as it goes.

---

## 5. Three configs you can run right now

All three live in `configs/runs/synthetic_games/`, are offline, free, and finish in about
two minutes on a laptop. None opens a socket or spends a cent. Read their headers — each one
explains what it is *for*, not just what it sets.

### 1. No grid — one condition, many episodes

[`synthetic_controlled_markov_repeated.yaml`](../../../configs/runs/synthetic_games/synthetic_controlled_markov_repeated.yaml)

One fixed control condition, 200 episodes, no `grid:` section at all. Produces the aggregate
curves and scalars — the "proportions of the decisions" over 200 runs:

```
converged_fraction        0.995
median_consensus_round    9.0   (IQR 5.0 – 15.0)
dominant_action_share  @ round 1   p10/p50/p90 = 0.50 / 0.67 / 0.83
                       @ round 20  p10/p50/p90 = 0.70 / 0.87 / 1.00
```

**It produces no empowerment, and not because it was switched off.** Empowerment is
`I(C ; outcome)`. This run has one condition, a variable with one value has zero entropy, so
there is nothing for the outcome to be informative *about*. The answer is not "small", it is
**undefined**. That is why its `sweep_metrics` list is empty — asking for them would give
NaN every time, by construction. Use this file to see what a single-cell run looks like,
in the output directory and in Comet.

### 2. One axis — the lagged empowerment, computed in-run

[`synthetic_controlled_markov_empowerment_lagged.yaml`](../../../configs/runs/synthetic_games/synthetic_controlled_markov_empowerment_lagged.yaml)

The same game, with the control input swept over three levels — push toward Q, push toward
M, push nowhere. 3 cells × 60 = 180 episodes. Those three levels *are* the condition
variable `C`.

```console
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
    --config configs/runs/synthetic_games/synthetic_controlled_markov_empowerment_lagged.yaml \
    --output-dir results
```

Per cell:

| cell | control | terminal outcomes | converged |
| --- | --- | --- | --- |
| 0000 | push to Q | 59 Q, 1 M | 0.98 |
| 0001 | push to M | 4 Q, 56 M | 1.00 |
| 0002 | push nowhere | 37 Q, 23 M | 0.38 |

And the whole MI block, written to `sweep_metrics.json` **by the run itself** — no second
command:

```json
"terminal_mi_estimate":     0.4887,
"terminal_mi_ground_truth": 0.5650,
"terminal_mi_gap":         -0.0763,
"terminal_mi_null_p95":     0.0236,

"lagged_cmi_h1_estimate":   0.0558,
"lagged_cmi_h2_estimate":   0.0959,
"lagged_cmi_h5_estimate":   0.1679,
"lagged_cmi_h10_estimate":  0.2322
```

Read it like this: the terminal estimate is **far outside** its own null band
(0.489 ≫ 0.024), so the signal is real; it sits 0.076 bits *below* the closed form, which is
the expected direction and size for Jeffreys smoothing at 180 episodes. The lagged CMI
*rises* with the lag — given where the population is now, the further ahead you look, the
more the condition tells you, because a control held fixed all episode keeps pushing the
trajectories apart.

**Why one axis:** `mi_ground_truth_gap` needs a single condition variable `C` to be the
channel input. With two axes there is no single `C`, so the metric reports nothing rather
than a wrong number.

**Which empowerment is this?** There are two in the codebase and they are different numbers
(see [`games/synthetic/effective_empowerment.py`](../../../src/mas_cc/games/synthetic/effective_empowerment.py)).
`lagged_cmi` is quantity **3a** — `I(C ; S_{t+h} | S_t)` with `C` drawn once per episode,
*estimated from the episodes*, which is why it can run inside a run. Quantity **3b**, the
empowerment paper's own `E[I(a_t ; s_* | s_t)]` with `τ ~ Geom(1−γ)`, needs
`control_mode: per_round` and is **exact linear algebra, not an estimate** — so it comes from
`mas-cc synthetic truth`, which runs nothing. See
[`synthetic_controlled_markov_per_round.yaml`](../../../configs/runs/synthetic_games/synthetic_controlled_markov_per_round.yaml).

### 3. Two axes — population size × committee size

[`configs/runs/synthetic_games/synthetic_controlled_markov_size_and_committee.yaml`](../../../configs/runs/synthetic_games/synthetic_controlled_markov_size_and_committee.yaml)

Rows are `game.population_size` (5, 6, 8); columns are how many agents the controller may
push, spelled as the list of indices, so its **length** is the committee size (1, 3, 5).
9 cells × 20 episodes = 180 episodes, ~2 min.

**Why the size axis stops at 8.** This game's `ground_truth()` is exact linear algebra on
the 2^N microstate chain and the synthetic runtime calls it once *per episode*: 0.10 s at
N = 6, 0.94 s at N = 8, and **24 s at N = 10**. It is a cliff, not a slope — one N = 10 cell
of 20 episodes is eight minutes on its own. If you need N = 10, cut `repetitions` to match.

```console
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
    --config configs/runs/synthetic_games/synthetic_controlled_markov_size_and_committee.yaml \
    --output-dir results
```

`converged_fraction` per cell, as produced here:

| N \ committee | 1 | 3 | 5 |
| --- | --- | --- | --- |
| **5** | 0.70 | 1.00 | 1.00 |
| **6** | 0.50 | 1.00 | 1.00 |
| **8** | 0.20 | 0.80 | 0.95 |

A bigger committee converges the population more reliably; a bigger population resists.
That is the finding, and it lives in the per-cell scalars — **not** in the MI:

```json
"terminal_mi_estimate": 0.0474,
"terminal_mi_null_p95": 0.0482
```

The estimate is *inside* its null band, and that is correct. `control_value` is fixed at 0
in every cell, so both axes change how *fast and how reliably* the population reaches Q,
never *which* action it reaches. A grid whose axes don't steer the outcome should report no
outcome information, and seeing the null band catch it is worth more than not asking.

**This is the general lesson:** decide before you run whether your axes move the *outcome*
(→ read the MI block) or the *dynamics* (→ read `converged_fraction` and
`median_consensus_round` per cell). Grid A is the first kind; grid B is the second.

---

## 6. Turning Comet on, and checking it worked

**The three `synthetic_games/synthetic_controlled_markov_*` configs ship with
`logging.comet: true`.** Nothing else is needed — no flag, no environment variable, no second
command. Run them and they upload.

There are exactly two switches, both in the config:

```yaml
logging:
  overrides:
    comet: true                       # SWITCH 1 — on/off. This is the whole thing.
    options:
      comet_project: mas-cc-synthetic
observability:
  comet:                              # SWITCH 2 — shape. Only read when switch 1 is true.
    heartbeat_seconds: 20
    sweep_experiment: true            # false = no liveness dashboard
    cell_reporting: experiments       # one experiment per cell. `master` puts them on the
                                      # master instead; `disabled` uploads no cell curves.
```

> **This is a real upload.** It has caught us out before: a validation run with a *mocked
> provider* still uploaded, because the provider mock and the Comet switch are independent.
> Mocking the LLM does not disable Comet. For a dry run, set `comet: false`.

**Every run now says which of these happened, in its own banner:**

```console
  Comet:         master -> project 'mas-cc-synthetic', cell experiments on
                 https://www.comet.com/<you>/mas-cc-synthetic/<key>
  Comet:         off  (set logging.comet: true in the config)
  Comet:         unavailable  (COMET_API_KEY is not set)
```

The line reports the **connection**, not the config, and it is printed after the connection
is attempted. That matters for one case in particular: `comet: true` with no API key in
scope used to be completely silent, and looked identical to a run that never meant to
upload.

What the master publishes — and only the master; workers write episode files and never touch
Comet, which is what leaves one writer per experiment key and no step-counter races:

| Object | `step` means | Carries |
| --- | --- | --- |
| **one sweep experiment** | `episodes_done` | heartbeat, progress, ETA, `grid_progress` image, per-cell headline scalars, the MI block |
| **one experiment per cell** | `round` | the aggregate curves — so cells overlay against each other natively |

Every sweep experiment also carries the run's **parameters** — experiment name, description,
tags, game type and every `game.options.*` knob, provider, model, seed, repetitions,
parallelism, prompt family/version, the grid axes and their levels, cell and episode counts,
and every aggregation setting. That is deliberately enough to reconstruct what was run from
the dashboard alone, without going back to the config file.

Three checks, in increasing order of trust:

1. **Did it connect at all?** The banner line above says so at launch; afterwards,
   `cat <run-dir>/comet_run_summary.json` — `status` is `active`, `disabled`, or
   `unavailable` with a `reason`. It also lists every cell experiment.
2. **Is the picture right?** `<run-dir>/grid_progress.png` is the *same figure* the dashboard
   gets, written locally whether or not Comet is on. Compare them.
3. **Are the numbers right?** `<run-dir>/sweep_metrics.json` and
   `<run-dir>/cells/*/aggregate.json` hold exactly what was published. Comet is a **view**,
   never the store — if it fails, the run is unaffected and you can re-log later from these
   files.

The heartbeat is on a **timer**, not a completion hook (`observability.comet.heartbeat_seconds`,
20 s in the shipped configs, 60 s by default). A metric that only moves when an episode finishes
cannot tell a dead master from a slow one; a flatlined heartbeat means dead.

---

## 7. More than two axes: what the grid picture can and cannot do

Short answer: **more than two axes runs fine, but the picture degrades, and you should know
how.**

Running is unaffected — the grid is the cartesian product however many axes you give it, and
every cell aggregates normally. It is only `grid_progress.png` that has a problem: colour is
already carrying "how full is this cell", so it cannot also carry a third variable. A third
axis has nowhere to go.

What actually happens today: the image is drawn over the **first two axes**, and the
remaining axes are **summed into them** as a marginal. With 2 × 3 × 2 axes you get a 2 × 3
picture whose cells read `4/4` instead of two separate `2/2`s. The fill fraction stays
correct, but you lose resolution — and, importantly, **a failure anywhere in a collapsed
group turns the whole square red**, without telling you which sub-cell failed.

So, in order of preference:

1. **Keep it to two axes for anything you want to watch as a picture.** This is why the
   design targets ≤ 2 swept parameters. Both configs above obey it.
2. **Run one grid per level of the third axis.** Three separate `experiment run`s, three
   sweep experiments, three unambiguous pictures. Costs nothing but a loop.
3. **Faceting (small multiples)** — a row of heatmaps, one per level of axis 3 — is the
   standard answer to exactly this, and would be a genuine improvement over the current
   marginal. It is *not built*. Ask if you want it; it's a contained change to
   `grid_progress_figure` in [`src/mas_cc/metrics/plotting.py`](../../../src/mas_cc/metrics/plotting.py).

What is *not* an option is using colour for the third axis. Colour means progress here, and
a picture where yellow sometimes means "done" and sometimes means "epsilon = 0.2" is worse
than no picture.

---

## 8. Reading the output directory

```
results/<game>/<experiment>/<run-id>/
├── grid_summary.json            per-cell completed/failed counts
├── grid_progress.png            the picture, same one Comet got
├── comet_run_summary.json       what was (or wasn't) published, and why
├── sweep_metrics.json           the grid-level MI block  <- only if sweep_metrics: is set
├── sweep_ground_truth.json      the closed-form answer key, if this sweep has one
├── resolved_base_config.yaml    the fully resolved config that actually ran
└── cells/
    └── cell-0000/
        ├── overrides.json       which axis values this cell is
        ├── aggregate.json       curves + scalars + the count tables  <- the cell's answer
        └── data/episodes/<id>/
            └── metrics/streaming.csv    the per-round rows everything is derived from
```

Two properties worth relying on:

**A cell is finalized at its own completion.** `aggregate.json` is written the moment a
cell's last episode lands, before anything is published. Kill a grid at 80% and every
finished cell still has complete, correct output.

**Aggregates are derived, so they are recomputable.** Changing a percentile band, a rolling
window, or the forward-fill rule is a re-read of files that already exist — never a re-run:

```console
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment aggregate \
    --run-dir <run-dir>                      # reproduce exactly
$ conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment aggregate \
    --run-dir <run-dir> --config other.yaml  # different rules
```

The second form takes only the `aggregation:` section from `other.yaml`, which is what keeps
the two sets of curves comparable rather than confusable. The design behind those rules —
why episodes are forward-filled, why curves are relabelled by winner, why bands are
percentiles and not ±std — is in
[`docs/tdd/architecture/05082026_master_logging_and_aggregate_metrics.md`](../../tdd/architecture/05082026_master_logging_and_aggregate_metrics.md).
