# Merging `dev/rsanchez` with `darius-MA-v1` — tested, and it is nearly clean

4 October 2026. Assessed by performing the merge in a throwaway worktree, not by
reading diffs. Nothing was pushed; the test merge was discarded.

## The short answer

**Yes, and it is easy.** One conflicted file, `.gitignore`, and the conflict is
two independent additions in the same place — keep both. `AGENTS.md` and
`README.md` auto-merged.

## The numbers

| | |
|---|---|
| merge base | `af33bdf` ("updated santa fe mechanics", 29 Sept) |
| our commits since | **13** — 27 files, +6,530 lines |
| their commits since | **39** — 186 files, +21,208 / −658 |
| files touched by **both** | **3** — `.gitignore`, `AGENTS.md`, `README.md` |
| actual conflicts | **1** — `.gitignore` |

No source file is touched by both sides. Our work is entirely new packages:
`src/bound_estimators/`, `src/rnd_init_metrics/`, `tests/bound_estimators/`,
`analysis/`. Theirs is the game runtime, configs and task artifacts.

## The one conflict, and how to resolve it

Two independent blocks added at the same point:

- **ours** — the `analysis/` rules (`analysis/**/results/`, report build
  products)
- **theirs** — a reworked `results/` rule:
  ```
  results/*
  !results/studies/
  results/studies/*
  !results/studies/musr_*_calibration_*/
  ```

**Resolution: keep both, theirs first.** Their `results/` rule is strictly more
general than the `task_001`-specific negations we had, and it is what lets the
frozen task directories be committed at all — which is how `task_003` and
`task_004` reached their branch. Verified after resolving: our packages and
`analysis/` tree survive, and so do their `task_003` and `task_004` directories.

## A pre-existing breakage, not caused by the merge

`tests/mas_cc/test_santa_fe.py` contains **committed git conflict markers** at
lines 226 / 237 / 425, so it is a `SyntaxError` and `pytest` aborts collection.

It is **on both branches already**, introduced by commit `3d451dd` ("santa fe
first run"). `dev/rsanchez`'s own suite has therefore been broken since then —
confirmed by running collection on `dev/rsanchez` directly. Nobody has run the
full suite recently.

**This needs fixing regardless of the merge.** It is a one-file repair:
reconcile the two sides of that conflict in `test_santa_fe.py`. Until then no
test run is meaningful without `--ignore=tests/mas_cc/test_santa_fe.py`.

## Test evidence: the merge adds no new failures

Ran the same four study-contract test files on three trees:

| tree | failures |
|---|---:|
| `dev/rsanchez` alone | **16** |
| `darius-MA-v1` alone | **16** |
| **merged** | **16** |

Identical. The merge introduces **zero** new failures.

Those 16 are a second pre-existing problem, separate from `test_santa_fe.py`:
`tests/mas_cc/test_study09{d,e,fg,jk}.py` reference config directories that no
longer exist, e.g.

```
ValueError: study config directory does not exist:
  configs/runs/relational_reasoning/population_study_09d
```

The configs were removed or moved without updating the tests. Also fixable
independently of anything here, and also not ours.

**So the repository has two latent breakages that predate this work:** committed
conflict markers in `test_santa_fe.py`, and 16 study-contract tests pointing at
deleted config directories. Neither blocks the merge; both should be fixed so
that future merges can be validated against a green suite.

## Naming: nothing of theirs or ours is overwritten

The new setups are **new task ids**, added alongside:

| task id | origin | status after merge |
|---|---|---|
| `task_003` | original | **untouched** |
| `task_004` | Darius (task_003 minus decisive facts) | **untouched** |
| `task003_symmetric_v2` | ours | new directory |
| `task003_nosolution_v2` | ours | new directory |

Our 12-fact pool goes in **our** task directories only. Darius's
`task_004/controller/balanced_fact_pool.json` (27 facts) and the original
`task_003/facts/controller_reportable_facts.json` (24 facts) are left exactly as
they are. Likewise our agent assignments are new `private/N24_assignment.json`
files inside our own task directories.

Nothing in `configs/` of theirs is edited either — our configs are new files.

## What we inherit by merging

Worth knowing, since it is 186 files:

- `controller_fact_pool_mode` (`frozen` / `all_nondecisive` / `balanced`) and
  the `controller_balanced_fact_ids` task field
- `controller_budget_scope` (`per_round` / `episode`) — **the only reason we
  need the merge**, for v0.4's `B`
- `controller_round_budget_mode`, `controller_memory_mode`,
  `controller_max_posts_per_round`
- the episode-scope relaxation of the budget-vs-pool-size bound
- `task_004` and its frozen artifacts, including `private/N15_assignment.json`
- `scripts/local/` plotting and Phoenix trace tooling
- `docs/design/musr_game_design_vs_implementation.md` and
  `swarm_scheduling_game_design.md`
- ~1,100 lines of new tests (`test_task004_balanced_pool.py`,
  `test_task004_controller_prompt.py`)

## Recommended sequence

1. Fix `tests/mas_cc/test_santa_fe.py` on `dev/rsanchez` first, so the merge can
   be validated against a green suite.
2. Merge `origin/darius-MA-v1` into `dev/rsanchez`, resolving `.gitignore` by
   keeping both blocks.
3. Run the full suite. 
4. Only then build the two new task directories.

Steps 1–3 are independent of the experiment design and can happen now.
