# Relational reasoning — population study 02 (adversarial follow-up)

A **10-cell** follow-up to study 01, derived from
`../population_study_01/overnight/relational_population_study01_d_adversarial_diagnostic_overnight.yaml`.

The game (`relational_imitation_round_feedback`), the controller
implementation (`relational_round_budgeted`), the prompts, the metrics and the
`results_only` artifact profile are **unchanged**. Nothing in `src/` was
modified for this study.

## The 10 cells

### Arm E — wrong-control budget scan (6 cells)

`relational_population_study02_e_wrong_budget_scan_L2_q1.yaml`

Fixed: `N=24`, `L=2`, `r=6`, `q=1`, `K=3`, 4 distractors, 10 population rounds,
`message_mode: recommendation_only`, `advocacy_schedule: always`, wrong target,
2 repetitions, worlds `task_0001` and `task_0002` of `pop24_L2_r06`.

| cell | b | c = b/N | world |
|---|---|---|---|
| cell-0000 |  6 | 0.25 | task_0001 |
| cell-0001 |  6 | 0.25 | task_0002 |
| cell-0002 | 12 | 0.50 | task_0001 |
| cell-0003 | 12 | 0.50 | task_0002 |
| cell-0004 | 18 | 0.75 | task_0001 |
| cell-0005 | 18 | 0.75 | task_0002 |

### Arm F — exploratory `L=3, q=2, r=6` (4 cells)

`relational_population_study02_f1_L3_q2_no_control.yaml` (2 cells, c = 0) and
`relational_population_study02_f2_L3_q2_wrong_b06.yaml` (2 cells, b = 6,
c = 0.25). Same two worlds (`task_0001`, `task_0002` of `pop24_L3_r06`) in both,
so the pair is matched by construction; the two files are byte-identical apart
from the `control:` block and the experiment identity.

Split across two files rather than one because `control.mechanism` cannot be a
grid axis: the uncontrolled cells must carry no `control.options` at all.

## The target is pinned, not drawn — read this before pooling anything

`control.options.target` is **`2`**, a zero-based index into the task's frozen
label-order `possible_answers`, not `random_incorrect`.

`random_incorrect` resolves from the *episode* seed, and the episode seed is
derived from the **grid cell index**
(`orchestrator.py:2124-2132`: `Seed(execution.seed).derive("grid-cell:i").derive("episode:j")`).
Two cells at different `b` therefore have no reason to draw the same wrong
relation. Study 01 arm D shows this happening *inside a single cell*: at r = 6,
`task_0002` drew `SOUTHEAST` in repetition 0 and `SOUTH` in repetition 1, and
those two repetitions ended at wrong-target shares of **0.333** and **1.000**.
Pooling them as one "b = 24" point averages two different experiments.

An option index is seed- and cell-independent, so every episode of every cell
here advocates one fixed wrong relation per world:

| dataset | world | `possible_answers` | correct | pinned target (index 2) |
|---|---|---|---|---|
| pop24_L2_r06 | task_0001 | `[SOUTHWEST, WEST, EAST]` | WEST | **EAST** |
| pop24_L2_r06 | task_0002 | `[SOUTHEAST, SOUTHWEST, SOUTH]` | SOUTHWEST | **SOUTH** |
| pop24_L3_r06 | task_0001 | `[NORTHWEST, WEST, SOUTHWEST]` | WEST | **SOUTHWEST** |
| pop24_L3_r06 | task_0002 | `[NORTHWEST, SOUTHEAST, EAST]` | SOUTHEAST | **EAST** |

"Correct sits at index 1" is a **per-task** fact, not a dataset invariant — in
`pop24_L2_r06` the correct option is at index 2 for `task_0006` and `task_0010`,
and in `pop24_L3_r06` for `task_0006`. Re-check before widening
`game.options.task_id`.

## Reusing the historical c = 0 and c = 1 anchors

Neither endpoint is re-run.

**c = 0** — study 01 arm A, r = 6, `task_0001`/`task_0002` (cells `cell-0010`,
`cell-0011`, 2 repetitions each; **all 4 episodes usable**). Arm A ran with no
controller, so `controller_target` is null there and `m_ctrl`/`analysis_target`
fall back to the *correct* answer — do not read `controller_target_share` from
those rows. The wrong-target share is still exactly recoverable per round
without re-running, because the round record carries `occupation_counts_after`
aligned to `possible_answers`:

```
share(target) = occupation_counts_after[2] / N
```

Verified against all four arm-A episodes (and cross-checked:
`occupation_counts_after[correct_index] / N == truth_vote_share` on every row).

**c = 1** — study 01 arm D, r = 6, `task_0001`/`task_0002` (cells `cell-0010`,
`cell-0011`). Only **3 of 4** episodes are target-matched:

| world | rep | realized target | index | matched to `target: 2`? |
|---|---|---|---|---|
| task_0001 | 0 | EAST | 2 | yes |
| task_0001 | 1 | EAST | 2 | yes |
| task_0002 | 0 | SOUTHEAST | 0 | **no — exclude** |
| task_0002 | 1 | SOUTH | 2 | yes |

No single fixed target matches all four (task_0002's two repetitions drew
different relations), so index 2 attains the maximum possible 3/4. The
`task_0002` repetition-0 episode must be **dropped** from the c = 1 point, or
that one cell re-run under the pinned target as an 11th cell.

## `L=3, q=2` still has a live peer-information channel

At q = 1 a controlled focal sees the controller and nothing else, so
"adversarial control" and "social isolation" are confounded. At q = 2 the
runtime draws exactly one `replaced_peer_slot`
(`runtime.py:540`, `replacement_rng.randrange(len(sampled_peers))`) and
`build_social_sources` substitutes the controller into that slot only
(`runtime.py:331`). Verified on a mock-provider run of the real F2 config
(N=24, q=2, b=6): every controlled update carried exactly 1 control source and
1 ordinary peer, `effective_peer_ids` had length 1, both slot positions were
replaced across the episode, peer evidence was still exposed on 16 of 18
controlled updates, and `controller_fact_exposures` stayed 0 throughout
(`recommendation_only` never cites a fact).

## The `pop24_L3_r06` dataset

Generated for this study with the same dataset seed as the `pop24_L2_r*` family:

```
python generate_dataset.py --num-tasks 10 --population-size 24 \
  --reasoning-depth 3 --support-redundancy 6 --distractors 4 \
  --distractor-redundancy 1 --num-options 3 --seed 20260818 \
  --no-single-agent-solution --output datasets/pop24_L3_r06
```

Fingerprint `559279be5049…`; `validate_dataset.py` (full reproducibility check)
reports **VALID**. In `task_0001`/`task_0002`: supporting facts `f1,f2,f3` each
held by exactly 6 of 24 agents, distractors `f4..f7` each by 1, and no agent
holds more than 1 of the 3 supporting facts — `no_single_agent_solution` holds
strictly, so `phi_t` starts at exactly 0.

At `L = 3` the knowledge strata are `|S| + 1 = 4` buckets, so this arm writes
`knowledge_share_k0..k3` and `truth_share_k0..k3` where study 01 wrote `k0..k2`.

## Observables

Unchanged from study 01, all written per round to
`round_records/<episode>/round_trajectory.jsonl`: `truth_vote_share`,
`controller_target_share`, `mean_supporting_fact_coverage` (κ_t),
`full_proof_agent_share` (φ_t), `truth_share_k*`, `knowledge_share_k*`,
`peer_fact_exposures` / `new_peer_facts`, `controller_fact_exposures` /
`new_controller_facts`, `controlled_target_adoption_rate`,
`occupation_counts_before/after`, `m_truth` / `m_ctrl` / `m_order`, `H_vote`.

## Cost

| config | cells | episodes | nominal | expected | conservative |
|---|---|---|---|---|---|
| arm E | 6 | 12 | 3,168 | 3,348 | 6,336 |
| arm F1 | 2 | 4 | 1,056 | 1,116 | 2,112 |
| arm F2 | 2 | 4 | 1,056 | 1,116 | 2,112 |
| **total** | **10** | **20** | **5,280** | **5,580** | **10,560** |

264 nominal calls per episode = 24 initial votes + 24 × 10 focal updates.
`expected` adds the 5 % validation-failure allowance; `conservative` is the
retry bound. `q` does not change the call count — control replaces a social
slot rather than adding one, so a focal update is one provider call whatever
`q` is (`game.py:801-804`).

## Running

Preflight makes no model calls. From the repository root:

```
for f in relational_population_study02_e_wrong_budget_scan_L2_q1 \
         relational_population_study02_f1_L3_q2_no_control \
         relational_population_study02_f2_L3_q2_wrong_b06; do
  conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
    --config configs/runs/relational_reasoning/population_study_02/$f.yaml \
    --output-dir inspection/study02_${f}_preflight
done
```

All three currently report `permitted`. Swap `preflight` for `run` to launch.
