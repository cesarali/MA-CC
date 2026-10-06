# Runbook: build and launch the two new experiments

**For an agent.** Read this file in full before acting. The two setups are
**frozen** (5 October 2026); the run settings are not. Do not freeze a study or
spend provider budget without confirming with the author (rsanchez).

> **Merge status.** `origin/darius-MA-v1` was merged into `dev/rsanchez` on
> 4 October (commit `4d570b3`). Statements below about what is "unmerged" or
> "darius only" describe the branch before that merge.

Context, in one line: we are running two MuSR blackboard-control setups that
differ *only* in whether the population of 24 agents collectively holds a proof
of the correct answer. Plain explanation in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## 0. Vocabulary you need

| term | meaning |
|---|---|
| **task_003** | The fixed world: 3 people, 2 jobs, 3 candidate allocations, 9 hidden values. `ALLOCATION_0` is correct; `ALLOCATION_2` is the wrong target we steer toward. |
| **fact** | One of 49 true propositions about that world. Nobody ever lies; steering is done purely by *which* true facts get said. |
| **lean** | The allocation a fact most favours when read alone. |
| **decisive facts** | A 6-fact set that proves `ALLOCATION_0`. Held by the agents in task003-symmetric, absent in task003-nosolution. |
| **ρ (persistence)** | Probability an agent remembers each fact it holds, per round. |
| **arm** | silent (no controller) / truth control / false control. |
| **pool** | The facts the controller may post. task003-symmetric has **two** 12-fact pools, one per target; task003-nosolution has **one**, shared by both targets. |
| **post-meeting plan** | `control_efficiency_project_plan_the post-meeting plan_post-meeting_plan.md`, the team's agreed plan. |

---

## 1. Blockers — check these first, stop if either fails

### 1a. The merge is NOT required for phase 1

*(Corrected 2026-10-03: an earlier version of this runbook said it was.)*

Phase 1 runs on `dev/rsanchez` unmerged. Use the **frozen-pool route**:

- `controller_report_pool_mode: frozen` (the default) returns
  `task.controller_reportable_fact_ids` **with no reference to the target** — it
  is already target-independent, on both branches.
- So the pool is chosen by the task directory: write the right pool into each
  directory's `facts/controller_reportable_facts.json` (§2 step 5).
- `validate_truthful_report_task` permits the target to be the task's declared
  target **or** the ground truth, so both arms pass.
- `intervention_budget: 3` is per-round on `dev/rsanchez` and is the only
  behaviour there.

**Do not use `controller_fact_pool_mode: balanced` for a `deterministic`
controller.** Darius's change forces `base_score = 0.0` whenever the pool mode
is not `frozen`, which makes the deterministic fact choice **target-blind** — both
arms then post the same facts in the same order and differ only in the
recommendation text. See `EXPERIMENTS.md` §3c.

**Merge `origin/darius-MA-v1` before phase 4** (episode-scope `B`), which needs
`controller_budget_scope`. That option does not exist on `dev/rsanchez` and is
**silently discarded** if set — verified by direct call, no error. So if phase 4
is ever configured, assert it took effect:

```python
c = RelationalRoundBudgetedControl.from_options(opts)
assert hasattr(c, "controller_budget_scope"), "darius-MA-v1 not merged"
```

Without that check a config asking for an episode reservoir runs per-round while
claiming otherwise.

### 1b. The three task directories do not exist yet

Build them per §2 before writing any config.

---

## 2. Build the three task directories

Source: the frozen `task_003` at
`results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/`.

| task directory | agents from | pool from | used for |
|---|---|---|---|
| `task003_symmetric_to_a0` | `../task003_symmetric/agents.json` | `../task003_symmetric/controller_pool_A0.json` → `fact_ids` | truth control, and the silent runs |
| `task003_symmetric_to_a2` | `../task003_symmetric/agents.json` | `../task003_symmetric/controller_pool_A2.json` → `fact_ids` | false control |
| `task003_nosolution` | `../task003_nosolution/agents.json` | `../task003_nosolution/controller_pool.json` → `fact_ids` | silent, truth and false control |

The two symmetric directories must be identical except for the pool file and
the scores in step 6. **Never run the A2 target from the A0 directory or the
reverse**: that hands the controller the other target's pool, with no error.

For each directory:

1. **Copy the whole `task_003` directory** to the new task id. Leave
   `task.json`, `hidden_world.json`, `facts/`, `symbolic/` and `generation/`
   **unchanged** — the world and the 49 facts are identical in both setups.
2. **Record provenance** in `task.json` with a `derived_from` field, the way
   `task_004` does.
3. **Write `private/N24_assignment.json`** from `agent_assignments` in
   the setup's `agents.json`. Copy
   `schema_version` and the `profiles` structure from the original
   `task_003/private/N24_assignment.json`.
4. **Write `controller/balanced_fact_pool.json`** from that directory's pool
   (table above). The runtime reads only the `fact_ids` list; the
   rest of the file is audit metadata. Follow the schema that
   `scripts/local/build_balanced_controller_pool.py` emits, but set `rule` to
   name *our* construction — ours adds a "proves no allocation" constraint that
   Darius's does not have.
5. **Write the same 12 fact IDs into `facts/controller_reportable_facts.json`**,
   replacing the copied `task_003` version, and update
   `controller_eligible_fact_count` in `task.json` from 24 to 12.

   **This is the mechanism, not bookkeeping.** With
   `controller_report_pool_mode: frozen` this file *is* the controller's pool,
   target-independently, which is exactly what the design wants — and it needs
   no merge. Leaving `task_003`'s version in place would hand the controller
   **the original defective 24-fact pool**, the bug this redesign exists to fix,
   with no error raised.

6. **Decide the scoring question — this changes what the experiment measures.**
   `task.controller_fact_scores` drives which facts a `deterministic` controller
   picks (`-base_score` in the ranking key). The tie-break hash does **not**
   depend on the target, so with absent or equal scores the fact choice is
   **target-blind** and the two arms differ only in the recommendation text.

   - **task003-symmetric**: already one directory per target, so score each
     directory's pool toward its own target.
   - **task003-nosolution**: one score field cannot point at two opposite
     targets. Either split it into two directories differing *only* in the
     score field, or accept target-blind fact choice (*does a recommendation
     steer with evidence held constant?*).

   Ask rsanchez which for task003-nosolution. See `EXPERIMENTS.md` §3c.

### Verify before going further

Recompute these from the built task and check each against the JSON. **If any
differs, stop** — the assignment was written wrong.

| check | `_to_a0` | `_to_a2` | `task003_nosolution` |
|---|---|---|---|
| agents, facts each | 24, 1 | 24, 1 | 24, 1 |
| distinct agent facts | 18 | 18 | 15 |
| agents per allocation A0/A1/A2 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 |
| agents' joint posterior | [1, 0, 0] | [1, 0, 0] | [0.5, 0, 0.5] |
| decisive facts held | 6 | 6 | 0 |
| minimal proofs assemblable | 12 | 12 | 0 |
| pool size | 12 | 12 | 12 |
| pool proves any allocation | **false** | **false** | **false** |
| pool facts the agents hold | 2 | 2 | **0** |

Each setup folder's `build.py` computes every value and stores it in the
`properties` block of its `agents.json` and pool files.

---

## 3. Write the configs

**Base:** [`config_template.yaml`](config_template.yaml) — annotated, one reason
per line, already validated as parseable. It *is* phase 1.

**Phase overrides:** [`variants.yaml`](variants.yaml) — replace exactly the keys
a variant lists, change nothing else.

### The grid

| factor | levels |
|---|---|
| setup | task003-symmetric (directories `_to_a0` and `_to_a2`), task003-nosolution |
| arm | `silent` (omit the whole `control:` block), `truth_control` (`target: ALLOCATION_0`), `false_control` (`target: ALLOCATION_2`) |
| `epistemic_persistence` | `0.75`, `1.0` |

**Phase 1 = 12 configs**: 4 silent (2 setups × 2 ρ) + 8 controlled
(2 setups × 2 directions × 2 ρ).

Phase 2 adds 8 controlled configs and **reuses phase 1's silent runs** — nothing
about the agents changed and a silent run has no controller. Phase 3 needs 4 new
silent runs because the agents' communication profile changed, plus 8 controlled
= 12.

### Settings that are easy to get wrong

| trap | correct value |
|---|---|
| `board.communication_profile` **defaults to `full_communication`** | set `report_only` explicitly in phases 1–2 |
| `social_group_size` must be ≤ `population_size - 1` | **23**, never 24. Inert under `sampling: full` |
| `board.sampling` defaults to `uniform`, `social_group_size` defaults to **1** | set `sampling: full` explicitly |
| `board.allow_participant_requests` | **never set it** — locked to `communication_profile`, conflicting values raise |
| five legacy fields illegal when `controller_authoring` is set | never set `controller_communication_policy`, `_version`, `_fallback_policy`, `allow_controller_requests`, `allow_controller_directives` |
| three peer-slot fields forbidden with `message_mode: recommendation_only` | never set `controller_evidence_strategy`, `controller_fact_id`, `controller_fact_selector` |
| `controller_authoring` requires `adaptive_communication` | keep `controller_actuation_mode: adaptive_communication` |
| the top-level `budget:` block is **money**, not control | the control budget is `intervention_budget` + `controller_budget_scope` |
| `controller_report_max_posts_per_fact` | must be ≥ the budget actually spent |
| **unknown option keys are silently discarded** on `dev/rsanchez` — no error | after any config change, assert the *resolved* control object actually carries `controller_fact_pool_mode` and `controller_budget_scope`; if `hasattr` is false, the merge is missing and the run will quietly use the wrong pool |

Copy `pricing`, `budget`, `logging`, `storage`, `analysis`, `metrics`,
`observability` and `experiment` from an existing study config unchanged.

---

## 4. Validate before spending anything

Run preflight on every config and resolve every issue. Preflight is designed to
fail **before** provider use — a config that only fails at runtime has already
cost money.

Expected checks to pass: the pool resolves to 12 facts; `intervention_budget`
(3) does not exceed the pool; the `communication_profile` /
`controller_authoring` pairing is satisfied for controlled runs; the derived
communication policy is `contextual_weighted_v1` in phase 1.

Then run **one** smoke episode per arm before launching a cell.

---

## 5. Launch

Use the `ma-cc-study-workflow` skill and read
`.codex/skills/ma-cc-study-workflow/SKILL.md` in full first, per the repository
`AGENTS.md`. For Cygnus, also the `ma-cc-cygnus-study-workflow` skill. Routing
through a skill does **not** by itself authorise a real submission or a paid
provider call — get that from rsanchez.

`repetitions: 100` in the template is a recommendation from the bootstrap
calculation, not the post-meeting plan's number (it says start at 50). It doubles the cost.
**Confirm the episode count before submitting.**

---

## 6. Do not do these

- **Do not** cross the later factors (budget level, budget scope, sensing,
  board sampling) into phases 1–3. Add one at a time, after phase 1 is
  analysed. Crossing everything at once is what made the previous batch
  uninterpretable.
- **Do not** use `advocacy_schedule: soft` here. In the archived batch it fired
  on 32% vs 54% of rounds, and that firing-rate gap was most of the apparent
  directional effect. The template uses `always`.
- **Do not** reuse the archived `task_003` controller pool for anything. Fact
  admission was computed once for `ALLOCATION_2` and reused for the truth arm,
  so its truth-arm numbers say nothing about truth-directed steering.
- **Do not** combine `deterministic` authoring with `full_communication` unless
  asked. It is legal, but with three permitted message types the deterministic
  chooser makes a seeded random draw among them, so "deterministic" stops
  meaning "no randomness". See `variants.yaml`, `phase_4_*`.
- **Do not** write anything into `/Users/rsanchez/Projects/agents_control` — it
  is read-only for this project.
- **Do not** present the two setups' accuracy numbers as directly comparable.
  task003-symmetric's ceiling is 1.0 and task003-nosolution's is 0.5; the
  latter needs rescaling first.

---

## 7. Where everything is

| what | where |
|---|---|
| plain explanation, cell counts | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| every property of both setups, how they were built | [`../README.md`](../README.md), then each setup folder's `README.md` |
| the settings, annotated | [`config_template.yaml`](config_template.yaml) |
| phase 2 / 3 overrides | [`variants.yaml`](variants.yaml) |
| the agents and pools | `../task003_symmetric/`, `../task003_nosolution/` (`agents.json`, `controller_pool*.json`) |
| why each protocol option; what Darius built; rounds and episodes | dated notes, deleted 6 October; see [`../../README.md`](../../README.md#recovering-deleted-material) |
| what `task_003` is, the 49 facts, the proofs | [`../../task_and_facts/task_003_analysis.md`](../../task_and_facts/task_003_analysis.md) |
