# Runbook: build and launch the two new experiments

**For an agent.** Read this file in full before acting. Everything here is
**PRELIMINARY** — do not freeze a study or spend provider budget without
confirming with the author (rsanchez).

Context, in one line: we are running two MuSR blackboard-control setups that
differ *only* in whether the population of 24 agents collectively holds a proof
of the correct answer. Plain explanation in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

---

## 0. Vocabulary you need

| term | meaning |
|---|---|
| **task_003** | The fixed world: 3 people, 2 jobs, 3 candidate allocations, 9 hidden values. `ALLOCATION_0` is correct; `ALLOCATION_2` is the wrong target we steer toward. |
| **fact** | One of 49 true propositions about that world. Nobody ever lies; steering is done purely by *which* true facts get said. |
| **lean** | The allocation a fact most favours when read alone. |
| **decisive facts** | A 6-fact set that proves `ALLOCATION_0`. Present in symmetric-v2, absent in nosolution-v2. |
| **ρ (persistence)** | Probability an agent remembers each fact it holds, per round. |
| **arm** | silent (no controller) / truth control / false control. |
| **pool** | The 12 facts the controller may post. Shared by both steering directions. |
| **v0.4** | `control_efficiency_project_plan_v0.4_post-meeting_plan.md`, the team's agreed plan. |

---

## 1. Blockers — check these first, stop if either fails

### 1a. `origin/darius-MA-v1` must be merged

Four controller options do **not** exist on `dev/rsanchez`:

| option | needed for |
|---|---|
| `controller_fact_pool_mode` | the target-independent 12-fact menu |
| `controller_budget_scope` | per-round vs episode budget |
| `controller_round_budget_mode` | — |
| `controller_memory_mode` | — |

Verify:

```bash
grep -c CONTROLLER_FACT_POOL_BALANCED \
  src/mas_cc/games/relational_reasoning/imitation_round_feedback/controller.py
```

`0` means not merged. **There is no workaround** —
`controller_report_pool_mode: target_aligned_v1` cannot express a shared pool
(it is a truth-arm-only override, and it raises unless the pool contains every
decisive fact, which ours does not).

**And an unmerged branch fails silently, not loudly.** Confirm on the resolved
control object, not just by reading the config:

```python
c = RelationalRoundBudgetedControl.from_options(opts)
assert hasattr(c, "controller_fact_pool_mode"), "darius-MA-v1 not merged"
assert hasattr(c, "controller_budget_scope"),   "darius-MA-v1 not merged"
```

Without the merge both keys are accepted and thrown away, and the run uses the
task's frozen pool and a per-round budget while the config claims otherwise.

### 1b. The two task directories do not exist yet

Build them per §2 before writing any config.

---

## 2. Build the two task directories

Source: the frozen `task_003` on `origin/darius-MA-v1` at
`results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/`.

For each of `task003_symmetric_v2` and `task003_nosolution_v2`:

1. **Copy the whole `task_003` directory** to the new task id. Leave
   `task.json`, `hidden_world.json`, `facts/`, `symbolic/` and `generation/`
   **unchanged** — the world and the 49 facts are identical in both setups.
2. **Record provenance** in `task.json` with a `derived_from` field, the way
   `task_004` does.
3. **Write `private/N24_assignment.json`** from `agents.agent_assignments` in
   the corresponding `task003_*_v2.json` beside this file. Copy
   `schema_version` and the `profiles` structure from the original
   `task_003/private/N24_assignment.json`.
4. **Write `controller/balanced_fact_pool.json`** from
   `controller_pool.fact_ids`. The runtime reads only the `fact_ids` list; the
   rest of the file is audit metadata. Follow the schema that
   `scripts/local/build_balanced_controller_pool.py` emits, but set `rule` to
   name *our* construction — ours adds a "proves no allocation" constraint that
   Darius's does not have.
5. **Write the same 12 fact IDs into `facts/controller_reportable_facts.json`**,
   replacing the copied `task_003` version, and update
   `controller_eligible_fact_count` in `task.json` from 24 to 12.

   **This step is a safety net, not bookkeeping.** `from_options` on
   `dev/rsanchez` silently accepts and discards keys it does not know — verified
   by direct call, no error and no warning. So if the merge in §1a is missing or
   incomplete, `controller_fact_pool_mode: balanced` is ignored, the pool falls
   back to `controller_report_pool_mode: frozen`, and `frozen` returns
   `facts/controller_reportable_facts.json`. Copied from `task_003` unchanged
   that is **the original defective 24-fact pool** — the bug this redesign
   exists to fix, reintroduced invisibly with no error. Writing our 12 facts
   there makes the fallback land on the right pool.

   The same trap applies to `controller_budget_scope: episode`, which on
   `dev/rsanchez` is silently ignored and runs **per-round** instead.

### Verify before going further

Recompute these from the built task and check each against the JSON. **If any
differs, stop** — the assignment was written wrong.

| check | symmetric-v2 | nosolution-v2 |
|---|---|---|
| agents, facts each | 24, 1 | 24, 1 |
| distinct facts | 16 | 15 |
| slot lean A0/A1/A2 | 8 / 8 / 8 | 8 / 8 / 8 |
| agents' joint posterior | [1.0, 0, 0] | [0.5, 0, 0.5] |
| decisive facts held | 6 | 0 |
| minimal proofs assemblable | 8 | 0 |
| pool size, lean | 12, 4/4/4 | 12, 4/4/4 |
| pool proves any allocation | **false** | **false** |
| agent ↔ pool overlap | **0** | **0** |

`../builders/build_designs_v2.py` shows how each was computed. It currently
reads the frozen tasks from a session scratchpad and **needs repointing** at a
permanent copy first.

---

## 3. Write the configs

**Base:** [`config_template.yaml`](config_template.yaml) — annotated, one reason
per line, already validated as parseable. It *is* phase 1.

**Phase overrides:** [`variants.yaml`](variants.yaml) — replace exactly the keys
a variant lists, change nothing else.

### The grid

| factor | levels |
|---|---|
| setup | `task003_symmetric_v2`, `task003_nosolution_v2` |
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
calculation, not v0.4's number (v0.4 says start at 50). It doubles the cost.
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
  symmetric-v2's ceiling is 1.0 and nosolution-v2's is 0.5; the latter needs
  rescaling first.

---

## 7. Where everything is

| what | where |
|---|---|
| plain explanation, cell counts | [`../EXPERIMENTS.md`](../EXPERIMENTS.md) |
| every property of all three setups, build steps | [`README.md`](README.md) |
| the settings, annotated | [`config_template.yaml`](config_template.yaml) |
| phase 2 / 3 overrides | [`variants.yaml`](variants.yaml) |
| the fact lists | `task003_symmetric_v2.json`, `task003_nosolution_v2.json`, `task003_symmetric_v0_reference.json` |
| why each protocol option, with code references | [`../protocols_2026-10-02.md`](../protocols_2026-10-02.md) |
| rounds / episodes / which controller first | [`../design_preliminary_2026-10-02.md`](../design_preliminary_2026-10-02.md) |
| what Darius built, what v0.4 says | [`../prior_art_2026-10-02.md`](../prior_art_2026-10-02.md) |
| what `task_003` is, the 49 facts, the proofs | [`../../10_task_and_facts/task_003_analysis.md`](../../10_task_and_facts/task_003_analysis.md) |
