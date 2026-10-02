# The two new experimental setups — build specification

**Status: PRELIMINARY.** These are candidate designs, not frozen
configurations. Nothing here has been written into a task directory or run.

This file is the single source of truth for the two setups. If you are an agent
asked to build them, read this file in full, then the JSON beside it.

---

## 1. What the two setups are

Both are the **same world**: `task_003`, latent vector
`(3,1,1,2,2,1,1,1,2)`, candidate scores 8 / 4 / 6, `ALLOCATION_0` correct,
`ALLOCATION_2` the designated wrong target. Both draw from the **same 49-fact
universe**. They differ only in **who holds which facts**.

| | `task003-symmetric` | `task003-nosolution` |
|---|---|---|
| question it answers | Can the swarm find a proof it collectively owns — and can a controller stop it? | Can a controller tip a swarm that is genuinely undecided? |
| population can prove the truth | **yes** | **no** |
| accuracy ceiling | **1.0** | **0.5** |
| the false controller must | defeat an available certainty | tip a coin flip |
| the truth controller's job | facilitation: help find the owned proof | persuasion |

Keep both. They are different questions, and the second is not a strictly
harder version of the first.

---

## 2. Every property, side by side

`v0` is the archived setup that was actually run. It is the reference, not a
target to build.

| property | **v0 (reference)** | **symmetric-v2** | **nosolution-v2** |
|---|---|---|---|
| agent number | 24 | **24** | **24** |
| facts per agent | 1 | 1 | 1 |
| # distinct facts | 21 | **16** | **15** |
| **slot lean** A0/A1/A2 | **16 / 5 / 3** | **8 / 8 / 8** | **8 / 8 / 8** |
| **distinct lean** A0/A1/A2 | 13 / 5 / 3 | 6 / 5 / 5 | **5 / 5 / 5** |
| slot multiplicity range | [1, 2] | [1, 2] | [1, 2] |
| agents' joint posterior | [1.0, 0, 0] | [1.0, 0, 0] | **[0.5, 0, 0.5]** |
| decisive facts held | 6 of 6 | **6 of 6** | **0 of 6** |
| proofs available to agents | 4 | **8** | **0** |
| pool size | 24 | **12** | **12** |
| pool lean A0/A1/A2 | 9 / 6 / 9 | **4 / 4 / 4** | **4 / 4 / 4** |
| pool joint posterior | [0.308, 0, 0.692] | **[⅓, ⅓, ⅓]** | [0.318, 0.364, 0.318] |
| pool proves any allocation? | no | **no** | **no** |
| proofs inside the pool | 0 | 0 | 0 |
| pool strength match (W₁) | n/a | 0.046 | **0.009** |
| pool shared by both targets | yes (by defect) | **yes (by design)** | **yes (by design)** |
| agent ↔ pool overlap | 8 of 21 | **0** | **0** |
| **is this already there?** | **yes, fully** | **no — must be built** | **no — must be built** |

### "Is this already there?" in detail

**v0 — yes, nothing to build.** The frozen task is on `origin/darius-MA-v1` at
`results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/`,
including `private/N24_assignment.json` and
`facts/controller_reportable_facts.json`. There are also 20,730 analysed
per-round rows from study `21-09-2026-full-vs-report-v1`. **Its controller pool
is defective**: admission was evaluated once for `ALLOCATION_2` and reused for
the `ALLOCATION_0` arm, so both arms quote the same 24 facts and the truth arm
recommends A0 while citing evidence picked to undermine it. Use v0 only as a
dynamics baseline and as the bridge back to our calibrated estimators. **Never
read its truth-arm numbers as evidence about truth-directed steering.**

**symmetric-v2 and nosolution-v2 — no, neither exists.** What is missing is
exactly two files per setup: a 24-agent private assignment and a 12-fact
controller pool. Everything else (the world, the 49 facts, the decisive set) is
reused unchanged from `task_003`.

The nearest existing thing to nosolution-v2 is Darius's `task_004`, which is
`task_003` with the decisive facts removed — **but it removed them together
with their holders**, dropping the population to 15 agents. That confounds proof
availability with population size, distinct-fact count (21 → 15) and
redundancy. nosolution-v2 holds the population at 24 and replaces the facts
instead.

---

## 3. Slot lean versus distinct lean

Every agent holds **exactly one fact**. If the same fact is given to two agents
it fills two **slots** but counts once as a **distinct** fact. A fact's *lean*
is the allocation it most favours when read alone, under the exact
14,388-world posterior.

- **Distinct lean** — how many *different* facts of each lean are in play. Sets
  what the population can *conclude*: the joint posterior, and whether a proof
  exists at all. This is the information content.
- **Slot lean** — how many *agents* hold a fact of each lean. Sets where the
  population *starts*: the opening spread of beliefs and the first vote split.

They come apart whenever facts repeat, and in v0 they come apart badly: all
three duplicated facts are truth-leaning, so distinct lean 13/5/3 becomes slot
lean **16/5/3** — **16 of 24 agents** open holding truth-leaning evidence
against 8 holding rival-leaning. The initial skew is worse than the distinct
count suggests. Both v2 setups fix this to 8/8/8.

**Multiplicity is a second, quieter knob.** A fact held by two agents must be
forgotten twice, so it survives epistemic decay at ρ < 1 better than a fact
held once. Uneven duplication therefore privileges whichever facts got doubled.
Both v2 designs keep multiplicity in **[1, 2]** and duplicate **evenly** within
each lean group by round-robin, not at random.

---

## 4. How to read the JSON

Three files, one per setup:

```
task003_symmetric_v0_reference.json
task003_symmetric_v2.json
task003_nosolution_v2.json
config_template.yaml                 <- the settings for BOTH experiments
```

`config_template.yaml` is annotated and shared: **only `task_id` differs between
the two experiments.** Lines marked `<<VARY>>` are the factor grid, lines marked
`<<DARIUS>>` need `origin/darius-MA-v1` merged. The reasoning for each setting is
in `../protocols_2026-10-02.md` §2, §3 and §5b.

Each has the same shape. The fields you need to build a setup:

| field | meaning |
|---|---|
| `population_size` | 24 |
| `facts_per_agent` | 1 |
| `agents.agent_assignments` | **the thing to write out** — `{"agent_001": ["cf_..."], ...}`, 24 entries, one fact each |
| `agents.distinct_facts` | the distinct facts, sorted |
| `agents.by_lean` | the same facts grouped `A0` / `A1` / `A2` |
| `agents.slot_multiplicity` | how many agents hold each fact |
| `controller_pool.fact_ids` | **the other thing to write out** — the 12 pool facts |

The rest are verification targets: `agents.joint_posterior`,
`agents.minimal_proofs_assemblable`, `agents.decisive_held`,
`controller_pool.joint_posterior`, `controller_pool.proves_any_allocation`,
`n_overlap`. Recompute them after building and check they match. If any differs,
stop — the assignment was written wrong.

---

## 5. How to build a setup

### Step 1 — copy the task directory

Start from the frozen `task_003` on `origin/darius-MA-v1`:

```
results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/
```

Copy it to a new task id. Keep `task.json`, `hidden_world.json`, `facts/`,
`symbolic/` and `generation/` **unchanged** — the world and the 49 facts are
identical across all three setups. Record the provenance in `task.json` the way
`task_004` does, with a `derived_from` field.

### Step 2 — write the private assignment

Write `agents.agent_assignments` from the JSON to
`private/N24_assignment.json`, matching the existing schema:

```json
{"schema_version": ..., "population_size": 24,
 "agent_assignments": {"agent_001": ["cf_x05_eq_1"], ...},
 "profiles": ...}
```

Copy `schema_version` and the `profiles` structure from the existing
`task_003/private/N24_assignment.json`.

### Step 3 — write the controller pool

Write `controller_pool.fact_ids` to `controller/balanced_fact_pool.json`, in the
schema Darius's builder emits (see
`scripts/local/build_balanced_controller_pool.py` on `darius-MA-v1`). The
runtime only reads the `fact_ids` list; the rest of that file is audit metadata.
Set `rule` to something that names *our* construction, not his — ours adds the
"proves no allocation" constraint his does not have.

### Step 4 — point the config at it

```yaml
control:
  options:
    controller_fact_pool_mode: balanced     # reads controller/balanced_fact_pool.json
```

**This mode does not exist on `dev/rsanchez`.** It was added on
`darius-MA-v1` (`CONTROLLER_FACT_POOL_BALANCED` in `controller.py`,
`controller_balanced_fact_ids` in `data.py`). So:

**Merging `darius-MA-v1` is a prerequisite, not a preference.** There is no
workaround. *(Corrected 2026-10-02: an earlier version of this file offered
`target_aligned_v1` as a fallback. It cannot work — see below.)*

`controller_report_pool_mode: target_aligned_v1`, the only alternative on
`dev/rsanchez`, is a **truth-arm-only override**: it substitutes an explicit
fact list when the resolved target is the truth target, and otherwise falls back
to the task's frozen pool. It therefore produces **two different menus**, which
is exactly what rule 2 in §6 forbids. It also enforces three hard checks that
these designs violate — each a `raise ValueError`, not a warning:

1. the explicit pool **must contain every decisive fact** — ours contains none,
   by construction;
2. it **must not overlap the frozen false-target pool**;
3. the target must be a truth target **distinct from** the task's declared
   controller target.

See `reportable_fact_ids_for_target` in
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/controller.py`.

The two option names are different things and are easy to confuse:

| option | what it selects | target-aware? | values |
|---|---|---|---|
| `controller_fact_pool_mode` *(darius only)* | **which facts are on the menu** | **no** — `reportable_fact_ids(task)` takes no target | `frozen`, `all_nondecisive`, `balanced` |
| `controller_report_pool_mode` *(both branches)* | **how the menu resolves against the target** | **yes** | `frozen`, `target_aligned_v1` |

`controller_report_pool_mode: frozen` returns the task's frozen pool whatever the
target — **that is the original defect**. Our designs need the target-independent
option, which only exists on `darius-MA-v1`.

**The budget protocol is also darius-only**, and so is v0.4's experimental
design. See `../protocols_2026-10-02.md` §4.

### Step 5 — set population size and verify

`population_size: 24` in the study config. Then recompute the verification
fields from §4 against the built task and confirm every one matches.

---

## 6. Design rules both v2 setups enforce

1. 24 agents, one fact each, **slot lean 8/8/8**, multiplicity in [1, 2],
   duplicated evenly within each lean group.
2. **One controller pool shared by both targets**, so the menu is identical
   whichever direction is steered and any directional difference is
   attributable to the *selection policy* rather than to the menu. Two
   target-specific pools cannot support that claim.
3. Pool lean `4/4/4`, with truth-leaning and rival-leaning strengths matched by
   Wasserstein-1 distance.
4. **The pool proves no allocation.** Darius's balanced pool does not meet this:
   its whole-pool joint posterior is 1.0 for the truth and it contains one
   size-6 proof, so his truth controller could hand over a proof outright and
   that arm becomes disclosure rather than persuasion.
5. **Zero agent ↔ pool overlap.**

---

## 7. Why the pool is only 12 facts — the scarcity bound

The task contains only **9 A1-leaning and 9 A2-leaning facts**, against 31
A0-leaning. Rival-leaning facts are the scarce resource, and both the agents and
the controller want them. With `kr` = the agents' distinct rival facts per lean
and `c` = the pool's per lean, zero overlap requires

```
kr + c <= 9
```

Both v2 setups sit at `kr = 5`, `c = 4`. Darius's balanced pool takes **all 18**
rival facts (its rule is "keep all rival-leaning facts"), which is why every one
of the 22 facts outside it is A0-leaning, and why any lean-balanced agent set
must overlap it by exactly two thirds. **His 10-of-15 overlap was forced by the
pool rule, not a tuning accident.**

A consequence to accept deliberately: at b = 3 posts per round over 30 rounds
the controller posts 90 times from a 12-fact menu, so its problem is **timing
and selection, not discovery**.

---

## 8. Two knobs that are decisions, not defaults

**Overlap is a mechanism switch.** At zero overlap the controller can only
**inject** facts new to the population; it can never **refresh** one an agent
already holds. That is clean for attribution — but refresh is the best
explanation we have for path divergence being *larger* at ρ = 0.75 than at
ρ = 1 despite fewer posts. Consider running overlap as a factor (0 / 4 / 8)
rather than always minimising it.

**Joint neutrality trades against strength matching.** symmetric-v2's pool sits
exactly on the prior (⅓ each) at W₁ = 0.046; nosolution-v2 gives up a little
neutrality for W₁ = 0.009. Both cannot be exact at a 12-fact pool. Which matters
more depends on whether the claim is about the menu's *composition* or its
*persuasive weight*.

---

## 9. Known gaps

- **symmetric-v2's six A0-leaning distinct facts are exactly the six decisive
  facts**, so truth-evidence and the proof set coincide perfectly. Setting
  `k0 = 8` decouples them; see `../builders/scan_design_menu.py`.
- **Nothing records the proof-assembly rate** — per round, the fraction of
  agents whose active memory holds a complete proof. That is the quantity
  symmetric-v2 exists to measure, and no current artifact logs it.
- **The builders read the frozen tasks from a session scratchpad.** They need
  repointing at a permanent copy of
  `results/studies/musr_truthful_selective_task_calibration_01/tasks/` from
  `origin/darius-MA-v1` before anyone else can run them.
- **These are 24-agent designs for `task_003` only.** Generality across worlds
  is untested; see `../../10_task_and_facts/task_003_analysis.md` §8.

---

## 10. Where the surrounding context lives

| | |
|---|---|
| what `task_003` is, the 49 facts, the proofs | `../../10_task_and_facts/task_003_analysis.md` (verified 2026-10-02) |
| why the old controller was unfair | `../../20_controller_redesign/` |
| the state representation for estimating information | `../../30_coarse_graining/`, `../../40_information_estimates/` |
| what Darius already built, and plan v0.4 | `../prior_art_2026-10-02.md` |
| rounds, episodes, which controller to run first | `../design_preliminary_2026-10-02.md` |
| the search that produced these designs | `../builders/` |
