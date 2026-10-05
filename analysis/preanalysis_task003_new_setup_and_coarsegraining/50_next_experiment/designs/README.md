# The two experimental setups — build specification

**Status: FROZEN 2026-10-05.** Both setups are fixed. Nothing has been written into
a task directory or run with language models yet.

This file is the single source of truth for the two setups. If you are an agent
asked to build them, read this file in full, then the JSON beside it, then
[`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md).

| file | what it is |
|---|---|
| `task003_symmetric.json` | **task003-symmetric**: 24 agents and **two** controller pools |
| `task003_nosolution.json` | **task003-nosolution**: 24 agents and **one** controller pool (rebuilt 5 October with eligible agent facts only) |
| `task003_archived_reference.json` | the setup actually run in study `21-09-2026-full-vs-report-v1`; a reference only |
| `config_template.yaml`, `variants.yaml` | the run settings, shared by both setups |

---

## 1. What the two setups are

Both use the **same world**: `task_003`, hidden vector `(3,1,1,2,2,1,1,1,2)`,
allocation scores 8 / 4 / 6, so `ALLOCATION_0` (A0) is correct and
`ALLOCATION_2` (A2) is the wrong target a false controller steers toward. Both
draw on the same 49 true facts.

| | **task003-symmetric** | **task003-nosolution** |
|---|---|---|
| agents hold the 6 decisive facts | **yes**, so together they can prove A0 | **no**, so they cannot prove anything |
| controller pools | **two**: the A0 pool, used only when the controller targets A0; the A2 pool, used only when it targets A2 | **one**, used for both targets: 4 facts favouring each of A0, A1, A2 |
| question it answers | Can the swarm assemble a proof it owns, and can a controller help or stop it? | Can a controller tip a swarm that is genuinely undecided? |
| accuracy ceiling | 1.0 | 0.5 |

Keep both. They ask different questions; the second is not a harder version of
the first.

---

## 2. Every property, side by side

The archived reference is what was actually run. It is not a target to build.

| property | **task003-symmetric** | **task003-nosolution** | archived reference |
|---|---|---|---|
| agents, facts per agent | 24, 1 | 24, 1 | 24, 1 |
| distinct facts held | 18 | 15 | 21 |
| agents per allocation A0 / A1 / A2 | **8 / 8 / 8** | **8 / 8 / 8** | 16 / 5 / 3 |
| decisive facts held | 6 of 6 | 0 of 6 | 6 of 6 |
| agents' joint posterior | [1, 0, 0] | [0.5, 0, 0.5] | [1, 0, 0] |
| minimal proofs the agents can assemble | 12 | 0 | 4 |
| agents per decisive fact / per rival fact | **1.33 / 1.33** | — / 1.6 | 1.5 / 1.0 |
| mean individual belief A0 / A1 / A2 | .367 / .312 / .321 | .349 / .291 / .361 | .377 / .295 / .328 |
| controller pool(s) | A0 pool 12, A2 pool 12 | one pool, 12 | one pool, 24 |
| pool favours A0 / A1 / A2 | A0 pool: A0; A2 pool: A2 | 4 / 4 / 4 | 9 / 6 / 9 |
| whole-pool posterior | A0 pool [.963, .038, 0]; A2 pool [.114, 0, .886] | [.329, .341, .329] | [.308, 0, .692] |
| a pool proves its target? | no | no | no |
| pool facts the agents already hold | 2 in each pool, 1 agent each | 0 | 8 |
| **new facts each controller brings** | **10 and 10** | 12 | 16 |
| ΣΔP of those new facts | A0 .957, A2 .969 | the pool's A0 facts .627, its A2 facts .629 | — |
| all agent facts private-eligible? | **yes** | **yes** | yes |

ΔP means: how much one fact, read alone, raises the probability of an allocation
above the prior of ⅓.

---

## 3. task003-symmetric — how it was built

### The two pools (designed 2 October)

Membership: [`../../10_task_and_facts/symmetric_pools.json`](../../10_task_and_facts/symmetric_pools.json).
Reasoning: [`../../20_controller_redesign/symmetric_controller_and_coarse_graining.md`](../../20_controller_redesign/symmetric_controller_and_coarse_graining.md) §2.

1. **Equal size**: 12 facts each.
2. **Matched weight**: each A0 fact is paired with an A2 fact of near-identical
   ΔP for its target; 10 of the 12 pairs match exactly.
3. **No decisive facts.**
4. **No shared facts.** Without this, 12 facts that only rule out A1 land in
   both pools.
5. **Neither pool can prove its target.** Without this, the A0 pool reaches
   P(A0) = 1 and the A2 pool cannot match it.

Only 12 facts in the whole task raise A2 while lowering A0. The A2 pool is all
of them.

### The agents (chosen 5 October)

Built by [`../builders/build_task003_symmetric.py`](../builders/build_task003_symmetric.py),
which writes `task003_symmetric.json`. Its rules, in order:

1. 24 agents, one fact each, every fact private-eligible, each fact held by 1 or 2 agents.
2. 8 agents per allocation. An agent counts for allocation k when its fact
   raises P(k) at least as much as any other allocation. **A fact tied between
   two allocations may count for either.**
3. The A0 group is exactly the 6 decisive facts, so the agents can prove A0.
4. **Equal new evidence for both controllers.** The pool facts the agents
   already hold must be matched pairs: an A0-pool fact together with its A2
   partner. Both controllers then bring the same number of new facts, of the
   same weight.
5. **Equal duplication.** A fact held by two agents survives forgetting better.
   So decisive facts and rival facts are held by the same mean number of agents
   (8/6 = 16/12 = 1.33). Otherwise, at ρ = 0.75 the proof would fade faster
   than the evidence against it, which favours the A2 controller.
6. Among what is left: fewest tied facts, then the smallest overlap, then rival
   groups of the most similar strength.

45,925 agent sets satisfy rules 1–5. Five others tie with the chosen one; they
swap facts of identical weight.

### The agents

| group (8 agents each) | facts; **bold** = held by 2 agents |
|---|---|
| A0 | **`cf_x00_ge_x04`**, `cf_x01_le_x04`, `cf_x01_le_x05`, `cf_x02_le_x03`, **`cf_x06_le_x08`**, `cf_x07_le_x08` — the 6 decisive facts |
| A1 | `cf_x00_ge_x03` (tie A0/A1), **`cf_x01_eq_x05`**, `cf_x02_ge_x05` (tie A1/A2), **`cf_x03_le_2`**, `cf_x06_le_2` (tie A0/A1), `cf_x06_le_x07` |
| A2 | `cf_x02_le_2` (tie A0/A2), **`cf_x02_le_x04`**, `cf_x03_eq_2` (tie A0/A2), `cf_x03_ge_x05`, **`cf_x06_ge_x07`**, `cf_x08_eq_2` (tie A1/A2) |

Pool facts the agents already hold: `cf_x00_ge_x03` and `cf_x06_le_x07` from the
A0 pool; their partners `cf_x02_ge_x05` and `cf_x08_eq_2` from the A2 pool.

### The price of rule 5

Exact duplication needs 6 tied facts among the 12 rival facts. In 4 of them the
tie is with A0. So "8 agents per allocation" really means: 8 agents favour A0
alone; 8 favour A1 and 8 favour A2, and 4 of those 16 favour A0 equally. The
effect on starting beliefs is small, because the tied facts are weak (ΔP .014
to .063): the mean individual belief in A0 is .367, against .358 for the best
design without ties. **State this in any write-up.**

Two of the decisive facts (`cf_x01_le_x04`, `cf_x02_le_x03`) are themselves tied
between A0 and A2. That was always true of the decisive set.

---

## 4. task003-nosolution — how it was built

Built by [`../builders/build_task003_nosolution.py`](../builders/build_task003_nosolution.py),
which writes `task003_nosolution.json` (about 2 minutes, same result every run).
Rules, in its docstring:

- **Agents**: 24, none of the 6 decisive facts, 8 per allocation, 15 distinct
  facts (5 per allocation), **every one private-eligible**: read alone, none
  makes an agent more than 45% sure. Together they put A0 and A2 at exactly 0.5
  each and can assemble no proof.
- **Ranking**: among the 10,438 agent sets that give that exact coin flip, the
  builder minimises the sum of three imbalances, all in probability units:
  the strength gap between the pool's A0 facts and its A2 facts (the pool serves
  both targets, so neither controller should get stronger facts); the whole
  pool's distance from the prior; and the gap between the agents' mean starting
  belief in A0 and in A2. Result: 0.0007, 0.004 and 0.012.
- **The pool may hold ineligible facts.** Eligibility limits what one agent
  holds. The pool includes the strongest A0 fact (`cf_x00_eq_3`, ΔP .252) and
  its mirror for A2 (`cf_x05_eq_1`, .252) as a matched pair.
- **Why 24 agents**: Darius's `task_004` removed the decisive facts *together
  with the agents holding them*, leaving 15 agents. That mixes up "no proof
  available" with "fewer agents". Here the decisive facts are *replaced*.
- **One pool**: 12 facts, 4 favouring each allocation, proving nothing, sharing
  no fact with the agents.
- **History**: the 2 October version gave two agents each the two strongest
  facts in the task (`cf_x01_eq_1`, `cf_x05_eq_1`, each making an agent 58.5%
  sure), breaking the eligibility rule. It is kept in
  `../archive/task003_nosolution_ineligible_superseded.json`. The rebuild also
  improved the pool's A0/A2 balance (ΣΔP .627 vs .629, against .335 vs .446
  before) and halved the agents' starting gap (.012, against .028).
- **The scarcity bound**: only 9 facts favour A1 and 9 favour A2, against 31
  for A0. Agents and pool compete for them, so zero overlap needs
  `kr + c ≤ 9`. Here `kr` = distinct rival facts the agents hold per allocation
  (5), and `c` = pool facts per allocation (4).

---

## 5. How to read the JSON

Both setup files carry the same `agents` block:

| field | meaning |
|---|---|
| `agents.agent_assignments` | **write this out**: `{"agent_001": ["cf_..."], ...}`, 24 entries |
| `agents.distinct_facts`, `agents.slot_multiplicity` | the facts, and how many agents hold each |
| `agents.joint_posterior`, `agents.minimal_proofs_assemblable`, `agents.decisive_held` | checks to recompute after building |

The pools differ:

| setup | field | meaning |
|---|---|---|
| task003-symmetric | `controller_pools.ALLOCATION_0.fact_ids` | the pool for a controller targeting A0 |
| task003-symmetric | `controller_pools.ALLOCATION_2.fact_ids` | the pool for a controller targeting A2 |
| task003-symmetric | `pool_pairs` | the 12 weight-matched pairs, and which the agents hold |
| task003-nosolution | `controller_pool.fact_ids` | the single pool, for either target |

---

## 6. How the runtime gets the right pool

With `controller_report_pool_mode: frozen`, the controller's pool is the task
directory's `facts/controller_reportable_facts.json`, **whatever the target**.
So a pool is chosen by choosing a task directory:

| task directory | agents | `facts/controller_reportable_facts.json` | used for |
|---|---|---|---|
| `task003_symmetric_to_a0` | symmetric agents | the A0 pool | truth control, and the silent runs |
| `task003_symmetric_to_a2` | symmetric agents | the A2 pool | false control |
| `task003_nosolution` | no-solution agents | the single pool | silent, truth and false control |

The two symmetric directories differ **only** in that one file (plus
`controller_fact_scores`, below), so a diff shows exactly what changed.

A bonus: the deterministic controller ranks facts by `task.controller_fact_scores`,
one field per task. In the symmetric setup each directory can carry scores
toward its own target, so the scripted controller's fact choice is no longer
blind to the target. In the no-solution setup the pool is shared, so that
problem remains; see `../EXPERIMENTS.md` §3c.

Step-by-step build instructions: [`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md) §2.

---

## 7. Open issues

- **The LLM-free simulations in `../../60_exact_simulation/` and
  `../../70_codex_simulations/` ran the superseded designs of both setups**
  (kept in `../archive/`). Their results do not describe the frozen setups and
  need a rerun with the new simulator specification.
- **Nothing logs the proof-assembly rate**: per round, the fraction of agents
  whose active memory holds a complete proof. That is what task003-symmetric
  exists to measure.
- **One world.** Every number here is tuned to `task_003`. Balance in one world
  does not show the result generalises.

---

## 8. Where the surrounding context lives

| | |
|---|---|
| what `task_003` is, the 49 facts, the proofs | `../../10_task_and_facts/task_003_analysis.md` |
| why the archived controller was unfair, and the two-pool design | `../../20_controller_redesign/` |
| how the experiment is run | `../EXPERIMENTS.md`, `AGENT_RUNBOOK.md` |
| superseded designs and builders | `../archive/` |
