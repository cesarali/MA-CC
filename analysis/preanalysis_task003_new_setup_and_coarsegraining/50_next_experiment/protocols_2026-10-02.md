# The three protocols, the full option surface, and what Darius set

2 October 2026. Read against the code, not from configs alone. Sources:
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/{controller,state}.py`
on `dev/rsanchez` and on `origin/darius-MA-v1`, and the twelve `task004` study
directories on that branch.

**Headline: `darius-MA-v1` must be merged before either new setup can be
configured at all.** Not for convenience — the options plan v0.4 requires do not
exist on `dev/rsanchez`. Details in §4.

---

## 1. The two pool options, properly

These are two different options that do two different jobs. Confusing them is
easy and was the original defect.

| option | job | target-aware? | branch | values |
|---|---|---|---|---|
| `controller_fact_pool_mode` | **which facts are on the menu** | **no** | **darius only** | `frozen`, `all_nondecisive`, `balanced` |
| `controller_report_pool_mode` | **how the menu resolves against the target** | **yes** | both | `frozen`, `target_aligned_v1` |

### `controller_fact_pool_mode` — the menu

Resolved by `reportable_fact_ids(task)`. Note the signature: **it takes no
target**. That is the point — one menu, whichever direction is being steered.

- `frozen` — the task's `facts/controller_reportable_facts.json`. For
  `task_003` that is the defective 24-fact pool (lean 9/6/9, joint posterior
  [0.308, 0, 0.692]).
- `all_nondecisive` — all 49 true facts minus the 6 decisive ones: 43 facts,
  leaning **25 / 9 / 9** toward truth. A controller given this starts with a
  menu slanted toward the answer it may be trying to hide.
- `balanced` — reads `controller/balanced_fact_pool.json` from the task
  directory into `task.controller_balanced_fact_ids`. Darius's is 27 facts at
  9/9/9. **Our 12-fact pools use this same mode and file name**, since the
  runtime only reads the `fact_ids` list.

### `controller_report_pool_mode` — the target resolution

Resolved by `reportable_fact_ids_for_target(task, episode_seed)`.

- `frozen` — returns the task's frozen pool **whatever the target is**. This is
  the original defect: admission was computed once for `ALLOCATION_2`, and this
  mode then handed the same 24 facts to the truth arm.
- `target_aligned_v1` — a **truth-arm-only override**. When the resolved target
  is the truth target *and* differs from the task's declared controller target,
  it substitutes the explicit `controller_report_pool_fact_ids` list. Otherwise
  it falls back to the frozen pool.

**Why `target_aligned_v1` cannot serve our designs.** It produces two different
menus by construction, which defeats the one-shared-pool rule. And it enforces
three hard checks — each a `raise ValueError` — that our designs violate:

1. the explicit pool must contain **every decisive fact**; ours contains none;
2. it must **not overlap** the frozen false-target pool;
3. the target must be a truth target **distinct from** the task's declared one.

So there is no fallback. The target-independent menu only exists on
`darius-MA-v1`.

---

## 2. Agent communication protocol

Set under `game.options`, mostly in the `board:` block. Parsed in `state.py`.

| option | values | Darius's task004 |
|---|---|---|
| `social_mode` | `peer`, `board` | `board` |
| `social_group_size` | integer | **7** |
| `board.sampling` | `uniform`, `full` | **`uniform`** |
| `board.message_lifetime_rounds` | integer | **1** |
| `board.communication_profile` | `report_only`, `full_communication` | `report_only` (one study uses `full_communication`) |
| `board.exclude_self_authored` | bool | `true` |
| `board.allow_no_post` | bool | `true` |
| `board.allow_participant_requests` | bool | — |
| `board.require_grounded_reports` | bool | **`true`** |
| `board.report_citation_scope` | `active_only`, `active_or_observed` | **`active_or_observed`** |
| `board.no_citable_fact_action` | — | `none` |
| `epistemic_persistence` | float | 0.6 / 0.75 / 1.0 |
| `vote_visibility` | — | `public` |
| `receiver_epistemic_disposition` | — | `vigilant` |
| `stop_on_consensus` | bool | `false` |
| `initialization.mode` | — | `paired_local_vote`, with a frozen `artifact_dir` and `require_artifact: true` |

Message types an agent may post: `REPORT`, `REQUEST`, `NONE`.
`communication_profile: report_only` restricts them to `REPORT` and `NONE`;
`full_communication` adds `REQUEST`.

### Three settings that matter more than they look

**`board.sampling: uniform` with `social_group_size: 7`.** Each agent reads a
**uniform sample of 7 messages**, not the whole board of 15. The agents are
already partially observing. Plan v0.4 discusses full-versus-partial sensing
only for the *controller* and says nothing about this. It changes everything
downstream — whether a fact reaches an agent at all is a sampling event each
round. `board.sampling: full` exists if we want to remove it.

**`report_citation_scope: active_or_observed`.** An agent may cite a fact it
*saw on the board*, not only one in its own active memory. **This is the
propagation mechanism** — it is how a fact spreads beyond its original holder,
and how a controller's post gets amplified by agents. With `active_only`,
facts could not propagate at all and the board would be inert.

**`require_grounded_reports: true` with `message_lifetime_rounds: 1`.** Every
report must cite a fact, and the board clears each round. Combined with
`epistemic_persistence < 1`, a fact survives only if some agent holds it in
active memory *or* it was re-posted last round. That is the decay channel our
coarse-graining tracks.

---

## 3. Controller communication protocol

Set under `control.options`. Parsed in `controller.py`.

| option | values | Darius's task004 |
|---|---|---|
| `target` | `ALLOCATION_0` … `_2`, or `correct` | `ALLOCATION_0` / `ALLOCATION_2` |
| `controller_actuation_mode` | `direct_recommendation`, `coordination_request`, `truthful_strategic_report`, `adaptive_communication` | `adaptive_communication` |
| `message_mode` | `recommendation_only`, `recommendation_plus_fact`, `silent` | `recommendation_only` |
| `controller_authoring` | `deterministic`, `llm_authored` | both, by study |
| `controller_timing` | `microscopic`, `dawn_only` | `dawn_only` |
| `sensing_mode` | `votes`, `board` | `board` |
| `sensor_sample_size` | integer | **100** (≥ 15, so effectively full) |
| `policy` | `soft_target`, … | `soft_target` |
| `threshold` | float | 0.5 |
| `beta` | float | 4.0 |
| `advocacy_schedule` | `soft`, `always`, `never` | `always` or `soft`, by study |
| `controller_report_selection_strategy` | `target_preserving_v1` | `target_preserving_v1` |
| `controller_report_cooldown_rounds` | integer | 0 |
| `controller_report_max_posts_per_fact` | integer | 90 |
| `controller_memory_mode` *(darius only)* | `none`, `public_ledger` | `none` |
| `controller_evidence_strategy` | `neutral`, `strategic` | — |
| `evidence_mode` | — | — |
| `controller_fact_id`, `controller_fact_selector` | `supporting` | — |
| `allow_controller_requests`, `allow_controller_directives`, `controller_allow_mixed_message_types` | bool | — |
| `controller_communication_policy`, `_version`, `_max_retries`, `_fallback_policy` | — | retries 2 |

Controller message types: `REPORT`, `REQUEST`, `DIRECTIVE` — a superset of what
agents may post.

### What to decide here

**`advocacy_schedule` is the activation gate, and it is a confound.** `soft`
gates posting on the sensed state crossing `threshold`; `always` removes the
gate. The archived batch used `soft` and fired on only 32% / 54% of rounds, and
that firing-rate difference — not the content — was most of the measured
directional effect. Fix the schedule across arms, or measure activation as a
separate quantity. **Do not condition on realised firing rounds and call the
result a causal effect.**

**`sensor_sample_size: 100` against 15 agents is full sensing.** v0.4's
partial-board arm means a smaller number here. That arm does not exist yet.

**`controller_authoring` is the choice we recommended deciding first**:
`deterministic` makes the posting sequence a known function of the sensed state;
`llm_authored` lets a model improvise. See
`design_preliminary_2026-10-02.md` §2.

---

## 4. Controller budget protocol — and why the merge is mandatory

**Three different things in this codebase are called a budget.** Keep them
apart:

| | what it limits | where |
|---|---|---|
| `intervention_budget` + `controller_budget_scope` | **the control budget** — how many facts the controller may post. v0.4's `B` and `b`. | `control.options` |
| top-level `budget:` block | **money and API calls** — `max_cost_per_run`, `max_provider_requests`, token caps | top of the config |
| `controller_report_max_posts_per_fact` | how often a *single* fact may be reposted | `control.options` |

### The scope option is darius-only

| option | values | on `dev/rsanchez`? |
|---|---|---|
| `controller_budget_scope` | `per_round`, `episode` | **no** |
| `controller_round_budget_mode` | `exact`, `zero_or_exact` | **no** |
| `controller_memory_mode` | `none`, `public_ledger` | **no** |

Without `controller_budget_scope`, there is no full-episode reservoir —
**v0.4's priority-1 experiment cannot be configured on `dev/rsanchez` at all.**

### And the budget is bounded by the pool, except under episode scope

On `dev/rsanchez`, `intervention_budget` is hard-checked against the number of
distinct pool facts and raises if it exceeds them. Darius relaxed that for
episode scope, with the reasoning in a code comment:

> a per-round quota cannot exceed the pool. A whole-game allowance is spent
> across rounds and may reuse facts, so it legitimately can; the runtime caps
> each round's offer at the pool instead.

**What this means for our 12-fact pools:**

| v0.4 cell | works with a 12-fact pool? |
|---|---|
| `b = 3`, per-round | yes (3 ≤ 12) |
| `b = 6`, per-round | yes (6 ≤ 12) |
| `B = 90`, episode | yes — **but only with darius's relaxation** |
| `B = 180`, episode | same |

Note `controller_report_max_posts_per_fact` must then be large enough to permit
the reuse that an episode budget implies: Darius sets it to 90, equal to `B`.

---

## 5. What Darius actually ran on task004

Ten study directories, all at **30 rounds, 15 agents, `board.sampling: uniform`,
`gpt-oss-120b` at temperature 1.0, 10 repetitions**, results under
`/shared/MA-CC-results` at execution site `cesar`.

| study | ρ | comms | pool mode | budget | scope | authoring | schedule |
|---|---|---|---|---|---|---|---|
| `24-09…no-control` | 0.75 | report_only | — | — | — | — | — |
| `24-09…rho075` | 0.75 | report_only | — | — | — | — | — |
| `24-09…rho1` | 1.0 | report_only | — | — | — | — | — |
| `25-09…false-rho075` | 0.75 | report_only | *(pre-balanced)* | 30 | episode | llm_authored | always |
| `25-09…false-rho1` | 1.0 | report_only | *(pre-balanced)* | 30 | episode | llm_authored | always |
| `28-09…balanced-b90…rho1` | 1.0 | report_only | **balanced** | 90 | episode | llm_authored | always |
| `28-09…false-balanced-pool` | 0.75 | **full_communication** | **balanced** | 3 | per_round | llm_authored | soft |
| `29-09…balanced-b90…rho06` | 0.6 | report_only | **balanced** | 90 | episode | llm_authored | always |
| `29-09…balanced-b90…rho075` | 0.75 | report_only | **balanced** | 90 | episode | llm_authored | always |
| `29-09…classical-b3-gated` | 0.6 | report_only | **balanced** | 3 | per_round | **deterministic** | soft |

### How that compares with plan v0.4

| v0.4 requirement | status |
|---|---|
| 30 rounds | **done** |
| front-page board (`message_lifetime_rounds: 1`) | **done** |
| ρ ∈ {0.75, 1.0} | **done**, plus 0.6 |
| truth and false control, shared no-control baseline | **done** |
| full-episode budget protocol | **done at B = 90** — missing B = 180 |
| per-round budget protocol | **done at b = 3** — missing b = 6 |
| full communication | **one study only** |
| report-only | **done** |
| full-board vs partial-board **controller** sensing | **missing** — every study is effectively full |
| 50 episodes per cell | **missing** — all are 10 |

So the structure is complete and the coverage is thin. The gaps are: the second
budget level of each pair, a partial-sensing arm, more `full_communication`, and
repetitions.

### Two things he changed that are easy to miss

No `results/` were committed — they live at `/shared/MA-CC-results` on César's
site. And the balanced pool is excluded from the paired-initialization hash, so
the `24-09` no-control baselines still pair against the later controlled arms.
That is what makes the shared baseline v0.4 asks for actually reusable.

---

## 6. What this leaves to decide

1. **Merge `darius-MA-v1` first.** Not optional; nothing else can proceed.
   Scope of what we need: `controller_fact_pool_mode` (+ `balanced` and the
   `data.py` loader), `controller_budget_scope`, `controller_round_budget_mode`,
   `controller_memory_mode`, and the episode-scope budget relaxation.
2. **Agent board sampling: keep `uniform` at 7 of 15, or switch to `full`?**
   Partial agent observation is a mechanism nobody has decided on — it is
   inherited, not chosen. With 24 agents the sample fraction changes too.
3. **`report_citation_scope`: keep `active_or_observed`?** It is the propagation
   mechanism. `active_only` would make the board inert.
4. **`advocacy_schedule`: fix it across arms**, or treat activation as a measured
   variable. The archived 32% / 54% split is why.
5. **Controller authoring: `deterministic` first** — see
   `design_preliminary_2026-10-02.md` §2.
6. **`sensor_sample_size`** for the partial-sensing arm.
