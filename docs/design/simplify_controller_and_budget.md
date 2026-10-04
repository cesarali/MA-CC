# Simplifying the blackboard controller

Four changes requested, in the order they should be made. Change 4 turned out to
be the opposite of what we assumed, and it changes what the other three are for.

---

## 4. Are the facts too informative? No — they are barely used

This was the worry: *"with a single fact, an agent could vote confidently in an
option."* The task data records the answer exactly, so we do not have to guess.

### What one controller fact does to a belief

`task_003/symbolic/individual_controller_fact_audit.json` stores, for each of the
24 controller-reportable facts, the posterior over the three allocations given
that fact **alone**. Prior is 0.333 each.

| statistic over the 24 facts | value |
|---|---|
| highest posterior any single fact reaches | 0.585 |
| median of the highest posterior | 0.423 |
| lowest | 0.347 |
| facts that rule out *any* allocation | **0 of 24** |
| facts whose best guess is the truth | 9 of 24 |

The single most informative controller fact (`cf_x05_eq_1`, 0.585) points at
**ALLOCATION_2 — the wrong answer**. The truth is ALLOCATION_0.

So a controller fact moves an agent from 33% to roughly 42%. It never settles
anything. The controller pool is deliberately weak.

### The decisive facts are not the controller's to post

Six facts *are* decisive — each one alone determines the answer:

    cf_x00_ge_x04  cf_x01_le_x04  cf_x01_le_x05
    cf_x02_le_x03  cf_x06_le_x08  cf_x07_le_x08

**None of them is in the controller pool.** All six sit in agents' private
holdings. Each agent starts with exactly one fact, and 9 of the 24 agents hold a
decisive one.

That asymmetry is the actual design of the game: agents can share the facts that
settle the question, the controller cannot. The controller can only shade
emphasis with facts that are individually inconclusive.

### The finding that matters: agents cannot use a decisive fact

Measured over the 20 finished no-control episodes, initial votes only (before any
communication):

| agent | votes for the truth |
|---|---|
| holds a decisive fact | **40.0%** |
| holds a non-decisive fact | 33.3% (chance) |

A fact that *logically determines the answer* buys 6.7 percentage points. The
model is essentially not reasoning from it.

This reframes everything upstream of it. The hidden-profile mechanism we believe
we are studying is not really running: the population's information is not being
converted into votes, so what the controller does on the board is a second-order
effect on top of agents that largely ignore evidence they already hold.

**Consequence for the other three changes.** Making the facts *less* informative
would push the game further toward noise. If anything the lever to pull is the
opposite — or the prompt, which is change 1.

Open question for the operator: is a controller-vs-agents experiment worth
running at all before the agents demonstrably use their own evidence? A cheap
diagnostic is proposed at the end.

---

## 1. Fix the controller prompt

### Bug: the budget is never given as a number

The instruction reads, literally:

    Choose exactly budget distinct eligible verified facts ...

The word `budget` is plain English text. The actual number appears only inside a
JSON blob further down, under `CONTROLLER INFORMATION`, as `"budget": 3`. The
model has to infer that the bare word refers to a JSON field.

Verified: `"6" in instruction_text` is `False` when the budget is 6.

This is survivable for a fixed per-round budget. It becomes untenable under
change 2, where the number differs every round.

**Fix:** interpolate the value, and say what it counts.

    Choose exactly 3 distinct eligible verified facts ...

### Other prompt problems

- **No sense of time.** The controller is told `round_index` but never the
  horizon. It cannot ration anything without knowing how many rounds remain.
  Required by change 2.
- **No statement of what it cannot do.** It is never told that the facts it may
  post are individually inconclusive, nor that agents hold evidence it does not.
  It behaves as if its facts should be persuasive on their own.
- **`reason` is underspecified.** "private strategy explanation" produces
  post-hoc narration. Asking *why these facts and not others* would make the
  field analytically useful.

---

## 2. Budget b for the whole game

Today `intervention_budget` is a **per-round** quota: on a round where the
controller speaks it posts exactly b (the `contextual_weighted_v1` and
`llm_authored_fixed_*` policies) or 1..b (the `llm_authored_report_only_v1` and
`llm_structured_v1` policies).

Wanted: b is the **total** for the episode, spent however the controller likes.

### Implementation sketch

New option, defaulting to today's behaviour so nothing existing changes:

```yaml
controller_budget_scope: per_round   # default, current meaning
controller_budget_scope: episode     # b is the whole-game allowance
```

- Runtime tracks `controller_budget_spent` across rounds.
- `ControllerCommunicationContext.budget` becomes **remaining** budget.
- Two new context fields: `rounds_remaining`, `budget_spent`.
- When remaining hits 0 the controller is silent for the rest of the episode.
- Record `controller_budget_remaining` per round so the spend curve is
  analysable.

### The ceiling that will bite

`controller_report_max_posts_per_fact: 3` over a 24-fact pool caps an episode at
**72 posts** regardless of b. Any total budget at or above ~72 is not a budget,
it is the cooldown rules. Total budgets should stay well under that.

---

## 3. Remove REPORT / REQUEST / DIRECTIVE

These labels are threaded through more than the prompt:

| where | what it does |
|---|---|
| `CommunicationMode` enum | REPORT / REQUEST / DIRECTIVE / MIXED |
| `allow_controller_requests`, `allow_controller_directives` | gate two of them |
| board messages | every message carries `message_type` |
| `message_type_counts` | per-round analysis field |
| agent prompts | agents are shown each message's type |
| analysis | `communication_funnel`, `directive_*` metrics, dashboards |

Dropping the labels from the *controller* is easy. Dropping them from the board
and the analysis breaks comparability with every existing study.

**Proposed middle path:** keep one mode internally (REPORT, since that is the
only one the recent studies allow anyway: `allow_controller_requests: false`,
`allow_controller_directives: false`), and remove the taxonomy from both prompts
so neither the controller nor the agents ever see a type label. Existing analysis
columns keep working; they simply become constant.

---

## Proposed order

1. Prompt fixes (1) — cheap, no schema change, needed by 2.
2. Budget scope (2) — new option, default preserves behaviour.
3. Label removal (3) — prompt-level only.
4. A cheap diagnostic for (4): give one agent one decisive fact and ask it to
   vote, 50 times, no population. If it cannot beat ~40%, fix that before
   running another controller study.

## Open decisions

- Total-budget values to sweep.
- Whether to keep `max_posts_per_fact` / cooldown under a total budget.
- Whether to run the decisive-fact diagnostic first.
- Which base config to fork (v1 is the full-fill, LLM-authored one).
