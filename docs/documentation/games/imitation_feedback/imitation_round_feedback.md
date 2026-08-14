# `hidden_bench_imitation_round_feedback`, end to end

This document explains **what the game in
[`src/mas_cc/games/hidden_bench/imitation_round_feedback/`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/)
actually does**: one step-by-step pass over an episode, then a full reference
for every controller option. Each section gives the plain-language version
first and the precise version second.

**How this fits with the other documents.**

| Document | Answers |
| --- | --- |
| [`games/hidden_bench/README.md`](../hidden_bench/README.md) | What the HiddenBench family is and how the three base games differ |
| [`games/hidden_bench/imitation_game_mechanics.md`](../hidden_bench/imitation_game_mechanics.md) | How the *event-clock* imitation game is played — evidence, assignment, prompts, focal update |
| [`games/hidden_bench/classical_dynamics_and_the_imitation_model.md`](../hidden_bench/classical_dynamics_and_the_imitation_model.md) | What the provider-free kernels mean physically |
| **this document** | **How the round-feedback variant is played, and every controller knob** |

Everything about *evidence, population construction, prompts, parsing, and the
local trajectory schema* is inherited unchanged from `hidden_bench_imitation`
and is not repeated here. What is new is the **clock structure** and the
**controller**.

---

## Table of contents

1. [Why this is a separate game](#1-why-this-is-a-separate-game)
2. [The two clocks](#2-the-two-clocks)
3. [Step by step through one episode](#3-step-by-step-through-one-episode)
4. [The controller: mechanisms](#4-the-controller-mechanisms)
5. [The controller: every option](#5-the-controller-every-option)
6. [The two dynamics modes](#6-the-two-dynamics-modes)
7. [What is recorded](#7-what-is-recorded)
8. [What is estimated](#8-what-is-estimated)
9. [A worked configuration](#9-a-worked-configuration)
10. [Failure modes and validation errors](#10-failure-modes-and-validation-errors)

---

## 1. Why this is a separate game

### Plain terms

In `hidden_bench_imitation` the controller looks at the population and decides
whether to speak **before every single agent update**. That is a controller
with unlimited hands: it can touch every event, and its sensing and its acting
happen on the same clock. You cannot then ask "how much did *sensing* buy you"
separately from "how much did *acting* buy you", because the two are never
varied independently.

This game splits them. Once per **population round**, the controller:

1. peeks at `q_c` randomly chosen agents' current votes (**sensing**),
2. makes **one** decision — stay quiet or advocate option `Z` (**policy**),
3. and, if it advocates, gets exactly `b` chances to speak during that round
   (**actuation budget**), placed at random positions.

`q_c` and `b` are then two independent dials — how much the controller *sees*
and how much it can *do* — and an experiment grid over the pair is the whole
scientific point.

### Technical terms

`hidden_bench_imitation_round_feedback` keeps the HiddenBench evidence
structure, population preparation, initialization, reasoning prompts, focal
update, population observables, and microscopic trajectory schema of
`hidden_bench_imitation`. It replaces the runtime with a two-clock loop and
adds one round-level record type.

- Game class:
  [`game.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/game.py)
  — `HiddenBenchImitationRoundFeedbackGame`, a subclass of
  `HiddenBenchImitationGame`.
- Runtime:
  [`runtime.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/runtime.py).
- Controller:
  [`controller.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/controller.py),
  building on the sensor/policy/actuator code in
  [`../imitation/controller.py`](../../../../src/mas_cc/games/hidden_bench/imitation/controller.py).
- Rules and record types:
  [`state.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/state.py).
- Provider-free kernel:
  [`classical.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/classical.py).
- Estimators:
  [`analysis.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py).

---

## 2. The two clocks

| Clock | Index | Length | What happens |
| --- | --- | --- | --- |
| Slow (population round) | `round_index` | `game.options.rounds` | Exactly **one** controller sense + decide + schedule |
| Fast (update position) | `within_round_index` | Exactly **`N`** per round | One focal-update opportunity |

An episode therefore contains `rounds x N` microscopic focal updates and
`rounds` controller decisions. The generic `game.horizon` field is still
required by the repository config schema, but
`RoundFeedbackRules.from_config` treats **`game.options.rounds` as the
authoritative slow-clock horizon** and sets the internal micro-horizon to
`rounds * N`. Keeping `horizon` equal to `rounds` in the YAML avoids
confusion — that is what every shipped config does.

A round that begins always runs all `N` positions. `stop_on_consensus` is
evaluated only at slow-clock boundaries (`turn % N == 0`), so consensus never
truncates a round mid-way and never leaves a partially actuated schedule.

---

## 3. Step by step through one episode

Symbols used throughout: `N` = population size, `q` =
`game.options.social_group_size`, `q_c` = `control.options.sensor_sample_size`,
`b` = `control.options.intervention_budget`, `Z` = the controller's target
option.

### 3.0 Setup and seeding

`RoundFeedbackRules.from_config` validates the config, reusing every inherited
`ImitationRules` check (population size, `social_group_size` in `[1, N-1]`,
prompt version, initialization block, paraphrase preparation, retry policy).
Two extra guards are its own: `rounds` must be a positive integer, and
`game.options.classical.kernel` must be `controlled_imitation_round_reference`.

The runtime also checks that `prompt.prompt_family` is a known HiddenBench
imitation family and that `prompt.prompt_version` equals
`game.options.prompt_version` — a mismatch is a hard error, not a silent
override.

Five independent RNG streams are derived from `execution.seed`, so any one of
them can change without disturbing the others:

| Stream | Derivation label | Governs |
| --- | --- | --- |
| Participants | `round-feedback-focal-and-peer-selection` | Which agent is focal, which `q` peers |
| Transition | `round-feedback-classical-transition` | Classical-mode kernel draws |
| Sensor/policy | `round-feedback-controller-sensor-policy` | Which `q_c` agents are peeked at, and the policy coin |
| Slot replacement | `round-feedback-controller-slot-replacement` | Which peer slot the controller displaces |
| Schedule | `round-feedback-schedule:{round_index}` | The `b` controlled positions of that round |

The schedule stream is derived **per round**, so the actuation schedule of
round `t` is reproducible on its own and its seed is written into the record.

### 3.1 Building the population

Inherited verbatim from `hidden_bench_imitation.initialize`: load the task set,
assign shared/private evidence to `N` agents under `assignment_scheme` (with
optional automatic paraphrase preparation from the annotation file), assert the
union invariant (the pooled evidence really does identify the correct answer),
apply the scenario variant, and shuffle each agent's presented facts unless
`shuffle_facts: false`.

### 3.2 Initial votes

Two paths, chosen by `game.options.initialization.mode`:

- `local_vote` **and** `dynamics_mode: reasoning` — every agent is asked once,
  in parallel, for a vote from its own evidence. These are the only provider
  calls outside the round loop, and they are recorded as `initial_decisions`.
- anything else (`uniform_random`, an explicit `initial_votes` list, or an
  `initial_distribution`) — votes are drawn provider-free from a derived
  stream. Classical mode *must* take this path; a classical run that would need
  provider calls to initialize is an error.

### 3.3 The round: sensing

At the top of each round the runtime snapshots
`population_state_before` and calls `Control.round_signal` **once**.

What the controller is allowed to see is not the game state — it is a stripped
projection built by `_controller_view`:

```text
GameState(
  agents = [ {agent_id, committed_action} ... ],   # votes only
  data   = { seed, task = {task_id, possible_answers, correct_answer} },
)
```

Evidence, rationales, dialogue transcript, and private memory are **removed
before sensing**. A controller therefore cannot be accused of steering on
information the population had and it should not have had.

The controller resolves its target `Z` (see
[§5.1](#51-target--which-option-the-controller-pushes)), samples `q_c` distinct
agents without replacement from the sensor stream, reads their committed votes,
and forms the sampled target share

```text
p_Z(Y_t) = #{sampled agents currently voting Z} / q_c
```

which is a hypergeometric estimate of the true share. Agents that have not yet
committed contribute `None` and are counted as non-target.

### 3.4 The round: the policy decision

The soft policy converts that share into one action:

```text
P(U_t = ADVOCATE_Z | Y_t) = sigma( beta * (theta - p_Z(Y_t)) )
U_t in {ADVOCATE_Z, NO_OP}
```

`sigma` is the logistic function, `theta` = `threshold`, `beta` = `beta`. It is
monotonically decreasing in the sampled support — the controller pushes harder
when `Z` looks like it is losing — and equals exactly `0.5` at the threshold.
The realized action, the probability it was drawn with, the sampled agent ids,
their opinions, and the count vector are all recorded.

If the action is `ADVOCATE_Z`, the controller also renders its message once for
the round (template version 3, [§5.6](#56-template_version--what-the-controller-says)).

### 3.5 The round: the actuation schedule

- `U_t = NO_OP` → the schedule is empty. Nothing is actuated this round even if
  `b > 0`.
- `U_t = ADVOCATE_Z` → `sample_controlled_positions` draws exactly `b` distinct
  positions uniformly without replacement from `{0, ..., N-1}`, sorted into
  canonical order.

The positions are hashed (SHA-256 over the canonical JSON list) and both the
hash and the schedule seed are stored, so a schedule can be replayed or
verified without trusting the stored list.

### 3.6 The `N` update positions

For each `within_round_index` in `0 .. N-1`:

1. **Select participants.** `q + 1` distinct agents are drawn in a single
   sample: the first is the **focal**, the remaining `q` are the sampled peers.
   Only the focal may change its vote.
2. **Is this position controlled?** `controlled_slot = within_round_index in
   controlled_positions`.
3. **If controlled, the controller displaces a peer.** One of the `q` peer
   slots is chosen uniformly from the replacement stream and its occupant is
   removed from the interaction. The remaining `q - 1` peers are the
   *effective* peers. This is the key design decision: **the controller
   competes for an existing influence slot; it does not add one.** A controlled
   focal is exposed to exactly as many voices as an uncontrolled focal, so the
   measured effect is substitution, not extra volume. For `q = 1` the single
   peer is fully replaced and the controller is the focal's only social input.
4. **Run the interaction** in the configured dynamics mode ([§6](#6-the-two-dynamics-modes)).
5. **Apply the transition** and write one microscopic row enriched with the
   round fields (`round_index`, `within_round_index`, `round_controller_action`,
   `round_controller_target`, `round_controller_advocate_probability`,
   `controlled_slot`, `intervention_budget`, the schedule hash, the sampled and
   effective peer ids, and the replaced peer id).

The controller never writes a vote. `Control.override` returns `None` by
design: feedback shapes the interaction the focal sees, and the focal — model
or kernel — still decides.

### 3.7 Closing the round

The population is snapshotted again and one `imitation_round_feedback` record
is written containing the before/after occupation vectors, the target and truth
counts, the order parameters (`m_ctrl`, `m_truth`, `m_order`, `H_vote`) and
their deltas, the full sensor observation, the schedule, and every controller
parameter in force. That single row is the unit of analysis
([§8](#8-what-is-estimated)).

The episode ends when `rounds` rounds have run (`max_rounds_reached`) or, if
enabled, on consensus at a round boundary.

---

## 4. The controller: mechanisms

`control.mechanism` selects one entry from
[`src/mas_cc/control/registry.py`](../../../../src/mas_cc/control/registry.py).
Only two are meaningful for this game.

| Mechanism | Clock | Usable here | What it does |
| --- | --- | --- | --- |
| `none` | — | ✅ **the uncontrolled baseline** | No sensing, no actuation. `controller_enabled` is `false` in every round record, and the analysis target defaults to the correct answer so baseline and controlled cells share one column set. |
| `round_soft_target_budgeted` | round | ✅ **the controller of this game** | Sense `q_c` once, draw one soft action, actuate at `b` random positions. |
| `soft_target` | interaction | ❌ | Event-clock controller of `hidden_bench_imitation`. Its `round_signal` is the inert base implementation returning `None`, so the runtime aborts with *"the selected control does not implement round-level signaling"*. |
| `threshold_target` | interaction | ❌ | Same, plus a deterministic policy. See [§5.4](#54-beta--how-soft-the-policy-is) for why a hard threshold is a poor fit for this game's estimators anyway. |

`RoundSoftTargetBudgetedControl` inherits the sensor and the soft policy from
`SoftTargetControl` and deliberately makes `interaction_signal` return `None`:
the class cannot be misused as an event-clock controller by accident.

---

## 5. The controller: every option

All of these live under `control.options`.

```yaml
control:
  mechanism: round_soft_target_budgeted
  options:
    target: correct            # §5.1
    sensor_sample_size: 2      # §5.2  (q_c)
    threshold: 0.5             # §5.3  (theta)
    beta: 4.0                  # §5.4
    intervention_budget: 6     # §5.5  (b)
    policy: soft_target        # §5.7  (fixed)
    template_version: 3        # §5.6  (fixed)
```

### 5.1 `target` — which option the controller pushes

| Value | Meaning |
| --- | --- |
| `correct` (default) | The task's correct answer. Requires the task to carry one. This is *beneficial* control: the controller pushes the population toward the truth. |
| `random_incorrect` | One wrong option, drawn uniformly from the incorrect options using a **stream derived from the episode seed and task id**. It is therefore fixed for the whole episode — the controller never changes direction mid-run — and reproducible without consuming the sensing stream. |
| An explicit option label, e.g. `West City` | That exact option. Used to target the *shared-information decoy* — the wrong answer the shared evidence already favours — which is a strictly harder adversarial condition than a random wrong option. |
| A non-negative integer | Zero-based index into `task.possible_answers`. |

The resolved target is written to every round record as `controller_target`,
and `analysis_target` is the column the estimators condition on
(`controller_target` when a controller is present, the correct answer
otherwise). The runtime rejects a target outside the task's option alphabet.

**Why the distinction matters.** With `target: correct`, "the controller
worked" and "the population got it right" are the same event, and `m_ctrl` and
`m_truth` move together. With a decoy target they are opposed, which is the
only way to tell a controller that *transmits information* from one that merely
*agrees with the truth*.

### 5.2 `sensor_sample_size` — how much the controller sees (`q_c`)

Number of agents peeked at per round, sampled **without replacement**, so the
sensor is hypergeometric. Must be a positive integer (validated when the control
is built) and cannot exceed `N` (checked by the runtime before the episode
starts, and again by the sensor itself on every round).

- `q_c = 1` — nearly blind; the sampled share is a single Bernoulli draw.
- `q_c` small (2–4) — the interesting regime: the sensor is noisy and the
  policy's mistakes are the observable of interest.
- `q_c = N` — perfect observation of the vote vector.

The sensing fraction `q_c / N` is recorded per round, and sensor error against
the true share is reported as `round_sensor_mae` / `round_sensor_mse`.

### 5.3 `threshold` — where the policy flips (`theta`)

A number in `[0, 1]`, default `0.5`. The advocacy probability equals exactly
`0.5` when the sampled target share equals `theta`, and rises as support falls
below it. Raising `theta` makes a controller that keeps pushing even when `Z`
is already well ahead; lowering it makes one that only intervenes in a crisis.

### 5.4 `beta` — how soft the policy is

Any positive number, default `4.0`. It sets the slope of the logistic:

- `beta → infinity` recovers the hard `threshold_target` step (advocate iff
  support `< theta`);
- `beta → 0` approaches an unconditional coin flip at probability `0.5`;
- intermediate values keep the action **stochastic in a given population
  state**.

That stochasticity is not decoration. The actuation statistic
`I(U_t ; N(t+1) | N(t))` is only identifiable if the same conditioning state
sometimes sees `ADVOCATE_Z` and sometimes `NO_OP`; a deterministic policy puts
every event in a slice with one action and leaves the estimator with no
within-slice contrast, however many episodes are run. The
`round_dual_action_state_fraction` / `round_dual_action_event_fraction`
diagnostics exist to report whether the chosen `beta` actually produced that
overlap.

### 5.5 `intervention_budget` — how much the controller can do (`b`)

A non-negative integer, `0 <= b <= N`. It is an **exact** budget, not a rate:
on an advocating round the controller speaks in exactly `b` of the `N` update
positions, chosen uniformly without replacement.

- `b = 0` — a **sensing-only** arm. The controller still senses, still decides,
  and `U_t` is still recorded, but nothing is actuated. This is the reference
  cell that separates "the controller's decisions correlate with the
  population" from "the controller's decisions moved the population".
- `0 < b < N` — the partial-actuation regime the pilot grids sweep.
- `b = N` — every position of an advocating round is controlled.

The actuation fraction `b / N` is recorded per round.

### 5.6 `template_version` — what the controller says

**Must be `3` for this game.** Any other value is a configuration error
(*"round feedback requires fixed peer-style template version 3"*), even though
the parent module implements three templates:

| Version | Text | Status here |
| --- | --- | --- |
| 1 | Announces itself: *"The external controller currently advocates option Z…"* | Rejected. An agent told an outside system is speaking can discount the turn, so the measured effect is not social influence. Kept in the codebase only to reproduce pre-2026-08-11 runs. |
| 2 | Four peer-voiced paraphrases, each citing one **shared** fact | Rejected here: drawing a paraphrase *and* a fact adds a second random channel `U_t → M_t` on top of `Y_t → U_t`, and the actuation estimate could no longer say which the population responded to. |
| 3 | One fixed line, no factual claim, no fact slot | **Required.** Deterministic given `Z`, so all randomness in the intervention sits in the policy. |

The version-3 message is:

> Weighing up the discussion so far, **Z** still looks like the strongest
> option to me, and I think it deserves more weight than it is getting. I'm
> voting **Z**.

Every rendered message is checked against the peer-style rules before it is
used: it must not contain `controller`, `external`, `experiment`, or
`simulation`; it must bold the option name; and it must end with an explicit
first-person vote. A violation raises rather than silently leaking the
controller's identity into the population. The controller is never given the
hidden facts, so it cannot manufacture evidence.

### 5.7 `policy`

Optional; if present it must be `soft_target`. It exists so a run record states
its policy explicitly rather than by implication from the mechanism name.

### 5.8 Related game-side options

These are `game.options`, not `control.options`, but they change what control
means:

| Option | Effect on control |
| --- | --- |
| `social_group_size` (`q`) | The number of influence slots a focal has. A controlled update replaces one of them, so the controller's share of a focal's inputs is `1/q`. At `q = 1` it is the focal's only input; at `q = 2` it must out-argue one surviving peer. |
| `dynamics_mode` | Whether the focal is an LLM or the provider-free kernel ([§6](#6-the-two-dynamics-modes)). |
| `classical.kernel` | Must be `controlled_imitation_round_reference`. |
| `decoy` | Recorded documentation of the task's shared-information decoy. `ImitationRules` does not read it; to actually target the decoy, set `control.options.target` to that label. |

---

## 6. The two dynamics modes

### `dynamics_mode: reasoning`

For each position:

1. The focal exchanges `messages_per_agent` rounds of private messages with
   **each effective peer** (both sides speak). A displaced peer slot produces no
   dialogue at all — that peer is simply not in the conversation.
2. The focal receives an update prompt containing its scenario, its own facts,
   its bounded private history, the dialogue, and — on a controlled position —
   the controller's message. When `q > 1` the prompt also carries the
   `influence_slots` structure, in which the controller occupies the replaced
   slot and ordinary peers occupy the rest.
3. The model returns JSON `{vote, rationale}`, parsed and normalized against the
   option list, with `invalid_response_retries` retries.

The focal's vote is whatever the model says. Nothing overwrites it.

### `dynamics_mode: classical`

Provider-free, and the reference null model. The kernel is a **strict-unanimity
`q`-voter** ([`classical.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/classical.py)):

- Effective inputs are the `q` sampled peer opinions, with the controller's
  target substituted into the replaced slot on a controlled position.
- The focal copies an option **only if every effective input agrees on it** and
  it differs from the focal's current vote. Otherwise it stays put.
- No spontaneous noise, no soft response, no anticonformity, no hidden control
  strength. Conditional on the sampled social context the response is
  deterministic, which is why the record carries
  `classical_transition_rate_or_weight: 1.0`.
- A controlled update can therefore only produce a **non-target → target**
  switch. For `q = 1`, a controlled non-target focal always switches to the
  target.

The module also exports closed forms for the controlled (`K1`) and uncontrolled
(`K0`) switch probabilities, `analytical_switch_probability` and
`analytical_mesoscopic_transition_probability`, so simulated cells can be
checked against theory without running anything.

---

## 7. What is recorded

| File | One row per | Contents |
| --- | --- | --- |
| `trajectory.jsonl` | microscopic update (`rounds x N` rows) | The inherited imitation event, enriched with the round fields of §3.6 — including `controlled_slot`, the schedule hash, and the sampled/effective/replaced peer ids |
| `round_trajectory.jsonl` | population round (`rounds` rows) | `record_type: imitation_round_feedback`, the analysis unit |

Each round record carries, among others:

- **Identity and design**: `episode_id`, `seed`, `task_id`, `K`, `N`,
  `dynamics_mode`, `social_group_size`, `sensor_sample_size`,
  `intervention_budget`, `sensing_fraction`, `actuation_fraction`.
- **Controller**: `controller_enabled`, `controller_target`, `analysis_target`,
  `controller_policy`, `controller_threshold`, `controller_beta`,
  `controller_action`, `controller_advocate_probability`.
- **Sensor**: `sensor_agent_ids`, `sensor_observed_opinions`,
  `sensor_count_vector`, `sensor_target_share`.
- **Schedule**: `controlled_positions`, `controlled_position_count`,
  `controlled_positions_seed`, `controlled_positions_hash_or_id`.
- **State before/after**: full vote vectors, occupation counts, shares,
  `target_count`, `truth_count`, `m_ctrl`, `m_truth`, `m_order`, `H_vote`, and
  the four deltas.

---

## 8. What is estimated

Run the provider-free analysis over a completed run or grid:

```bash
python -m mas_cc.cli.main analysis hidden-bench-round-feedback \
  --run-dir results/<run>
```

With `analysis.per_cell_reports: true` the same pass runs automatically after
each cell finishes.

**Information statistics** (bits, with bootstrap CIs and permutation nulls):

| Estimator | Reads |
| --- | --- |
| `round_sensing_mi` | `I(N(t) ; Y(t))` — how much the `q_c`-sample tells the controller about the population |
| `round_population_actuation_cmi` | `I(U(t) ; N(t+1) \| N(t))` on the full occupation vector — sparse for large `N`, `K` |
| `round_target_actuation_cmi` | Same, projected onto the target count — the primary channel |
| `round_truth_actuation_cmi` | Same, projected onto the truth count |
| `round_order_actuation_cmi` | Same, projected onto the largest faction |

**Diagnostics** (plain floats):

- `round_controller_action_entropy`, `round_controller_action_entropy_given_population`
  — how much the policy actually varies, unconditionally and within a state.
- `round_target_information_fraction`, `round_population_information_fraction`
  — actuation CMI divided by the conditional action entropy: the share of the
  controller's decision variability that reached the population.
- `round_dual_action_state_fraction`, `round_dual_action_event_fraction`,
  `round_single_action_slice_fraction`, `round_conditioning_state_count`,
  `round_singleton_fraction` — the support/overlap diagnostics that say whether
  a CMI number is identifiable at all.
- `round_target_signed_actuation`, `round_truth_signed_actuation`,
  `round_order_signed_actuation` — the *signed* effect: within each conditioning
  state, mean delta under `ADVOCATE_Z` minus mean delta under `NO_OP`, averaged
  over states that contain both. This is what tells direction; CMI only reports
  magnitude.
- `round_sensor_mae`, `round_sensor_mse` — sampled target share versus the true
  share.

Two caveats the module is explicit about: full-state direct-counting CMI is
sparse for large `N` and `K` (prefer the projections and read the support
diagnostics), and raw sensing MI is not comparable across different `q_c`
without an alphabet caveat, because the alphabet of `Y` changes with `q_c`.

---

## 9. A worked configuration

[`configs/runs/imitation_round_feedback/hidden_bench_imitation_round_feedback_qwen_first_llm_pilot_grid_A.yaml`](../../../../configs/runs/imitation_round_feedback/hidden_bench_imitation_round_feedback_qwen_first_llm_pilot_grid_A.yaml)
is the reference example. Reading only the parts this document covers:

```yaml
game:
  type: hidden_bench_imitation_round_feedback
  population_size: 24          # N
  horizon: 10                  # schema mirror of rounds
  options:
    rounds: 10                 # 10 controller decisions, 240 focal updates
    dynamics_mode: reasoning   # the focal is the LLM
    social_group_size: 1       # q = 1: a controlled focal hears only the controller
    memory_size: 0
    stop_on_consensus: false
    initialization:
      mode: uniform_random     # provider-free start, no initial-vote calls
    classical:
      kernel: controlled_imitation_round_reference

control:
  mechanism: round_soft_target_budgeted
  options:
    target: correct            # beneficial control
    sensor_sample_size: 2      # q_c = 2 of 24: a very noisy sensor
    threshold: 0.5
    beta: 4.0
    intervention_budget: 0     # overridden by the grid
    template_version: 3

grid:
  game.options.social_group_size: [1, 2]
  control.options.intervention_budget: [0, 6]
```

That grid is the design in miniature: four cells crossing "how contested is the
focal's attention" (`q`) with "how often can the controller speak" (`b`), where
`b = 0` is the sensing-only reference and `q = 2` forces the controller to
out-argue a surviving peer. Sibling files B–D extend it to larger budgets,
larger sensors, and a decoy target.

---

## 10. Failure modes and validation errors

| Message | Cause |
| --- | --- |
| `only classical.kernel 'controlled_imitation_round_reference' is implemented` | `game.options.classical.kernel` was left at the event-clock game's kernel name. |
| `game.options.rounds must be a positive integer` | Missing or non-integer slow-clock horizon. |
| `prompt.prompt_version is X but game.options.prompt_version is Y` | The prompt block and the game block disagree; they must match exactly. |
| `round feedback requires fixed peer-style template version 3` | `template_version` set to 1 or 2. |
| `control.options.intervention_budget must be a non-negative integer` | `b` was negative, a float, or a bool. |
| `controller intervention_budget must be between 0 and N` | `b > population_size`. |
| `controller sensor_sample_size cannot exceed the population size` | `q_c > N`. |
| `the selected control does not implement round-level signaling` | An event-clock mechanism (`soft_target`, `threshold_target`) was configured for this game. |
| `controller target is outside the task option alphabet` | An explicit `target` label that is not one of the task's `possible_answers` — check spelling and case. |
| `round controller action must be NO_OP or ADVOCATE_Z` | A custom control returned some other action. |
| `classical initialization must never require provider decisions` | `dynamics_mode: classical` combined with `initialization.mode: local_vote`. |
