# Implementation Plan: Epistemic Persistence for `relational_imitation_round_feedback`

## Purpose

Extend the existing relational imitation round-feedback game with **finite epistemic persistence** while preserving the current game exactly at \(\rho=1\). The new mechanism should make facts that an agent has previously learned *not necessarily remain available in its current reasoning context forever*. This is an extension of the epistemic state only. It must **not** change the controller, sensing, social-slot replacement, task generator, frozen truth, option shuffling, voting semantics, strategic evidence semantics, or the existing soft feedback policy.

The implementation should be deliberately minimal and backward-compatible. Existing runs/configs that do not mention persistence must behave exactly as they do today.

---

## 1. Scientific semantics

Introduce one game-level parameter:

```yaml
game:
  options:
    epistemic_persistence: 1.0
```

Denote this parameter by \(\rho\).

- `rho = 1.0`: every currently active fact remains active forever. This must reproduce the existing game.
- `0 < rho < 1`: each currently active fact independently remains active from one population round to the next with probability `rho`.
- `rho = 0`: allowed if convenient; every active fact becomes inactive at the persistence boundary unless it is subsequently communicated again.

The frozen task truth does not change. A fact that becomes inactive is **not deleted from history and does not become false**. It simply ceases to be available in the agent's current reasoning context until it is communicated again.

The central state distinction is therefore:

```text
historical known facts = facts the agent has ever legitimately received
active facts           = facts currently available to the agent's reasoning prompt
```

At initialization:

```text
active_facts == known_facts == task-assigned initial facts
```

For `rho = 1.0`, the invariant is always:

```text
active_facts == known_facts
```

and the runtime must follow the legacy behavior.

---

## 2. Hard backward-compatibility contract

This is the most important requirement.

### 2.1 Configuration compatibility

- `epistemic_persistence` must default to `1.0` when absent.
- Every existing YAML file must remain valid without modification.
- Existing study configs must not need a migration.
- Validate `0.0 <= epistemic_persistence <= 1.0`.

### 2.2 Runtime compatibility at `rho = 1`

When `epistemic_persistence == 1.0`:

1. Do **not** consume any new random numbers.
2. Do **not** change the existing focal-agent RNG stream.
3. Do **not** change peer sampling.
4. Do **not** change controller sensing or soft-policy draws.
5. Do **not** change controlled-position sampling.
6. Do **not** change semantic option shuffling.
7. Do **not** change prompts sent to the provider.
8. Do **not** change which facts ordinary peers can expose.
9. Do **not** change controller evidence selection.
10. Do **not** change votes, knowledge acquisition, termination, or call counts.

Prefer an explicit fast path:

```python
if epistemic_persistence == 1.0:
    # legacy behavior; no persistence RNG and no state mutation
```

This is stronger and safer than sampling Bernoulli(1) values, because even harmless RNG consumption can shift later stochastic streams.

### 2.3 Regression requirement

Add a regression test demonstrating that, for the same frozen task, execution seed, mock/provider outputs, and config:

```text
legacy config with no persistence key
==
config with epistemic_persistence: 1.0
```

for all legacy scientific behavior, including:

- prompts;
- focal/peer/controller selections;
- controller actions;
- controlled positions;
- vote trajectories;
- historical fact trajectories;
- legacy round-record fields;
- call counts.

If exact byte identity of complete artifact files is impossible because new additive schema fields are introduced, document this explicitly; **all pre-existing fields and scientific trajectories must nevertheless be identical**.

---

## 3. State model

The existing relational state stores each agent's known fact IDs. Preserve this historical state and add the active state.

Suggested conceptual representation in:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/state.py
```

```python
known_fact_ids: set[str]          # existing historical truth/provenance state
active_fact_ids: set[str]         # new currently usable epistemic state
```

Do not rename or repurpose `known_fact_ids`; existing analyses and provenance may depend on its historical meaning.

Add explicit helpers rather than scattering conditional logic throughout the game, e.g. conceptually:

```python
def reasoning_fact_ids(agent_state, rho):
    return agent_state.known_fact_ids if rho == 1.0 else agent_state.active_fact_ids
```

The exact API should follow repository conventions.

### Required invariants

For every agent and every time:

```text
active_fact_ids subseteq known_fact_ids
```

At `rho = 1`:

```text
active_fact_ids == known_fact_ids
```

A persistence event may remove a fact only from `active_fact_ids`, never from `known_fact_ids`.

---

## 4. Which facts are rendered and transmitted

Persistence is meaningful only if inactive information is genuinely unavailable to the current interaction.

### 4.1 Focal-agent private knowledge

The private facts rendered into a focal agent's decision prompt must come from its **active facts** when `rho < 1`.

At `rho = 1`, the rendered prompt must be identical to the current prompt.

Implementation preference: keep `prompts.py` ignorant of persistence if possible. Resolve the effective fact set in `game.py` and pass the same prompt structure as today.

### 4.2 Ordinary peer evidence

An ordinary peer may expose at most one fact it **currently has active**, not merely one it saw historically.

This is essential. Otherwise an inactive fact could re-enter the social channel from an agent that is supposedly not currently reasoning from it.

At `rho = 1`, active facts equal historical known facts, so peer evidence behavior is unchanged.

### 4.3 Controller evidence

Do **not** subject the external controller to epistemic persistence. The controller's evidence selection remains exactly as currently implemented:

- recommendation-only remains unchanged;
- recommendation-plus-fact remains unchanged;
- strategic evidence remains a deterministic/production selection among **true frozen task facts**;
- no fabricated facts are introduced.

If the controller exposes a true fact that is inactive for the focal agent, that exposure may reactivate the fact as described below.

---

## 5. Fact acquisition versus reactivation

The current code records newly acquired facts. Persistence introduces a new event type that must not be conflated with first acquisition.

When a focal agent is exposed to a valid fact `f`, distinguish three cases.

### Case A: first acquisition

```text
f not in known_fact_ids
```

Then:

```text
add f to known_fact_ids
add f to active_fact_ids
```

This retains the existing meaning of a `new_*_fact` event.

### Case B: reactivation

```text
f in known_fact_ids
f not in active_fact_ids
```

Then:

```text
leave known_fact_ids unchanged
add f to active_fact_ids
```

Record this as a **reactivation**, not a newly acquired fact.

Suggested additive diagnostics:

```text
reactivated_peer_facts
reactivated_controller_facts
```

or equivalent repository-consistent names.

### Case C: redundant active exposure

```text
f in active_fact_ids
```

No knowledge-state change.

Do not change the historical semantics of existing `new_peer_facts` / `new_controller_facts` fields.

---

## 6. Persistence transition and timing

Apply persistence **once per population round**, not once per microscopic update.

Recommended round order:

```text
1. state at start of round
2. controller senses votes
3. controller samples U_t
4. schedule controlled positions
5. execute N microscopic updates exactly as today
6. complete all vote/fact acquisition/reactivation transitions
7. apply epistemic persistence to active facts
8. emit the canonical end-of-round state
9. next round begins from exactly that state
```

This timing has two advantages:

1. the game still has a clean population-round clock for persistence;
2. the stored invariant `after(t) == before(t+1)` can remain exact.

### Important observability detail

Because newly communicated facts may be deactivated at the end-of-round persistence step, retain enough information to distinguish:

```text
active state immediately after social/controller interactions
active state after persistence
```

This can be done with additive round-record fields rather than changing the logical transition order.

For example, record:

```text
active_*_before
active_*_after_interactions
active_*_after_persistence
persistence_deactivated_fact_count
```

The canonical `after` state used as the next round's `before` state should be **post-persistence**.

At `rho = 1`, the interaction and post-persistence snapshots are identical.

---

## 7. Randomness and reproducibility

Persistence must use its **own deterministic RNG stream** so that enabling `rho < 1` does not perturb any existing stochastic mechanism except through the resulting changed epistemic state.

Derive the persistence stream from stable episode provenance, conceptually from something like:

```text
(execution_seed, episode_seed, round_index, "epistemic_persistence")
```

Use repository-native seed derivation utilities if they exist.

Requirements:

- never use Python container iteration order for RNG ordering;
- sort agent IDs and fact IDs before drawing survival events;
- one survival draw per currently active `(agent_id, fact_id)` pair is acceptable;
- log/retain enough seed provenance to reproduce the transition;
- `rho = 1` must bypass this stream entirely.

Do not reuse the focal/peer/controller/controlled-position/option-shuffle streams.

---

## 8. Metrics and terminology

The current relational game defines, for supporting-fact set `S`:

```text
coverage_i = |K_i intersect S| / |S|
kappa      = mean_i coverage_i
phi        = count_i[K_i contains S] / N
```

For persistence studies, the scientifically relevant `K_i` is the **active** fact set.

### Canonical active observables

When persistence is enabled, define the main epistemic observables from active facts:

```text
active_coverage_i
kappa_active
phi_active
```

The existing user-facing metrics `mean_supporting_fact_coverage` and `full_proof_agent_share` may continue to represent the *effective reasoning state* if this can be done without breaking old analyses:

- at `rho = 1`, they are numerically identical to the legacy metrics;
- at `rho < 1`, they should reflect active facts.

### Historical diagnostics

Also retain separate monotone diagnostics based on `known_fact_ids`, e.g.:

```text
historical_mean_supporting_fact_coverage
historical_full_proof_agent_share
```

This separation is scientifically important. It allows us to distinguish:

```text
"the population has never received the proof"
```

from

```text
"the population has received it historically, but it is not currently active in reasoning"
```

Do not silently reinterpret historical provenance fields.

---

## 9. Round-record additions

The existing round records already retain vote, controller, knowledge, exposure, and provenance data. Extend them additively.

At minimum persist:

```text
epistemic_persistence
active_mean_supporting_fact_coverage_before
active_mean_supporting_fact_coverage_after_interactions
active_mean_supporting_fact_coverage_after
active_full_proof_agent_share_before
active_full_proof_agent_share_after_interactions
active_full_proof_agent_share_after
historical_mean_supporting_fact_coverage_after
historical_full_proof_agent_share_after
persistence_deactivated_fact_count
persistence_deactivated_supporting_fact_count
reactivated_peer_fact_count
reactivated_controller_fact_count
```

Exact names should follow the repository's current naming conventions. Avoid duplicate aliases if equivalent fields already exist.

If artifact profiles retain exact fact IDs, optionally include the deactivated/reactivated IDs as provenance. If the standard profile intentionally stores only counts, preserve that storage policy.

---

## 10. Resume/checkpoint behavior

Persistence changes state, therefore active facts must be part of any resumable/checkpointed episode state.

A resumed episode must reproduce an uninterrupted episode exactly for the same seed and provider/mock outputs.

Test a checkpoint immediately after a persistence boundary and verify:

```text
uninterrupted continuation == resumed continuation
```

for active facts, prompts, controller actions, votes, and records.

---

## 11. Files expected to change

Use the actual repository structure and minimize the diff. Based on the current game map, likely touch only the following areas.

### `imitation_round_feedback/state.py`

- add active fact state;
- add state invariants/helpers;
- ensure serialization/checkpoint support.

### `game.py`

- initialize `active_fact_ids` from initial known facts;
- render focal private knowledge from active facts;
- restrict ordinary peer evidence to active facts;
- distinguish acquisition vs reactivation;
- keep controller evidence semantics unchanged.

### `runtime.py`

- read `epistemic_persistence` with default `1.0`;
- add the round-boundary persistence transition;
- use a dedicated RNG stream;
- bypass all persistence RNG/mutation at `rho == 1.0`;
- emit persistence diagnostics.

### `metrics.py`

- compute effective/active `kappa` and `phi`;
- add historical diagnostics without removing existing quantities.

### relational `analysis.py`

Only make additive reader/schema changes required to consume the new fields. Do **not** change estimators, control-information definitions, or theory comparisons as part of this task.

### Config/schema validation

- add `epistemic_persistence` as a game option, default `1.0`;
- validate bounds;
- ensure it is grid-sweepable through the existing generic grid mechanism without special infrastructure.

### Tests

Add focused persistence tests near the existing relational game/runtime tests. Do not create a new generic framework unless the current repository requires one.

---

## 12. Tests that must exist before merge

Implement tests in this order where practical.

### T1 — default is legacy

Config without `epistemic_persistence` resolves to `1.0`.

### T2 — exact `rho=1` behavioral regression

For fixed seed and deterministic provider/mock outputs:

```text
no persistence key
```

and

```text
epistemic_persistence: 1.0
```

produce identical legacy behavior.

Also assert the persistence RNG path is not invoked.

### T3 — initialization

For every agent:

```text
active_fact_ids == known_fact_ids
```

at episode initialization.

### T4 — `rho=0` boundary behavior

After one persistence boundary:

```text
active_fact_ids == empty
known_fact_ids unchanged
```

for facts not reacquired afterward.

### T5 — first acquisition

A genuinely unseen true fact enters both historical and active sets and increments the existing `new_*` counter.

### T6 — reactivation

An historically known but inactive fact becomes active when exposed again and increments a reactivation counter but **not** a `new_*` counter.

### T7 — inactive peer fact cannot leak

A peer cannot expose a fact that exists only in its historical set but is inactive.

### T8 — controller semantics unchanged

The same controller strategy selects the same true task fact as before. Exposure may reactivate that fact for the focal agent, but controller selection itself is unaffected by persistence.

### T9 — RNG isolation

For a test where persistence outcomes are injected/fixed, verify the dedicated persistence RNG does not alter controller sensing, policy draw, focal selection, peer selection, controlled positions, or option shuffles.

### T10 — round continuity

For active and historical observables:

```text
after(round t) == before(round t+1)
```

exactly.

### T11 — active versus historical metrics

Construct a small state where every agent has historically seen the complete proof but only half currently have it active. Assert:

```text
historical phi = 1
active phi = 0.5
```

and analogous coverage behavior.

### T12 — resume determinism

An episode resumed from checkpoint after a persistence transition matches the uninterrupted episode.

### T13 — grid compatibility

A normal config grid can sweep, e.g.:

```yaml
grid:
  game.options.epistemic_persistence: [0.94, 0.97, 1.0]
```

without a study-specific launcher or schema hack.

### T14 — existing test suite

Run the existing relational/runtime/analysis tests and then the full repository suite appropriate for this change. Existing studies at default `rho=1` must remain valid.

---

## 13. Acceptance checks on a tiny deterministic run

Before any provider-backed study, run a small mock/deterministic episode and show a round table with at least:

```text
round
rho
historical kappa
active kappa
historical phi
active phi
facts acquired
facts reactivated
facts deactivated
controller action
controlled positions
```

Verify visually and programmatically:

1. historical knowledge is monotone;
2. active knowledge can decrease when `rho < 1`;
3. inactive facts can re-enter only through ordinary valid communication;
4. frozen task truth never changes;
5. the controller still exposes only valid true task facts;
6. `rho=1` restores monotone active knowledge and exactly the legacy scientific trajectory.

---

## 14. Explicit non-goals

Do **not** include any of the following in this implementation:

- lying or fabricated facts;
- source reputation or trust scores;
- changes to naive/vigilant receiver semantics;
- changes to `q`, social-slot semantics, or number of controller slots;
- changes to `b`, `q_c`, `beta`, `theta`, or controller scheduling;
- changes to the task generator or existing frozen datasets;
- changes to the existing controller target semantics;
- changes to MI/CMI estimators;
- a new theory implementation;
- modifications to `theory_revised.py`;
- fitting or calibrating a mean-field model;
- a new study-level framework or submission layer.

This task is **only** the epistemic-persistence extension of the production game.

For any persistence study run before the theory is extended, use no matched theoretical reference rather than forcing the existing `q=1` theory onto the new dynamics.

---

## 15. Implementation principle

The desired behavior can be summarized by one state transition:

```text
TRUE FACT RECEIVED
      |
      v
historical known = yes  ------------------------------+
      |                                                |
      v                                                |
active = yes -- survives round with probability rho --+
      |
      +-- fails persistence --> active = no
                                  |
                                  +-- same valid fact is communicated again
                                      --> active = yes
```

The world remains unchanged throughout. Persistence controls only whether previously learned true information is **currently available to reasoning**.

---

## 16. Definition of done

The extension is ready only when all of the following are true:

- existing configs require no edits;
- `rho=1.0` follows the legacy runtime path and consumes no persistence RNG;
- legacy prompts and scientific trajectories are regression-tested unchanged at `rho=1`;
- `rho<1` can make active `kappa` and `phi` decrease while historical knowledge remains monotone;
- valid peer/controller exposure can reactivate inactive facts;
- ordinary peers cannot transmit inactive facts;
- controller semantics remain untouched;
- persistence uses an isolated deterministic RNG stream;
- round records contain sufficient active/historical/deactivation/reactivation provenance;
- resume/checkpoint behavior is deterministic;
- current theory code is untouched;
- the full relevant test suite passes.

**Do not start a new provider-backed persistence study as part of this task. Stop after implementation, tests, and a deterministic/mock demonstration, then report the exact diff and validation results for review.**
