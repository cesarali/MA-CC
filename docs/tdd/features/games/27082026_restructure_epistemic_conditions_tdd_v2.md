# TDD Revision Brief: Factorized Epistemic Conditions for the Relational Round-Feedback Game

**Purpose:** revise the epistemic-condition implementation before Study 08 so that it matches the scientific design we actually want.

The previous four prompt classes (`naive`, `distributed_information`, `strategic_uncertainty`, `evidence_calibrated`) should NOT be treated as the final Study 08 design.

The new design separates two different mechanisms:

1. **receiver epistemic disposition** — how the focal LLM is instructed to evaluate social information;
2. **controller evidence strategy** — how the controller selects the real task fact attached to its recommendation.

The controller's truth/false target is a THIRD, independent axis and must not be bundled into either of these.

The four epistemic conditions used later in Study 08 are therefore the Cartesian product:

| receiver disposition | controller evidence strategy | derived condition label |
|---|---|---|
| naive | neutral | `naive_neutral` |
| naive | strategic | `naive_strategic` |
| vigilant | neutral | `vigilant_neutral` |
| vigilant | strategic | `vigilant_strategic` |

These four labels are scientific grouping labels. Do NOT implement them as four duplicated prompt families or four separate controller types if the clean implementation is two independent categorical axes.

---

## 1. Repository facts that must be preserved

The relational game already supports:

```yaml
message_mode: recommendation_only
```

and

```yaml
message_mode: recommendation_plus_fact
```

In `recommendation_plus_fact`, the controller exposes exactly one real fact from the frozen task through the same structured evidence channel used by peers. The fact is not paraphrased or invented, its exact ID is recorded, and it enters the focal agent's `K_i`.

The existing implementation resolves the controller fact deterministically once per episode. Preserve that general reproducibility property unless a very strong implementation reason requires otherwise.

The game also deliberately keeps free-form reasons private. The only inter-agent task-information channel is `shared_fact_id`.

These invariants are load-bearing and must not change.

---

# 2. New axis A: receiver epistemic disposition

Introduce one explicit categorical coordinate:

```text
receiver_epistemic_disposition
```

Allowed values:

```text
naive
vigilant
```

Use the repository's native config location/prompt architecture after inspection. Do not create two game types.

## 2.1 `naive`

Scientific meaning:

The focal agent receives the normal task instructions, its own known facts, current vote, and the visible participants' votes/shared facts, but receives no explicit warning that other participants may have different objectives or that a recommendation may be strategically selected.

The environment is still distributed. We are NOT changing fact allocation. We are only removing the special epistemic warning.

The naive condition must not add language such as:

```text
participants may have different objectives
sources may be strategic
recommendations may be unreliable
check whether evidence is selectively presented
```

Do not remove mechanical instructions about the task, fact sharing, privacy, or the response contract.

## 2.2 `vigilant`

Scientific meaning:

The focal agent is explicitly warned that another participant's recommendation may reflect a different objective and that the recommendation should be evaluated against the evidence actually supplied.

The key idea is **epistemic vigilance**, not blanket distrust.

Use wording close in spirit to:

```text
Some participants may have objectives that differ from yours. Do not accept a recommendation merely because another participant gives it. Evaluate whether the explicit evidence you have actually supports the conclusion, and remain aware that a source may present only part of the relevant information.
```

Important constraints:

- do not identify the controller;
- do not tell the agent which participant is strategic;
- do not say to ignore social information;
- do not say minority claims are automatically correct;
- do not reveal the truth;
- do not alter the fact-sharing contract.

### Compatibility with the old strategic prompt

The current main relational prompt already contains the warning that some participants may have objectives that differ.

Inspect the exact current prompt and the in-progress prompt-class implementation.

Where practical, make `vigilant` the compatibility representation of the current strategic-warning behavior. Existing real Study 06/07 behavior must remain reproducible.

Do not silently alter historical prompt semantics.

---

# 3. New axis B: controller evidence strategy

Introduce a second independent categorical coordinate:

```text
controller_evidence_strategy
```

Allowed values:

```text
neutral
strategic
```

This axis is only meaningful when:

```yaml
message_mode: recommendation_plus_fact
```

If `controller_evidence_strategy` is set while `message_mode` is `recommendation_only`, config/preflight should refuse the combination or otherwise make the invalidity explicit. Do not silently ignore the evidence strategy.

For Study 08, `recommendation_only` will never be used.

## 3.1 Shared invariant for BOTH evidence strategies

Both `neutral` and `strategic` must:

- expose exactly one REAL fact from the frozen task;
- use the existing structured peer-compatible evidence renderer;
- record the exact controller fact ID;
- never fabricate, paraphrase, mutate, or negate a task fact;
- never insert free prose as a second evidence channel;
- preserve `K_i` as an exact set of true task facts;
- keep controller identity indistinguishable from the existing ordinary-participant rendering;
- use deterministic/replayable selection;
- preserve `NO_OP` semantics: no controller slot, no controller fact.

The difference between the strategies is ONLY **which true fact is selected**.

## 3.2 `neutral`

Scientific meaning:

The controller attaches a true task fact using a target-independent selection rule.

The fact must NOT be selected because it favors the controller's target.

Preferred behavior:

- define a deterministic admissible fact pool for the task;
- select one fact according to a fixed, target-independent rule;
- for paired truth-vs-false controller runs on the same task/seed, the neutral evidence should be the SAME fact whenever possible.

This is important because target semantics is a separate experimental axis.

The existing fixed `controller_fact_id` or deterministic target-independent selector may already provide most of this behavior. Reuse existing code if scientifically correct.

Document the exact neutral rule.

## 3.3 `strategic`

Scientific meaning:

The controller still tells only the truth at the fact level, but it **selects which true fact to reveal in a target-aware way**.

The selected fact should be the real task fact that is most favorable to, or most locally compatible with, the controller's current target.

This is selective disclosure / cherry-picking, NOT false evidence.

For a false controller, the desired behavior is:

```text
wrong target + true but incomplete/cherry-picked fact that makes the target locally plausible
```

For a truth controller, the same target-aware selection machinery should choose the real fact most favorable/informative for the correct target.

The selection rule must be deterministic and based on the task's symbolic structure, not delegated to an LLM.

### Critical task-admissibility requirement

Do not pretend that every frozen task necessarily contains a strategically useful fact for every possible target.

Before implementing Study 08, audit the intended Study 06/07 task set.

For each task and each target semantics needed by Study 08, determine whether a valid strategic fact exists under the documented selection criterion.

If no valid fact exists:

- mark the task/target as strategically inadmissible;
- fail Study 08 preflight for that task;
- report the issue.

Do NOT silently fall back to neutral/random evidence.
Do NOT fabricate a fact.
Do NOT change the target just to make the task pass.

If the symbolic notion of "most favorable" is not uniquely determined by the current task representation, implement a small explicit annotation/derivation layer with tests and provenance rather than hiding a heuristic in runtime code.

---

# 4. Truth/false target is a separate axis

Do not encode controller objective inside either of the new epistemic axes.

The existing target mechanism should remain authoritative, conceptually:

```text
target_semantics ∈ {truth, false}
```

using the repository's actual existing representation (`correct`, `random_incorrect`, explicit option, etc.).

The full scientific factorization is:

```text
receiver_epistemic_disposition
    ∈ {naive, vigilant}

controller_evidence_strategy
    ∈ {neutral, strategic}

controller target semantics
    ∈ {truth, false}
```

The four epistemic conditions are produced by the first two axes and can each be run under BOTH truth and false control.

---

# 5. `recommendation_plus_fact` is mandatory for the new design

This must be explicit in code validation, docs, and Study 08.

All new epistemic-condition experiments use:

```yaml
message_mode: recommendation_plus_fact
```

There is no `recommendation_only` cell in Study 08.

`recommendation_only` remains scientifically valuable as the historical bare-control reference from Studies 04/06/07, but it is NOT one of the new Study 08 conditions.

Do not silently inherit `recommendation_only` from a Study 06/07 template.

---

# 6. TDD tests to write first

## 6.1 Receiver disposition config tests

Verify:

```text
naive
vigilant
```

resolve successfully.

Unknown values fail clearly.

If legacy/in-progress keys such as:

```text
distributed_information
strategic_uncertainty
evidence_calibrated
social_distrust
```

still exist, inspect actual usage and migrate/reconcile them explicitly.

Do not leave contradictory independent knobs.

## 6.2 Receiver prompt-content tests

For identical live game state:

### Naive
Must preserve all mechanical/task/evidence blocks but contain no strategic-source warning.

### Vigilant
Must contain the strategic-objective/evidence-evaluation warning.

Across the two, all non-epistemic prompt blocks must be identical.

## 6.3 Evidence-strategy config tests

Both:

```text
neutral
strategic
```

resolve only with:

```text
message_mode = recommendation_plus_fact
```

Unknown strategies fail.

Evidence strategy + `recommendation_only` fails preflight/config validation.

## 6.4 Neutral evidence tests

For a fixed task/episode seed:

- selected fact is real;
- selection is deterministic;
- selection is target-independent;
- paired truth/false target runs receive the same neutral fact when the task allows the intended pairing;
- fact is rendered through the standard structured evidence channel;
- exact ID is recorded.

## 6.5 Strategic evidence tests

For a fixed task and target:

- selected fact is real;
- selected fact satisfies the documented target-aware strategic criterion;
- selection is deterministic;
- changing target may change selected fact;
- no LLM is used for evidence selection;
- no false/mutated fact can enter the prompt or `K_i`;
- inadmissible task/target combinations fail explicitly rather than falling back.

Include at least one synthetic/frozen fixture where the strategic fact is obvious from the symbolic geometry.

## 6.6 Single-information-channel regression

Parameterize across:

```text
receiver ∈ {naive, vigilant}
strategy ∈ {neutral, strategic}
```

and verify:

- peer/controller reasons never become social prose;
- `shared_fact_id` remains the only task-information channel;
- controller fact enters only focals who actually see a controlled slot;
- facts remain exact members of the frozen task;
- knowledge propagation/provenance remain exact;
- `NO_OP` adds nothing.

## 6.7 RNG/dynamics invariance

Using a class-independent mock provider, changing receiver disposition or evidence strategy must not alter:

- focal selection;
- peer selection;
- controller sensing;
- controller action RNG;
- controlled-position schedules;
- option shuffles;
- round counts;
- provider call counts.

The controller fact itself may differ under neutral vs strategic by design.

## 6.8 Provenance

Canonical scientific output must expose at least:

```text
receiver_epistemic_disposition
controller_evidence_strategy
message_mode
controller target semantics
controller_fact_id
```

without requiring prompt-text inspection.

Prompt definition hashes must distinguish naive vs vigilant.

Evidence-strategy provenance must be auditable without duplicating large prompt strings.

---

# 7. Derived four-condition labels

For plotting/aggregation convenience, derive:

```text
naive_neutral
naive_strategic
vigilant_neutral
vigilant_strategic
```

from:

```text
(receiver_epistemic_disposition, controller_evidence_strategy)
```

Do not make this derived label a second source of truth.

---

# 8. Acceptance criteria

The restructuring is complete when:

- [ ] receiver disposition has exactly the scientific values `naive` and `vigilant`;
- [ ] controller evidence strategy has exactly `neutral` and `strategic`;
- [ ] truth/false target remains a separate existing axis;
- [ ] the four derived epistemic conditions are cleanly recoverable;
- [ ] every evidence-bearing condition requires `recommendation_plus_fact`;
- [ ] neutral evidence is deterministic and target-independent;
- [ ] strategic evidence is deterministic, target-aware, and always true;
- [ ] strategically inadmissible task/target pairs fail explicitly;
- [ ] no false fact is ever inserted;
- [ ] `shared_fact_id` remains the only inter-agent task-information channel;
- [ ] current historical Study 06/07 prompt behavior remains reproducible;
- [ ] existing recommendation-only studies are not modified;
- [ ] all new unit/regression tests pass;
- [ ] a 2×2 mock smoke grid over receiver × evidence strategy runs successfully with `recommendation_plus_fact`.

---

# 9. Required handoff

At completion report:

1. exact config paths for `receiver_epistemic_disposition` and `controller_evidence_strategy`;
2. exact naive/vigilant prompt text;
3. exact neutral evidence-selection rule;
4. exact strategic evidence-selection rule;
5. audit of the intended Study 08 frozen tasks, including strategic admissibility for truth and false targets;
6. files changed;
7. backward-compatibility handling of the previous/in-progress epistemic prompt classes;
8. confirmation that `recommendation_plus_fact` is mandatory for these new conditions;
9. confirmation that no false facts are introduced;
10. test results;
11. mock 2×2 smoke-grid result;
12. minimal example YAML for all four derived epistemic conditions.

Do not launch Study 08 as part of this restructuring task. Study 08 should only be prepared after this factorized implementation is complete and audited.
