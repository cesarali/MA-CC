# Study 08 Revised Implementation Prompt: Epistemic Vigilance × Evidence Selection

Create the revised **Relational Population Study 08** after the factorized epistemic-condition implementation is complete and all of its tests pass.

This replaces the earlier Study 08 design based on four prompt classes.

The scientific design is now:

```text
receiver epistemic disposition:
    naive
    vigilant

controller evidence strategy:
    neutral
    strategic

controller target semantics:
    truth
    false

actuation budget:
    b = 4, 8, 12, 16, 20, 24
```

The first two axes produce four epistemic conditions:

```text
naive_neutral
naive_strategic
vigilant_neutral
vigilant_strategic
```

Each of those four must be run under BOTH truth and false controller targets.

Most importantly:

# EVERY SINGLE STUDY 08 CONTROLLED CONDITION MUST USE

```yaml
message_mode: recommendation_plus_fact
```

There must be **zero** `recommendation_only` cells in Study 08.

Studies 04/06/07 already provide the historical recommendation-only reference. Study 08 is specifically about what happens when control carries explicit task evidence.

---

# 1. Scientific question

Study 08 asks:

> How do receiver epistemic vigilance and target-aware evidence selection change the efficiency of feedback control in a multi-agent LLM population, and how do those effects depend on whether the controller targets truth or a false answer?

This is a three-factor semantic experiment:

```text
receiver disposition
× evidence-selection strategy
× controller target semantics
```

with actuation budget `b` used to trace the full control-strength response.

The factors must remain conceptually separate.

Do NOT equate:

- `strategic evidence` with `false target`;
- `neutral evidence` with `truth target`;
- `vigilant` with resistance to false control only.

All four receiver/evidence combinations must be evaluated under both truth and false targets.

---

# 2. Meaning of the factors

## 2.1 Receiver disposition

### `naive`

No explicit warning that a visible participant may have a different objective or may selectively present evidence.

### `vigilant`

The focal LLM is explicitly instructed to evaluate whether the supplied evidence actually supports a recommendation and to remain aware that another participant may have a different objective or may present only part of the relevant information.

This is epistemic vigilance, not blanket distrust.

## 2.2 Controller evidence strategy

Both strategies use exactly one **real frozen task fact** through the normal structured controller evidence channel.

Both therefore require:

```yaml
message_mode: recommendation_plus_fact
```

### `neutral`

The fact is selected by the new deterministic target-independent neutral rule.

For matched truth/false runs on the same task, the neutral fact should be identical whenever that is part of the implemented pairing contract.

### `strategic`

The fact is selected by the new deterministic target-aware strategic rule.

The controller is allowed to cherry-pick/selectively disclose a true fact favorable to its target, but it is NOT allowed to fabricate evidence.

For false-target control this should produce:

```text
wrong recommendation
+
true but incomplete / target-favorable evidence
```

when the task is strategically admissible.

For truth-target control the same selector should choose true evidence most favorable/informative for the correct target.

## 2.3 Controller target semantics

Use the repository's existing target mechanism for:

```text
truth
false
```

Do not create a new target mechanism unless inspection proves necessary.

Truth/false is an independent scientific axis.

---

# 3. Study 08 factorial design

Use:

```text
receiver ∈ {naive, vigilant}
evidence_strategy ∈ {neutral, strategic}
target_semantics ∈ {truth, false}
b ∈ {4, 8, 12, 16, 20, 24}
```

Therefore:

```text
2 receiver dispositions
× 2 evidence strategies
× 2 target semantics
× 6 intervention budgets
= 48 scientific conditions per task
```

Use the expected Study 06/07 convention:

```text
4 matched tasks
× 10 repetitions
```

if the actual current configs confirm that convention.

Expected total:

```text
48 conditions/task
× 4 tasks
= 192 scientific cells

192 cells
× 10 repetitions
= 1,920 episodes
```

This scale is intentional. It should fit approximately in an overnight run under the existing cluster workflow, subject to preflight.

Do not reduce the factorial design before preflight.

If estimated runtime is unexpectedly well beyond an overnight run, preserve all factors and all six `b` values first; reduce repetitions only after reporting the estimate.

---

# 4. Fixed settings

Use the established Study 06/07 relational setup as the template.

Keep fixed unless inspection of the current canonical configs shows a different authoritative value:

```text
N = 24
q = 1
q_c = 12
theta = 0.5
beta = 4
round horizon = same as Study 06/07
initialization = same as Study 06/07
provider/model = same as the current main studies
artifact retention = results_only
controller schedule/policy = same as Study 06/07
```

Keep unchanged:

- task mechanics;
- frozen initial `K_i(0)`;
- response contract;
- per-call semantic option shuffle;
- focal/peer sampling;
- controller sensing;
- soft policy;
- controlled-slot scheduling;
- single-information-channel guarantee;
- fact validation;
- knowledge propagation;
- MI/CMI estimator engine;
- bootstrap/null machinery;
- existing information-response efficiency definition;
- standardized study submit/aggregate workflow.

The semantic factors above must be the intended experimental changes.

---

# 5. Mandatory `recommendation_plus_fact` constraint

This is a hard Study 08 validation rule.

Every controlled Study 08 config/cell must resolve to:

```yaml
message_mode: recommendation_plus_fact
```

Before launch, scan/resolved-check all cells and fail if any cell uses:

```yaml
message_mode: recommendation_only
```

or omits the evidence-bearing mode.

Do not inherit `recommendation_only` accidentally from Study 06/07 templates.

The controller fact must use the same structured evidence renderer as ordinary peer-shared facts and must enter `K_i` through the existing provenance-aware mechanism.

`NO_OP` rounds continue to transmit no controller message/fact.

---

# 6. Task audit before finalizing configs

Study 08 requires strategic evidence to be scientifically meaningful.

Audit the intended four frozen Study 06/07 tasks using the new strategic-evidence admissibility check.

For each task, verify both:

```text
truth target: valid strategic fact exists
false target: valid strategic fact exists
```

Also verify the neutral selection rule resolves deterministically.

Produce an audit table:

```text
task
truth target
false target
neutral fact id
strategic truth fact id
strategic false fact id
admissible yes/no
reason
```

Only use tasks that pass the preregistered strategic criterion.

Prefer the same four Study 06/07 tasks for comparability.

If one or more do not pass, STOP before changing the task set and report exactly which condition fails. Do not silently substitute tasks, generate new ones, relax the strategic criterion, or fabricate evidence.

---

# 7. Pairing / common random numbers

The comparison should be maximally paired.

For fixed:

```text
task
repetition
b
```

the following should share the same underlying frozen world, initial fact assignment, and matching execution-seed structure as far as the existing framework allows:

```text
naive_neutral_truth
naive_neutral_false
naive_strategic_truth
naive_strategic_false
vigilant_neutral_truth
vigilant_neutral_false
vigilant_strategic_truth
vigilant_strategic_false
```

Do not alter the established RNG stream derivation merely to force pairing.

Under a class-independent mock provider, nonsemantic mechanics should replay identically.

---

# 8. Canonical scientific coordinates

The standardized Study 08 output must expose:

```text
receiver_epistemic_disposition
controller_evidence_strategy
derived_epistemic_condition
target_semantics
message_mode
controller_fact_id
b
q_c
theta
beta
task
repetition/episode seed
```

The derived condition should be one of:

```text
naive_neutral
naive_strategic
vigilant_neutral
vigilant_strategic
```

but it must be derived from the two authoritative axes rather than becoming a competing config source.

---

# 9. Primary state variables

Use pre-round states for conditioning.

Social state:

```text
x = controller_target_share_before = n_Z/N
```

Primary epistemic state:

```text
phi = full_proof_agent_share_before
```

Secondary epistemic state:

```text
kappa = mean_supporting_fact_coverage_before
```

Do not condition controller response on post-round epistemic variables.

---

# 10. Primary analysis quantities

Reuse the existing authoritative estimator implementation.

Do not reimplement MI/CMI or efficiency.

For each Study 08 condition estimate the existing quantities:

```text
T_pi
chi
eta_IR
```

plus the existing target/truth response and knowledge observables.

Retain:

```text
round_memory_target_actuation_cmi
round_epistemic_target_actuation_cmi
round_phi_target_actuation_cmi
round_kappa_target_actuation_cmi
```

and corresponding signed responses.

Also retain:

```text
phi(t)
kappa(t)
truth_vote_share(t)
controller_target_share(t)
peer/controller fact exposures
new peer/controller facts
```

The key scientific distinction is between:

```text
prompt/evidence strategy changes epistemic accumulation
```

and:

```text
prompt/evidence strategy changes control response at matched epistemic state
```

---

# 11. Primary figures

Generate the figures automatically during Study 08 aggregation. Do not leave the comparison to manual post-processing.

Use a consistent 2×2 layout for the four epistemic conditions:

```text
                    neutral evidence        strategic evidence

naive receiver      naive_neutral           naive_strategic

vigilant receiver   vigilant_neutral        vigilant_strategic
```

Truth and false targets should be shown as separate matched figure families or otherwise clearly faceted as the independent third axis.

Use common axes, bins, support masks, and color scales for directly compared panels.

## 11.1 x × b maps

For TRUTH target, produce 2×2 panels for:

```text
T_pi(x,b)
chi(x,b)
eta_IR(x,b)
```

For FALSE target, produce the same:

```text
T_pi(x,b)
chi(x,b)
eta_IR(x,b)
```

## 11.2 x × phi maps

This is a central new analysis.

For TRUTH target, produce 2×2 panels for:

```text
T_pi(x,phi)
chi(x,phi)
eta_IR(x,phi)
```

For FALSE target, produce the same.

Use pre-round `phi`.

Mask unsupported regions using the established support rules.

Where practical distinguish:

```text
unvisited
visited but insufficient ADV/NO_OP overlap
supported estimate
```

Do not smooth/fill unsupported cells.

## 11.3 x × kappa robustness

Produce secondary/appendix versions:

```text
T_pi(x,kappa)
chi(x,kappa)
eta_IR(x,kappa)
```

for truth and false targets.

`phi` remains the primary epistemic coordinate.

## 11.4 b = 24 profiles

At minimum at `b = 24`, overlay or facet the four epistemic conditions across `x` for:

```text
T_pi(x)
chi(x)
eta_IR(x)
```

separately for truth and false targets.

These should make differences immediately legible without reading heatmaps.

## 11.5 Epistemic evolution

Plot by round/time:

```text
phi(t)
kappa(t)
truth_vote_share(t)
controller_target_share(t)
```

for all four epistemic conditions, separately or cleanly faceted by truth/false target.

## 11.6 Factorial contrasts

In addition to the four raw conditions, compute descriptive matched contrasts for existing metrics.

At fixed target semantics:

Vigilance effect:

```text
vigilant - naive
```

computed separately under neutral and strategic evidence.

Evidence-strategy effect:

```text
strategic - neutral
```

computed separately under naive and vigilant receivers.

Optional interaction diagnostic:

```text
(vigilant_strategic - naive_strategic)
-
(vigilant_neutral - naive_neutral)
```

This is a difference-of-differences diagnostic, NOT a new efficiency definition.

Do not introduce a new bounded metric unless explicitly requested later.

## 11.7 Truth vs false as separate semantic contrast

Because target semantics is an independent axis, also provide matched truth-vs-false comparisons within EACH of the four epistemic conditions.

This may include:

```text
Delta_chi_truth_false
Delta_eta_IR_truth_false
```

as descriptive differences on common supported state regions.

Again, do not name these as new efficiencies.

---

# 12. Historical recommendation-only reference

Do not rerun recommendation-only as part of Study 08.

Studies 04/06/07 already provide the bare recommendation reference.

Where scientifically useful, Study 08 reports may show those prior values as clearly labeled external/reference comparisons, but they are not part of the Study 08 factorial cell count.

Do not pool old and new episodes into one estimator without an explicit matched analysis.

---

# 13. Theory comparison

Keep the same prompt-blind coarse q-voter/control reference where applicable.

Do not add receiver vigilance or evidence strategy to the theoretical model.

That is scientifically useful:

```text
same coarse control theory
different LLM epistemic protocol/evidence selection
```

Compare empirical departures from the same matched control reference at the same:

```text
N, q, q_c, b, theta, beta
```

with the usual caveat that `recommendation_plus_fact` changes the LLM epistemic dynamics in a way the q-voter reference does not model.

---

# 14. Study folder and execution

Create:

```text
configs/runs/relational_reasoning/population_study_08/
```

using the same standardized study mechanism as Study 06/07.

Do not create study-specific SLURM logic.

Use the established:

```text
mas-cc study submit --config-dir ...
mas-cc study aggregate --study-dir ...
```

or the exact current canonical commands after inspecting the repository.

Use as few config files as the architecture naturally supports.

---

# 15. Preflight

Before any real-provider launch, run the normal Study 08 preflight and report:

```text
2 receiver dispositions
× 2 evidence strategies
× 2 target semantics
× 6 b
× tasks
× repetitions
```

Expected with 4 tasks × 10 reps:

```text
48 conditions/task
192 cells
1,920 episodes
```

Also report:

- number of configs;
- expected provider calls;
- prompt-token difference relative to recommendation-only Study 06/07;
- estimated cost;
- requested concurrency;
- estimated runtime;
- evidence-admissibility audit;
- confirmation that every cell uses `recommendation_plus_fact`.

The practical goal is an overnight run, roughly around the previous-study scale.

If projected runtime is clearly beyond an overnight window, do not silently alter the design. Report the estimate first. If reduction is necessary, reduce repetitions before deleting factorial conditions or `b` levels.

---

# 16. Validation before launch

Require all of the following:

1. receiver `naive` and `vigilant` resolve;
2. evidence strategies `neutral` and `strategic` resolve;
3. all intended tasks pass strategic-evidence admissibility for both truth and false targets;
4. all 48 condition types per task resolve with `message_mode: recommendation_plus_fact`;
5. no Study 08 cell uses recommendation-only;
6. neutral evidence is target-independent under the implemented pairing contract;
7. strategic evidence is target-aware but always a true frozen task fact;
8. controller fact IDs are logged;
9. single-information-channel tests pass;
10. prompt/evidence provenance is retained in canonical output;
11. matched seed structure is verified with a mock provider;
12. requested x×b and x×phi plots can be generated from a smoke fixture;
13. support masks behave correctly;
14. standardized aggregation succeeds;
15. no new execution architecture has been introduced.

Only after all validation passes should the real-provider Study 08 be submitted.

---

# 17. Final handoff

Report:

1. Study 08 config folder/files;
2. exact receiver axis values;
3. exact evidence-strategy axis values;
4. exact truth/false target representation;
5. confirmation that EVERY Study 08 controlled condition uses `recommendation_plus_fact`;
6. neutral evidence rule;
7. strategic evidence rule;
8. task admissibility table;
9. tasks and repetitions;
10. six-value `b` grid;
11. total cells;
12. total episodes;
13. expected calls/cost/runtime;
14. paired-seed validation;
15. configured analysis/figure families;
16. preflight result;
17. launch command;
18. job IDs/status if launched.

The intended Study 08 total, assuming four tasks and ten repetitions, is:

```text
2 receiver dispositions
× 2 evidence strategies
× 2 target semantics
× 6 budgets
× 4 tasks
× 10 repetitions
= 1,920 episodes
```

The central interpretation must remain:

> Study 08 tests whether epistemic vigilance changes how LLM agents respond to neutral versus strategically selected TRUE evidence, under both truth-aligned and false-target feedback control.

No false evidence is introduced, and `recommendation_plus_fact` is mandatory throughout Study 08.
