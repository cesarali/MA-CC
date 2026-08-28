# TDD / Implementation Plan: Local Controller-Retention and Exposure Probe

## Purpose

Build a **provider-backed but game-free local probe** that measures whether an LLM focal agent actually responds to the controller message as the visible social context becomes larger (`q > 1`).

This probe must **not run population dynamics, rounds, sensing, or the feedback policy**. Instead, it constructs isolated focal-agent decision prompts using the **same relational task representation, prompt blocks, option shuffling, response contract, controller/peer rendering, evidence rendering, and parsing code as the real game**, then asks one LLM call per local condition.

The motivation is diagnostic: before interpreting a future `q > 1`, `L > 2` population study, rule out the simpler failure mode that the controller becomes behaviorally negligible when its message is diluted among multiple social sources.

The probe should support **exactly four configurable LLM model/provider specifications** in one run and produce model-stratified statistics and plots.

---

# 1. Scientific questions

The probe should answer five separate questions.

### Q1. Runtime-faithful controller retention

With the current mechanism, one controlled social slot is present and the remaining `q-1` slots are ordinary peers. Does the probability that the focal adopts the controller target decrease as `q` increases?

Primary object:

```text
Delta_ctrl(q) = P(X' = Z | controlled local prompt)
                - P(X' = Z | matched NO_OP local prompt)
```

where all task facts, focal state, peer background, option permutation, and non-controller prompt content are paired.

### Q2. Is an apparent q-effect actually controller dilution?

The current protocol gives the controller a visible-source fraction

```text
rho_C = 1 / q
```

on a controlled microscopic update. Add a **probe-only synthetic exposure arm** in which more than one of the `q` social slots carries the controller intervention so that `rho_C` is approximately held fixed as `q` grows.

If the one-slot response collapses with `q` but the fixed-exposure response is rescued, interpret this as evidence for **controller dilution / salience loss**, not immediately as q-voter-like nonlinear social susceptibility.

### Q3. Does epistemic vigilance change controller responsiveness?

Test the existing independent receiver axis:

```yaml
game.options.receiver_epistemic_disposition: naive
# or
vigilant
```

The scientific meaning must remain exactly the established one:

- `naive`: no explicit warning that another participant may have a different objective or selectively present evidence;
- `vigilant`: warn that recommendations may reflect different objectives and should be evaluated against the explicit evidence, without telling the model to distrust all social information.

### Q4. Does evidence-selection strategy change controller responsiveness?

In an evidence-bearing arm, test:

```yaml
control.options.controller_evidence_strategy: neutral
# or
strategic
```

This axis is valid **only** with:

```yaml
message_mode: recommendation_plus_fact
```

Preserve the established semantics:

- `neutral`: exactly one real frozen-task fact selected deterministically by a target-independent rule;
- `strategic`: exactly one real frozen-task fact selected deterministically in a target-aware way to be favorable / locally compatible with the controller target;
- strategic evidence is selective disclosure / cherry-picking, **never false evidence**;
- no synthetic, paraphrased, negated, or LLM-generated task fact is allowed.

### Q5. Does task reasoning depth affect these local responses?

Use:

```text
L in {1, 2}
```

as a diagnostic before moving to deeper population tasks. `L=1` is a simple positive-control regime; `L=2` is the first genuinely compositional relational regime and should expose whether controller evidence / vigilance behave differently once one fact is not sufficient to establish the final relation.

---

# 2. Important separation from Study 08

This is **not Study 08** and must not modify Study 08 configs or historical Study 04/06/07 outputs.

Study 08 continues to require `recommendation_plus_fact` in every controlled condition. This local probe is allowed to include a separate `recommendation_only` diagnostic arm because its purpose is specifically to determine whether the bare controller recommendation is being ignored as `q` grows.

Do not reinterpret the local probe as a population-control efficiency experiment. There is no sensor, no controller policy, no `b`, no trajectory-level MI/CMI, and no thermodynamic efficiency in this assay.

---

# 3. Core design: two staged assays, not one huge factorial

Implement the script so all axes are configurable, but the default scientific execution should be split into **Stage A** and **Stage B**. This avoids an unnecessary full Cartesian explosion while still identifying the relevant mechanisms.

## 3.1 Stage A — bare recommendation retention / q dilution

Purpose: establish whether the LLM still responds to the controller recommendation when `q > 1` before adding controller evidence strategy.

Use:

```text
message_mode = recommendation_only
model          = exactly 4 configured LLMs
L              in {1, 2}
q              in {1, 2, 3, 4}
receiver       in {naive, vigilant}
target         in {truth, false}
exposure_mode  in {one_slot, approx_half}
```

`controller_evidence_strategy` MUST be absent / null in Stage A. Do not silently set `neutral` or `strategic` when there is no controller fact.

### Runtime-faithful exposure: `one_slot`

For a controlled local prompt:

```text
controller_slots = 1
ordinary_peer_slots = q - 1
rho_C = 1/q
```

This is the primary, scientifically faithful condition.

### Probe-only exposure control: `approx_half`

Use:

```text
controller_slots = max(1, round_half_up(q / 2))
ordinary_peer_slots = q - controller_slots
rho_C = controller_slots / q
```

This is **not** the current game mechanism and must be marked in every row as:

```text
probe_only_multi_controller_slot = true
```

The only purpose of this arm is to ask whether preserving controller exposure rescues responsiveness. Do not use it as a matched q-voter theory condition and do not silently feed it into population-study analysis.

If the production prompt architecture cannot represent more than one controlled social slot without invasive changes, implement a probe-specific observation adapter that still calls the canonical social-source renderer. Do not alter the production game runtime merely to support this diagnostic.

## 3.2 Stage B — evidence-bearing epistemic modifier assay

Purpose: test whether receiver vigilance and controller evidence selection modify the local controller response.

Use:

```text
message_mode = recommendation_plus_fact
model          = exactly 4 configured LLMs
L              in {1, 2}
q              in {1, 2, 3, 4}
receiver       in {naive, vigilant}
evidence       in {neutral, strategic}
target         in {truth, false}
```

Primary Stage B exposure must be:

```text
exposure_mode = one_slot
```

because this is the current runtime-faithful intervention.

Add only a **focused exposure-rescue subset**, rather than the full second exposure factorial:

```text
q in {2, 4}
exposure_mode = approx_half
```

for the same four receiver/evidence conditions. This is enough to tell whether strategic/neutral evidence itself survives controller dilution without doubling the whole Stage B grid.

---

# 4. Local vignette construction

## 4.1 Reuse production prompt machinery

The probe must not hand-write an alternative version of the relational prompt.

Inspect the real game and reuse, as directly as possible:

- frozen relational task loader / symbolic task representation;
- focal private-knowledge rendering;
- current-vote rendering;
- social participant rendering;
- controller participant rendering;
- `recommendation_only` vs `recommendation_plus_fact` message behavior;
- `receiver_epistemic_disposition` prompt block;
- structured `shared_fact_id` evidence renderer;
- semantic option shuffling;
- parse / validate action code;
- response contract.

The local harness should construct an observation/state object and call the same prompt builder instead of concatenating free-form prompt strings.

## 4.2 Preserve controller identity semantics

The real controller is rendered as another participant, not an authority. Preserve that behavior.

Do not add labels such as:

```text
controller
expert
system recommendation
trusted source
```

unless the current production renderer already contains them.

The probe is intended to measure the behavior of the mechanism we actually use, not a stronger controller invented for testing.

## 4.3 Task fixtures

Create or reuse a frozen local-probe task set with exact symbolic validation.

Required depths:

```text
L = 1
L = 2
```

Use enough distinct task worlds to make the statistical unit the **task/vignette**, not merely repeated stochastic generations of one prompt. Default target:

```text
12 frozen tasks per L
```

If reusable validated datasets already exist, prefer them. Otherwise generate a dedicated frozen probe fixture directory using the existing relational task generator and record its manifest/fingerprint.

Do not dynamically generate a new world per API call.

## 4.4 Receiver private knowledge regimes

Use two explicit local knowledge regimes, but do not cross both through every expensive condition unless required.

### Primary: `zero_support`

The focal starts with no supporting proof fact.

This isolates social recommendation / controller evidence susceptibility and makes `L=1` versus `L=2` a genuine difference in how much reasoning information is absent.

### Calibration subset: `one_missing`

The focal starts with exactly `L-1` supporting facts and lacks exactly one proof fact.

Run this subset primarily in Stage B. It is a positive-control condition for whether a supplied true fact can complete an otherwise nearly complete proof.

For `L=1`, `one_missing` coincides with knowing zero support facts. Deduplicate rather than issuing duplicate calls.

## 4.5 Focal current vote

The local probe must not allow the initial focal vote to become an uncontrolled nuisance variable.

Use a balanced deterministic schedule across fixtures:

```text
initial_vote in {controller_target, truth, other_non_target}
```

subject to semantic deduplication when `controller_target == truth`.

Record both the semantic initial vote and final vote. This enables separate measurement of:

```text
adoption of controller target
retention of controller target
switch toward controller
switch away from controller
```

## 4.6 Ordinary peer background

For every `(q, controller_slots)` condition, explicitly vary how many **ordinary peers** support the controller target.

Let:

```text
m = ordinary_peer_slots
j = number of ordinary peers voting for controller target
```

Use every feasible:

```text
j in {0, ..., m}
```

and define:

```text
peer_target_fraction = j/m
```

when `m > 0`.

For `m = 0`, store `peer_target_fraction = NaN` and a categorical label `controller_only_social_context`.

This is the local analogue of asking whether the controller response depends on pre-existing social support. It is **not** the same as the population-state coordinate `x=n_Z/N` and must not be labeled as such.

For the two non-target semantic alternatives in `K=3`, use a deterministic balanced assignment rule and store the exact peer vote-count vector. Do not collapse the raw fixture to target/non-target in the saved data.

---

# 5. Matched counterfactual construction

Every controlled vignette needs a matched local baseline.

Create a stable `vignette_pair_id` from the components that are identical across the pair:

```text
model-independent task fixture
L
focal private knowledge
initial semantic vote
q
ordinary peer background
receiver disposition
target semantics
option permutation seed
replicate seed
```

Then render two versions.

### Controlled version

Contains the configured controller slot(s), recommendation, and controller fact if `recommendation_plus_fact`.

### Matched NO_OP version

Contains no controller message/fact. Replace the removed controlled slot(s) with pre-specified ordinary-peer slots from the same deterministic counterfactual background generator so that the total number of visible social slots remains `q`.

The pairing rule must be deterministic and documented. Do not select the NO_OP replacement after observing a model response.

The primary controller effect is the paired change in final target adoption, not the uncontrolled marginal frequency from unrelated prompts.

---

# 6. Evidence strategy rules and admissibility

This probe must reuse the authoritative `neutral` and `strategic` selectors implemented for the factorized epistemic design.

Do not reimplement strategic evidence heuristically inside the probe.

For every Stage B task and each target semantic, perform a preflight audit:

```text
task_id
L
truth_target
false_target
neutral_fact_id
strategic_truth_fact_id
strategic_false_fact_id
strategic_truth_admissible
strategic_false_admissible
reason
```

Critical rule:

- if a strategically useful true fact does not exist for a task/target, mark it inadmissible;
- do not fall back to neutral evidence;
- do not fabricate a fact;
- do not change the target;
- do not silently drop the failure from the audit.

### Special warning for L=1

A one-step relation can make false-target strategic evidence scientifically impossible under a strict target-compatible true-fact criterion. Treat this as a meaningful admissibility result, not as an implementation inconvenience.

If false-target strategic cells are inadmissible at `L=1`, keep `L=1` in Stage A and in admissible Stage B cells, but do not force a fake fully balanced strategic-false `L=1` factorial.

---

# 7. Four-model execution contract

The probe must accept exactly four explicit model specifications, for example via a config section such as:

```yaml
models:
  - provider: ...
    model: ...
    label: model_1
  - provider: ...
    model: ...
    label: model_2
  - provider: ...
    model: ...
    label: model_3
  - provider: ...
    model: ...
    label: model_4
```

Do not hard-code model names in the analysis layer.

Use the same generation settings that the corresponding real relational game would use for that provider/model unless a probe config explicitly overrides them. Record temperature / sampling settings and any provider-specific seed support.

All model comparisons must use the same frozen task/vignette schedule and semantic option-permutation schedule wherever the provider APIs permit.


## 7.1 Bounded multiprocessing / concurrent provider execution

The probe must **not execute provider calls serially**. Use a bounded worker pool with configurable concurrency:

```yaml
execution:
  workers: 4
  backend: process_pool
  resume: true
  max_retries: 2
```

Default:

```text
workers = 4
```

The unit scheduled to a worker is **one fully specified local prompt call**. Calls from different models, tasks, `q`, `L`, receiver dispositions, targets, and exposure arms may therefore be in flight simultaneously.

Implementation requirements:

- Prefer `concurrent.futures.ProcessPoolExecutor(max_workers=workers)` or the repository's existing equivalent bounded process-pool abstraction if one already exists and is suitable.
- Do not share a live provider/client object between processes. Each worker initializes its own provider client from a serializable provider/model specification.
- The worker receives an immutable call specification and returns a result object; it must not mutate shared experiment state.
- **Only the parent/master process writes canonical output files** (`raw_calls.jsonl`, CSV summaries, checkpoints, and the final report).
- Results may finish out of order, but every call carries a deterministic `call_id`; final artifacts are sorted/indexed by that ID.
- All design and pairing seeds are generated before scheduling. Concurrency must never affect task selection, option permutation, peer composition, controlled/NO_OP pairing, or replicate identity.
- A failed provider call is retried using the same immutable call specification. If retries are exhausted, record a failed row rather than substituting another vignette.
- Resume mode skips already completed `call_id`s and schedules only missing calls.
- Provider rate limits remain authoritative. Support optional per-provider concurrency caps without altering the scientific grid.
- Multiprocessing is an execution optimization only; it must have **zero effect on prompt contents or estimands**.

Interleave models and conditions rather than completing one entire model first. This gives balanced partial results and prevents one slow provider from blocking the whole run.

Preflight must print:

```text
execution backend
requested workers
effective workers
per-provider concurrency caps, if any
total scheduled calls
already completed calls
remaining calls
```

Use a single master-process progress display with completed/total calls, errors, elapsed time, and ETA. Do not let worker processes print independent progress bars.

---

# 8. Replication and suggested scale

The statistical unit should be a **frozen task/vignette**, with repeated generations used only to estimate within-vignette model stochasticity.

Recommended starting values:

```text
12 tasks per L
3 stochastic replicates per vignette
```

Before launch, print exact projected call counts separately for Stage A and Stage B.

Do not silently reduce tasks or factors. If the full requested model grid is unexpectedly expensive, report the preflight and reduce stochastic replicates before removing scientific axes.

Because this probe makes one provider call per local prompt rather than hundreds of calls per population episode, it should be substantially cheaper than a full game study.

---

# 9. Primary estimands

All quantities below are descriptive probabilities / paired contrasts. They are **not** the existing population-level susceptibility `chi` unless explicitly named `local_*`.

## 9.1 Local target-adoption probability

For a condition `c`:

```text
R_C(c) = P(X' = Z | controlled, c)
```

Report this because it is directly interpretable, but do not use it alone as evidence of controller effect.

## 9.2 Paired local controller effect — PRIMARY

```text
Delta_C(c)
  = P(X' = Z | controlled, c)
    - P(X' = Z | matched NO_OP, c)
```

Estimate from paired vignette outcomes.

This is the primary quantity for the controller-retention plot.

## 9.3 Switch-toward-controller effect

Among cases with initial vote not equal to `Z`:

```text
P(X' = Z | controlled)
- P(X' = Z | matched NO_OP)
```

This distinguishes active persuasion from merely retaining a pre-existing target vote.

## 9.4 Controller-target retention effect

Among cases with initial vote equal to `Z`:

```text
P(X' = Z | controlled)
- P(X' = Z | matched NO_OP)
```

This measures whether the controller prevents abandonment of its target.

## 9.5 q-dilution ratio

For each model and matched semantic condition:

```text
D_q = Delta_C(q) / Delta_C(q=1)
```

when the denominator is supported and sufficiently far from zero.

Do not report unstable ratios when `|Delta_C(q=1)|` is below a configured tolerance; report the raw contrast instead.

## 9.6 Exposure-rescue contrast

For `q > 1`:

```text
R_exposure(q)
  = Delta_C(approx_half, q)
    - Delta_C(one_slot, q)
```

Positive values indicate that preserving controller exposure rescues local influence.

This is the key diagnostic separating controller dilution from a more intrinsic loss of response under larger social groups.

## 9.7 Truth-selectivity contrast

```text
A_truth
  = Delta_C(target=truth)
    - Delta_C(target=false)
```

A model that follows correct controller interventions more than false ones exhibits positive target selectivity. Blind controller compliance would tend to make this contrast small even if `Delta_C` itself is large.

## 9.8 Vigilance contrast

At fixed evidence strategy / target semantics:

```text
V = Delta_C(vigilant) - Delta_C(naive)
```

Interpret separately for truth and false targets. Epistemic vigilance is not defined as globally lowering controller response.

## 9.9 Strategic-evidence contrast

In `recommendation_plus_fact` only:

```text
S = Delta_C(strategic) - Delta_C(neutral)
```

Again interpret separately for truth and false targets.

## 9.10 Vigilance × strategic-evidence interaction

```text
I_VS
  = [Delta_C(vigilant, strategic) - Delta_C(naive, strategic)]
    - [Delta_C(vigilant, neutral) - Delta_C(naive, neutral)]
```

Treat as a descriptive difference-in-differences, not as a new efficiency.

## 9.11 L effect

Report:

```text
Delta_L = Delta_C(L=2) - Delta_C(L=1)
```

within common admissible / supported conditions.

Do not pool `L=1` strategic-false cells with `L=2` if the former are structurally inadmissible.

---

# 10. Descriptive aggregation

This is a controlled local-response probe. The main scientific objects are the **paired displacements themselves**, not confidence intervals or hypothesis tests.

## 10.1 Primary displacement summary

For every frozen matched vignette `j`, retain the local paired displacement:

```text
delta_j
  = 1[X'_controlled = Z] - 1[X'_NOOP = Z]
```

and summarize within each experimental cell as:

```text
Delta_C = mean_j(delta_j)
```

Equivalently for binary target adoption:

```text
Delta_C
  = P(X' = Z | controlled)
    - P(X' = Z | matched NO_OP)
```

The Markdown report should show the direct displacement `Delta_C`, the two raw target rates, and the number of matched local pairs.

Useful secondary descriptive diagnostics may be stored in CSV, for example:

```text
mean(delta_j)
std(delta_j)
fraction(delta_j > 0)
fraction(delta_j = 0)
fraction(delta_j < 0)
```

These are descriptive stability checks across constructed local situations. They are not required in the main Markdown tables.

If identical prompts are intentionally sampled repeatedly at nonzero temperature, retain replicate-level variability in the raw/CSV artifacts. **Do not add confidence intervals to the main Markdown report or primary plots.**

## 10.2 Per-model first

Every primary plot and displacement summary must be produced **separately for each of the four LLMs**.

A pooled-across-model displacement may be shown only as a secondary descriptive summary because the four models are not exchangeable replicates of one model family.

## 10.3 Optional descriptive factorial model

As a secondary diagnostic only, a per-model Bernoulli logistic model may be fit for final target adoption / switch-toward-target using:

```text
q
rho_C
L
receiver
target_semantics
evidence_strategy   # Stage B only
peer_target_fraction
```

Use this only to compactly describe interaction structure. Do not make regression p-values, significance stars, or confidence intervals part of the primary scientific conclusion.

# 11. Required plots

## 11.1 Primary controller-retention plot

For each model, produce a plot of:

```text
y = Delta_C
x = q
```

with separate curves for:

```text
one_slot
approx_half exposure
```

with the raw displacement curves only; do not add confidence bands.

Facet or create separate figure families for:

```text
target = truth / false
L = 1 / 2
receiver = naive / vigilant
```

Do not mix all semantic conditions into one unreadable panel.

This is the main diagnostic requested by this study.

## 11.2 Peer-support response plot

For each model and selected `q`, plot:

```text
Delta_C versus peer_target_fraction
```

for the runtime-faithful `one_slot` condition.

This shows whether the controller only works when ordinary peers already support its target.

## 11.3 Exposure-rescue plot

For `q > 1`, plot:

```text
R_exposure(q)
```

with zero reference line.

A large positive rescue is direct evidence that the one-slot controller is being diluted by the larger social context.

## 11.4 Epistemic 2×2 plot family

For Stage B, use the established 2×2 organization:

```text
                    neutral evidence        strategic evidence

naive receiver      naive_neutral           naive_strategic

vigilant receiver   vigilant_neutral        vigilant_strategic
```

Within each panel show `Delta_C(q)` for the primary one-slot exposure.

Produce separate figure families for truth and false targets, and for `L=1` / `L=2` where admissible.

## 11.5 Truth-selectivity plot

Plot:

```text
A_truth(q)
```

for each model.

This is crucial for distinguishing "the model ignores the controller" from "the model selectively rejects a false controller because it can reason from the evidence."

---

# 12. Interpretation logic / decision table

The analysis report should explicitly classify the following patterns.

### Pattern A: one-slot and approx-half both remain strong as q rises

Interpretation: controller retention is robust; `q > 1` population experiments are not obviously threatened by recommendation neglect.

### Pattern B: one-slot collapses, approx-half is rescued

Interpretation: controller dilution / salience loss is a major confound. A future `q > 1` population study must either treat this as the scientific object or redesign controller exposure before calling the result nonlinear social susceptibility.

### Pattern C: both collapse as q rises

Interpretation: larger social contexts suppress the intervention even after controller exposure is increased. This is stronger evidence for a genuine social-context interaction rather than a simple `1/q` salience dilution.

### Pattern D: false-target response collapses but truth-target response remains

Interpretation: do not call this controller failure. It is consistent with evidence-sensitive / epistemically selective behavior and should be read with the truth-selectivity contrast and disposition/evidence factors.

### Pattern E: `vigilant` reduces false-target response much more than truth-target response

Interpretation: evidence-sensitive vigilance is behaving as intended rather than inducing blanket social distrust.

### Pattern F: `strategic` increases false-target response using only true facts

Interpretation: selective disclosure can steer the model even without false evidence; this is the local precursor of the planned Study 08 population question.

---

# 13. Saved row schema

Every model call should produce one canonical row containing at least:

```text
probe_version
stage
model_label
provider
model_id
generation_settings_hash
task_id
task_fingerprint
L
K
truth_semantic
controller_target_semantic
target_semantics                 # truth / false
receiver_epistemic_disposition
message_mode
controller_evidence_strategy     # null in Stage A
controller_fact_id               # null when absent
strategic_admissible
knowledge_regime
known_fact_ids
initial_vote_semantic
q
controller_slots
ordinary_peer_slots
controller_exposure_fraction
probe_only_multi_controller_slot
ordinary_peer_vote_count_vector
peer_target_fraction
option_permutation_seed
prompt_definition_hash
vignette_pair_id
replicate
condition                         # controlled / NO_OP
final_vote_semantic
final_is_controller_target
final_is_truth
switched
switched_to_controller_target
parse_ok
provider_error
latency
input_tokens
output_tokens
```

If the response contract records a private reason / chosen shared fact, retain them in the raw artifact according to current privacy/artifact conventions, but do not create a new social-information channel from them.

---

# 14. Output artifacts

Create a dedicated analysis directory containing at least:

```text
controller_retention_probe/
  resolved_probe_config.yaml
  preflight.json
  task_admissibility.csv
  raw_calls.jsonl
  local_response_rows.csv
  paired_controller_effects.csv
  model_summary.csv
  plots/
    controller_retention_by_q_*.png
    peer_support_response_*.png
    exposure_rescue_*.png
    epistemic_2x2_*.png
    truth_selectivity_*.png
  controller_retention_probe_report.md
```

The principal human-readable deliverable is:

```text
controller_retention_probe_report.md
```

It must be self-contained enough to understand the experiment without opening the CSVs. It must explain every reported quantity, show the main numerical results in Markdown tables, and link to the corresponding plots using relative paths.

The report must start with the runtime-faithful `one_slot` result. Probe-only multi-controller-slot results are secondary diagnostics and must be clearly labeled.

## 14.1 Required report structure

Use:

```text
# Controller-retention local probe

## 1. Experimental design
## 2. How to read the statistics
## 3. Cross-model summary
## 4. Model: <LLM 1>
## 5. Model: <LLM 2>
## 6. Model: <LLM 3>
## 7. Model: <LLM 4>
## 8. Cross-model interpretation
## 9. Execution / data-quality diagnostics
## 10. Reproducibility
```

Do **not** merge all four LLMs into one enormous primary table. Each model gets its own section and its own tables. A compact cross-model table is allowed only as an executive summary.

## 14.2 Table explaining the statistics

Near the beginning include:

| Quantity | Definition | Interpretation |
|---|---|---|
| `P_controlled` | `P(X'=Z | controlled)` | Raw probability of ending on controller target |
| `P_NOOP` | `P(X'=Z | matched NO_OP)` | Matched baseline probability |
| `Delta_C` | `P_controlled - P_NOOP` | Primary local controller effect |
| `D_q` | `Delta_C(q) / Delta_C(q=1)` | Relative retention as social context grows |
| `R_exposure` | `Delta_C(approx_half)-Delta_C(one_slot)` | Rescue from preserving controller exposure |
| `A_truth` | `Delta_C(truth)-Delta_C(false)` | Truth-selective response |
| `V` | `Delta_C(vigilant)-Delta_C(naive)` | Vigilance contrast |
| `S` | `Delta_C(strategic)-Delta_C(neutral)` | Strategic-evidence contrast |

This is an explanatory table, not a result table.

## 14.3 Cross-model executive-summary table

Produce one compact table with exactly one row per LLM:

| Model | One-slot q trend | Largest exposure rescue | Truth selectivity | Vigilance effect | Strategic-evidence effect | Overall retention diagnosis |
|---|---:|---:|---:|---:|---:|---|
| `<model_1>` | ... | ... | ... | ... | ... | ... |
| `<model_2>` | ... | ... | ... | ... | ... | ... |
| `<model_3>` | ... | ... | ... | ... | ... | ... |
| `<model_4>` | ... | ... | ... | ... | ... | ... |

`One-slot q trend` should be a simple descriptive endpoint contrast or slope, with the definition stated in text. Never infer the diagnosis from a pooled-across-model estimate.

## 14.4 Separate Stage A result table for each LLM — PRIMARY

For each LLM, the first numerical table must contain runtime-faithful `one_slot` results:

| L | q | Receiver | Target | N pairs | P controlled | P NO_OP | Delta_C | D_q | Interpretation |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 1 | naive | truth | ... | ... | ... | ... | 1.00 | ... |
| 1 | 2 | naive | truth | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Do not average over `L`, receiver disposition, or truth/false target in this primary table.

Immediately below it link/embed:

```text
plots/controller_retention_by_q_<model>.png
```

Then answer in prose:

1. Does `Delta_C` remain non-negligible for `q > 1`?
2. Does it decrease with `q`?
3. Is the behavior different for truth versus false targets?
4. Is the behavior different under `naive` versus `vigilant`?

## 14.5 Separate exposure-rescue table for each LLM

For each LLM:

| L | q | Receiver | Target | Delta_C one_slot | Delta_C approx_half | R_exposure | Diagnosis |
|---:|---:|---|---|---:|---:|---:|---|
| 1 | 2 | naive | truth | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

This is the main table for distinguishing simple `1/q` controller dilution from a broader social-context suppression.

## 14.6 Stage B epistemic tables for each LLM

Avoid one opaque 64-row factorial table. For each model, create separate tables by `(L, target_semantics)`, with `q` as rows and the 2x2 epistemic design as columns:

| q | naive / neutral | naive / strategic | vigilant / neutral | vigilant / strategic |
|---:|---:|---:|---:|---:|
| 1 | `Delta_C` | `Delta_C` | `Delta_C` | `Delta_C` |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... |

Create separately for:

```text
L=1, truth target
L=1, false target        # admissible cells only
L=2, truth target
L=2, false target
```

Any inadmissible cell must say, for example:

```text
N/A — no admissible strategic true fact
```

rather than being blank or silently omitted.

After these tables summarize:

```text
A_truth
V
S
I_VS
Delta_L
```

as direct displacement contrasts.

## 14.7 Data-quality / execution table per model

At the end of each model section include:

| Calls scheduled | Calls successful | Provider errors | Parse failures | Success rate | Median latency | Input tokens | Output tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|

This prevents weak effects from being confused with provider/parsing failures.

## 14.8 Cross-model interpretation section

After the four model sections include:

| Scientific question | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---|---|---|---|
| Controller retained at q=2? | ... | ... | ... | ... |
| Controller retained at q=3/4? | ... | ... | ... | ... |
| Exposure rescue present? | ... | ... | ... | ... |
| Truth-selective response? | ... | ... | ... | ... |
| Vigilance selective rather than blanket? | ... | ... | ... | ... |
| Strategic evidence increases steering? | ... | ... | ... | ... |
| Safe to proceed to q>1, L>2 population study? | ... | ... | ... | ... |

The final row is a diagnostic recommendation based on the observed displacement patterns, not an automated scientific truth.

## 14.9 No fabricated numbers

Unit/smoke tests may use synthetic fixture values. The production Markdown report must contain only statistics computed from actual completed probe rows. Missing or unsupported quantities must be printed as `N/A` with a reason.

---

# 15. Preflight requirements

Before any real-provider calls, the probe command must report:

```text
number of models = exactly 4
number of frozen L=1 tasks
number of frozen L=2 tasks
Stage A condition count
Stage B admissible condition count
number of inadmissible strategic task/target cells
calls per model
calls total
estimated input/output tokens
estimated cost if pricing metadata is available
requested concurrency
```

Also verify:

- all task fixtures validate symbolically;
- option shuffling is semantic and deterministic under the recorded seed;
- controlled / NO_OP prompt pairs differ only in the intended social intervention blocks;
- `naive` and `vigilant` preserve all non-epistemic prompt content;
- `neutral` evidence is deterministic and target-independent;
- `strategic` evidence is deterministic, target-aware, and always a real true task fact;
- evidence strategy is never used with `recommendation_only`;
- `NO_OP` never inserts a controller fact;
- no production game runtime semantics are changed to support the probe-only multi-slot condition.

---

# 16. Tests to write first

## 16.1 Prompt-pair identity test

For a matched controlled / NO_OP pair, assert that task, private knowledge, current vote, option mapping, receiver disposition, and ordinary peer data are identical. Only the configured controller social slot(s) / replacement peer slot(s) may differ.

## 16.2 q slot-count test

For every prompt:

```text
number of visible social slots == q
```

For `one_slot` controlled prompts:

```text
controller_slots == 1
```

For `approx_half` prompts:

```text
controller_slots == max(1, round_half_up(q/2))
```

## 16.3 Production-renderer reuse test

Assert that the probe uses the same canonical prompt/social-source renderer and parser as the relational game. Do not allow a second hand-written prompt template to drift independently.

## 16.4 Receiver disposition test

With identical semantic state:

- naive contains no strategic-source warning;
- vigilant contains the canonical vigilance warning;
- all non-epistemic prompt blocks are identical.

## 16.5 Evidence strategy validation

- evidence strategy + recommendation-only fails;
- neutral and strategic each expose exactly one real fact when controlled;
- neutral is target-independent;
- strategic is target-aware;
- no false or mutated fact can pass validation;
- inadmissible strategic task/target pairs are explicit failures/omissions with audit provenance.

## 16.6 Option-shuffle invariance

Changing display letters must not change saved semantic task/peer/controller votes.

## 16.7 K=3 peer-vector provenance

Store the full three-option peer count vector; do not save only target/non-target counts.

## 16.8 Mock-provider smoke test

A deterministic mock provider must run the complete Stage A and a small Stage B subset without population runtime, producing valid paired rows and all analysis tables.

## 16.9 Plot smoke test

Synthetic fixtures with known controller effects must generate all required plot families with common scales and correct labels.

---

# 17. Command / integration recommendation

Prefer a provider-independent analysis/probe entry point rather than creating a new `Game` type.

A reasonable shape after repository inspection would be conceptually similar to:

```bash
mas-cc probe controller-retention \
  --config configs/probes/controller_retention.yaml
```

or an equivalent script under the repository's established experiment tooling.

Do not force this exact CLI if it conflicts with the repository architecture. The important boundary is:

```text
frozen local vignette builder
-> canonical game prompt builder
-> provider call
-> canonical parser
-> paired local analysis
```

with no population transition loop.

---

# 18. Acceptance criteria

The implementation is complete when:

- [ ] the probe performs local one-call decisions without running the game population dynamics;
- [ ] exactly four model specifications can be executed in the same frozen vignette grid;
- [ ] provider calls execute through a bounded multiprocessing/concurrent worker pool with default `workers=4`;
- [ ] concurrency is deterministic with respect to call design, pairing, option shuffling, and seeds;
- [ ] only the master process writes canonical output artifacts;
- [ ] interrupted runs can resume by deterministic `call_id` without repeating completed calls;
- [ ] `q in {1,2,3,4}` is supported;
- [ ] runtime-faithful `one_slot` exposure is the primary condition;
- [ ] probe-only `approx_half` exposure is implemented and unmistakably labeled;
- [ ] `L in {1,2}` frozen fixtures are supported;
- [ ] `naive` and `vigilant` reuse the authoritative receiver-disposition implementation;
- [ ] Stage A uses `recommendation_only` with no evidence-strategy field;
- [ ] Stage B uses `recommendation_plus_fact` with `neutral` / `strategic` evidence;
- [ ] truth and false controller targets remain a separate independent axis;
- [ ] strategic evidence is always true and task-admissibility failures are explicit;
- [ ] controlled / NO_OP prompt pairs are deterministically matched;
- [ ] primary `Delta_C(q)` is reported directly from matched local displacements, without confidence intervals in the Markdown report or primary plots;
- [ ] results are reported separately for all four LLM models;
- [ ] controller exposure rescue, truth selectivity, vigilance, strategic-evidence, and L contrasts are produced;
- [ ] the primary controller-retention plot is generated automatically;
- [ ] `controller_retention_probe_report.md` is generated automatically;
- [ ] the Markdown report contains one compact cross-model summary plus separate detailed result tables for each of the four LLMs;
- [ ] Stage A one-slot results are the first numerical table in every model section;
- [ ] Stage B is rendered as readable per-model 2x2 epistemic tables rather than one opaque pooled factorial;
- [ ] no population-level `chi`, MI/CMI, `eta_IR`, or thermodynamic-efficiency label is incorrectly attached to these local quantities;
- [ ] production game semantics and historical study configs remain unchanged.

---

# 19. Required handoff after implementation

Report:

1. files added / changed;
2. exact probe config path;
3. exact four configured model labels and provider/model IDs;
4. frozen L=1 and L=2 fixture locations and fingerprints;
5. exact prompt-building functions reused from production;
6. exact local NO_OP pairing rule;
7. exact implementation of `one_slot` and `approx_half` exposure;
8. exact naive/vigilant prompt provenance;
9. exact neutral/strategic fact-selector provenance;
10. strategic admissibility audit, especially false-target `L=1`;
11. multiprocessing/concurrency backend, requested/effective worker count, and provider caps;
12. exact path to `controller_retention_probe_report.md` and the four per-model report sections;
11. Stage A and Stage B call counts;
12. cost/runtime preflight;
13. test results;
14. raw/analysis output paths;
15. controller-retention plots for each of the four models;
16. a short conclusion classifying each model according to Patterns A–F above;
17. recommendation on whether a subsequent `q > 1` population study can be interpreted cleanly or requires a controller-exposure redesign first.

---

# Bottom line

The primary question is deliberately simple:

```text
When one local LLM decision is held fixed in every other respect,
does the controller still move the answer as q increases?
```

The `one_slot` condition tests the mechanism we actually use. The `approx_half` condition diagnoses whether any loss is primarily exposure dilution. `naive` versus `vigilant`, `neutral` versus `strategic` true evidence, truth versus false targets, and `L=1` versus `L=2` then explain *why* the controller is followed or resisted rather than collapsing all forms of non-response into a single "the model ignored the controller" conclusion.
