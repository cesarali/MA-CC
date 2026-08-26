# Study 08 Implementation Prompt

Please create a new Relational Population Study 08 whose purpose is to compare the four epistemic prompt classes under otherwise matched control conditions.

Study 08 should be based directly on the current Study 06/07 relational setup and should use the same standardized study-level launch, parallelization, results-only retention, aggregation, and analysis workflow already used there.

Do not redesign the execution architecture. Inspect Study 06 and Study 07 first and create Study 08 in the same way they were created and launched.

Do not launch anything until the new `epistemic_prompt_class` implementation and its tests are present and passing.

## Scientific question

Study 08 should answer:

How does the epistemic reasoning prompt of the agents change the efficiency with which an external controller can steer the population, and does that effect differ between incorrect/adversarial and truth-aligned control?

The four prompt classes are:

- `naive`
- `distributed_information`
- `strategic_uncertainty`
- `evidence_calibrated`

These should come from the new prompt-class implementation. Do not reproduce the prompt texts inside the Study 08 configs; use the categorical config coordinate implemented by the preceding TDD task.

The central comparison should be:

prompt class × controller semantic alignment × actuation budget

while keeping the rest of the system fixed.

## Experimental design

Use the same relational game, population size, tasks/worlds, model/provider, round horizon, initialization, recommendation-only controller, and scientific retention settings used in the main Study 06/07 experiments unless there is a concrete technical reason not to.

Keep fixed:

- `N`: same as Study 06/07
- `q = 1`
- `q_c = 12`
- `theta = 0.5`
- `beta = 4`
- same number of rounds as Study 06/07
- same frozen relational task family
- same model/provider as the current main studies
- same `recommendation_only` controller behavior
- same fact-sharing mechanism
- same knowledge propagation
- same response contract
- same single-information-channel guarantee
- same analysis estimators
- same bootstrap/null/support settings
- same results-only scientific retention
- same execution/parallelization protocol

Sweep:

- `epistemic_prompt_class = [naive, distributed_information, strategic_uncertainty, evidence_calibrated]`
- `b = [4, 8, 12, 16, 20, 24]`
- controller semantic alignment = `[wrong/adversarial, truth]`

Use the repository's existing mechanism for specifying:

- the same incorrect/adversarial target used in the relevant Study 06/07 experiment
- the correct/truth target used in the Study 07 truth-aligned experiment

Do not invent a new semantic-target mechanism if the repository already supports both.

This gives the core experimental design:

4 prompt classes × 6 intervention budgets × 2 controller semantic targets = 48 scientific policy conditions per task.

This is intentionally not a large factorial sweep.

Do not vary:

- `q_c`
- `theta`
- `beta`
- `q`

in Study 08.

The purpose is to isolate the prompt-class effect cleanly.

## Expected scale

If Study 08 uses the same 4-task × 10-repetition convention as the current Study 06/07 template, the expected workload is:

- 4 prompt classes
- 6 intervention budgets
- 2 controller semantic targets
- 4 tasks
- 10 repetitions

Therefore:

- 48 policy conditions per task
- 192 scientific cells total
- 1,920 episodes total

Treat these as the expected numbers and verify them against the actual Study 06/07 template during preflight. If the repository currently uses a different task/repetition convention, preserve the established template and report the resulting totals explicitly before launch.

## Tasks and repetitions

Use the same task set and repetition convention as Study 06/07 unless inspection of the actual configs shows that one of them is substantially better suited as the direct template.

Prefer enough matched tasks to avoid making the result task-specific.

My expectation is that Study 08 should reuse the Study 06/07 task set and repetitions rather than creating new tasks.

Preserve common random numbers and paired seeds as much as technically possible.

For the same:

- task
- repetition
- `b`
- target semantics

the four prompt classes should begin from the same frozen world, initial fact assignment, and corresponding execution seed.

Likewise, where technically possible, truth-vs-wrong comparisons should use matched task/repetition seeds.

The scientific comparison should therefore be as paired as the existing execution framework allows.

Do not alter RNG-stream derivation merely to force pairing if the existing Study 06/07 mechanism already handles it.

## Critical invariance

Across prompt classes, the only scientific change should be the epistemic prompt class.

Do not change:

- visible facts
- initial `K_i`
- peer selection
- controller sensing
- controller scheduling
- intervention mechanics
- response schema
- fact-sharing rules
- controller visibility
- target semantics except for the explicit truth/wrong axis
- any estimator definitions

This study is meant to measure the effect of epistemic framing, not prompt plus dynamics changes.

## Primary observables

Retain the full established analysis set, but Study 08 should be organized around the following quantities.

### Social state

Use:

`x = controller_target_share_before = n_Z / N`

### Epistemic state

Use:

- `phi = full_proof_agent_share_before`
- `kappa = mean_supporting_fact_coverage_before`

`phi` should be the primary epistemic coordinate.

`kappa` should be retained as a robustness / secondary coordinate.

Always use the pre-round values for state conditioning.

## Primary scientific diagrams

The final aggregation should automatically generate comparison-ready figures.

The most important requirement is that the four prompt classes appear in directly comparable panels with common axes and, where appropriate, common color scales.

Do not make me reconstruct these diagrams manually afterward.

### A. Classical state-local control maps

For each controller semantic target separately:

- wrong/adversarial
- truth-aligned

produce, for each prompt class:

- `T_pi(x,b)`
- `chi(x,b)`
- `eta_IR(x,b)`

at fixed:

- `q_c = 12`
- `theta = 0.5`
- `beta = 4`

The preferred presentation is one four-column figure per observable:

`naive | distributed_information | strategic_uncertainty | evidence_calibrated`

with the same `x` bins, `b` values, support masking, and color scale across columns.

At minimum produce:

Wrong/adversarial control:
- `T_pi(x,b)` with 4 prompt-class panels
- `chi(x,b)` with 4 prompt-class panels
- `eta_IR(x,b)` with 4 prompt-class panels

Truth-aligned control:
- `T_pi(x,b)` with 4 prompt-class panels
- `chi(x,b)` with 4 prompt-class panels
- `eta_IR(x,b)` with 4 prompt-class panels

These are the direct prompt-class analogues of the Study 06 state-local phase diagrams.

### B. Epistemic efficiency maps

This is the central new Study 08 analysis.

For each prompt class and each controller semantic target, estimate and plot the state-local quantities on the social–epistemic plane:

- `T_pi(x,phi)`
- `chi(x,phi)`
- `eta_IR(x,phi)`

using the already-recorded pre-round epistemic state.

Again, use common binning and common color scales across prompt classes.

Primary comparison layout:

`naive | distributed_information | strategic_uncertainty | evidence_calibrated`

for a fixed controller target semantics.

Produce:

Wrong/adversarial:
- `T_pi(x,phi)`
- `chi(x,phi)`
- `eta_IR(x,phi)`

Truth-aligned:
- `T_pi(x,phi)`
- `chi(x,phi)`
- `eta_IR(x,phi)`

If support is insufficient for a full `x × phi` map under some condition, do not fabricate or smooth unsupported values.

Use the established support diagnostics and mask unsupported regions.

If practical, distinguish visually between:

- unvisited state region
- visited but insufficient ADVOCATE/NO_OP overlap
- supported estimate

rather than rendering all missing or unsupported cells identically.

### C. Kappa robustness

Produce the same analysis using `kappa` as a secondary epistemic coordinate:

- `T_pi(x,kappa)`
- `chi(x,kappa)`
- `eta_IR(x,kappa)`

These can go into secondary plots or appendix-style outputs.

Do not let `kappa` double the entire headline figure set if `phi` already tells the same story.

`phi` is primary.

### D. Prompt-class profile summaries

At fixed `b`, particularly `b = 24`, produce profile plots across `x` for the four prompt classes:

- `T_pi(x)`
- `chi(x)`
- `eta_IR(x)`

separately for truth and wrong control.

The goal is to make prompt-class differences immediately visible without relying only on heatmaps.

If support allows, do the same for representative epistemic slices, for example low/intermediate/high `phi`.

### E. Semantic selectivity

A major Study 08 question is whether a prompt can remain responsive to useful/truthful control while suppressing unsupported/wrong control.

Prepare matched truth-vs-wrong comparisons for each prompt class.

At minimum produce diagnostics such as:

`Delta_chi_semantic = chi_truth - chi_wrong`

and

`Delta_eta_IR_semantic = eta_IR_truth - eta_IR_wrong`

on supported state-local regions.

Prefer:

`Delta_eta_IR_semantic(x,b)`

and, where adequately supported:

`Delta_eta_IR_semantic(x,phi)`

Do not call this a new efficiency unless we explicitly decide later to define one.

For now it is simply a diagnostic difference between matched truth and wrong-control efficiencies.

The interpretation we care about is:

large useful/truth response + small wrong/adversarial response

but do not hard-code this as a new estimator definition.

### F. Epistemic evolution

The prompt classes may alter the knowledge dynamics themselves.

Therefore retain and plot:

- `phi(t)`
- `kappa(t)`
- `truth_vote_share(t)`
- `controller_target_share(t)`

by prompt class and target semantics.

Also retain existing fact-exposure and fact-acquisition observables.

This allows us to distinguish:

prompt changes epistemic accumulation

from

prompt changes response even at matched epistemic state.

This distinction is scientifically important.

### G. Existing memory-conditioned analysis

Do not remove the established memory/epistemic-conditioned CMI diagnostics.

Retain:

- `round_memory_target_actuation_cmi`
- `round_epistemic_target_actuation_cmi`
- `round_phi_target_actuation_cmi`
- `round_kappa_target_actuation_cmi`

and their corresponding signed-response quantities.

These are complementary to the phase diagrams above.

Do not implement new CMI estimators if the existing machinery already supports the required conditioning.

### H. Theory comparison

Keep the existing matched q-voter/reference-theory outputs where scientifically applicable.

The theory has no epistemic prompt class.

That is useful.

For each prompt class, compare the empirical `x`-dependent quantities against the same matched classical reference at the same:

- `N`
- `q`
- `q_c`
- `b`
- `theta`
- `beta`

where applicable.

The purpose is to show whether epistemic framing changes the empirical deviation from the common coarse-grained theory.

Do not modify the theory to contain a prompt-class parameter.

The theoretical reference should remain prompt-blind.

Conceptually:

same coarse control theory, different LLM epistemic behavior.

## Analysis philosophy

Reuse all existing estimators and support machinery.

Do not reimplement:

- MI
- CMI
- bootstrap
- randomization nulls
- `eta_IR`
- signed response
- `h_eff`
- `gamma_eff`
- currents
- memory-conditioned estimators

Only add grouping and plotting logic necessary to compare prompt classes and semantic target conditions.

Use the new standardized compact analysis package once that implementation is available:

- canonical Parquet scientific data
- compact estimator summaries
- support diagnostics
- plots
- reports
- validation/provenance
- no persistent caches
- no raw null/bootstrap draws

## Study organization

Create:

`configs/runs/relational_reasoning/population_study_08/`

following the same actual organization used by Study 06/07.

Use as few configs as is natural under the current study system.

Conceptually the study contains two semantic blocks:

- Study 08A: wrong/adversarial controller
- Study 08B: truth-aligned controller

Each block sweeps:

`epistemic_prompt_class × b`

If the existing config system can cleanly encode both target semantics in one study-level configuration without making provenance confusing, that is fine.

The important point is that both belong to one Study 08 and are aggregated together in a way that makes matched comparisons straightforward.

## Preflight

Before provider launch, run the normal study-level preflight and report:

- number of configs
- number of scientific cells
- episodes per cell
- total episodes
- expected provider calls
- expected concurrency
- estimated cost if available
- estimated runtime relative to Study 06/07

Explicitly show the arithmetic:

4 prompt classes × 6 budgets × 2 target semantics × number_of_tasks × repetitions

so that the total workload is auditable.

If the workload is unexpectedly much larger than the corresponding Study 06/07 runs, stop and report why before launching.

Do not silently increase repetitions, tasks, or grid size.

## Validation before launch

Before real-model launch, verify with tests or smoke execution that:

1. all four prompt classes resolve;
2. `strategic_uncertainty` reproduces the existing Study 06/07 prompt behavior;
3. the same task/repetition across prompt classes uses matched initial conditions and seeds where expected;
4. prompt class is recorded as a canonical scientific coordinate;
5. truth/wrong target semantics resolve correctly;
6. results-only retention contains `x`, `phi`, `kappa`, target/truth state, and all existing fields required for post-processing;
7. the standardized aggregator can group by prompt class and target semantics;
8. the requested `x × b` and `x × phi` candidate diagrams can be generated from a mock or small fixture;
9. support masking works;
10. no new study-specific SLURM or job architecture is introduced.

## Launch

Once:

- the new epistemic prompt-class implementation passes its tests;
- Study 08 configs pass preflight;
- the four-class mock smoke comparison succeeds;
- expected cost/runtime is reasonable;

launch Study 08 using the same study-level launch mechanism used for Study 06/07.

Do not create a new execution protocol.

## Deliverables

At the end of preparation and launch, report:

1. Study 08 folder/configs created;
2. Study 06/07 config(s) used as templates;
3. exact four prompt classes;
4. exact `b` grid;
5. exact truth/wrong semantic-target configuration;
6. tasks;
7. repetitions;
8. total cells;
9. total episodes;
10. expected calls/cost/runtime;
11. confirmation of paired/common-random-number structure;
12. confirmation that all other scientific settings are unchanged;
13. analysis/plot families configured;
14. launch command;
15. submitted job ID(s) and launch status.

## Final intended design

The intended scientific design is:

- prompt class in `{naive, distributed_information, strategic_uncertainty, evidence_calibrated}`
- `b in {4,8,12,16,20,24}`
- target semantics in `{wrong/adversarial, truth}`
- `q = 1`
- `q_c = 12`
- `theta = 0.5`
- `beta = 4`

with everything else matched to the established Study 06/07 relational setup.

The key output of Study 08 should not merely be an accuracy table.

It should give us a clean family of control-efficiency landscapes:

`eta_IR(x,b | prompt_class, target_semantics)`

and, most importantly,

`eta_IR(x,phi | prompt_class, target_semantics)`

with the corresponding `T_pi` and signed response maps.

The scientific question is whether changing how agents reason about distributed evidence changes the effective controllability of the population, and whether that creates selective responsiveness to truth-aligned versus incorrect control.
