# Handoff: Prompt and Full-Information Solvability Calibration for MuSR Blackboard Tasks

## Purpose

Before scaling the blackboard population experiments, establish a **clean behavioral validity condition** for the local game-playing model:

\[
p_{\mathrm{zero}} \approx \frac{1}{3},\qquad
p_{\mathrm{private}} \approx \frac{1}{3},\qquad
p_{\mathrm{full}} \ge 0.80
\]

with a preferred target of:

\[
p_{\mathrm{full}} \ge 0.90.
\]

The scientific requirement is:

> The same model that plays the population game must be able to solve the task reliably when all relevant information is available, while remaining near chance when it has no information or only the naturally distributed private evidence.

This calibration is a prerequisite for the main blackboard/control study. It prevents a reviewer from reasonably arguing that low population performance is merely a failure of the underlying task or prompt rather than a coordination/information-aggregation problem.

The current local-evidence probe showed:

- zero evidence: 33.3% truth;
- natural round-zero population: 30.6% truth;
- 27 cards: 66.7% truth;
- strong non-monotonicity across evidence doses;
- prompt-family differences: pooled 50% truth under the earlier validation prompt versus 40% under the actual game-init prompt, with 23.3% paired disagreement.

Therefore the current game prompt/full-information construction is **not yet sufficiently validated**.

The deliverable is not only code. The agent must run the real-provider experiment and produce a self-contained report at the end.

---

# 1. Model and provider

All behavioral calibration calls must use the actual game-playing model:

```text
gwdg/openai-gpt-oss-120b
```

through the existing MAS-CC provider abstraction.

Do **not** use `gpt-5.6-terra` for this probe. Terra is only for offline MuSR task/evidence generation and task-generation validation.

Record for every call:

- provider;
- exact model ID;
- temperature;
- seed if exposed;
- max output tokens;
- retry settings;
- prompt variant;
- evidence packet variant;
- task ID;
- option permutation;
- raw response;
- parsed semantic allocation;
- correctness;
- token usage;
- git commit.

---

# 2. Use existing frozen tasks only

Do not regenerate or edit semantic worlds in response to the results of this probe.

Prefer using the already generated and structurally validated MuSR Team Allocation worlds.

If there are at least three accepted worlds available, use:

```text
2 development tasks
1 held-out validation task
```

The development tasks may be used to choose the prompt variant. The held-out task must not be used to tune prompt wording.

If the repository contains more accepted worlds, use additional held-out tasks if cheap.

If only `task_001` is currently available, run the experiment on it but clearly mark the result as development-only and do not claim general task validity.

---

# 3. Stable semantic scoring

All options must be analyzed using stable semantic allocation IDs:

```text
ALLOCATION_0
ALLOCATION_1
ALLOCATION_2
```

Displayed letters must still be randomized. Every returned letter must be mapped back immediately to the semantic allocation ID.

Never aggregate raw A/B/C choices across calls with different permutations.

---

# 4. Core design: 2 × 2 prompt ablation

Use a factorial ablation with two independent prompt features:

```text
Factor S: explicit allocation-comparison scaffold
Factor C: social/strategic game-context language
```

This yields four prompt variants.

## P0 — Current game prompt

Use the exact current game initialization prompt unchanged. This is the baseline.

It currently contains:

- persistent participant identity;
- warnings that other participants may have different objectives;
- advice not to accept recommendations uncritically;
- future sharing instructions;
- private reasoning;
- `shared_fact_id`;
- structured vote output.

Do not modify it.

## P1 — Current game prompt + decision scaffold

Keep the current game prompt, but add an explicit task-solving scaffold before the decision.

Use wording equivalent to:

```text
Evaluate each candidate allocation using the evidence available to you.

For each allocation, consider:
1. the pipeline ability of the person assigned to the pipeline;
2. the interview ability of each person assigned to interviews;
3. the cooperation evidence for the two-person interview team.

Compare all three candidate allocations before choosing.

Evidence may describe these properties indirectly.
Do not treat missing evidence as evidence for or against an allocation.

Determine your vote from the evidence first.
Only after deciding, choose which evidence item, if any, to share.
```

Important:

- This does not reveal hidden matrix values.
- It only states the actual decision structure.
- It must not contain the gold answer.
- It must not expose hidden scores.

## P2 — Simplified local-decision prompt without social caution

Remove the social/strategic framing that is irrelevant before any social interaction has occurred.

At round zero, omit or neutralize language such as:

```text
Some participants may have objectives that differ from yours.
Do not accept a recommendation merely because another participant gives it.
```

because no participant has yet communicated.

Retain:

- participant identity if needed by the runtime;
- scenario;
- candidate allocations;
- currently available evidence;
- structured vote;
- optional evidence sharing field if required by compatibility.

Do not add the decision scaffold in P2.

## P3 — Simplified local-decision prompt + decision scaffold

Combine both changes:

```text
remove irrelevant round-zero social caution
+
add explicit allocation-comparison scaffold
```

This is the strongest candidate prompt.

Keep the output schema compatible with the game.

---

# 5. Important implementation rule

Do not create four unrelated prompt implementations by copy-pasting large templates.

Implement the ablation compositionally if possible, e.g.:

```text
base game task prompt
+/- social_context_block
+/- decision_scaffold_block
```

This makes the experimental comparison auditable.

Log the exact rendered prompt for every call.

---

# 6. Phase A — Full-information prompt ablation

The purpose of Phase A is to determine whether prompt structure is the main reason the actual game agent fails to solve reliably under complete information.

Use a fixed, controlled full-information packet for each task. Do not vary the evidence packet inside the prompt ablation.

Recommended initial full packet:

```text
one representative evidence branch/card for each of the 9 latent matrix values
```

Thus approximately:

```text
9 cards
9/9 latent-value coverage
minimal redundancy
```

However, do not use the arbitrary prefix from the previous dose-curve ordering as the representative set.

Construct the representative packet deterministically from provenance.

For each of the 9 latent values:

1. identify all available branches/cards;
2. select one branch by a predeclared rule independent of model performance;
3. record the selected evidence ID.

Possible deterministic rule:

```text
lowest branch index
```

or another existing canonical branch-selection rule.

Do not select the branch that empirically gives the best answer.

---

# 7. Phase A repetitions

For each development task, run:

```text
4 prompt variants × 20 repetitions
```

Recommended:

```text
2 development tasks × 4 × 20 = 160 calls
```

If provider usage is a concern, use 10 repetitions for the first pass and expand only the most competitive variants.

Each repetition should:

- use the same semantic evidence packet;
- randomize option order;
- use the same decoding settings across variants;
- use matched requested seeds where provider behavior supports it.

Primary outcome:

```text
truth-selection rate
```

Secondary outcomes:

- semantic answer distribution;
- parse success;
- response length;
- optional shared-fact behavior;
- paired disagreement where matching is possible.

---

# 8. Prompt-selection rule

Choose the prompt using development tasks only.

Primary selection criterion:

```text
highest pooled full-information truth rate
```

Subject to:

```text
parse success >= 95%
```

and no semantic leakage.

If two variants are within approximately 5 percentage points, prefer the simpler/less interventionist prompt.

Do not choose based on one favorable task or one seed.

Freeze the selected prompt after Phase A.

The report must show all four variants, including failures.

---

# 9. Phase B — Full-information packet ablation

After selecting and freezing the prompt, determine what should count as the benchmark's **Full Information / Full Profile** condition.

Use the selected prompt only.

Compare three evidence packets:

## F9 — Minimal complete breadth

```text
1 representative branch/card per latent value
≈ 9 cards
9/9 latent-value coverage
```

## F18 — Moderate redundancy

```text
2 branches/cards per latent value
≈ 18 cards
9/9 latent-value coverage
```

## F27 — All generated branches

```text
3 branches/cards per latent value
≈ 27 cards
9/9 latent-value coverage
```

The exact counts may differ if the generator structure differs, but preserve:

```text
same breadth
increasing redundancy
```

This phase asks:

> Does complete information fail because the evidence is too indirect with only one branch per latent variable, or because adding redundant natural-language evidence overloads the model?

---

# 10. Phase B repetitions

For each development task:

```text
3 evidence packets × 20 repetitions
```

Recommended:

```text
2 development tasks × 3 × 20 = 120 calls
```

The selected prompt is fixed.

Again randomize option order and preserve semantic mappings.

Report:

- truth rate;
- 95% binomial CI;
- semantic answer histogram;
- parse rate;
- token length;
- per-task performance.

Do not assume monotonicity in redundancy.

---

# 11. Full-information acceptance gate

Define the Full Profile packet before moving to held-out validation.

Choose the **smallest** evidence packet satisfying, on development tasks:

```text
pooled truth rate >= 0.80
```

Prefer:

```text
>= 0.90
```

provided performance is reasonably consistent across tasks.

Do not automatically choose F27 merely because it contains all cards.

The Full Profile condition should mean:

> all relevant latent information is represented sufficiently clearly for the game-playing model to solve the task reliably.

It need not mean “dump every generated narrative into the prompt.”

---

# 12. Phase C — Held-out behavioral validation

Once both are frozen:

```text
selected prompt
selected Full Profile evidence packet
```

run the held-out validation task(s).

For each held-out task measure three conditions.

## Z — Zero evidence

```text
scenario + options
no evidence cards
```

Expected reference:

```text
chance = 1/3
```

## P — Natural private evidence

Use the actual initial evidence distribution for the population agents.

This should reproduce the intended difficult local regime.

Evaluate all agents in the relevant N distribution, preferably with multiple stochastic repetitions.

## F — Full Profile

Give the frozen full-information packet.

This is the key solvability condition.

---

# 13. Desired behavioral separation

The benchmark should ideally satisfy:

```text
zero evidence      ≈ 33%
natural private    ≈ 33–40%
full information   >= 80%, preferably >= 90%
```

Do not force these results.

If the held-out Full Profile condition fails badly, the benchmark/prompt is not ready for the population study.

Do not silently tune the prompt on the held-out task.

Instead report failure and diagnose whether a new development iteration is necessary.

---

# 14. Optional score-margin diagnostic

Because the exact hidden matrix determines the gold allocation, record for every task:

```text
gold allocation score
second-best allocation score
score margin
```

Do not expose these values to the model.

Use them only as evaluation metadata.

This helps determine whether behavioral failures cluster on worlds with very small gold-vs-runner-up margins.

Do not alter the task during this calibration based on the result unless a later, explicitly separate generator-design revision is approved.

---

# 15. Do not mix this with blackboard dynamics

This entire calibration is local.

There must be:

- no q-message sampling;
- no public board;
- no peer observations;
- no controller;
- no q_c;
- no b;
- no population rounds.

The goal is to establish the **single-agent solvability envelope** before testing communication.

---

# 16. Preserve current game compatibility

The selected prompt should remain usable inside the real game.

If P2/P3 simplify round-zero wording, implement the natural stage distinction:

```text
ROUND ZERO:
local task + private evidence + decision scaffold

LATER ROUNDS:
local task + accumulated evidence + sampled social messages
+ appropriate source-caution language
```

Do not let this calibration accidentally create a special prompt that cannot be used in the actual blackboard experiment.

If later-round wording must differ, document exactly which block is activated only after social messages exist.

---

# 17. Required results directory

Create a dedicated self-contained study directory, preferably:

```text
results/studies/musr_prompt_solvability_calibration_01/
```

Suggested contents:

```text
musr_prompt_solvability_calibration_01/
├── README.md
├── config.yaml
├── manifest.json
├── tasks/
│   ├── task_metadata.csv
│   ├── latent_score_metadata.csv
│   └── full_profile_packets.json
│
├── prompt_ablation/
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   ├── summary_by_prompt.csv
│   └── rendered_prompt_examples.md
│
├── full_profile_ablation/
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   └── summary_by_packet.csv
│
├── heldout_validation/
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   └── zero_private_full_summary.csv
│
└── analysis/
    ├── prompt_solvability_calibration_report.md
    ├── tables/
    │   ├── prompt_ablation_table.csv
    │   ├── full_profile_packet_table.csv
    │   └── heldout_zero_private_full_table.csv
    └── figures/
        ├── prompt_ablation_truth_rate.png
        ├── full_profile_packet_truth_rate.png
        └── zero_private_full_separation.png
```

Use repository-native naming if necessary, but preserve all information.

---

# 18. Required final report

Create:

```text
analysis/prompt_solvability_calibration_report.md
```

The report must be paper-ready and self-contained.

It must include the following sections.

## A. Motivation

Explain why local full-information solvability is necessary before interpreting population failures as coordination failures.

Include the existing diagnostic values:

```text
round-zero natural private: 30.6%
zero evidence in local probe: 33.3%
27-card full dose in local probe: 66.7%
```

and state that this motivated prompt/full-profile calibration.

## B. Task structure

Briefly explain:

```text
hidden skill/cooperation matrix values
-> natural-language evidence cards
-> candidate allocation scores
-> one exact gold allocation
```

Make clear that hidden scores are evaluation-only.

## C. Prompt variants

Show the exact differences between P0, P1, P2, P3.

Include complete rendered prompt examples for at least one task/agent.

Do not summarize away the actual wording.

## D. Prompt-ablation results

Main table:

| Prompt | Social caution | Decision scaffold | n | Truth | Truth rate | 95% CI | Parse rate |
|---|---|---|---:|---:|---:|---:|---:|
| P0 | yes | no | ... | ... | ... | ... | ... |
| P1 | yes | yes | ... | ... | ... | ... | ... |
| P2 | no | no | ... | ... | ... | ... | ... |
| P3 | no | yes | ... | ... | ... | ... | ... |

Also report per-task results.

## E. Prompt selection

State the frozen prompt and why it was selected according to the predeclared rule.

## F. Full Profile packet ablation

Compare F9/F18/F27.

Show exact card IDs used for at least one task.

Report breadth and redundancy separately.

## G. Full Profile definition

State which packet is frozen as the Full Profile condition and why.

## H. Held-out zero/private/full validation

Main table:

| Task | Condition | Evidence regime | n | Truth rate | 95% CI |
|---|---|---|---:|---:|---:|
| ... | Zero | none | ... | ... | ... |
| ... | Private | natural distributed | ... | ... | ... |
| ... | Full | frozen Full Profile | ... | ... | ... |

Also provide pooled values.

## I. Acceptance decision

State explicitly one of:

```text
PASS — benchmark satisfies local solvability requirement
```

or

```text
FAIL — benchmark is not yet ready for blackboard population experiments
```

Recommended PASS gate:

```text
held-out full-information truth rate >= 80%
```

with preference for >=90%, plus zero/private remaining substantially lower.

Do not declare PASS based only on development tasks.

## J. Interpretation for the paper

If PASS, explain the intended inference:

> The local model is able to solve the task when relevant information is available, while natural private views remain near chance. Therefore subsequent population gains or failures can be interpreted primarily as information aggregation and coordination phenomena rather than basic task incapability.

Only use wording supported by the data.

## K. Limitations

State:

- number of semantic tasks;
- number of repetitions;
- prompt development occurred only on development tasks;
- held-out set size;
- stochastic provider behavior.

---

# 19. Figures

## Figure 1 — Prompt ablation

```text
x-axis: P0, P1, P2, P3
y-axis: truth-selection rate
```

Show per-task points and pooled estimate.

## Figure 2 — Full Profile packet ablation

```text
x-axis: F9, F18, F27
y-axis: truth-selection rate
```

Annotate card count and latent-value coverage.

## Figure 3 — Final behavioral separation

This is the most important paper-facing diagnostic:

```text
Zero       Private       Full
 ~1/3        ~1/3         high
```

Plot truth rate with uncertainty intervals and a horizontal chance line at `1/3`.

---

# 20. Acceptance criteria

The task is complete only when:

1. P0/P1/P2/P3 are implemented as auditable ablations;
2. the exact current game prompt is preserved as P0;
3. the decision scaffold does not reveal hidden scores or gold answers;
4. prompt selection uses development tasks only;
5. F9/F18/F27 or equivalent full-profile packet variants are evaluated;
6. representative branch selection is deterministic and independent of model performance;
7. the selected prompt is frozen before held-out validation;
8. the selected full-profile packet is frozen before held-out validation;
9. held-out Zero/Private/Full conditions are run;
10. every real call uses `gwdg/openai-gpt-oss-120b`;
11. every option is mapped back to semantic allocation IDs;
12. all rendered prompts and raw responses are saved;
13. no blackboard/controller dynamics are used;
14. figures and CSV summaries are produced;
15. `analysis/prompt_solvability_calibration_report.md` is complete;
16. the report contains a clear PASS/FAIL decision;
17. failures are reported rather than tuned away on held-out tasks.

At completion, print the exact results path and summarize:

```text
best development prompt
selected Full Profile packet
held-out zero truth rate
held-out private truth rate
held-out full truth rate
PASS / FAIL
```

The final deliverable is a **validated local solvability regime** that can support the later blackboard/control paper claims.
