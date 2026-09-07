# Handoff: Private-Evidence Redistribution Calibration for MuSR Blackboard Tasks

## Purpose

Before running the full blackboard population/control study, test whether the current MuSR Team Allocation tasks can be made into a clean distributed-information benchmark **without regenerating the semantic task or evidence cards**.

The current calibration already established:

- selected local prompt: `P2`;
- Full Profile packet: `F9`;
- held-out Full Profile truth rate: `100%`;
- held-out natural-private truth rate: `78.3%`;
- held-out zero-evidence truth rate: `60.0%`.

The current benchmark therefore fails because the **lower-information regimes are too easy**, not because the full-information task is unsolvable.

The primary hypothesis is:

> The current private assignments expose each agent to too many distinct hidden skill/cooperation values. The key quantity is not only raw card count, but how many distinct latent matrix values are represented in an agent's private evidence.

The goal is to determine whether we can fix this by **redistributing the existing evidence cards only**.

Do not regenerate tasks or evidence during the main experiment.

---

## 1. Freeze validated components

Keep fixed:

```text
model = gwdg/openai-gpt-oss-120b
prompt = P2
Full Profile = F9
semantic task worlds = existing validated MuSR Team Allocation tasks
evidence-card texts = unchanged
gold answers = unchanged
```

Do not perform new prompt tuning here.

Do not alter Full Profile based on private-regime performance.

---

## 2. Desired benchmark separation

The intended behavior is approximately:

```text
Zero evidence      ~ 33%
Private evidence   ~ 33–45%
Full Profile       >= 80%, preferably >= 90%
```

The scientific interpretation we want is:

> Individual agents are weak because their information is incomplete, while the same model solves reliably when the relevant information is integrated.

---

## 3. Terminology

### Evidence card

A natural-language observation generated from one hidden latent value.

Example:

```text
e_skill_p0_t0_b00
```

### Latent value

One hidden skill/cooperation entry of the exact task world:

```text
6 skill values
+
3 cooperation values
=
9 latent values
```

### Private evidence

The evidence cards assigned to one agent at round zero.

### Latent coverage

The number of distinct latent values represented by an agent's cards.

For example:

```text
6 cards -> 6 latent values
```

is much more globally informative than:

```text
6 cards -> 3 latent values with 2 redundant branches each
```

even though the raw card count is identical.

---

# 4. Phase A — Audit the current natural private assignments

Before constructing alternatives, audit the existing private distributions.

For every task and agent record:

- number of cards;
- exact evidence IDs;
- number of distinct latent values covered;
- exact latent IDs covered;
- fraction of the 9 latent values covered;
- branch redundancy per latent value;
- whether all terms needed to evaluate `ALLOCATION_0` are represented;
- whether all terms needed to evaluate `ALLOCATION_1` are represented;
- whether all terms needed to evaluate `ALLOCATION_2` are represented;
- whether the agent has enough complementary evidence to compare all three allocations structurally.

Create:

```text
structural_audit/current_private_coverage.csv
```

Suggested columns:

```text
task_id
agent_id
num_cards
num_latent_values
latent_fraction
latent_ids
redundancy_profile
covers_all_terms_allocation_0
covers_all_terms_allocation_1
covers_all_terms_allocation_2
num_fully_scoreable_allocations
```

The report must explicitly answer:

> Is the original 78.3% private accuracy plausibly explained by excessive latent-value breadth per agent?

---

# 5. Phase B — Construct controlled redistribution regimes

Using only existing evidence cards, construct three alternative initial distributions.

Only change:

```text
which agent initially receives which evidence cards
```

Do not change:

```text
task
gold
matrix
evidence text
prompt
model
```

## R2 — narrow local coverage

Target approximately:

```text
2 distinct latent values per agent
```

## R3 — moderate local coverage

Target approximately:

```text
3 distinct latent values per agent
```

## R4 — broader local coverage

Target approximately:

```text
4 distinct latent values per agent
```

The exact card count per agent may differ.

The controlled quantity is primarily **distinct latent-value coverage**, not card count.

Prefer giving redundant branches for already assigned latent values rather than increasing breadth beyond the regime target.

---

# 6. Structural constraints for every redistribution

Every R2/R3/R4 assignment must satisfy:

### A. Collective completeness

Across the whole population:

```text
all 9 latent values are represented
```

### B. Redundancy

Prefer:

```text
each latent value held by at least 2 agents
```

or the closest balanced feasible assignment.

Record holder counts.

### C. No single-agent completeness

No agent may cover all 9 latent values.

### D. Avoid accidental global comparison

Where feasible, avoid giving one agent all latent terms needed to score all three candidate allocations.

Flag any agent that can fully score:

```text
0
1
2
or 3 allocations
```

### E. Balance

Report for both card count and latent coverage:

```text
mean
std
min
max
```

across agents.

---

# 7. Deterministic redistribution algorithm

Implement the assignment deterministically from a recorded seed.

Suggested procedure:

1. group all evidence cards by latent value;
2. assign target latent values to agents;
3. balance holder counts across the population;
4. assign branch/card instances inside each selected latent value;
5. add redundancy within assigned latent values before increasing breadth;
6. validate structural constraints;
7. freeze the assignment;
8. only then run behavioral evaluation.

Do not alter assignments based on LLM accuracy.

The benchmark construction must remain independent of behavioral outcomes.

---

# 8. Behavioral conditions

For every tested semantic task evaluate:

```text
Z   = Zero evidence
NAT = original natural private distribution
R2  = ~2 latent values/agent
R3  = ~3 latent values/agent
R4  = ~4 latent values/agent
F9  = frozen Full Profile
```

Use frozen prompt `P2` in all conditions.

Displayed answer letters must be randomized and mapped immediately back to:

```text
ALLOCATION_0
ALLOCATION_1
ALLOCATION_2
```

All analysis must use semantic allocation IDs.

---

# 9. Use multiple semantic worlds

Do not rely on only one held-out task.

Use as many already validated frozen tasks as are available.

Recommended minimum:

```text
3 semantic tasks
```

Prefer more if already generated.

Report the gold semantic-allocation frequencies across tasks.

This is important because zero-evidence accuracy may reflect task-specific semantic priors.

---

# 10. Repetitions

For private regimes:

```text
all N agents x at least 3 stochastic repetitions
```

per task/regime.

For Zero and F9:

```text
at least 10 repetitions per task
```

Prefer 20 if provider budget allows.

Use the same temperature and decoding configuration as the real game.

Save requested provider seeds if supported.

---

# 11. Primary outcome

For every condition compute:

```text
truth-selection rate
```

with 95% binomial confidence intervals.

Main comparison:

```text
Zero | Natural | R2 | R3 | R4 | Full
```

Do not assume R2/R3/R4 must be monotone, but inspect the pattern.

---

# 12. Observation-level metadata

For every call save:

```text
task_id
agent_id
regime
num_cards
num_latent_values
latent_ids
evidence_ids
semantic_option_mapping
raw_response
parsed_semantic_answer
correct
parse_success
prompt length / token usage
provider metadata
```

This must allow analysis of:

```text
truth vs latent-value coverage
truth vs raw card count
truth vs candidate-score coverage
```

---

# 13. Candidate-score coverage diagnostic

For every private view compute evaluation-only indicators:

```text
fraction of ALLOCATION_0 score terms represented
fraction of ALLOCATION_1 score terms represented
fraction of ALLOCATION_2 score terms represented
```

Also compute:

```text
number of candidate allocations fully scoreable from the represented latent values
```

Never expose hidden matrix values or score structure to the provider.

This is analysis metadata only.

The report should assess whether performance depends more on:

```text
raw card count
latent breadth
or complementary score-term coverage
```

---

# 14. Zero-evidence diagnostic

The previous held-out zero result was:

```text
6/10 = 60%
```

on one semantic world, so do not treat it as established.

For every task:

1. run zero-evidence calls;
2. randomize displayed option letters;
3. record semantic allocation choices;
4. compute semantic choice histograms;
5. inspect whether the model has a stable semantic allocation preference.

Report pooled zero-evidence accuracy across worlds.

State whether the earlier 60% appears to be:

```text
sampling noise
task-specific semantic prior
or persistent cross-task bias
```

---

# 15. Selection rule for the private regime

The goal is not to choose the regime with the lowest possible accuracy.

Prefer the **largest local-information regime that still remains sufficiently difficult**.

Development selection rule:

```text
if R4 private truth <= 50%:
    select R4
elif R3 private truth <= 50%:
    select R3
elif R2 private truth <= 50%:
    select R2
else:
    redistribution FAILS
```

Preferred performance band:

```text
33%–45%
```

Also require:

```text
collective 9/9 latent coverage
no individual 9/9 coverage
Full Profile >= 80%
```

with Full Profile >=90% preferred.

If separate held-out worlds are available, freeze the selected regime before held-out validation.

---

# 16. Held-out validation

If enough frozen tasks exist, split them into:

```text
development tasks
held-out validation tasks
```

Use development tasks to select R2/R3/R4.

Then freeze the selected redistribution algorithm and evaluate on held-out tasks.

Do not tune the assignment rule on held-out behavioral results.

Final held-out comparison:

```text
Zero
Selected Private
Full F9
```

Desired separation:

```text
~1/3 | ~1/3–0.45 | >=0.8
```

---

# 17. Failure criteria

Declare redistribution alone insufficient if:

- R2 remains too accurate;
- zero-evidence semantic priors remain strongly above chance across worlds;
- very small latent subsets reliably reveal the winner;
- private and Full Profile performance remain poorly separated;
- task structure itself makes incomplete views too predictive.

If redistribution fails, do not silently regenerate data in this study.

Instead recommend a separate generator revision.

Possible future interventions, not part of this handoff:

```text
more indirect evidence
larger hidden matrices
different score margins
balanced semantic worlds
distractor evidence
new generated tasks
```

---

# 18. Required results directory

Create:

```text
results/studies/musr_private_redistribution_calibration_01/
```

or the repository-native equivalent.

Suggested contents:

```text
musr_private_redistribution_calibration_01/
├── README.md
├── config.yaml
├── manifest.json
│
├── assignments/
│   ├── natural_assignments.json
│   ├── R2_assignments.json
│   ├── R3_assignments.json
│   ├── R4_assignments.json
│   └── assignment_summary.csv
│
├── structural_audit/
│   ├── current_private_coverage.csv
│   ├── redistributed_private_coverage.csv
│   ├── latent_holder_counts.csv
│   └── candidate_score_coverage.csv
│
├── behavioral/
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   └── summary_by_regime.csv
│
└── analysis/
    ├── private_redistribution_calibration_report.md
    ├── tables/
    │   ├── regime_truth_rates.csv
    │   ├── truth_by_latent_coverage.csv
    │   ├── truth_by_card_count.csv
    │   ├── candidate_score_coverage_results.csv
    │   └── zero_semantic_preferences.csv
    └── figures/
        ├── regime_truth_rate.png
        ├── truth_vs_latent_coverage.png
        ├── truth_vs_card_count.png
        └── latent_coverage_distribution.png
```

---

# 19. Required final report

Create:

```text
analysis/private_redistribution_calibration_report.md
```

The report must be self-contained and paper-ready.

It must include:

## A. Motivation

State the previous calibration:

```text
P2 selected
F9 solvable
Natural private too easy
```

and report:

```text
Zero = 60.0%
Natural private = 78.3%
Full F9 = 100.0%
```

Explain why redistribution is tested before new data generation.

## B. Existing-distribution audit

Show:

- cards per agent;
- latent values per agent;
- redundancy;
- candidate-score coverage.

Answer explicitly:

> Did the original natural private assignment expose too many distinct latent values per agent?

## C. Redistribution algorithm

Describe R2/R3/R4 and the structural constraints.

## D. Structural validation

Include:

| Regime | Mean cards/agent | Mean latent values/agent | Global 9/9? | Min holders/value | Agents fully scoring all candidates |
|---|---:|---:|---|---:|---:|
| NAT | ... | ... | ... | ... | ... |
| R2 | ... | ... | ... | ... | ... |
| R3 | ... | ... | ... | ... | ... |
| R4 | ... | ... | ... | ... | ... |

## E. Behavioral results

Include:

| Regime | n | Truth | Truth rate | 95% CI |
|---|---:|---:|---:|---:|
| Zero | ... | ... | ... | ... |
| Natural | ... | ... | ... | ... |
| R2 | ... | ... | ... | ... |
| R3 | ... | ... | ... | ... |
| R4 | ... | ... | ... | ... |
| F9 | ... | ... | ... | ... |

Also report per-task results.

## F. Coverage-response analysis

Analyze:

```text
truth vs latent-value coverage
truth vs raw card count
truth vs candidate-score coverage
```

State which appears most predictive.

## G. Zero-evidence semantic prior

Show semantic answer histograms under zero evidence.

State whether the previous 60% zero result appears to be noise, task-specific prior, or persistent bias.

## H. Recommended private regime

If a regime passes, state:

- selected regime;
- target latent coverage per agent;
- deterministic assignment algorithm;
- structural constraints to freeze for future blackboard studies.

## I. PASS / FAIL

Conclude explicitly with one of:

```text
PASS — redistribution is sufficient
```

or:

```text
FAIL — redistribution alone does not create a clean distributed-information benchmark
```

## J. Consequence

If PASS:

```text
Freeze the selected distribution and proceed to blackboard:
no control / direct recommendation / coordination request.
```

If FAIL:

```text
Do not scale the blackboard study.
A separate task/evidence-generation revision is required.
```

## K. Limitations

Report:

- number of semantic worlds;
- number of agents;
- repetitions;
- provider stochasticity;
- gold-allocation balance.

---

# 20. Main figures

## Figure 1 — Regime separation

```text
x-axis:
Zero | Natural | R2 | R3 | R4 | Full

y-axis:
truth-selection rate
```

Include 95% CIs and a horizontal chance line at:

```text
1/3
```

## Figure 2 — Latent-coverage response

```text
x-axis:
number of distinct latent values available to the agent

y-axis:
truth-selection rate
```

This is the direct test of the redistribution hypothesis.

## Figure 3 — Card-count response

```text
x-axis:
number of evidence cards

y-axis:
truth-selection rate
```

Compare this with Figure 2 to show whether latent breadth is more informative than raw card count.

---

# 21. Acceptance criteria

The study is complete only when:

1. the natural private distribution is structurally audited;
2. latent-value coverage is computed for every agent;
3. R2/R3/R4 are constructed from existing cards only;
4. assignments are deterministic and saved;
5. collective 9/9 latent coverage is verified;
6. holder redundancy is verified;
7. candidate-score coverage is audited;
8. P2 remains frozen;
9. F9 remains frozen;
10. all behavioral calls use `gwdg/openai-gpt-oss-120b`;
11. multiple semantic tasks are used where available;
12. Zero/Natural/R2/R3/R4/F9 are behaviorally compared;
13. all raw prompts and outputs are saved;
14. summary CSVs and figures are produced;
15. `analysis/private_redistribution_calibration_report.md` is complete;
16. the report gives a clear PASS/FAIL decision;
17. no blackboard/controller dynamics are introduced;
18. no task/evidence regeneration occurs inside this study.

At completion print:

```text
original natural mean latent coverage
R2 mean latent coverage
R3 mean latent coverage
R4 mean latent coverage
truth rate for every regime
selected private regime
PASS / FAIL
results directory
report path
```

The scientific deliverable is a **validated private-information regime** in which individual agents are locally weak while the population collectively possesses the complete information needed to solve the task.
