# Handoff: Systematic Real-Provider Validation of the Native MuSR Team Allocation Generator

## Goal

The native MuSR-style Team Allocation generator is now implemented and mechanically tested. The next step is **not another redesign**.

Run a small but systematic **real-provider validation study** using `gpt-5.6-terra`, so that by the end we have:

1. three accepted, frozen MuSR-style Team Allocation worlds;
2. matched evidence distributions for populations `N = 12` and `N = 24`;
3. full-information, partial-information, and zero-information LLM validation results;
4. a complete results directory containing the generated data, raw validation outputs, summary tables, plots, and a self-contained Markdown report;
5. one clearly worked example of an actual generated task and its distributed evidence;
6. outputs organized well enough that the validation results can already be cited/described in the paper as a **pilot task-validation study**.

Do not modify the population game or implement the blackboard in this task.

---

# 1. Use the real MAS-CC provider

Use the existing MAS-CC provider abstraction and:

```text
model = gpt-5.6-terra
```

No direct MuSR/OpenAI client should be introduced.

All real generation and validation calls must go through the normal MAS-CC provider stack.

Record in metadata:

- provider;
- model;
- generation temperature / decoding parameters;
- validation temperature / decoding parameters;
- seed;
- prompt version;
- MAS-CC git commit;
- pinned MuSR upstream commit;
- generator parameters.

---

# 2. Generate exactly three accepted base worlds

Generate **three independent Team Allocation worlds**:

```text
task_001
task_002
task_003
```

Use the current native MuSR-style generator.

Each world must satisfy all existing exact/structural QA:

- unique exact optimum;
- nonzero margin over second-best allocation;
- nine hidden skill/cooperation facts;
- multiple independent evidence branches per hidden fact;
- no answer leakage;
- valid reasoning-tree provenance;
- complete evidence pool;
- parseable frozen JSON.

Use the generator's current configurable:

```text
tree_depth
branches_per_latent_fact
```

Do **not** invent an abstract `L` variable for this task family.

Store the exact values used in the manifest and report.

If a generated world fails structural QA or repeated full-information validation, regenerate until **three accepted worlds** exist.

---

# 3. Use the same three worlds for N=12 and N=24

Do **not** independently generate different semantic worlds for the two population sizes.

For each accepted base world, create two matched evidence-distribution variants:

```text
N = 12
N = 24
```

The following must remain identical across the two variants:

- scenario;
- people/tasks;
- latent matrices;
- candidate allocations;
- gold answer;
- natural-language evidence pool;
- reasoning-tree provenance.

Only the assignment of evidence cards to agents changes.

This gives a clean matched comparison:

```text
same reasoning problem
same evidence
different population / evidence partition
```

Validate the no-single-agent condition separately for both `N=12` and `N=24`.

---

# 4. Evidence distribution requirements

Distribute coherent **evidence cards/branches**, not isolated individual sentences.

For both population sizes report:

- total evidence cards;
- total explicit evidence snippets;
- mean / min / max cards per agent;
- mean / min / max distinct latent facts touched per agent;
- evidence redundancy;
- fraction of agents touching each latent fact;
- whether any agent structurally certifies the complete solution.

Required:

```text
no single agent structurally solves the task
```

for all six `(task, N)` variants.

At the same time, avoid meaningless views: each agent should receive at least one informative evidence bundle.

---

# 5. Full-information solvability validation

This is the crucial semantic QA.

For each of the three tasks:

1. give a validation LLM the **global scenario + all natural-language evidence + the K=3 candidate allocations**;
2. ask it to solve exactly the same allocation problem the experimental agents will eventually solve;
3. map the response back to the semantic allocation/gold index;
4. store the complete raw response and parsed answer.

Use:

```text
gpt-5.6-terra
```

Run **5 independent validation calls per task**.

Randomize/permutate the displayed order/labels of the three options independently on each call, then map responses back to the semantic allocations. This prevents a position/letter artifact from contaminating the validation.

Acceptance criterion for a task:

```text
at least 4 / 5 full-information calls correct
```

If a task does not satisfy this after the configured retry policy, reject/regenerate it.

Report both:

- per-task full-information accuracy;
- pooled full-information accuracy.

The **exact latent solver remains the source of truth**. This LLM test only demonstrates that the generated language communicates enough information to solve the task.

---

# 6. Zero-information baseline

For each of the same three accepted tasks, run a zero-information condition.

The validation model receives:

- scenario / task framing;
- the three candidate allocations;
- **no private evidence cards**.

Use `gpt-5.6-terra`.

Run:

```text
5 calls per task
```

Again independently permute option order/labels per call.

Expected reference level:

```text
chance = 1 / 3
```

Do not force the observed accuracy to equal chance; measure and report it.

Also report the semantic answer histogram to detect systematic option bias.

---

# 7. Partial-information validation: evaluate every initial agent view

This is the most important comparison with the previous relational task.

For each accepted task and each population size:

```text
N = 12
N = 24
```

evaluate **every agent's initial evidence view once**.

For each agent, show:

- common scenario/task introduction;
- that agent's assigned evidence cards only;
- the same K=3 semantic candidate allocations;
- no peer information;
- no controller;
- no board;
- no other agents' evidence.

Ask `gpt-5.6-terra` to choose the best allocation.

Randomize option order/labels independently for every call and resolve back to the semantic allocation.

This produces:

```text
3 tasks × 12 agents = 36 partial-information observations
3 tasks × 24 agents = 72 partial-information observations
```

for a total of:

```text
108 partial-information calls
```

This is intentionally systematic: every initial agent view is tested, not a convenience sample.

No need to repeat every partial view multiple times in this pilot. The population of agent views itself provides the main descriptive sample.

---

# 8. Primary validation pattern

The main task-validity result we want to examine is:

```text
zero information
      <
partial information
      <
full information
```

More precisely, we hope to see:

```text
zero ≈ chance
partial > chance
partial < full
full ≈ high / near-perfect
```

Do not massage, filter, or regenerate tasks based on their partial-information accuracy.

Only structural QA and the explicit full-information acceptance criterion may reject a task.

The partial and zero results are empirical outcomes and must be reported even if they are not what we hoped for.

This is important for paper credibility.

---

# 9. Metrics to compute

At minimum compute the following.

## Behavioral validation

For:

```text
full
zero
partial_N12
partial_N24
```

report:

- `n`;
- number correct;
- accuracy;
- 95% binomial confidence interval;
- parse success rate;
- semantic answer distribution across the three allocations.

For partial-information conditions additionally report:

- accuracy per task;
- accuracy by `N`;
- mean accuracy across tasks;
- matched per-task difference `partial_N12 - partial_N24`;
- accuracy as a function of number of evidence cards if there is sufficient variation;
- accuracy as a function of number of distinct latent facts touched if there is sufficient variation.

Because there are only **three semantic tasks**, do not claim strong statistical significance from task-level comparisons.

Treat this as a systematic **pilot validation**, and show per-task results explicitly.

## Structural distribution diagnostics

For each `(task, N)`:

- number of agents;
- total evidence cards;
- cards per agent: mean/min/max;
- distinct latent facts touched per agent: mean/min/max;
- evidence redundancy;
- number of agents violating no-single-agent constraint (must be zero);
- fraction of global evidence represented in each agent view;
- gold-vs-second-best exact score margin.

---

# 10. Save all raw model outputs

Every provider call must be recoverable.

For each call save at least:

```json
{
  "task_id": "...",
  "condition": "full|zero|partial",
  "population_size": 12,
  "agent_id": "...",
  "call_index": 0,
  "displayed_options": [...],
  "semantic_option_mapping": {...},
  "gold_index": ...,
  "raw_response": "...",
  "parsed_semantic_answer": "...",
  "correct": true,
  "parse_success": true
}
```

Also save prompt/model metadata sufficient to reproduce the call.

Do not put hidden latent matrices into the prompt sent to the validation model.

---

# 11. Results-directory contract

At completion there must be one self-contained directory, preferably:

```text
results/studies/musr_team_allocation_validation_01/
```

Use the repository's established result conventions if there is already a standard equivalent.

It should contain approximately:

```text
musr_team_allocation_validation_01/
├── README.md
├── config.yaml
├── manifest.json
│
├── tasks/
│   ├── task_001/
│   │   ├── base_task.json
│   │   ├── distribution_N12.json
│   │   └── distribution_N24.json
│   ├── task_002/
│   │   ├── base_task.json
│   │   ├── distribution_N12.json
│   │   └── distribution_N24.json
│   └── task_003/
│       ├── base_task.json
│       ├── distribution_N12.json
│       └── distribution_N24.json
│
├── raw/
│   ├── generation_calls.jsonl
│   ├── full_information.jsonl
│   ├── zero_information.jsonl
│   ├── partial_N12.jsonl
│   └── partial_N24.jsonl
│
└── analysis/
    ├── validation_report.md
    ├── behavioral_summary.csv
    ├── per_task_summary.csv
    ├── distribution_summary.csv
    ├── agent_level_results.csv
    └── figures/
        ├── accuracy_by_information_condition.png
        └── partial_accuracy_by_population.png
```

Exact filenames may follow existing MAS-CC conventions, but the same information must be present.

The **results folder is the final artifact of this task**.

---

# 12. `validation_report.md` must be paper-ready

Create:

```text
analysis/validation_report.md
```

It must be readable without inspecting code.

Include the following sections.

## A. Study design

Briefly state:

- 3 independently generated semantic worlds;
- exact Team Allocation ground truth;
- MuSR-style LLM-generated evidence;
- `gpt-5.6-terra` generation/validation;
- matched `N=12` and `N=24` evidence distributions;
- K=3 choices;
- full / partial / zero information conditions;
- option-order randomization.

## B. Generator configuration

Table with:

```text
number of tasks
tree_depth
branches_per_latent_fact
number of latent facts
provider/model
generation seed(s)
population sizes
```

## C. Task-generation / structural QA table

One row per task, with columns such as:

```text
task
gold allocation
exact margin
# evidence cards
# explicit snippets
full-info validation
leakage check
N12 no-single-agent
N24 no-single-agent
```

## D. Behavioral validation table

Something like:

| Condition | N | Observations | Correct | Accuracy | 95% CI | Parse rate |
|---|---:|---:|---:|---:|---:|---:|
| Zero evidence | — | ... | ... | ... | ... | ... |
| Partial evidence | 12 | 36 | ... | ... | ... | ... |
| Partial evidence | 24 | 72 | ... | ... | ... | ... |
| Full evidence | — | 15 | ... | ... | ... | ... |

Also provide the same results **per task**.

## E. Distribution diagnostics

Table comparing `N=12` and `N=24`:

```text
cards/agent
latent facts touched/agent
redundancy
no-single-agent violations
```

## F. Interpretation

State plainly whether the desired ordering:

```text
zero < partial < full
```

is observed.

Do not hide failures.

If partial evidence remains close to chance, flag that explicitly as a task-design issue.

If full evidence is not reliably solved, flag that as a generator/evidence-quality issue.

## G. Limitations

Explicitly state that this is a **three-world pilot validation**. Agent-level observations give useful diagnostics, but only three independently generated semantic tasks are insufficient for broad inferential claims.

---

# 13. Include one complete, human-readable task example

The report must contain a section:

```text
## Worked example: task_XXX
```

Choose one of the three accepted tasks and show clearly:

1. scenario;
2. the three candidate allocations;
3. correct allocation;
4. exact latent skill/cooperation table **clearly marked as hidden evaluation metadata**;
5. exact candidate scores;
6. examples of several generated reasoning branches;
7. representative evidence assigned to:
   - one `N=12` agent;
   - one `N=24` agent;
8. one full-information validation response, preferably shortened to the relevant reasoning/answer;
9. explanation of why no individual agent has the complete solution.

Do not dump enormous JSON into the Markdown report.

Make the example readable enough that it can be used while drafting the paper to explain what the new task actually looks like.

---

# 14. Figures

Produce two minimal diagnostic figures.

## Figure 1

```text
Accuracy vs information condition
```

Show:

```text
zero
partial N=12
partial N=24
full
```

Include chance level `1/3`.

## Figure 2

```text
Partial-information accuracy by population size
```

Show per-task points if practical so the reader can see the fact that only three semantic worlds were used.

Keep plotting code simple and reproducible.

---

# 15. Do not wire the population game yet unless needed for validation

This stage is about validating the **task substrate**.

Do not yet implement:

- q-message blackboard;
- controller experiments;
- population rounds;
- thermodynamic analysis.

The partial-information validation calls are independent single-agent probes of the frozen initial views.

Once this study passes, the next task is the thin integration into:

```text
games/relational_reasoning/imitation_round_feedback
```

followed by the q-message/blackboard extension.

---

# 16. Final acceptance criteria

This handoff is complete only when all of the following exist:

- three accepted real-provider-generated tasks;
- frozen base tasks;
- matched `N=12` distributions;
- matched `N=24` distributions;
- structural QA for all six distributions;
- 15 full-information validation calls;
- 15 zero-information validation calls;
- 36 `N=12` partial-information calls;
- 72 `N=24` partial-information calls;
- raw outputs for every real LLM call;
- CSV summary tables;
- diagnostic figures;
- `analysis/validation_report.md`;
- one clear worked example;
- a complete manifest/config with model, prompts, seeds, commits, and hashes.

At the end, print the exact path to:

```text
results/studies/musr_team_allocation_validation_01/
```

and give a concise status summary including the observed:

```text
zero accuracy
partial N=12 accuracy
partial N=24 accuracy
full accuracy
```

Do not report success merely because the code ran. The scientific validation results themselves are the deliverable.
