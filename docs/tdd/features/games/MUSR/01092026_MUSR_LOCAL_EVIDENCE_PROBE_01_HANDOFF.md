# Handoff: Self-Contained Local Evidence-Dose and Prompt-Equivalence Probe

## Purpose

Run a systematic, paper-ready local validation study of how `gwdg/openai-gpt-oss-120b` behaves on the MuSR Team Allocation task **before any social interaction occurs**.

This probe should establish the local response curve underlying the later blackboard experiment:

\[
	ext{available evidence} \longrightarrow 	ext{single-agent decision quality}.
\]

The probe must be completely self-contained in its results directory, including the frozen task, exact evidence subsets, exact rendered prompts, raw provider outputs, parsed semantic answers, tables, figures, and a Markdown report suitable for paper drafting.

The primary scientific question is:

> Does increasing locally available evidence move an isolated game-playing agent from near-chance performance toward reliable truth selection?

The secondary diagnostic question is:

> Does the actual game initialization prompt produce systematically different decisions from the earlier validation prompt when the semantic task and evidence are held fixed?

This is **not** a population study: no board, no social messages, no controller, and no population rounds.

---

## 1. Baseline result to preserve

The existing round-zero calibration on validated `task_001`, `N=12`, found:

- model: `gwdg/openai-gpt-oss-120b`;
- 36 independent initialization decisions;
- 11/36 truth votes = **30.6%**;
- three-option chance = **33.3%**;
- no social interactions;
- no population rounds.

Replica counts:

| Replica | ALLOCATION_0 | ALLOCATION_1 | ALLOCATION_2 (truth) | Truth share |
|---|---:|---:|---:|---:|
| 1 | 3 | 5 | 4 | 33.3% |
| 2 | 4 | 5 | 3 | 25.0% |
| 3 | 4 | 4 | 4 | 33.3% |

The new probe should explain this local near-chance behavior mechanistically.

Use the same validated task unless there is a concrete implementation reason not to:

```text
task_001
N=12 distribution
truth = ALLOCATION_2
```

---

## 2. Model/provider

All decision calls in this probe must use:

```text
gwdg/openai-gpt-oss-120b
```

through the existing MAS-CC provider abstraction.

Do **not** use `gpt-5.6-terra` here. Terra is only for offline MuSR task/evidence generation and task validation.

Record:

- provider;
- exact model ID;
- temperature and decoding settings;
- seed if exposed;
- max tokens;
- retry configuration;
- MAS-CC commit;
- task hash;
- prompt-family hash/version.

---

## 3. Two experiments

Run:

```text
A. Prompt-equivalence probe
B. Nested evidence-dose curve
```

Both use the same frozen semantic task.

Always randomize displayed option order/letters independently, then map the returned letter back to stable semantic IDs:

```text
ALLOCATION_0
ALLOCATION_1
ALLOCATION_2
```

All analysis must use semantic allocation IDs, never raw letters.

---

# Experiment A — Prompt-equivalence probe

## 4. Goal

The previous report found a concrete prompt discrepancy:

- Agent 1 chose the truth in the earlier validation prompt, but chose `ALLOCATION_1` in all three actual game initializations.
- Agent 4 chose the truth in both settings.

This is not consistent with a simple global option-letter mapping bug.

The actual game prompt additionally introduces:

- persistent participant identity;
- strategic caution;
- future information sharing;
- private reasoning;
- `shared_fact_id`;
- a more demanding three-field response.

The earlier comparison also used stochastic sampling, so prompt wording and sampling variability were confounded.

The new probe should isolate this as cleanly as practical.

---

## 5. Probe agents

Select at least three fixed agents from the `N=12` distribution.

Include:

```text
Agent 1
Agent 4
```

because their contrasting behavior is already documented.

Choose one additional agent with a meaningfully different initial evidence profile.

For each selected agent:

1. use exactly the same evidence-card IDs in both prompt families;
2. use the same semantic candidate allocations;
3. use a matched displayed option permutation for each paired call;
4. use identical decoding settings;
5. repeat enough times to characterize stochastic variability.

Recommended:

```text
10 paired repetitions × 3 agents × 2 prompt families = 60 calls
```

If provider usage must be reduced, use at least 5 paired repetitions.

---

## 6. Earlier validation prompt

Use the exact existing validation implementation if available. Do not silently rewrite it.

Its structure is:

```text
You are solving a Team Allocation reasoning task in the partial information condition.
One person must do the first task and the remaining two jointly do the second task.
The strongest allocation depends on the relevant individual skills and on how well
the two-person team cooperates. Choose exactly one displayed option. Do not invent
additional evidence.

Scenario:
<scenario>

Candidate allocations:
A) ...
B) ...
C) ...

Available evidence:
- <evidence card 1>
- <evidence card 2>
...

Return JSON only with this shape:
{"option_label": "A", "rationale": "brief reasoning based only on the available information"}
```

Store the full rendered prompt for every call.

---

## 7. Actual game initialization prompt

Use the exact round-zero prompt family from `relational_imitation_round_feedback`.

The relevant structure is:

```text
You are Agent <id>, one participant in a group reasoning problem.

Your goal is to identify the correct answer.

Participant identities persist across interactions.

Some participants may have objectives that differ from yours. Do not accept a
recommendation merely because another participant gives it. Evaluate whether
the explicit evidence you actually have supports the conclusion, and remain
aware that a source may present only part of the relevant information.

Make your own decision, using the facts you currently know and nothing else.
No other participant has stated a position yet.

QUESTION

SCENARIO
<scenario>

QUESTION
Which allocation is expected to be most effective?

The available answers are:
- A) ...
- B) ...
- C) ...

Exactly one of these answers is correct. Vote by its letter.

YOUR CURRENT KNOWLEDGE

- <evidence ID>: <evidence text>
...
```

followed by the current decision message:

```text
DECISION

Work out which option the facts available to you support, and vote for it.

Your reason should briefly explain your choice, for your own record.

Sharing a fact is the only way to pass information to other participants.
You may share exactly one of the facts you currently know by giving its identifier
in `shared_fact_id`, so that the participants who see your position can use it too.

Return only valid JSON:

{
    "vote": "<A | B | C>",
    "reason": "<brief private reason>",
    "shared_fact_id": "<valid evidence id | none>"
}
```

Again, prefer calling the actual game prompt renderer rather than reproducing this manually.

Store the complete rendered message sequence for every call.

---

## 8. Prompt-equivalence outputs

For every call save at least:

```json
{
  "task_id": "task_001",
  "agent_id": 1,
  "pair_id": "...",
  "prompt_family": "validation|game_init",
  "displayed_options": [],
  "semantic_option_mapping": {},
  "evidence_ids": [],
  "raw_response": "...",
  "parsed_semantic_answer": "ALLOCATION_2",
  "correct": true,
  "parse_success": true
}
```

Report per agent and pooled:

- truth-selection frequency;
- semantic answer histogram;
- parse rate;
- paired prompt-family disagreement rate;
- exact paired agreement count;
- validation-correct/game-wrong count;
- game-correct/validation-wrong count.

Treat this as a prompt-behavior diagnostic, not a broad statistical claim.

---

# Experiment B — Nested evidence-dose curve

## 9. Goal

Measure how the **actual game initialization prompt** responds to systematically increasing local evidence.

The central question is:

\[
	ext{How much local evidence is needed before isolated agents reliably identify the truth?}
\]

Use only the actual game initialization prompt for the main dose curve.

---

## 10. Evidence doses

Use the following target card counts where feasible:

```text
0, 3, 6, 9, 12, 18, 27
```

At every dose record both:

```text
number of evidence cards
number of distinct latent facts represented, out of 9
```

Raw card count and latent-fact breadth are different quantities and must not be conflated.

---

## 11. Nested evidence construction

Evidence sets must be nested for each fixed probe agent:

```text
E0 ⊂ E3 ⊂ E6 ⊂ E9 ⊂ E12 ⊂ E18 ⊂ E27
```

Every larger condition contains all cards from the preceding condition plus additional cards.

Do not independently resample unrelated evidence subsets at each dose.

Use a deterministic seeded rule.

---

## 12. Breadth before redundancy

When adding cards, maximize coverage of previously unseen latent facts before adding redundant branches whenever possible.

Desired progression:

```text
0 cards  -> no latent facts
3 cards  -> ~3 latent facts
6 cards  -> ~6 latent facts
9 cards  -> broad coverage, ideally close to all 9
12/18/27 -> increasing redundant branches after broad coverage
```

Do not force an impossible exact mapping if one evidence card touches multiple facts or the branch structure differs. Record realized coverage.

This is essential for separating:

```text
information breadth
from
evidence redundancy
```

---

## 13. Probe agents and repetitions

Use:

```text
3 fixed probe agents
```

and at least:

```text
3 repetitions per dose
```

giving:

```text
7 doses × 3 agents × 3 repetitions = 63 calls
```

If feasible, increase repetitions, but keep the semantic evidence sets fixed within each agent/dose. Only provider sampling and option presentation should vary across repetitions.

---

## 14. Zero-evidence condition

For dose `0`:

- show the scenario;
- show candidate allocations;
- show no evidence cards;
- use the actual game initialization framing as faithfully as possible;
- preserve the structured game decision output.

Reference chance level:

```text
1 / 3
```

Do not force the observed result to equal chance.

---

## 15. Primary outcomes

At every dose compute:

```text
fraction choosing ALLOCATION_2
```

where `ALLOCATION_2` is the exact truth for this task.

Also report:

- n;
- number correct;
- 95% binomial CI;
- semantic answer distribution;
- parse success rate;
- mean distinct latent facts represented.

Keep agent identity in the dataset and plots because the same probe agents are reused across nested conditions.

---

## 16. Important interpretation warning

Do **not** require monotonic performance.

More cards can be:

- redundant;
- conflicting in salience;
- informative mainly for eliminating different options;
- longer and harder to attend to.

Therefore a non-monotone curve is a result, not an implementation failure.

Do not discard or regenerate non-monotone observations.

---

## 17. Secondary analysis

If supported by the realized doses, report both:

```text
truth rate vs card count
truth rate vs distinct latent-fact coverage
```

The second relation may be more scientifically meaningful than raw prompt length.

Do not fit elaborate models to this small pilot.

---

# 18. Results-directory contract

Create one self-contained directory, preferably:

```text
results/studies/musr_local_evidence_probe_01/
```

Use repository-native result conventions if an equivalent structure already exists.

It should contain approximately:

```text
musr_local_evidence_probe_01/
├── README.md
├── config.yaml
├── manifest.json
│
├── task/
│   ├── task_001.json
│   ├── evidence_catalog.csv
│   └── agent_initial_views.csv
│
├── prompt_equivalence/
│   ├── raw_calls.jsonl
│   ├── paired_results.csv
│   └── summary.csv
│
├── evidence_dose/
│   ├── dose_definitions.json
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   └── summary.csv
│
└── analysis/
    ├── local_evidence_probe_report.md
    ├── tables/
    │   ├── prompt_equivalence_table.csv
    │   ├── dose_curve_table.csv
    │   └── evidence_coverage_table.csv
    └── figures/
        ├── evidence_dose_truth_curve.png
        ├── evidence_dose_by_agent.png
        └── prompt_family_comparison.png
```

The exact filenames can follow current MAS-CC conventions, but all information above must be preserved.

---

## 19. Save every prompt and raw response

Every real provider call must be recoverable.

Save:

- full rendered prompt/message sequence;
- evidence IDs and texts;
- displayed options;
- semantic option mapping;
- raw response;
- parsed answer;
- correctness;
- parse status;
- model/provider metadata.

Do not rely on transient external logs.

---

# 20. Paper-ready report

Create:

```text
analysis/local_evidence_probe_report.md
```

It must be understandable without reading source code.

Include:

## A. Motivation

Explain the existing near-chance round-zero result:

```text
30.6% truth vs 33.3% chance
```

and motivate the local evidence-response calibration.

## B. Task

Show:

- scenario;
- three semantic allocations;
- truth allocation;
- nine hidden latent facts;
- evidence-card structure;
- original `N=12` initial distribution.

Clearly mark hidden latent information as **evaluation-only metadata**.

## C. Exact prompt examples

Include one complete rendered example of:

1. earlier validation prompt;
2. actual game initialization prompt plus decision message;

using the **same agent and same evidence cards**.

This is mandatory. The report must make the prompt difference inspectable without source code.

## D. Prompt-equivalence design

State agents, repetitions, evidence matching, option-permutation matching, model, and decoding settings.

## E. Prompt-equivalence results

Provide a table such as:

| Agent | Validation truth rate | Game-init truth rate | Paired disagreement |
|---|---:|---:|---:|
| 1 | ... | ... | ... |
| 4 | ... | ... | ... |
| ... | ... | ... | ... |

Also show pooled semantic answer histograms.

## F. Evidence-dose design

Explain nested evidence and breadth-before-redundancy.

Show the exact nested evidence IDs for at least one probe agent.

## G. Evidence-dose results

Provide:

| Cards | Mean latent facts covered | n | Truth choices | Truth rate | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 6 | ... | ... | ... | ... | ... |
| 9 | ... | ... | ... | ... | ... |
| 12 | ... | ... | ... | ... | ... |
| 18 | ... | ... | ... | ... | ... |
| 27 | ... | ... | ... | ... | ... |

Also show results by probe agent.

## H. Interpretation

Answer explicitly:

1. Does local truth selection rise above chance with increasing evidence?
2. At approximately what evidence breadth does improvement begin?
3. Does redundancy beyond broad coverage help?
4. Is the response monotone?
5. Does the actual game prompt behave differently from the earlier validation prompt?

Do not hide irregular or negative results.

## I. Connection to blackboard dynamics

Explain the mechanistic chain:

```text
round-zero private evidence
        -> near-chance decisions

blackboard communication
        -> accumulation of evidence / semantic information

local evidence-dose curve
        -> calibrated relationship between information and truth selection
```

This report is the **local-response calibration** for later communication/control experiments.

## J. Limitations

State clearly:

- one semantic task;
- three probe agents;
- limited repetitions;
- stochastic LLM;
- descriptive pilot rather than broad inferential evidence.

---

# 21. Figures

## Figure 1 — Evidence-dose truth curve

```text
x-axis: evidence-card count
y-axis: fraction choosing ALLOCATION_2
```

Include the chance line:

```text
1 / 3
```

Annotate or otherwise show mean latent-fact coverage.

Show individual agent trajectories/points behind the aggregate if practical.

## Figure 2 — Truth rate vs latent-fact coverage

If enough distinct coverage levels exist:

```text
x-axis: number of distinct latent facts represented, 0–9
y-axis: truth-selection rate
```

## Figure 3 — Prompt-family comparison

Show per-agent truth rates for:

```text
validation prompt
actual game initialization prompt
```

Keep all plots descriptive.

---

# 22. Paper-facing interpretation

Do not write the conclusion in advance.

The report should support, if justified by the data, statements like:

> With naturally allocated private evidence, isolated agents begin near chance. As the same game-playing model receives progressively broader evidence, truth selection increases, establishing the local information-response curve underlying the subsequent communication experiment.

If the data do not support this, say so.

If the actual game prompt substantially degrades performance relative to the simpler validation prompt, report that clearly.

---

# 23. Acceptance criteria

The probe is complete only when:

1. the exact validated task is archived;
2. all evidence cards and latent-fact provenance are archived;
3. exact rendered validation and game prompts are archived;
4. at least 3 agents are used for prompt-equivalence;
5. prompt-equivalence uses paired evidence and matched option permutations;
6. evidence-dose sets are nested;
7. doses include `0, 3, 6, 9, 12, 18, 27` where feasible;
8. latent-fact coverage is recorded at every dose;
9. all real calls use `gwdg/openai-gpt-oss-120b`;
10. all outputs are mapped back to semantic allocation IDs;
11. every raw prompt and response is saved;
12. summary CSV tables exist;
13. diagnostic figures exist;
14. `analysis/local_evidence_probe_report.md` is complete;
15. the report includes full prompt examples;
16. the report includes at least one explicit nested evidence example;
17. the report compares against the `1/3` chance baseline;
18. the report connects the local evidence-response curve to the later blackboard experiment;
19. no social interaction, controller, or population dynamics are introduced.

At completion, print the exact path:

```text
results/studies/musr_local_evidence_probe_01/
```

and summarize:

```text
existing round-zero truth rate
zero-evidence truth rate
truth rate at each evidence dose
truth rate at broad/full evidence
prompt-equivalence disagreement rate
```

The **scientific behavior of the local model**, not merely successful code execution, is the deliverable.
