# Handoff: Symbolically Ambiguous MuSR World Construction and End-to-End Validation

## Goal

Prepare, implement, and test a new benchmark-construction gate for the MuSR Team Allocation task so that:

1. zero information stays near chance;
2. intended private views are genuinely ambiguous;
3. full information remains reliably solvable;
4. the same frozen prompt/model used in the blackboard game exhibits that separation.

The previous redistribution calibration still failed:

```text
Zero:    33.3%
NAT:     63.9%
R2:      50.9%
R3:      50.0%
R4:      53.7%
F9:      90.0%
```

This shows that structural incompleteness alone is insufficient. Even two latent values can make the correct allocation too predictable.

The new construction principle is therefore:

```text
PARTIALLY AMBIGUOUS
+
GLOBALLY DECISIVE
```

The agent must implement the complete symbolic filter, generate accepted tasks, run real-provider validation, and produce a self-contained final scientific report.

---

## 1. Freeze the components that already work

Do not redesign the blackboard, controller, or prompt in this study.

Keep fixed:

```text
game-playing model = gwdg/openai-gpt-oss-120b
round-zero prompt = P2
Full Profile representation = F9
board/controller implementation = unchanged
```

For offline MuSR natural-language evidence generation, keep the existing generation provider/configuration. If the generator currently uses `gpt-5.6-terra`, use it only for data/evidence generation and validation, never as the population game-playing model.

---

## 2. Exact hidden world

Each Team Allocation world contains nine hidden values:

```text
6 skill values
3 pairwise cooperation values
```

Represent the hidden world as:

```text
z = (z1, ..., z9)
```

Use the generator's actual discrete latent-value support and sampling distribution. Do not hard-code a support if the repository already defines it.

For every candidate allocation `a`, reuse the existing exact scorer:

```text
S_a(z)
```

The exact gold answer is:

```text
A*(z) = argmax_a S_a(z)
```

Only worlds with a unique optimum are valid.

Do not duplicate scoring logic if an exact scorer already exists.

---

## 3. New concept: private-view ambiguity

A private view reveals only a subset of latent values.

Let:

```text
I   = indices of visible latent values
z_I = their observed values
```

All other latent values are unknown.

The key question is:

> Given only this partial latent state, how predictable is the eventual optimal allocation over possible completions of the unknown values?

For each partial view, exactly enumerate possible completions whenever feasible.

Because the world has only nine latent values and a small discrete support, exact enumeration should be cheap.

---

## 4. Completion distribution

For a partial view `z_I`, define the completion set:

```text
C(z_I) = {z' : z'_I = z_I}
```

Weight completions according to the same latent-world prior used by the generator.

If the generator samples latent values uniformly and independently, uniform completion weighting is correct.

If the generator uses a non-uniform prior or structural constraints, respect them.

Condition on the structural rules used to define valid candidate worlds, including at minimum:

```text
valid latent support
unique exact optimum
```

and any pre-existing minimum score-margin rule.

The report must state exactly how completions were weighted and how ties/invalid completions were handled.

---

## 5. Exact private predictability

For each partial view compute the conditional answer distribution:

```text
p_I(a) = P(A* = a | z_I)
```

Then compute:

### Maximum posterior predictability

```text
M_I = max_a p_I(a)
```

Interpretation:

```text
M_I ~ 1/3  -> nearly uninformative private view
M_I ~ 1    -> private view nearly determines the winner
```

### Conditional answer entropy

```text
H_I = -sum_a p_I(a) log p_I(a)
```

Normalized:

```text
Hbar_I = H_I / log(3)
```

Interpretation:

```text
Hbar_I ~ 1 -> highly ambiguous
Hbar_I ~ 0 -> nearly determined
```

Save both metrics.

---

## 6. Why the previous structural checks were insufficient

The redistribution study enforced:

```text
no agent has all nine latent values
no agent can explicitly score all candidates
```

but private truth remained around 50%.

A private view can therefore be structurally incomplete while still being highly predictive.

Example:

```text
BAD private view

possible valid completions:
A0 wins   5%
A1 wins  10%
A2 wins  85%

M = 0.85
```

Desired:

```text
GOOD private view

possible valid completions:
A0 wins 34%
A1 wins 31%
A2 wins 35%

M = 0.35
```

The benchmark must control decision ambiguity explicitly.

---

## 7. Symbolic scan before any language generation

Implement a symbolic candidate-world scan requiring no LLM calls.

Recommended first scan:

```text
10,000 candidate hidden worlds
```

or more if computationally cheap.

For every candidate world:

1. verify unique exact optimum;
2. compute all candidate scores;
3. compute gold score;
4. compute second-best score;
5. compute score margin;
6. evaluate private-view ambiguity;
7. record the gold semantic allocation.

This symbolic scan must happen before natural-language evidence is generated.

---

## 8. Exhaustive evaluation of private subset sizes

For every candidate world, evaluate all latent-index subsets of sizes:

```text
k = 2
k = 3
k = 4
```

There are only:

```text
C(9,2) = 36
C(9,3) = 84
C(9,4) = 126
```

For each k, summarize:

```text
mean M_I
median M_I
max M_I
95th percentile M_I
mean Hbar_I
minimum Hbar_I
```

This gives both:

```text
average private ambiguity
worst-case private ambiguity
```

Keep both.

---

## 9. Candidate ambiguity criteria

Test the following preferred symbolic criterion first:

```text
M_I <= 0.45
Hbar_I >= 0.90
```

for the intended private views.

Do not silently relax thresholds.

First report how many candidate worlds satisfy the preferred gate.

If almost no worlds pass, perform a documented feasibility check with:

```text
M_I <= 0.50
```

as a fallback threshold.

The final report must distinguish clearly between:

```text
preferred criterion
fallback feasibility criterion
```

Do not choose these thresholds after observing game-playing LLM accuracy.

---

## 10. Select the intended private breadth

Use the symbolic scan to determine whether robust ambiguity is feasible for:

```text
k = 2
k = 3
k = 4
```

Prefer the largest k that yields enough accepted worlds:

```text
if k=4 is feasible:
    prefer k=4
elif k=3 is feasible:
    prefer k=3
else:
    use k=2
```

Do not automatically choose the smallest private view.

The goal is to keep local reasoning rich while maintaining genuine ambiguity.

---

## 11. Full-world decisiveness

Private ambiguity alone is not sufficient.

For every complete world compute:

```text
Delta = gold score - second-best score
```

Compare at least:

```text
minimum score margin >= 1
minimum score margin >= 2
```

Report how the score-margin requirement trades off against private ambiguity and candidate acceptance rate.

The purpose is to find worlds that are:

```text
ambiguous when partially observed
decisive when fully observed
```

Do not assume margin 2 is automatically optimal.

---

## 12. Balance the semantic gold answers

The final task set should be approximately balanced across:

```text
ALLOCATION_0
ALLOCATION_1
ALLOCATION_2
```

For a small final validation set, prefer exact balance.

Example:

```text
6 tasks:
2 gold ALLOCATION_0
2 gold ALLOCATION_1
2 gold ALLOCATION_2
```

This reduces zero-information semantic-prior artifacts.

---

## 13. Select worlds before language generation

The generation pipeline must become:

```text
sample hidden matrix
        |
        v
exact gold + score margin
        |
        v
private ambiguity analysis
        |
        v
PASS symbolic gates?
        |
       yes
        |
        v
generate natural-language evidence
```

Do not spend LLM generation calls on worlds rejected symbolically.

---

## 14. Generate natural-language evidence for accepted worlds

For symbolically accepted worlds, use the current MuSR-style evidence generator.

Preserve:

```text
latent value
-> multiple reasoning/evidence branches
-> natural-language evidence cards
```

Keep the current branch-count/depth configuration unless there is a concrete existing frozen setting.

Do not redesign evidence depth, introduce distractors, or otherwise change the language-generation mechanism in this study unless the final behavioral validation proves symbolic filtering alone insufficient.

Archive for every accepted task:

```text
hidden latent world
exact candidate scores
gold answer
score margin
all evidence-card provenance
generation metadata
hashes
```

---

## 15. Full Profile packet

Construct the existing F9 Full Profile:

```text
one deterministic representative evidence card
for each of the nine latent values
```

Representative-card selection must be independent of gpt-oss behavioral performance.

Use the same canonical deterministic rule already established.

F9 should therefore have:

```text
9/9 latent-value coverage
minimal redundancy
```

---

## 16. Private assignment construction

Once a target private breadth k is selected, construct population assignments.

Requirements:

```text
each agent gets approximately k distinct latent values
population union covers all 9 latent values
each latent value has multiple holders
no agent has 9/9 coverage
assignment is deterministic from a stored seed
```

Prefer extra branches within already assigned latent values rather than increasing breadth beyond k.

Most importantly:

```text
every realized agent private view must satisfy
the selected symbolic ambiguity threshold
```

Save for every agent:

```text
latent IDs
evidence IDs
p(A0 | view)
p(A1 | view)
p(A2 | view)
M_I
Hbar_I
```

---

## 17. Exact validation of realized private views

Even if a world passes broad subset-level screening, validate the exact agent private views used in the final assignment.

For every agent require:

```text
M_I <= selected threshold
```

and record the full conditional allocation distribution.

This makes the final benchmark auditable.

---

## 18. Real-provider behavioral validation

Only after the symbolic benchmark is frozen, evaluate it with:

```text
gwdg/openai-gpt-oss-120b
```

using the frozen:

```text
P2 prompt
```

No blackboard and no controller yet.

For every accepted task evaluate:

```text
Z = zero evidence
P = final private distribution
F = F9 Full Profile
```

---

## 19. Behavioral sample size

Use multiple semantic worlds.

Recommended minimum final validation set:

```text
6 accepted semantic worlds
```

balanced across gold allocations.

For Zero:

```text
at least 10 repetitions per task
```

For Full:

```text
at least 10 repetitions per task
```

For Private:

```text
all N agents
x at least 3 repetitions
```

If provider budget allows, use 20 repetitions/task for Zero and Full.

Use exactly the same decoding configuration as the actual game.

---

## 20. Behavioral acceptance target

Preferred benchmark behavior:

```text
Zero:       near 1/3
Private:    0.33–0.45
Full:       >= 0.80
```

with:

```text
Full >= 0.90
```

preferred.

Recommended hard benchmark-level gate:

```text
Zero <= 0.45
Private <= 0.45
Full >= 0.80
```

with confidence intervals reported.

Do not modify individual hidden worlds after observing their provider calls.

If behavioral filtering is used, it must follow a predeclared rule and all rejection counts must be reported.

---

## 21. Compare symbolic ambiguity with empirical LLM behavior

For each realized private view compare:

```text
symbolic predictability M_I
```

with:

```text
empirical gpt-oss probability of selecting truth
```

Produce a plot/table.

The key hypothesis is:

> Lower symbolic private predictability should correspond to lower local game-playing accuracy, while F9 should remain highly solvable.

This validates that the new symbolic criterion is controlling the right quantity.

---

## 22. Diagnostic: dangerous partial views

For each world identify the subset with highest:

```text
M_I
```

Record which latent types it contains, e.g.:

```text
skill + skill
skill + cooperation
cooperation + cooperation
```

This is diagnostic only.

Do not hand-edit individual worlds based on this result.

---

## 23. Automated tests

Add tests for the symbolic ambiguity machinery.

At minimum:

1. exact completion enumeration on a tiny synthetic case;
2. conditional probabilities sum to 1;
3. semantic gold IDs are correct;
4. ties/invalid completions follow the documented rule;
5. nonuniform latent priors are respected if applicable;
6. `Hbar_I` is in `[0,1]`;
7. `M_I` is in the valid three-choice range;
8. observing all nine latent values yields certainty in the actual gold answer;
9. all k=2/3/4 subset scans are deterministic;
10. accepted private assignments satisfy the chosen ambiguity threshold;
11. population union covers all nine values;
12. answer-letter shuffling does not alter semantic scoring;
13. hidden matrix values never leak into provider prompts.

Run relevant existing generator/game regression tests as well.

---

## 24. Configuration

Expose the new behavior through explicit configuration.

Suggested conceptual shape:

```yaml
symbolic_ambiguity:
  enabled: true
  private_breadth_candidates: [2, 3, 4]
  preferred_max_predictability: 0.45
  fallback_max_predictability: 0.50
  min_normalized_entropy: 0.90
  evaluate_all_subsets: true

world_filter:
  require_unique_optimum: true
  min_score_margin_candidates: [1, 2]
  balance_gold_allocations: true

behavioral_validation:
  model: gwdg/openai-gpt-oss-120b
  prompt_variant: P2
  zero_max_truth_rate: 0.45
  private_max_truth_rate: 0.45
  full_min_truth_rate: 0.80
  preferred_full_truth_rate: 0.90
```

Adapt names to repository conventions.

---

## 25. Required study directory

Create:

```text
results/studies/musr_symbolic_ambiguity_calibration_01/
```

or the repository-native equivalent.

Suggested structure:

```text
musr_symbolic_ambiguity_calibration_01/
├── README.md
├── config.yaml
├── manifest.json
│
├── symbolic_scan/
│   ├── candidate_worlds.csv
│   ├── acceptance_summary.csv
│   ├── subset_metrics.parquet
│   ├── ambiguity_by_k.csv
│   └── margin_ambiguity_tradeoff.csv
│
├── accepted_tasks/
│   ├── task_*.json
│   ├── generation_manifest.json
│   ├── full_profile_packets.json
│   └── private_assignments.json
│
├── behavioral_validation/
│   ├── raw_calls.jsonl
│   ├── observation_level_results.csv
│   ├── summary_by_task_condition.csv
│   └── summary_pooled.csv
│
└── analysis/
    ├── symbolic_ambiguity_calibration_report.md
    ├── tables/
    │   ├── symbolic_acceptance_table.csv
    │   ├── final_task_table.csv
    │   ├── zero_private_full_table.csv
    │   └── symbolic_vs_empirical.csv
    └── figures/
        ├── ambiguity_by_private_breadth.png
        ├── candidate_acceptance_rates.png
        ├── score_margin_vs_ambiguity.png
        ├── zero_private_full_separation.png
        └── symbolic_vs_empirical_predictability.png
```

---

## 26. Required final report

Create:

```text
analysis/symbolic_ambiguity_calibration_report.md
```

This report is mandatory and must be written as a scientific validation report, not a developer log.

It must contain:

### A. Motivation

Report the previous failure:

```text
Zero = 33.3%
R2 = 50.9%
R3 = 50.0%
R4 = 53.7%
F9 = 90.0%
```

Explain why structural incompleteness is insufficient.

### B. Exact world definition

Explain:

```text
9 latent values
-> exact allocation scores
-> unique gold answer
```

### C. Private ambiguity definition

Define and explain:

```text
p_I(a)
M_I
H_I
Hbar_I
```

and the completion distribution.

### D. Symbolic scan

Report:

- number of candidate matrices;
- gold-answer balance;
- margin distribution;
- acceptance rate for k=2/3/4;
- preferred/fallback threshold feasibility.

### E. Ambiguity versus private breadth

Include a table like:

| k | Candidate worlds | Pass M<=0.45 | Pass rate | Median worst-case M | Median Hbar |
|---:|---:|---:|---:|---:|---:|
| 2 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |

### F. Score-margin tradeoff

Show whether stricter full-world margins improve or hurt private ambiguity feasibility.

### G. Frozen construction rule

State the selected:

```text
private breadth k
max-predictability threshold
entropy threshold
minimum score margin
gold-allocation balance rule
```

The choice should be based on symbolic feasibility, not later LLM behavior.

### H. Accepted task set

List:

```text
task ID
gold allocation
score margin
worst private M
mean private M
private entropy
```

### I. Real-provider validation

Main table:

| Condition | n | Truth rate | 95% CI |
|---|---:|---:|---:|
| Zero | ... | ... | ... |
| Private | ... | ... | ... |
| Full F9 | ... | ... | ... |

Also show per-task values.

### J. Symbolic versus empirical relation

Assess whether lower `M_I` corresponds to lower empirical private accuracy.

### K. PASS / FAIL

Conclude explicitly:

```text
PASS — symbolically ambiguous worlds create the required distributed-information benchmark
```

or:

```text
FAIL — symbolic ambiguity filtering alone is insufficient
```

### L. Consequence

If PASS:

```text
Freeze the accepted tasks, assignments, P2, and F9.
Proceed to blackboard:
no control / direct recommendation / coordination request.
```

If FAIL:

```text
Do not scale the blackboard experiment.
The next revision must modify evidence generation or task complexity.
```

### M. Limitations

State:

- number of symbolic candidates;
- number of accepted worlds;
- real-provider call count;
- provider stochasticity;
- any approximation in completion weighting.

---

## 27. Main figures

### Figure 1 — Symbolic ambiguity versus private breadth

```text
x-axis: k = 2, 3, 4
y-axis: distribution of worst-case M_I
```

Show threshold 0.45.

### Figure 2 — Score margin versus ambiguity

Show the tradeoff between:

```text
full-world decisiveness
private-view ambiguity
```

### Figure 3 — Final behavioral separation

```text
Zero | Private | Full
```

with truth rate, uncertainty intervals, and chance line at `1/3`.

### Figure 4 — Symbolic versus empirical predictability

```text
x-axis: M_I
y-axis: empirical truth-selection probability
```

---

## 28. Hard rules

Do not:

- tune P2 again;
- modify blackboard/controller code;
- use Terra as the game-playing model;
- generate evidence for symbolically rejected worlds;
- hand-pick evidence cards based on gpt-oss answers;
- hand-edit matrices after behavioral evaluation;
- hide rejected candidate worlds;
- claim success from symbolic metrics alone.

The benchmark must pass both:

```text
symbolic ambiguity
+
real-provider Zero/Private/Full separation
```

---

## 29. Completion criteria

The task is complete only when:

1. exact completion enumeration is implemented;
2. `p_I(a)`, `M_I`, and normalized entropy are implemented;
3. all k=2/3/4 latent subsets can be scanned;
4. the symbolic machinery is tested;
5. a large candidate-world scan is run;
6. score-margin/ambiguity tradeoffs are reported;
7. a frozen construction criterion is selected;
8. accepted worlds are approximately balanced across gold allocations;
9. language evidence is generated only after symbolic acceptance;
10. final private assignments satisfy the ambiguity criterion;
11. P2 and F9 remain frozen;
12. Zero/Private/Full validation is run with `gwdg/openai-gpt-oss-120b`;
13. all raw prompts/responses are saved;
14. all summary tables and figures are generated;
15. `analysis/symbolic_ambiguity_calibration_report.md` is complete;
16. the report gives an explicit PASS/FAIL decision;
17. the final frozen task pack and assignment paths are printed.

At completion summarize:

```text
candidate worlds scanned
symbolic acceptance rate
selected private breadth
selected ambiguity threshold
selected score-margin rule
number of accepted/final tasks
Zero truth rate
Private truth rate
Full truth rate
PASS / FAIL
results directory
report path
```

The scientific deliverable is a **validated family of Team Allocation tasks whose small private views are genuinely ambiguous while their complete information state is decisive and solvable**.
