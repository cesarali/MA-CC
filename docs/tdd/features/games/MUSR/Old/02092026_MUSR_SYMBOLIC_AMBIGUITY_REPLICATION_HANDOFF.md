# Handoff: Frozen Benchmark Replication and Task-Heterogeneity Check

## Purpose

Run a **final replication of the currently frozen MuSR symbolic-ambiguity benchmark** before making any further generator changes.

The current validated benchmark has:

```text
6 semantic tasks
balanced gold allocations: 2 × ALLOCATION_0, 2 × ALLOCATION_1, 2 × ALLOCATION_2
private breadth k = 4
symbolic gate M <= 0.45
normalized entropy Hbar >= 0.90
minimum score margin = 2
prompt = P2
Full Profile = F9
game-playing model = gwdg/openai-gpt-oss-120b
```

The current behavioral result is:

```text
Zero    = 36.7%   (60 observations)
Private = 45.8%   (216 observations)
Full F9 = 80.0%   (60 observations)
```

The prespecified gate was:

```text
Zero <= 45%
Private <= 45%
Full >= 80%
```

Therefore the benchmark formally failed only because Private was **0.8 percentage points above the threshold**, while Full passed exactly at the lower bound.

Do **not** redesign the generator, prompt, evidence, assignments, or symbolic gate yet.

The goal is to determine whether the current benchmark is actually stable, and whether the aggregate result is driven by one or two problematic tasks.

---

# 1. Freeze everything

Do not change:

```text
accepted six tasks
hidden matrices
natural-language evidence
private assignments
symbolic ambiguity threshold
private breadth k=4
P2 prompt
F9 Full Profile
model/provider
temperature/decoding settings
option-shuffling logic
```

This study is a **replication only**.

Do not regenerate evidence.

Do not rerun the 10,000-world symbolic scan unless required only to read metadata.

Do not introduce blackboard communication or control.

---

# 2. First: analyze the existing data before making new calls

Before launching any additional provider requests, inspect the existing behavioral validation data.

For each of the six tasks report:

```text
Zero truth rate
Private truth rate
Full F9 truth rate
```

Also report for Private:

```text
per-agent truth rates
mean across agents
min/max agent accuracy
```

The first question is:

> Are the aggregate 45.8% Private and 80.0% Full values approximately consistent across all six tasks, or are they driven by a small number of outlier tasks?

Create a table:

| Task | Gold | Margin | Zero | Private | Full | Worst private M |
|---|---|---:|---:|---:|---:|---:|
| task_001 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

Do not exclude any task.

---

# 3. Additional replication target

Extend the existing behavioral validation to:

## Private

Current:

```text
3 repetitions per task-agent view
```

Increase to:

```text
6 repetitions per task-agent view
```

There are:

```text
6 tasks × 12 agents × 3 additional repetitions
= 216 additional calls
```

## Zero

Current:

```text
10 repetitions per task
```

Increase to:

```text
20 repetitions per task
```

Additional:

```text
6 tasks × 10
= 60 additional calls
```

## Full F9

Current:

```text
10 repetitions per task
```

Increase to:

```text
20 repetitions per task
```

Additional:

```text
6 tasks × 10
= 60 additional calls
```

Total additional calls:

```text
216 + 60 + 60 = 336 gpt-oss calls
```

This is intentionally the same order as the existing behavioral validation and should remain a small run.

---

# 4. Provider

All new behavioral calls must use:

```text
gwdg/openai-gpt-oss-120b
```

through the existing MAS-CC provider abstraction.

Use the exact same:

```text
temperature
output limits
validation/parser
retry logic
prompt P2
option-randomization logic
```

as in the existing calibration.

Do not use Terra.

---

# 5. Preserve call identity and pairing

The existing observations must remain part of the final dataset.

Do not overwrite them.

Append only the new repetitions.

For every new call save:

```text
task_id
agent_id where applicable
condition = Zero | Private | Full
replicate_id
evidence_ids
latent_ids where applicable
option permutation
semantic answer
correctness
raw response
parsed response
provider metadata
seed/requested seed
token usage
retry status
```

The final analysis must combine:

```text
existing observations
+
new replication observations
```

---

# 6. Final sample sizes

After replication, the intended totals are:

```text
Zero:
6 tasks × 20 = 120 observations

Private:
6 tasks × 12 agents × 6 reps = 432 observations

Full:
6 tasks × 20 = 120 observations
```

Total:

```text
672 behavioral observations
```

---

# 7. Primary analysis

Recompute pooled truth rates and 95% binomial confidence intervals:

```text
Zero
Private
Full
```

Main table:

| Condition | n | Truth | Truth rate | 95% CI |
|---|---:|---:|---:|---:|
| Zero | 120 | ... | ... | ... |
| Private | 432 | ... | ... | ... |
| Full F9 | 120 | ... | ... | ... |

Include the chance reference:

```text
1/3 = 33.3%
```

---

# 8. Task-heterogeneity analysis

This is essential.

For every task separately compute:

```text
Zero accuracy
Private accuracy
Full accuracy
```

with descriptive confidence intervals.

Create a plot with six task-specific triplets:

```text
Zero | Private | Full
```

for each task.

Explicitly answer:

1. Is one task responsible for most of the elevated Private accuracy?
2. Is one task responsible for most of the Full-information failures?
3. Are the six tasks qualitatively consistent?
4. Does score margin appear related to Full accuracy?
5. Does symbolic private predictability M show any useful relationship to empirical Private accuracy?

Do not remove outlier tasks in this study.

---

# 9. Agent-level heterogeneity

For the Private condition, compute per-agent-view accuracy using all six repetitions.

Report:

```text
distribution of empirical private accuracy across the 72 task-agent views
```

Compare this with the archived symbolic:

```text
M_I
Hbar_I
```

Recompute descriptive associations between:

```text
M_I and empirical accuracy
Hbar_I and empirical accuracy
```

Do not overinterpret correlations.

The previous correlation between M and empirical truth frequency was approximately zero; verify whether that remains true with six repetitions per view.

---

# 10. Decision logic

Do not mechanically redesign the benchmark if Private remains slightly above 45%.

Interpret the final result using both:

```text
prespecified gate
effect separation
task heterogeneity
uncertainty
```

Report the formal gate exactly:

```text
Zero <= 45%
Private <= 45%
Full >= 80%
```

Then also report:

```text
Full - Private
Private - Zero
Full - Zero
```

The central scientific separation is:

```text
Zero -> Private -> Full
```

---

# 11. Final decision categories

At the end assign one of these outcomes.

## A. PASS — freeze benchmark

Use if the replicated result satisfies the prespecified gate:

```text
Zero <= 45%
Private <= 45%
Full >= 80%
```

and no major task pathology is found.

## B. BORDERLINE PASS / ACCEPTABLE FOR BLACKBOARD PILOT

Use if:

```text
Zero remains near chance
Private is only slightly above 45%
Full remains >=80%
```

and the aggregate separation is strong and no single task is catastrophically invalid.

Do not silently relabel the original hard-gate failure. Clearly state that the strict gate is missed but the benchmark may still be scientifically adequate for a blackboard pilot.

## C. FAIL — further benchmark revision needed

Use if any of the following is clearly observed:

```text
Private remains substantially above ~50%
Full falls below 80%
Zero is consistently far above chance
one or more tasks are fundamentally pathological
the Full/Private separation becomes weak
```

If FAIL, diagnose the specific failure mode.

Do not implement another redesign in this study.

---

# 12. Required figures

Create at least:

## Figure 1 — Final pooled separation

```text
Zero | Private | Full
```

with:

```text
truth rate
95% CI
chance line at 1/3
```

## Figure 2 — Per-task separation

For each of the six tasks show:

```text
Zero | Private | Full
```

## Figure 3 — Private task-agent distribution

Show the empirical accuracy distribution across the 72 task-agent private views.

## Figure 4 — Symbolic vs empirical private predictability

```text
x-axis: symbolic M_I
y-axis: empirical truth frequency over 6 repetitions
```

---

# 13. Results directory

Extend the existing study if repository conventions allow, or create a dedicated replication directory such as:

```text
results/studies/musr_symbolic_ambiguity_replication_01/
```

Do not overwrite the original calibration results.

Suggested contents:

```text
musr_symbolic_ambiguity_replication_01/
├── README.md
├── config.yaml
├── manifest.json
├── existing_data_reference.json
│
├── behavioral_validation/
│   ├── new_raw_calls.jsonl
│   ├── combined_observation_level_results.csv
│   ├── pooled_summary.csv
│   ├── per_task_summary.csv
│   └── per_task_agent_private_summary.csv
│
└── analysis/
    ├── symbolic_ambiguity_replication_report.md
    ├── tables/
    │   ├── final_zero_private_full.csv
    │   ├── task_heterogeneity.csv
    │   ├── agent_view_heterogeneity.csv
    │   └── symbolic_empirical_association.csv
    └── figures/
        ├── final_zero_private_full.png
        ├── per_task_zero_private_full.png
        ├── private_agent_view_distribution.png
        └── symbolic_vs_empirical.png
```

---

# 14. Required final report

Create:

```text
analysis/symbolic_ambiguity_replication_report.md
```

The report must be self-contained and scientific, not a developer log.

Include:

## A. Motivation

State the previous result:

```text
Zero = 36.7%
Private = 45.8%
Full = 80.0%
```

and explain that the purpose is to test stability rather than redesign the benchmark.

## B. Frozen benchmark

List:

```text
6 tasks
gold balance
k=4
M<=0.45
Hbar>=0.90
margin>=2
P2
F9
gwdg/openai-gpt-oss-120b
```

## C. Existing per-task diagnostic

Show the six task-specific results before adding new calls.

## D. Replication design

State the additional sample sizes:

```text
+216 Private
+60 Zero
+60 Full
= 336 new calls
```

## E. Final pooled results

Show the final 120 / 432 / 120 sample sizes and confidence intervals.

## F. Task heterogeneity

Identify whether the aggregate behavior is uniform or driven by specific tasks.

## G. Agent-view heterogeneity

Summarize the 72 private task-agent views.

## H. Symbolic vs empirical relation

Reassess M_I/Hbar versus empirical private accuracy.

## I. Gate and separation

Report:

```text
strict PASS/FAIL under the original gate
Full - Private
Private - Zero
Full - Zero
```

## J. Final recommendation

Conclude explicitly with:

```text
PASS
BORDERLINE PASS / ACCEPTABLE FOR BLACKBOARD PILOT
or
FAIL
```

and explain why.

If PASS or BORDERLINE PASS:

```text
freeze benchmark and proceed to the first blackboard population study
```

If FAIL:

```text
do not redesign automatically; state the specific remaining problem
```

## K. Limitations

State:

```text
6 semantic worlds
provider stochasticity
repeated calls from identical task-agent views
small number of semantic worlds despite larger call count
```

---

# 15. Hard rules

Do not:

- regenerate tasks;
- regenerate evidence;
- change P2;
- change F9;
- modify private assignments;
- modify symbolic thresholds;
- change k=4;
- tune score margins;
- introduce board communication;
- introduce the controller;
- remove poorly performing tasks;
- alter the benchmark based on the new calls.

This is a pure replication and heterogeneity study.

---

# 16. Completion criteria

The task is complete only when:

1. existing per-task results are inspected first;
2. 336 additional gpt-oss calls are run;
3. the combined dataset contains 672 behavioral observations;
4. pooled Zero/Private/Full estimates are recomputed;
5. task heterogeneity is analyzed;
6. private task-agent heterogeneity is analyzed;
7. symbolic-vs-empirical association is recomputed;
8. all raw new calls are archived;
9. combined CSV tables are saved;
10. all required figures are produced;
11. `analysis/symbolic_ambiguity_replication_report.md` is complete;
12. the report gives a clear PASS / BORDERLINE PASS / FAIL recommendation.

At completion print:

```text
new calls completed
final Zero truth rate
final Private truth rate
final Full truth rate
Full - Private separation
strict gate PASS/FAIL
recommended benchmark status
results directory
report path
```

The scientific deliverable is a **stability check of the frozen benchmark**, not another benchmark redesign.
