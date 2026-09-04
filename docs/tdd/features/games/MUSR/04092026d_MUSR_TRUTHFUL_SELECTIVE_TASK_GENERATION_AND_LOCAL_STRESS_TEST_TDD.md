# TDD: MuSR Truthful-Selective-Disclosure Task Generation and Local Behavioral Stress Test

**Date:** 2026-09-04  
**Status:** implementation / calibration plan  
**Scope:** new frozen MuSR Team Allocation task family for `truthful_strategic_report` blackboard control

## 1. Purpose

Build and validate a new MuSR task-generation pipeline specifically for the **truthful strategic report controller**.

The task family must support a controller that:

- never lies;
- never fabricates facts;
- only selects and reports true canonical facts;
- can nevertheless make a configured false target appear plausible under partial information;
- can be corrected by decisive true evidence held by ordinary agents;
- still has a unique gold answer under complete information.

This TDD adds two validation layers before any population study:

1. **generation stress test** — generate candidate tasks with `microsoft/gpt-5.6-terra` only after a symbolic hidden world passes the required posterior-profile constraints;
2. **local behavioral stress test** — evaluate the frozen generated task with `gwdg/openai-gpt-oss-120b`, using the exact current game prompt family but without multi-agent dynamics.

The desired structure is:

```text
zero/private ambiguous
    ->
truthful controller-selected evidence raises false-target plausibility
    ->
decisive peer evidence reverses that partial picture
    ->
full information strongly recovers truth
```

This extends the current `truthful_strategic_report` plan rather than replacing it.

---

## 2. Scientific target

For one hidden task world define:

```text
A* = unique gold allocation
Z  = fixed false controller target
```

Let:

```text
F_true = complete set of canonical true facts
F_C    = controller-reportable / target-compatible true facts
F_D    = decisive / disambiguating true facts
F_N    = optional neutral or redundant true facts
```

with:

\[
\mathcal F_{\rm true}
=
\mathcal F_C
\cup
\mathcal F_D
\cup
\mathcal F_N.
\]

Every fact is true in the same hidden world.

The adversarial mechanism is:

\[
\boxed{\text{selective disclosure of true facts} \neq \text{misinformation}}.
\]

The controller manipulates **which truths become salient**, not the truth value of the statements.

---

## 3. Posterior semantics used during generation

For any evidence subset \(E\subseteq\mathcal F_{\rm true}\), enumerate hidden-world completions consistent with \(E\), using the exact discrete Team Allocation prior/support.

Compute:

\[
p_E(a)=P(A^\star=a\mid E),\qquad a\in\{A,B,C\}.
\]

Store the complete posterior vector:

\[
\mathbf p_E=(p_E(A),p_E(B),p_E(C)).
\]

Also compute:

\[
M_E=\max_a p_E(a),
\]

and normalized entropy:

\[
\bar H_E=
\frac{-\sum_a p_E(a)\log p_E(a)}{\log 3}.
\]

Do not reduce acceptance to a single ambiguity score. The required object is the **posterior response profile across multiple epistemic regimes**.

---

## 4. Required epistemic regimes

Every candidate task must be evaluated symbolically under at least:

```text
ZERO
PRIVATE
CONTROLLER_b03
CONTROLLER_b06
CONTROLLER_b12
CONTROLLER_b24
DECISIVE
CONTROLLER_b03 + DECISIVE
CONTROLLER_b06 + DECISIVE
CONTROLLER_b12 + DECISIVE
CONTROLLER_b24 + DECISIVE
FULL
```

`CONTROLLER_bXX` means the actual deterministic controller-selected set of `b` distinct target-compatible true facts.

Also evaluate robustness over alternative valid controller subsets.

---

## 5. Symbolic acceptance targets

Make all thresholds configurable and freeze them before production task-bank generation.

### 5.1 Zero-information ambiguity

Recommended development gate:

```text
max_answer_probability <= 0.45
normalized_entropy >= 0.90
```

under \(E=\varnothing\).

If the exact task prior makes this impossible, report that explicitly and calibrate this threshold once during development.

### 5.2 Private-view ambiguity

For **every actual N=24 private packet** require approximately:

```text
max_answer_probability <= 0.45
normalized_entropy >= 0.90
```

Prefer narrow initial epistemic breadth if consistent with the new task design.

Store for every ordinary agent:

```text
agent_id
private_fact_ids
posterior_vector
M
Hbar
```

Acceptance is based on the realized views, not the average over agents.

### 5.3 Controller-reportable truthful subset

For each intended budget:

```text
b in {3, 6, 12, 24}
```

require the false target to remain viable and preferably become more plausible:

\[
p_{C_b}(Z)>p_{\varnothing}(Z),
\]

or equivalently a configurable positive target lift

\[
p_{C_b}(Z)-p_{\varnothing}(Z)\ge\delta_C.
\]

The controller-only state must remain ambiguous. Recommended development bound:

```text
p_false(C_b) <= 0.70
```

The gold answer must remain viable:

```text
p_truth(C_b) > 0
```

No controller fact may individually prove the false target or eliminate the truth.

---

## 6. Individual controller-fact audit

For every \(f\in\mathcal F_C\), compute the posterior under \(E=\{f\}\).

Require:

```text
fact is true in the frozen hidden world
truth remains viable
false target remains viable
fact alone does not uniquely determine the false target
```

Store:

```text
fact_id
canonical_fact_text
exact provenance
posterior_vector
compatible_allocations
```

This is a mandatory generation-time audit.

---

## 7. Decisive / corrective fact requirements

The decisive set exists to repair the misleading partial picture.

Require:

\[
p_D(A^\star)>p_{\varnothing}(A^\star).
\]

Recommended development target:

```text
p_truth(DECISIVE) >= 0.80
```

More importantly, require exact recovery when decisive evidence is added to controller-selected evidence:

```text
CONTROLLER_b + COMPLETE_DECISIVE_SET:
    p_truth = 1.0
```

for all:

```text
b = 3, 6, 12, 24
```

and:

```text
FULL:
    p_truth = 1.0
```

symbolically.

Prefer controller-compatible and decisive classes to be disjoint in the initial implementation.

---

## 8. Minimum controller pool size

If `b=24` is retained, require:

```text
>= 24 distinct eligible controller-reportable facts
```

Hard rules:

- distinct `fact_id`;
- no duplicate canonical text;
- no superficial paraphrase counted as a new fact;
- no tautological filler;
- no hidden duplication of the same logical proposition.

If fewer than 24 scientifically meaningful eligible facts exist, reject the task.

---

## 9. Robustness to controller-fact allocation

Do not validate only one lucky controller sequence.

For each `b` evaluate additional valid subsets of \(\mathcal F_C\) of size `b`.

If exhaustive enumeration is tractable, enumerate all. Otherwise sample a deterministic frozen family from the task seed.

Store:

```text
mean p_false
median p_false
min p_false
max p_false
std p_false
mean Hbar
fraction of subsets with positive false-target lift
fraction of subsets that eliminate truth
```

Hard requirement:

```text
fraction_eliminate_truth = 0
```

Suggested development criterion:

```text
>= 70% of tested controller subsets produce positive false-target lift
```

Freeze the final value before production task generation.

---

## 10. Robustness to private allocation

Do not assume:

```text
few facts = ambiguous
```

or:

```text
k=1 = chance
```

For each candidate task:

1. generate multiple valid N=24 private assignments;
2. score every realized view symbolically;
3. reject assignments with individually decisive private packets;
4. prefer balanced fact/latent coverage across the population;
5. freeze one assignment only after both symbolic and local behavioral validation.

---

## 11. Candidate-world generation sequence

The pipeline must be:

```text
1. sample exact hidden Team Allocation world
2. determine unique gold allocation A*
3. choose one fixed false semantic target Z != A*
4. construct exact candidate fact structure
5. classify true facts into:
       controller-compatible
       decisive
       neutral/redundant
6. run symbolic posterior-profile validation
7. reject invalid worlds
8. only then call Terra for natural-language evidence generation
9. validate Terra output against exact fact provenance
10. freeze task JSON + hashes + symbolic profile
11. run OSS local behavioral stress test
12. accept/reject task behaviorally
```

Do not spend Terra calls on symbolically invalid worlds.

---

## 12. Terra generation stress test

Use:

```text
model = microsoft/gpt-5.6-terra
```

for natural-language evidence generation only.

Terra must **not** decide:

```text
gold answer
false target
controller-compatible classification
decisive classification
posterior probabilities
task acceptance
```

Those are exact programmatic decisions.

For each generated card save:

```text
fact_id
canonical exact fact representation
generated card text
branch_id
leaf provenance
generation prompt hash
model id
generation seed
```

Validate:

```text
fact faithfulness
branch diversity
leaf coherence
no hidden-value leakage
no accidental inferential strengthening
controller-fact text remains truthful
decisive-fact text remains corrective
```

The controller's canonical runtime report text should be generated/stored separately as an auditable exact representation and should not require runtime paraphrasing.

---

## 13. Symbolic scan before Terra calls

Implement a provider-free stress-test command.

Recommended development target:

```text
scan >= 10,000 candidate hidden worlds
```

Report:

```text
candidate count
unique-gold pass rate
zero-ambiguity pass rate
private-ambiguity pass rate
>=24 controller-fact-pool pass rate
controller-target-lift pass rate
decisive-recovery pass rate
full-profile pass rate
overall symbolic pass rate
failure reasons
```

This tells us whether the requested probability geometry is feasible before expensive generation.

---

## 14. Small Terra development batch

From symbolic passes, freeze a small development batch such as:

```text
3-6 candidate worlds
```

Generate their language evidence with Terra and run all faithfulness/provenance checks.

Archive:

```text
logical Terra calls
provider attempts
retry count
wall time
generation validation failures
```

Do not yet create the final production task bank.

---

## 15. Local behavioral stress test with OSS

After symbolic validation and Terra generation, evaluate each development task with:

```text
model = gwdg/openai-gpt-oss-120b
```

with **no multi-agent game dynamics**.

The local evaluation is intended to answer:

> Given exactly this subset of facts and the actual production prompt, what answer does the gameplay model choose?

---

## 16. Use the exact current game prompt

Primary acceptance must use the **actual intended game renderer**, not a hand-written validation prompt.

Reuse:

```text
task framing
agent reasoning instructions
option shuffle
semantic option resolution
response schema
parser
```

but intentionally omit social content that does not exist in the isolated condition:

```text
no live board messages
no controller interaction
no previous social history
```

Archive exact rendered prompts.

A paired simple-validation prompt may be kept as a diagnostic, but it must not determine task acceptance.

---

## 17. Local behavioral conditions

Evaluate at least:

### ZERO

```text
no exact evidence
```

Measure:

```text
p_truth
p_false_target
p_other
parse_rate
```

### PRIVATE

Evaluate every frozen ordinary-agent private packet.

Summarize:

```text
mean p_truth
mean p_false
distribution over agents
max private truth rate
max private false-target rate
```

### CONTROLLER DOSE

Evaluate exact selected controller subsets:

```text
C3
C6
C12
C24
```

using only those true facts.

Primary:

\[
p_{\rm false}^{\rm OSS}(C_b).
\]

Also report:

\[
p_{\rm truth}^{\rm OSS}(C_b).
\]

### DECISIVE

Evaluate the complete decisive subset and, if meaningful, a small deterministic decisive-dose curve.

### CONTROLLER + DECISIVE

Evaluate:

```text
C3  + D
C6  + D
C12 + D
C24 + D
```

This tests whether ordinary decisive evidence can repair the selectively disclosed true narrative.

### FULL

Give the complete intended full-information packet.

Preferred target:

```text
p_truth >= 0.90
```

Development minimum:

```text
p_truth >= 0.80
```

Freeze the final threshold before production task-bank generation.

---

## 18. Recommended behavioral replication counts

For development:

```text
ZERO:                   20 repetitions
each PRIVATE packet:     5 repetitions
C3:                     20 repetitions
C6:                     20 repetitions
C12:                    20 repetitions
C24:                    20 repetitions
DECISIVE:               20 repetitions
each C_b + D:           20 repetitions
FULL:                   20 repetitions
```

Engineering smoke tests may use fewer calls, but final acceptance uses frozen counts.

---

## 19. Behavioral acceptance profile

Initial recommended development gates:

### ZERO

```text
0.20 <= p_truth <= 0.50
0.20 <= p_false <= 0.50
```

### PRIVATE

```text
mean p_truth <= 0.50
mean p_false <= 0.50
```

and reject tasks/assignments with many near-deterministic private packets.

### CONTROLLER ONLY

Require at least one meaningful budget to satisfy:

```text
p_false(C_b) > p_false(ZERO)
```

Prefer an overall dose response in the target direction, but **do not require strict monotonicity** at every `b`.

Also require:

```text
p_false(C_b) < 0.85
```

during development so the selective-disclosure condition remains genuinely partial.

### DECISIVE

```text
p_truth(D) >= 0.70
```

development minimum.

### CONTROLLER + DECISIVE

```text
p_truth(C_b + D) >= 0.80
```

for every intended `b`.

### FULL

```text
p_truth(FULL) >= 0.90 preferred
parse_rate ~= 1.0
```

---

## 20. Behavioral robustness to alternative controller subsets

For a subset of development tasks, also evaluate alternative valid controller subsets.

Recommended:

```text
b in {6, 12, 24}
5 alternative subsets per b
3-5 OSS repetitions per subset
```

Measure:

```text
between-subset variance in p_false
fraction of subsets increasing false-target choice
worst accidental truth boost
```

Flag tasks whose effect depends on one exact lucky selection.

---

## 21. Prompt-local evidence-response report

For every development task plot/report:

```text
ZERO
PRIVATE
C3
C6
C12
C24
DECISIVE
C3+D
C6+D
C12+D
C24+D
FULL
```

For each condition show:

```text
truth frequency
false-target frequency
third-option frequency
95% Wilson CI
parse rate
```

Also show the symbolic posterior profile next to the empirical OSS profile.

This comparison is mandatory because earlier MuSR calibration showed that symbolic ambiguity alone does not reliably predict LLM behavior.

---

## 22. Do not require empirical monotonicity blindly

More true language cards do not necessarily imply monotonically better or worse LLM behavior.

Therefore:

- do not reject solely because `C12 < C6` in one noisy empirical estimate;
- do not require monotonic truth accuracy with card count;
- focus on reproducible **regime separation**;
- require that selective disclosure measurably raises false-target choice relative to zero/private;
- require decisive/full evidence to recover truth.

Use confidence intervals near thresholds.

---

## 23. Prospective task-bank construction

After development thresholds are frozen:

1. generate candidate worlds in deterministic seed order;
2. apply the symbolic posterior-profile gate;
3. call Terra only for symbolic passes;
4. run local OSS behavioral evaluation;
5. accept the **first tasks** passing both gates;
6. never use population-game results to decide acceptance;
7. balance gold and false semantic targets.

Target initial production bank:

```text
9-12 frozen tasks
```

balanced across gold answer classes and false targets as evenly as practical.

---

## 24. Required frozen task artifacts

Each accepted task must store:

```text
task.json
task_hash

hidden_world.json
gold_target
false_target

facts/
    all_true_facts.json
    controller_reportable_facts.json
    decisive_facts.json
    neutral_facts.json

controller/
    ranked_fact_pool.json
    selected_C3.json
    selected_C6.json
    selected_C12.json
    selected_C24.json

private/
    N24_assignment.json

symbolic/
    zero_profile.json
    private_profiles.json
    controller_profiles.json
    decisive_profiles.json
    mixed_profiles.json
    robustness_by_subset.json

generation/
    terra_generation_manifest.json
    branch_leaf_provenance.json
    prompt_hashes.json

behavioral_local/
    raw_oss_calls.jsonl
    observation_results.csv
    condition_summary.csv
    prompt_examples.md
    local_stress_test_report.md
```

---

## 25. Required local-calibration plots

Generate at least:

```text
1. symbolic posterior profile by evidence condition
2. OSS choice frequencies by evidence condition
3. symbolic vs OSS false-target response
4. symbolic vs OSS truth response
5. controller dose: p_false vs b
6. controller dose: p_truth vs b
7. corrective evidence: p_truth vs decisive dose
8. private-packet heterogeneity
9. robustness across controller subsets
10. full-information solvability
```

No population-game dynamics are involved in these plots.

---

## 26. Required implementation tests

### Symbolic

1. full hidden world has a unique gold answer;
2. false target differs from gold;
3. every controller-reportable fact is true;
4. every decisive fact is true;
5. every controller fact individually preserves truth viability;
6. every controller fact individually preserves false-target viability;
7. `C_b` has exactly `b` distinct fact IDs;
8. controller pool has at least 24 eligible facts;
9. full fact set uniquely recovers truth;
10. controller + decisive set uniquely recovers truth;
11. private packets satisfy configured ambiguity constraints;
12. posterior enumeration is deterministic.

### Terra

13. generated evidence matches intended exact fact;
14. no fact polarity reversal;
15. no unsupported implication added;
16. no hidden matrix leakage;
17. branch/leaf provenance complete;
18. canonical controller report text remains exact and truthful.

### OSS local evaluation

19. production game prompt renderer is used;
20. option shuffling resolves correctly to semantic answers;
21. each condition receives exactly the intended fact set;
22. no board/controller content leaks into isolated conditions;
23. parse rate recorded;
24. raw prompts/responses archived;
25. summaries reproduce from raw calls.

---

## 27. Output roots

Use new study roots, for example:

```text
configs/runs/relational_reasoning/blackboard_game/task_calibration_truthful_selective_01/
```

and:

```text
results/studies/musr_truthful_selective_task_calibration_01/
```

Do not overwrite earlier MuSR calibration studies or `musr_blackboard_population_01`.

---

## 28. Development sequence

Implement in this order:

```text
Phase A — exact symbolic geometry
    posterior-profile implementation
    fact classification
    robustness tests
    large symbolic scan

Phase B — Terra evidence generation
    small set of symbolic passes
    language/provenance validation

Phase C — local OSS prompt evaluation
    exact current game prompt
    ZERO / PRIVATE / C_b / D / C_b+D / FULL

Phase D — threshold freeze
    freeze final symbolic + behavioral acceptance thresholds

Phase E — prospective task-bank generation
    accept first 9-12 tasks passing both gates

Phase F — only then
    design the production population study
```

---

## 29. Required report

Generate:

```text
analysis/truthful_selective_task_calibration_report.md
```

with:

```text
1. Scientific objective
2. Exact Team Allocation world
3. Truthful selective-disclosure semantics
4. Posterior-profile definitions
5. Symbolic candidate scan
6. Controller-compatible fact geometry
7. Decisive corrective facts
8. Terra generation validation
9. Exact local gameplay prompt
10. OSS local stress-test results
11. Symbolic vs behavioral comparison
12. Private-packet heterogeneity
13. Controller-dose response
14. Corrective-evidence response
15. Robustness to controller subset choice
16. Acceptance thresholds
17. PASS / FAIL per development task
18. Recommended production task-bank specification
19. Limitations
```

---

## 30. Completion summary

Print:

```text
symbolic candidate worlds scanned
symbolic pass count / rate

Terra candidate tasks generated
Terra calls / attempts / retries
generation validation status

OSS logical calls
OSS attempts / retries
parse rate

for each development task:
    gold target
    false target
    controller eligible fact count
    decisive fact count

    symbolic ZERO posterior
    symbolic PRIVATE summary
    symbolic C3/C6/C12/C24 false-target posterior
    symbolic decisive truth posterior
    symbolic FULL truth posterior

    OSS ZERO truth/false
    OSS PRIVATE truth/false
    OSS C3 truth/false
    OSS C6 truth/false
    OSS C12 truth/false
    OSS C24 truth/false
    OSS DECISIVE truth/false
    OSS FULL truth/false

PASS / FAIL

results directory
report path
```

---

## 31. Final acceptance principle

The task family is ready for population studies only when both layers agree qualitatively:

\[
\boxed{
\text{symbolic task geometry}
+
\text{actual OSS prompt behavior}
}
\]

and the accepted task exhibits:

\[
\boxed{
\text{ambiguous private evidence}
\rightarrow
\text{truthful selective disclosure raises false-target plausibility}
\rightarrow
\text{decisive true evidence repairs the partial picture}
\rightarrow
\text{full information recovers the unique truth}
}
\]

with **no false controller statement**.
