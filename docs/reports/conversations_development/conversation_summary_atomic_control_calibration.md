# Conversation Summary — From Population Control Failure to Atomic Controller Calibration

**Date:** 2026-08-14  
**Project:** Byzantine Fault Tolerance in Language Games / HiddenBench controlled imitation  
**Purpose of this note:** Carry enough context into a new chat to continue once the coding agents return results, without having to reload this long conversation.

---

## 1. Overall research goal

The main scientific goal is **control of multi-agent language systems**.

The intended paper is not primarily about prompt engineering, and it is not primarily about HiddenBench accuracy. The core question is:

> **When, how, and at what cost can an external feedback signal steer a population of reasoning language agents?**

The broader framing is:

- **Control** is the scientific problem.
- **Statistical physics / information theory** provide a quantitative language for studying it.
- **HiddenBench** provides a semantic substrate with distributed private information, a known correct answer, and non-trivial reasoning.
- The key comparison is between a classical controlled q-voter / imitation process and an LLM population with language, private evidence, reasoning, and possibly history.

The long-term empirical goal is to recover a meaningful **phase diagram of controllability** and quantify it with observables such as order parameters, signed response / susceptibility, currents and activities, convergence times, controller success/failure, transfer entropy / conditional mutual information, and possibly later more complex memory-aware information measures.

The physics is not the end in itself. It is there to make the collective-control phenomenon precise and measurable.

---

## 2. Current theoretical model

The current classical theory is a **round-level controlled q-voter**:

- Population size: `N`
- Social group size: `q`
- Controller sensing size: `q_c`
- Controller intervention budget: `b`
- Controller target: `Z`
- One controller decision per population round
- If the controller advocates, exactly `b` update positions are controlled
- At a controlled update, the controller replaces one of the `q` social input slots

The theory currently provides microscopic uncontrolled and controlled kernels `K0`, `K1`, whole-round kernels `R0`, `R1`, exact finite-`N` transfer entropy, a mean-field / weak-control approximation, special behavior for `q = 1`, and scaling predictions in `q`, `b/N`, and controller stochasticity.

Important theoretical point:

\[
T_{U\to N} = I(U_k;N_{k+1}\mid N_k)
\]

is treated as a **directional measure of control information**, not as the definition of control itself.

---

## 3. What the first population sweep showed

The first real phase-space sweep used the LLM implementation with:

- `N = 24`
- `K = 3` options
- `q ∈ {1,2}`
- `q_c ∈ {2,12,16,24}`
- `b ∈ {0,6,12,18}`
- aligned and decoy targets
- round-level controller
- Qwen model
- HiddenBench evacuation task

The main empirical result was unexpectedly simple:

> **The controller was generally not able to steer the population.**

Across the measured cells, the population converged to the correct answer even when the controller pushed a wrong decoy. Increasing sensing quality did not make the controller more effective. Increasing intervention budget often did not help, and at `q=1`, stronger intervention could even slow convergence.

The strongest robust conclusion from the first sweep is therefore not yet a rich phase transition, but a **weak actuation regime**.

---

## 4. What the information measures told us

The infrastructure for mutual information / CMI appears to be doing something meaningful.

### Sensing

The sensing side behaved sensibly:

- larger `q_c` improved the controller's estimate of the population,
- sensor MAE decreased,
- full census sensing gave essentially zero sensor error.

So we are less worried than before that the information-theoretic machinery is fundamentally broken.

### Actuation

The key issue was the other direction:

> **The controller can observe the population, but that does not imply that it can move the population.**

The controller-to-population CMI / TE signal was weak and often not distinguishable from the finite-sample null.

We agreed not to overstate this as “there is mathematically no actuation channel.” The safer statement is:

> **If an actuation effect exists under the current prompting/mechanism, it is small relative to the classical reference and below the resolution of the present experiment.**

This led to an important distinction:

- **sensing**: does the controller know the state?
- **actuation**: does its intervention actually change the next state?
- **outcome control**: does it eventually change where the population goes?

Transfer entropy is only one measurement of the second point.

---

## 5. Why we decided to step back before larger population grids

The population sweep changes structural parameters such as `q`, `q_c`, and `b`, but we realized that we do **not yet know the empirical microscopic response law of one LLM agent**.

In the classical model, the controlled microscopic kernel is known analytically.

In the LLM case, it is not.

The real local object is more like:

\[
P(X_i^{t+1}\mid X_i^t, E_i, H_i, S_i)
\]

where:

- `X_i^t` = current committed vote,
- `E_i` = private task evidence,
- `H_i` = recent history,
- `S_i` = current social inputs, including the controller if present.

This is qualitatively richer than a finite-state q-voter.

The immediate calibration question became:

> **What kind of social intervention produces a measurable but non-deterministic change in an LLM's vote?**

This local study is **not** a change of research direction. It is a model-identification / actuator-calibration step before returning to the full population study.

---

## 6. Important implementation discovery: talk is not the vote

We inspected a saved prompt and initially thought that an utterance such as “I now recommend East Town” was the actual committed decision.

That was wrong.

The runtime makes two different LLM calls:

1. `hidden_bench_imitation_message`
   - discussion turn
   - free-form social message

2. `hidden_bench_imitation_update`
   - actual focal vote update
   - returns `{vote, rationale}`

The prompt logging system had been saving the discussion calls and shadowing the update calls because of deduplication by round index.

A separate inspection confirmed a real update example where:

- the controller advocated North Hill,
- an ordinary peer also advocated North Hill,
- the focal agent still committed East Town.

This is direct evidence that **local resistance exists at the actual transition level**.

There is also a leakage issue in some histories where text such as `partner/controller` can appear. This should not be present in the new atomic experiment.

---

## 7. Why simplify the interaction prompt

The current simulation includes private evidence, current vote, free-form dialogue, generated peer messages, controller messages, accumulated textual history, and a separate vote-update call.

This makes the microscopic dynamics hard to identify.

We therefore decided that the local calibration experiment should use:

> **one prompt → one response → one committed vote**

The prompt should contain only the minimal ingredients that can later appear in the full simulation:

- task / scenario,
- available options,
- focal private evidence,
- current committed vote,
- a compact recent history,
- current social inputs,
- one controller-like social source,
- final vote request.

The rationale may be omitted entirely or stored only for debugging. It should not feed back into the dynamics during this calibration.

---

## 8. HiddenBench's role

We explicitly do **not** want the paper to become too close to *Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs*.

That paper studies collective reasoning, failure to surface unshared information, communication depth, group size, prompting strategies, final task accuracy, and structured communication protocols.

Our differentiating factor should remain:

> **control + statistical physics**

HiddenBench is useful because it gives us semantically meaningful tasks, private and shared evidence, a correct answer, a decoy structure, and validated task diversity.

But the main scientific object in our paper is not “can agents communicate hidden information?” It is:

> **how externally controllable a reasoning population is, locally and globally.**

---

## 9. “Agent sociology” as a missing model layer

We identified three conceptually different kinds of parameters.

### Physics / structure

Examples: `N`, `q`, `q_c`, `b`, topology, update schedule.

### Semantics

Examples: task, private evidence, correct answer, decoy, paraphrases, evidence strength / ambiguity.

### Agent sociology

Examples:

- whether sources are identifiable,
- persistent identity,
- source reputation,
- what others say about a source,
- strategic incentives,
- whether agents know that others may mislead them.

This sociological layer is potentially important because the controller acts **through a social information channel**.

The key intuition is:

> agents need enough trust/openness to aggregate distributed information, but too much trust may make them easy to manipulate.

This could later produce a non-trivial collective trade-off between truth aggregation, controllability, and robustness to adversarial control.

However, we do **not** want this atomic study to become a full psychology or prompt-engineering project.

---

## 10. Inspiration from classic social/behavioral experiments

We discussed several historical inspirations, only as methodological inspiration:

- **Asch**: social consensus / conformity
- **Milgram**: source authority
- **Sherif**: social influence under private uncertainty
- **Kahneman & Tversky**: very small controlled decision problems and counterfactual changes

The most useful methodological lesson from Kahneman/Tversky was:

> keep the decision state almost identical and change one controlled aspect of the social environment.

This is the logic behind the atomic calibration dataset.

---

## 11. The six local calibration buckets

We decided **not** to use six different persuasive controller wordings.

Instead, the controller message should stay essentially fixed while the **social interpretation of the source** changes.

The six buckets are:

### 1. Anonymous

The focal receives social information but no persistent source identities.

### 2. Persistent identity

Stable labels such as `Agent 2`, `Agent 7`.

No reliability information.

### 3. Positive personal reputation

The focal's previous experience indicates that the controller-source agent has usually been useful/correct.

### 4. Negative personal reputation

The focal's previous experience indicates that the controller-source agent has previously been misleading/incorrect.

### 5. Social reputation

The focal has heard another participant describe the controller-source as reliable.

The focal does not independently know whether that assessment is correct.

### 6. Strategic uncertainty

The focal is told that some participants may have objectives different from its own and may advocate strategically.

The focal is **not** told which source is strategic.

Important:

> The controller is present in every bucket.

There is no “no control” bucket in this first calibration study.

The controller is never called “controller.” It appears simply as another source such as `Agent 7`.

---

## 12. Why `q` is not a local experimental axis here

We explicitly decided **not** to sweep `q` in the atomic calibration.

Reason:

- `q` belongs to the later population phase diagram,
- the local study is supposed to calibrate the actuator, not duplicate the population physics sweep.

For this first calibration:

```text
q = 2
```

So each focal prompt sees one ordinary social source and one controller-generated social source.

---

## 13. The 600-prompt atomic calibration dataset

The working agent was instructed to generate:

- **10 HiddenBench tasks**
- **10 atomic states per task**
- **100 base states total**
- **6 matched social buckets**
- **600 prompts total**

Crucially:

> the same 100 base states are reused in all six buckets.

Therefore `state_0037` in all six buckets must have the same task, scenario, options, focal private evidence, paraphrase realization, current vote, recent history, ordinary peer message, controller target, and controller message.

Only the social-context condition changes.

This gives paired counterfactual comparisons.

---

## 14. How the 100 base states are constructed

The dataset generation is **programmatic**.

The working agent must create:

```text
generate_atomic_control_calibration.py
```

The 600 prompts are **not** written manually.

For each of the 10 tasks, generate 10 realistic atomic states.

Balance:

- 5 controller targets = truth
- 5 controller targets = incorrect
- controller target must differ from the focal's current vote

Use multiple focal evidence assignments when possible, validated paraphrases when available, realistic current votes, one ordinary peer, and one controller-like source.

Do not invent HiddenBench facts.

---

## 15. History in the local study

History matters because the eventual population simulation is sequential.

However:

> **history is part of the sampled state, not another experimental bucket.**

For each task:

- 5 states have no previous interaction
- 5 states have exactly one compact previous decision event

Example:

```text
Previous event:
- Agent 2 supported East Town.
- Agent 7 supported North Hill.
- Your committed option after that event was East Town.
```

Rules:

- maximum one previous event for this first calibration,
- no full transcripts,
- no `partner/controller`,
- persistent identities must be internally consistent,
- the exact same history is reused across all six bucket variants of a state.

---

## 16. Controller message in the calibration

The controller does not have private HiddenBench evidence.

It must not fabricate evidence.

Use one fixed form such as:

```text
Based on the discussion so far, I think <CONTROL_TARGET> deserves more weight.
I currently believe <CONTROL_TARGET> is the strongest option.
I am voting <CONTROL_TARGET>.
```

The first calibration study is therefore testing the **social interpretation of the source**, not six different persuasion styles.

---

## 17. Ordinary peer message

The ordinary peer should be allowed to communicate valid task information.

A standardized form is preferred, for example:

```text
Based on the information available to me, I currently think <OPTION> is the best choice.
One relevant fact I have is: <VALID HIDDENBENCH FACT OR PARAPHRASE>.
I am voting <OPTION>.
```

The ordinary peer can support the focal's current vote, the controller target, or the third option, depending on the generated atomic state.

But again, this is held fixed across the six matched bucket versions of that state.

---

## 18. Execution architecture

The prompt construction and the expensive LLM calls are separate stages.

### Working agent

Creates:

1. frozen 600-prompt dataset
2. generic runner script
3. post-processing script

The working agent should **not** spend the API budget on the real multi-model run.

### Frozen prompt dataset

The 600 prompts are generated once, validated, hashed, and frozen.

All models must receive exactly the same dataset.

### Generic runner

The working agent creates:

```text
run_atomic_control_calibration.py
```

with arguments such as:

```text
--provider
--model
--input-dir
--output-dir
--shard-index
--num-shards
--concurrency
```

The runner should use the repository's existing LLM-provider abstraction.

---

## 19. Parallel execution

The actual LLM evaluations can then happen independently:

```text
                         frozen 600 prompts
                                |
             ┌──────────────────┼──────────────────┐
             ↓                  ↓                  ↓
        Qwen worker         GPT worker        Gemini worker
             ↓                  ↓                  ↓
      responses/qwen/     responses/gpt/    responses/gemini/
             └──────────────────┴──────────────────┘
                                ↓
                         post-processing
```

Different models/providers may run at different times, on different machines, in different HPC jobs, and finish asynchronously.

One model can also be sharded over several workers.

The runner must be deterministic, resumable, non-duplicating, and isolated by model/output directory.

---

## 20. Post-processing

The working agent must create:

```text
analyze_atomic_control_calibration.py
```

The main operational quantity is:

\[
C = P(X' = Z)
\]

where `X'` is the new focal vote and `Z` is the controller target.

This is called:

```text
control_target_adoption_rate
```

Interpretation:

> When the controller advocates an option different from the focal's current vote, how often does the focal end up adopting the controller target?

This is the primary **local controllability** statistic.

---

## 21. Aligned vs adversarial control

The 100 base states are balanced:

- 50 correct controller targets
- 50 incorrect controller targets

Therefore compute:

```text
aligned_target_adoption_rate
adversarial_target_adoption_rate
adversarial_resistance_rate
```

This is essential.

A controller that is adopted only when it happens to advocate the truth is not the same as a controller that can genuinely steer the focal agent toward an incorrect target.

---

## 22. Complementary local metrics

Also compute:

```text
truth_rate
stay_rate
switch_rate
switch_to_other_rate
```

These distinguish controller capture, general instability, resistance, and truth correction.

---

## 23. Main expected table

The first table to inspect after all runs is:

| Model | Anonymous | Identity | +Reputation | -Reputation | Social reputation | Strategic uncertainty |
|---|---:|---:|---:|---:|---:|---:|
| Model A | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Model B | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Model C | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |

Each cell is:

```text
control_target_adoption_rate
```

with a 95% confidence interval where practical.

There is also a second table:

| Model | Bucket | Aligned adoption | Incorrect-target adoption | Truth rate |
|---|---|---:|---:|---:|

This will show whether a social setting makes the model generally influenceable, selectively truth-sensitive, robust to adversarial control, or socially deaf.

---

## 24. Statistical comparison

Because the same `state_id`s are reused across buckets and models, comparisons are paired.

The current plan is:

- bootstrap at the task level,
- preserve all 10 atomic realizations within a sampled task,
- use at least 2000 bootstrap resamples,
- compute paired bucket differences,
- compute paired model differences.

The task-level bootstrap avoids pretending that 10 realizations from the same HiddenBench task are completely independent.

---

## 25. Minimal plots

Only two primary plots are currently required.

### Controllability heatmap

Rows = models  
Columns = six social buckets  
Value = controller target adoption rate

### Control-vs-truth plot

For every `(model, bucket)`:

```text
x = adversarial_target_adoption_rate
y = truth_rate
```

This makes the central trade-off immediately visible.

---

# 26. Most important: what we are trying to achieve with these experiments

This local experiment is **not the final paper experiment**.

It is not intended to establish a complete theory of trust, reputation, deception, theory of mind, social psychology, or prompt engineering.

Its role is much narrower and much more important for the main project:

> **We need to identify a microscopic social interaction rule under which an LLM agent is actually controllable to a measurable but non-trivial degree.**

The first population sweep suggested that the current controller has too little effective actuation.

Before spending more compute on large phase diagrams, we therefore want to determine:

1. **Can an individual reasoning agent be moved by a social control signal at all?**
2. **Under which minimal social assumptions does this happen?**
3. **Is the effect large enough to measure, but not so strong that the controller deterministically overwrites the agent?**
4. **Does this behavior generalize across HiddenBench tasks, paraphrases, and different LLM families?**
5. **Can the agent remain receptive enough to useful distributed information while still resisting incorrect or strategic control?**

The desired outcome is a **calibrated microscopic controller mechanism**.

Once we identify a useful regime, we freeze that interaction rule and return immediately to the real scientific problem:

\[
oxed{\text{local agent controllability}}
\quad\longrightarrow\quad
oxed{\text{collective population controllability}}
\]

Then we re-run the full multi-agent system and ask:

- Does weak local control amplify collectively?
- Does the population suppress local control?
- Can local resistance coexist with global steering?
- Does local susceptibility predict global susceptibility?
- Where are the controllable / resistant / delayed / backfire regions?
- How do those regions depend on `q`, `q_c`, `b`, `N`, target alignment, and task semantics?
- How does the LLM population differ from the matched classical q-voter?
- How much directional information flows from the controller into the population?

That is where the **physics and information theory return to center stage**:

\[
T_{U\to N}=I(U_k;N_{k+1}\mid N_k),
\]

together with signed response, currents, order parameters, convergence behavior, and phase-space structure.

The strongest possible eventual result would be a **micro–macro mismatch**, for example:

- local control works but global control fails → collective resistance,
- local control is weak but global control succeeds → emergent amplification,
- adversarial control slows but cannot redirect the population,
- social trust helps information aggregation but simultaneously opens a control vulnerability,
- or some non-monotonic regime where intermediate openness is optimal.

The atomic calibration experiment exists only to make sure that, when we go back to the population, we are studying a **real control channel** rather than sweeping structural parameters around an actuator that barely affects the LLM.

---

## 27. Files produced during this conversation

The most up-to-date implementation specification for the coding agent is:

```text
hiddenbench_atomic_control_calibration_working_agent_v3.md
```

It specifies the 100 base states, six matched prompt buckets, 600 total prompts, programmatic generation, frozen dataset/hash, generic multi-provider runner, deterministic sharding, resumable parallel execution, post-processing, required tables, confidence intervals, plots, and responsibility separation between the working agent and execution workers.

This is the file to give to the coding/working agent.

---

## 28. Recommended starting point for the next chat

When the coding agents return, begin by checking:

1. Did the generator actually produce 100 base states and 600 paired prompts?
2. Are the six versions of each `state_id` identical except for the intended social condition?
3. Are there any leaks such as `controller` / `partner/controller`?
4. Are the histories compact and internally consistent?
5. Is the controller message fact-free and identical in substantive content across buckets?
6. Are truth and incorrect targets balanced 50/50?
7. Can the runner execute providers independently and resume?
8. What models were actually run?
9. Inspect the main controllability table.
10. Inspect aligned vs adversarial adoption separately.
11. Only after that decide which microscopic social rule should be frozen for the next full population experiment.

The next scientific decision should be based on those results, not on further prompt speculation.
