# ASTRA Plan 1 — Improving Adaptive Game

**Status:** implementation plan and launch gate  
**Date:** 7 September 2026  
**Scope:** repair the adaptive blackboard controller, add a causal response estimator, and measure how communication resources become population response.

## Goal

Use the new MuSR task to test a controller whose decisions are based on meaningful information, whose truthful reports may be repeated when useful, and whose effect can be measured against silence.

This work has two implementation phases:

- **Phase 1:** Section 1 — improve the adaptive game and controller.
- **Phase 2:** Sections 2 and 3 — add causal-response and communication-efficiency analysis.

Do not launch the full study until both phases pass their tests and a small matched smoke experiment succeeds.

---

# Section 1 — Improving Adaptive Game

## 1.1 Use the calibrated task

Use **MuSR task 003, candidate 130** as the main follow-up task.

Reason:

- full-evidence correctness was 6/6 in the isolated OSS evaluation;
- private evidence left substantial uncertainty;
- the strategic true reports produced a strong false-target response at budgets 6 and 9.

Task 001 may be retained as a backup. Task 002 is not the first choice because matched random reports often moved agents toward the false target more strongly than the strategic reports.

This task choice was informed by `results/musr_truthful_selective_isolated_oss_01_analysis.zip`. The new population study is therefore a **follow-up test**, not independent confirmation that task 003 is effective.

The ZIP is consistent with the implementation prepared by:

- `docs/tdd/features/games/MUSR/TaskPreparation/20260907_MuSR_01_Local_Preparation.md`
- `docs/tdd/features/games/MUSR/TaskPreparation/20260907_MuSR_02_Cluster_Launch.md`

It contains the expected model, candidates, budgets, answer orders, random controls, and all 10,818 completed logical requests. The reduced ZIP does not contain enough raw runtime material to prove every cluster-launch step independently.

## 1.2 Correct the controller’s information timing

The current controller has two separate decisions:

1. sampled votes determine **whether** it acts;
2. live blackboard contents largely determine **how** it acts.

At dawn, one-round messages from the previous round are already unavailable and current-round agents have not posted. The adaptive mode chooser therefore normally sees an empty board.

Change the round order so that the controller receives a read-only snapshot of the previous day’s board **before expiration**. Then:

1. sample current votes;
2. capture the previous-board snapshot;
3. decide whether to act;
4. if acting, choose REPORT, REQUEST, or DIRECTIVE from both inputs;
5. expire old public messages before daytime delivery;
6. publish the controller’s new messages;
7. run the agents’ daytime updates.

Old messages must inform the controller without being delivered for an extra round.

## 1.3 Add optional LLM-selected communication

Keep the existing coded stochastic policy for the binary intervention gate. It continues to sample $U$ and record the exact probability $e_t$ without modification. When $U=0$, post nothing and make no controller LLM call. When $U=1$, make one LLM call to select REPORT, REQUEST, or DIRECTIVE and its permitted content.

The LLM controller input contains only:

- current sampled votes and answer counts;
- previous-day message types and contents allowed by the controller’s information rules;
- the target allocation;
- the configured budget;
- eligible canonical fact IDs and their canonical text;
- its own public posting history;
- the current round.

Exposure, acquisition, and retention are researcher diagnostics, not automatically controller-visible information. If a controller variant receives any of them as feedback, treat that as an explicit expansion of its sensing access. Configure, label, and record that variant separately rather than silently adding those fields to the ordinary controller view.

Private agent reasoning, private evidence, acquisitions, retention, and other researcher-only diagnostics remain hidden. Use the configured gameplay provider for the controller call.

Use this controller instruction:

> Your objective is to increase support for the target allocation through truthful communication. Review the sampled votes and previous public discussion. Choose REPORT to supply useful verified evidence, REQUEST to ask participants for relevant evidence, or DIRECTIVE to focus a comparison. Choose what best serves the target in the present context. You may select only supplied eligible fact IDs; never invent or alter verified evidence. Return the required structured decision.

For REPORT, the LLM may choose between 1 and $b$ distinct eligible fact IDs. Code renders their canonical text; the LLM may not invent or rewrite evidence. REQUEST and DIRECTIVE each create one message through the existing structured-response and validation contract. Keep the ordinary public controller identity `Agent 25`.

Use the existing bounded structured-response retries. If they fail, invoke a documented algorithmic fallback under exactly the same visibility and fact-eligibility restrictions. The existing algorithmic chooser remains available as a normal configured controller option for comparison.

The fallback uses its own reproducible random stream, derived separately from the episode and round. It must never consume or perturb the random stream used to sample $U$. Record the fallback seed or stream identity and the selected fallback result.

Record the exact controller-visible prompt/input, structured output, eligible pool, chosen mode, selected fact IDs, actual posts, retry outcomes, fallback status, fallback random-stream identity, action probability, and controller token usage. Keep researcher-only diagnostics in separate fields.

Retain $U$ as the originally assigned binary action even when the LLM decision, validation, fallback, or delivery fails. Record decision validity and realized delivery separately.

## 1.4 Allow truthful repetition

Remove the rule that permanently forbids reuse of a selected controller fact. Enforce eligibility in code, not in the LLM.

A true fact may be reposted when, for example:

- target support remains low;
- too few agents previously received it;
- it was seen but not acquired or retained;
- its earlier influence has decayed.

Use these initial limits:

- at most three total posts per fact per episode;
- one complete intervening round before the same fact may be reused;
- distinct fact IDs within one REPORT action.

Record fact identity, prior post count, reason for reuse, and realized delivery. A repeated post remains a real communication cost even if it adds no new evidence.

## 1.5 Separate communication failures

For each controller action, preserve enough data to distinguish:

1. **selection:** what the controller chose;
2. **posting:** what reached the board;
3. **exposure:** which updates sampled it;
4. **acquisition:** whether it added or reactivated evidence;
5. **response:** whether votes moved toward the target;
6. **persistence:** whether the response survived later rounds.

This separates poor evidence, poor delivery, lack of persuasion, and later recovery.

## 1.6 Phase 1 tests and acceptance gate

Add focused tests for:

- previous-day messages are visible to the chooser before expiration;
- old messages are not delivered for a second day;
- current votes can change mode/content selection;
- no controller LLM call occurs when $U=0$;
- exactly one controller LLM decision call occurs when $U=1$, apart from bounded schema retries;
- private evidence and private prose never enter the controller view;
- acquisitions, retention, and researcher-only diagnostics never enter the ordinary controller view;
- REPORT output can select only eligible supplied fact IDs and canonical rendering cannot be altered;
- truthful facts can repeat within configured limits;
- repetition state survives resume and is deterministic under the same seed;
- fallback randomness is reproducible and does not change binary intervention samples;
- failed decision or delivery does not rewrite the assigned value of $U$;
- controller prompt, response, retries, fallback, and token usage are retained;
- zero-post REPORT rounds and realized post counts are recorded correctly;
- existing non-adaptive controllers remain unchanged unless explicitly configured.

Before a full launch, run a small matched experiment from identical complete saved population states:

- silence;
- current adaptive controller;
- repaired adaptive controller.

Use task 003 and keep the gameplay model, participant prompt, decoding settings, and complete starting snapshot identical within each comparison. Change only the intended controller configuration. Compare matched actual post counts descriptively where possible. Measure immediate response and response after two and three rounds. Include a condition where later controller actions are disabled, which isolates persistence of the first intervention.

---

# Section 2 — Propensity-Weighted Causal Response

## 2.1 Add a new estimator family

For round $t$, define:

$$
\widehat{R}_{t,h}=
\left[
\frac{U_t}{e_t}-\frac{1-U_t}{1-e_t}
\right]
\left(x_{t+h}-x_t\right),
\qquad
e_t=P(U_t=1\mid I_t),
$$

where:

- $U_t=1$ means the controller intervened and $U_t=0$ means silence;
- $I_t$ is the complete information visible to the controller when it makes the action decision;
- $e_t$ is the exact recorded probability of intervention conditional on $I_t$;
- $x_t$ is the population share supporting the controller’s target;
- $h=1,2,3$ gives immediate and delayed response.

If the action policy uses previous-board history as well as sampled votes, that history is part of $I_t$. The logged $e_t$ must be the probability produced by that exact policy from that exact controller-visible information—not a probability reconstructed from sampled votes alone.

Averaging this quantity estimates **activation of the LLM communication policy, including its fallback, versus silence** over histories encountered by the randomized binary policy, under the implemented randomization assumptions.

Give this estimator a separate name such as **propensity-weighted causal response**. Do not silently replace susceptibility $\chi$, information-response efficiency, $T_\pi$, or thermodynamic efficiency.

## 2.2 Required outputs

Produce:

- one round-level estimator-input table;
- one cell-level effect table;
- immediate, two-round, and three-round estimates;
- confidence intervals based on complete episodes and shared-initialization blocks;
- support diagnostics for the distribution of $e_t$;
- missing-lag counts and incomplete-episode counts;
- an effect-versus-lag plot.

Estimate causal intervention-versus-silence effects where support permits by pre-intervention or configured variables such as:

- truthful versus false target arm;
- controller type;
- nominal budget;
- pre-intervention round or state summaries declared before analysis.

Do not publish unstable fine strata with little action/silence overlap. Show their support diagnostics instead.

Do **not** directly stratify this causal estimator by chosen communication mode or actual posts. They are outcomes that exist only after intervention, while silence has no corresponding mode or positive post count. Mode-specific causal comparisons require the joint assignment probability for action and mode, or a separate experiment that randomizes mode explicitly. Mode and realized-post breakdowns remain descriptive in Section 3.

## 2.3 Existing-data implementation

The required fields already survive in the canonical round data:

- `U_k` or its canonical $U_t$ equivalent;
- `P_U1_given_Y` or the canonical intervention probability;
- target share before and after a round;
- episode and round identities;
- initialization repetition and physical-state hash.

This means existing studies can be reaggregated without new model calls. The implementation must verify probability bounds, round ordering, target orientation, and lag availability before estimation.

## 2.4 Section 2 tests

Test that:

- a constructed randomized example recovers a known intervention effect;
- action and silence weights use the recorded probability correctly;
- truth and false arms use their own target orientation;
- lagged outcomes never cross episode boundaries;
- shared-initialization blocks remain intact during resampling;
- missing rounds are reported rather than silently treated as zero response;
- estimates are unchanged by row order and execution sharding.

---

# Section 3 — Communication-to-Response Efficiency

## 3.1 Build a round-level communication funnel

Create one analysis table linking:

```text
controller action
  -> actual posts
  -> message exposures
  -> unique readers in the round
  -> new or reactivated evidence
  -> vote response
  -> delayed response
```

Include nominal budget, communication mode, target arm, episode, round, cell, and shared-initialization identifiers.

## 3.2 Add operational efficiency measures

Report response against:

- actual controller posts;
- exposures;
- unique readers per round;
- newly acquired controller facts;
- reactivated controller facts.

Useful descriptive ratios include:

- exposures per post;
- unique readers per post;
- new acquisitions per exposure;
- expected response divided by expected communication cost for a configured policy;
- persistent expected response divided by expected communication cost for a configured policy.

Ratios with zero denominators must be missing and explicitly counted, not forced to zero.

Do **not** divide each round’s propensity-weighted response by that round’s actual posts. Silent rounds have zero posts, and removing them destroys the intervention-versus-silence comparison. Estimate causal response and expected communication cost separately for each configured policy, then form a clearly named aggregate ratio. Treat response versus cost as the primary result and aggregate ratios as secondary summaries.

Keep sensing cost separate from posting cost. Do not assign an arbitrary conversion between them.

## 3.3 Response–cost frontier

Create plots showing:

- response versus nominal budget;
- response versus actual posts;
- response versus exposures;
- response versus new evidence acquired;
- immediate and delayed response on the same cost scale;
- separate truthful and false target arms;
- descriptive controller-mode overlays where statistical support is adequate and their non-causal status is explicit.

Add a **response–cost frontier**, meaning the best observed causal response achieved at each communication cost. Include uncertainty and the number of contributing initialization blocks.

The frontier is an operational control-efficiency result. Do not label it thermodynamic efficiency.

## 3.4 What existing canonical data supports

Existing canonical Parquet tables are sufficient for:

- actual controller post counts and post IDs;
- exposure counts and sampled controller message IDs;
- per-round unique-reader counts;
- aggregate new and reactivated evidence counts;
- individual vote-change counts from before/after micro-slot opinions;
- $U_t$, intervention probability, and current/lagged target shares;
- episode and shared-initialization matching.

Therefore, most Section 3 tables and plots require only reaggregation.

## 3.5 Recording gaps for future runs

Add future recording for:

- controller/public-message token counts;
- focal agent identity in compact micro-slot records;
- stable message and fact identities linked to each reader across rounds;
- whether a reader saw the same message or fact previously.

Without these fields, canonical compact packages cannot recover:

- true episode-wide unique readers;
- repeated exposure of the same evidence to the same agent across rounds;
- controller-message-specific token cost.

Do not reconstruct token cost from provider input/output token totals. Those totals include more than public controller communication.

## 3.6 Update the aggregation contract

Update:

`docs/documentation/metrics/study_aggregation_contract.md`

to document:

- the new causal-response estimator family;
- the communication-funnel and response–cost outputs;
- the exact retained fields used by each output;
- which reader/token questions require richer future recording;
- block-based uncertainty and missing-data behavior.

Also reconcile the document with the current implementation: current canonical analysis output uses Parquet, and analysis compaction exists. Code and tests are authoritative if the document conflicts with them.

## 3.7 Phase 2 tests and acceptance gate

Add focused tests proving that:

- existing canonical Parquet packages can produce Sections 2 and 3 without run trees;
- communication counts agree with their source round/micro-slot records;
- per-round unique-reader counts are not mislabeled as episode-wide unique readers;
- new evidence and reactivation remain separate;
- response-cost ratios handle zero cost and missing outcomes correctly;
- causal estimates are not conditioned on realized mode or post count without a valid joint randomization design;
- plots mask unsupported estimates rather than presenting zeros;
- reaggregation is deterministic and makes no provider calls;
- the final ZIP contains the new tables, plots, report sections, provenance, and validation results.

---

# Section 4 — Theoretical Scope and Compatibility

## 4.1 Preserve existing definitions

Preserve the existing binary action $U$, target-share coordinate $x$, transition-information quantity $T_\pi$, susceptibility $\chi$, and information–response efficiency $\eta_{\mathrm{IR}}$.

For the blackboard, define empirical action-conditioned transitions as

$$
Q_u(x'\mid s),
$$

where $s$ is an explicitly stated pre-intervention conditioning state. Every comparison must say exactly what $s$ contains.

Do not assume:

$$
Q_0=I
\qquad\text{or}\qquad
Q_1=K_{h,\gamma}^{b}.
$$

Silence does not freeze the population: autonomous agent and peer dynamics continue when $U=0$. In the blackboard game, $b$ is a message allowance, not a number of controlled microscopic updates.

## 4.2 Preserve the general information–response bound

The general bound remains

$$
T_\pi(s)\geq
\frac{2a_s(1-a_s)}{\ln 2}\chi(s)^2,
\qquad
a_s=P(U=1\mid s),
$$

with

$$
\chi(s)=
\mathbb{E}[x'\mid U=1,s]
-
\mathbb{E}[x'\mid U=0,s].
$$

Compute $T_\pi(s)$, $a_s$, and $\chi(s)$ from the same empirical distributions, conditioning state, support, and weighting. Do not substitute the new propensity-weighted causal response into the existing $\eta_{\mathrm{IR}}$ formula while retaining an observational $T_\pi$ computed under different conditioning or weights.

## 4.3 Limit thermodynamic claims

Retain the single-affinity model as a reference only under its stated assumptions. Do not fit new constants merely to force agreement with blackboard results. Do not extend the thermodynamic-efficiency interpretation to the blackboard without a justified derivation.

No new blackboard thermodynamic theory is required before implementing Sections 2 and 3. Their primary claims are causal response, delivery, persistence, and operational communication cost.

---

# Implementation sequence

## Phase 1 — Game and controller

1. Freeze task 003 inputs and record their hashes.
2. Add the previous-board snapshot at the correct point in the round lifecycle.
3. Extend the adaptive controller input and decision policy.
4. Add controlled truthful repetition.
5. Extend recording for controller inputs, repetition, tokens, and reader/message identity.
6. Add focused unit and integration tests.
7. Run the matched local/mock smoke with the same gameplay model, prompt, decoding settings, and complete starting snapshots across controller conditions, then inspect its records.

## Phase 2 — Analysis and aggregation

1. Add propensity-weighted causal response and support diagnostics.
2. Add lagged response and block-based confidence intervals.
3. Add the communication-funnel table and efficiency measures.
4. Add response–cost and persistence plots.
5. Update the aggregation contract and final analysis report.
6. Reaggregate one existing retained study as a backward-compatibility test.
7. Validate final ZIP contents and reproducibility.

---

# Launch preparation

After both phases pass:

1. Create a small scientific study config using task 003.
2. Include silence, current-controller, and repaired-controller comparisons from matched initializations.
3. Preflight the complete grid and report episodes, calls, tokens, cost assumptions, concurrency, rate limits, and expected wall time.
4. Run a small live smoke on the approved cluster environment.
5. Verify that matched smoke conditions use the same gameplay model, prompt, decoding settings, and complete starting snapshots, changing only the intended controller configuration.
6. Inspect actual posts, exposures, acquisitions, vote response, persistence, retries, and resume behavior.
7. Freeze the full study manifest before examining full-study outcomes.
8. Submit through the generic study launcher; do not create a study-specific SLURM job.
9. Aggregate strictly and package the canonical Parquet tables, new metrics, plots, reports, and provenance.

## Final launch gate

Do not launch the full study unless all of the following hold:

- the controller sees the intended previous-board snapshot and current votes;
- controller-visible inputs are separated from researcher-only delivery diagnostics;
- old messages are not delivered twice;
- truthful repetition is bounded and recorded;
- the action probability matches the implemented policy conditional on the complete controller-visible information;
- action and silence both have empirical support;
- causal estimates are not conditioned on post-intervention mode or realized post count;
- response and communication cost are estimated separately before any aggregate efficiency ratio is formed;
- existing information–response quantities use internally consistent distributions, conditioning, and weighting;
- causal-response and communication-funnel calculations pass constructed tests;
- matched initialization identifiers are retained;
- the smoke run resumes safely and produces a complete analysis package;
- the result destination and provider limits pass the standard study preflight.
