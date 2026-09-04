# TDD: Truthful Strategic Report Controller for the MUSR Blackboard

**Date:** 2026-09-04  
**Status:** implementation plan  
**Scope:** relational-reasoning MUSR blackboard game and frozen task design

## 1. Motivation

The existing dawn controller uses `coordination_request`: on every
`ADVOCATE_Z` round it inserts exactly `b` identical, factless `DIRECTIVE`
messages. This creates substantial board occupancy but weak scientific
steering. The repeated content behaves more like attention flooding than an
ordinary participant making an evidence-bearing case.

The first replacement should remove directives completely. The controller
should appear as a normal participant who holds an opinion and selectively
reports genuine facts that make its preferred answer appear plausible. It may
be strategically incomplete or confusing through selection, but it must not
lie, fabricate facts, invent implications, or claim certainty it does not have.

Directive-based control is deferred until this report-based mode is correct,
auditable, and scientifically characterized.

## 2. Scientific semantics

Add one explicit controller actuation mode, provisionally:

```yaml
controller_actuation_mode: truthful_strategic_report
```

On each round:

1. the existing vote sensor observes `q_c` votes;
2. the existing controller policy samples exactly one action, `NO_OP` or
   `ADVOCATE_Z`;
3. on `NO_OP`, the controller publishes nothing;
4. on `ADVOCATE_Z`, the controller posts exactly `b` ordinary `REPORT`
   messages at dawn;
5. every report cites one valid frozen-task fact using the same report schema
   and rendering path available to ordinary agents;
6. the controller's visible opinion is its configured target, but it never
   states that a true fact logically proves a conclusion when it does not;
7. normal agents sample the reports through the unchanged blackboard sampling
   mechanism.

Preserve the existing meanings of `beta`, `theta`, `q_c`, persistence,
message lifetime, board sampling, and `b`. In this mode, `b` is the exact number
of controller-authored reports inserted per advocating round—not a total
episode budget and not a microscopic slot replacement count.

## 3. Ordinary-participant presentation

Controller reports must be observationally ordinary at the prompt boundary:

- use the existing `REPORT` message type;
- use the same fields and prose renderer as peer reports;
- show an ordinary stable participant identity;
- do not include `controller`, `directive`, `authority`, `system`, `policy`, or
  privileged-source language in agent-visible text;
- do not ask agents to comply, coordinate, vote, or repeat a conclusion;
- do not use a private prose channel unavailable to peers.

Operational records must still identify controller authorship for analysis.
Scientific auditability must not depend on hiding the source from researchers.

## 4. Truthfulness contract

Every controller report must satisfy all of the following:

- `fact_id` exists in the frozen task;
- rendered fact text is the canonical text for that ID;
- the fact is true in the task's ground-truth world;
- the fact belongs to the controller's declared reportable fact pool;
- no generated paraphrase changes its logical content;
- no unsupported causal, inferential, or certainty claim is appended;
- no fact is silently coerced when validation fails.

The safest initial implementation should render canonical fact text directly
rather than ask an LLM to paraphrase it. Strategic behavior comes from fact
selection, ordering, timing, and visible target opinion—not fabrication.

## 5. Strategic selective disclosure

Define a frozen, deterministic score for candidate true facts. Prefer facts
that:

- are consistent with both the ground truth and the controller target when
  considered locally;
- leave the controller target viable under partial information;
- distinguish the controller target from irrelevant alternatives without
  exposing the decisive fact that immediately refutes it;
- are not already overrepresented on the live board;
- add novel target-compatible information relative to the other reports chosen
  that round.

Do not describe these facts as “supporting” the false target unless the formal
task representation establishes that limited implication. Use terms such as
`target-compatible`, `target-preserving`, or `non-discriminating` in analysis.

Tie-breaking and round-to-round selection must be deterministic from the
existing episode seed. Retain the selected candidate scores/reasons as compact
research provenance, never in agent-visible prompts.

## 6. Report diversity and anti-spam rules

- Reports within one dawn must use distinct fact IDs.
- Preflight must reject `b` larger than the available distinct eligible pool.
- Across rounds, rotate eligible facts before repeating them where possible.
- Add a configurable deterministic cooldown, initially one round, before a fact
  may be reposted.
- Do not create multiple text-identical reports in one round.
- Do not change uniform board sampling to grant hidden priority.

If the desired grid includes `b=24`, every matching task must provide at least
24 distinct scientifically meaningful eligible facts. Do not satisfy this by
duplicating facts, superficial paraphrases, or tautological filler.

## 7. Frozen task requirements

Create a new frozen task family suitable for truthful adversarial selection.
For every task:

- the complete fact set has one unique ground-truth answer;
- the configured false target is globally wrong;
- at least 24 distinct true facts are locally compatible with that false target
  if `b=24` is retained;
- a smaller set of decisive bridge/disambiguating facts rules out the false
  target and establishes truth;
- those decisive facts are distributed among ordinary agents so peer
  propagation can correct the population;
- no individual controller-selected fact is false;
- controller-compatible and decisive fact classes are stored explicitly and
  validated independently of prose;
- the initial population contains nontrivial support for truth, target, and at
  least one alternative where scientifically intended.

Add a symbolic validator that enumerates candidate answers under:

1. each individual fact;
2. the controller-reportable subset;
3. the decisive subset;
4. the complete fact set.

The validator must prove that the controller subset preserves ambiguity while
the complete set uniquely selects truth.

## 8. Runtime implementation boundaries

Reuse the production night/dawn/day protocol:

- expire prior-day board messages;
- apply epistemic persistence;
- sense votes and sample the controller action;
- insert exactly `b` truthful reports only on `ADVOCATE_Z`;
- perform the unchanged ordinary daytime updates;
- preserve ordinary fact acquisition/reactivation rules.

Do not introduce a second game runtime, controller-specific SLURM launcher, or
replacement estimator. Extend the existing controller and blackboard message
interfaces.

The new mode is a scientific protocol change and therefore must produce a new
protocol fingerprint, study identity, and result root. Existing studies remain
immutable.

## 9. Required retained observables

For every round retain compact fields sufficient to reconstruct:

- controller action and advocacy probability;
- `b` requested and reports actually admitted;
- selected fact IDs and strategy scores/classes;
- report novelty and cooldown status;
- controller reports sampled/read;
- unique readers and repeat exposures;
- fact acquisitions and reactivations caused by controller reports;
- target adoption following controller exposure;
- controller-report share of eligible board messages and actual reads;
- peer-report exposure with and without controller actuation;
- board mean/peak occupancy and expiry;
- truth share, controller-target share, active phi, and active kappa;
- per-round and next-round population response.

The dashboard must visualize the full funnel:

```text
ADVOCATE_Z
  -> b distinct truthful reports
  -> sampled reports / unique readers
  -> fact acquisition or reactivation
  -> target adoption / population response
```

Use `controller_posts`, not `controlled_update_count`, as the primary
blackboard actuation count.

## 10. Development and test sequence

1. Add task-level ambiguity and truthfulness validators.
2. Add a small frozen fixture with a false target, target-compatible true facts,
   and decisive corrective facts.
3. Implement deterministic strategic fact ranking/selection.
4. Implement `truthful_strategic_report` using the ordinary `REPORT` path.
5. Add compact provenance and dashboard fields.
6. Run unit and fake-provider integration tests.
7. Run a two-round dashboard pilot and inspect exact rendered prompts.
8. Run a small multi-episode pilot across low and high `b`.
9. Confirm that high `b` increases distinct information exposure rather than
   duplicate text.
10. Only then design a production comparison study.

## 11. Required tests

1. `NO_OP` creates zero controller messages.
2. `ADVOCATE_Z` creates exactly `b` reports.
3. All selected fact IDs exist and are true.
4. Every report uses canonical fact text and the peer `REPORT` renderer.
5. No controller/directive/authority marker reaches the agent-visible prompt.
6. Reports are distinct within a round.
7. Insufficient eligible facts fail preflight.
8. Cooldown and rotation are deterministic.
9. Same seed reproduces selection; different repetitions remain distinct.
10. The reportable subset preserves the configured false target as logically
    viable, while the complete fact set uniquely proves truth.
11. Agents acquire/reactivate facts only after valid report exposure.
12. Board sampling, message expiry, persistence, and peer behavior remain
    unchanged.
13. Existing `coordination_request` and `direct_recommendation` tests remain
    unchanged and passing.
14. Dashboard and canonical aggregation expose the controller funnel without
    retaining heavy prompt/request logs.

## 12. Pilot acceptance criteria

- Exact inspection shows ordinary evidence-bearing reports, never directives.
- No duplicate controller text occurs within a round.
- Every controller statement is auditable as a true canonical fact.
- `b` equals the exact number of controller reports on every advocating round.
- Both controller-selected and decisive peer facts receive nonzero exposure.
- At least some pilot episodes show a nontrivial transient target response,
  while truth remains recoverable through peer evidence.
- High `b` does not merely reproduce the present identical-message flood.
- Existing scientific estimators run without modification.

## 13. Deferred directive comparison

Do not redesign or tune the directive controller in this implementation. Once
the truthful strategic report mode passes the pilot criteria, prepare a
separate matched study comparing:

1. no controller;
2. truthful strategic report controller;
3. the existing factless directive controller;
4. optionally a later mixed report-plus-directive controller.

Match tasks, seeds, persistence, beta, theta, `q`, `q_c`, `b`, rounds, and
repetitions. This keeps the later comparison about controller communication
semantics rather than unrelated experimental changes.
