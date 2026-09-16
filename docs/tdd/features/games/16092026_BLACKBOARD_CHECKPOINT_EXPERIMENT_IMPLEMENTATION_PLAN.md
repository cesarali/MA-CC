# Blackboard checkpoint experiment: two-phase implementation plan

Date: 2026-09-16  
Status: proposed implementation plan; no authorization for paid execution.

Companion scientific specification:
`docs/tdd/features/games/16092026_BLACKBOARD_CHECKPOINT_EXPERIMENT_SPEC.md`.

## 1. Goal

Implement the checkpoint experiment without duplicating the blackboard game,
the generic study scheduler, or the existing information-estimator engine.

Each parent population runs for `L=2` uncontrolled preparation rounds. Its
complete round-boundary state is then reused by nine continuation branches,
each of which runs for `M=10` further population rounds:

1. `none`;
2. `always_truth`, `b=3`;
3. `always_truth`, `b=12`;
4. `always_false`, `b=3`;
5. `always_false`, `b=12`;
6. `sensing_truth`, `b=3`;
7. `sensing_truth`, `b=12`;
8. `sensing_false`, `b=3`;
9. `sensing_false`, `b=12`.

The implementation is divided into two phases. Phase 1 establishes that the
alternate futures are mechanically valid. Phase 2 makes the design runnable,
analyzable, and costable as a complete study.

## 2. Cross-phase invariants

- The parent checkpoint is taken after preparation round `L` and before the
  next round's expiry and evidence-persistence operations.
- All nine children of a parent start from the exact same immutable checkpoint
  hash.
- The no-control continuation exists once per parent/copy and is reused in
  analysis; it is not simulated separately for each posting budget.
- Continuations share their starting state but use distinct documented future
  random streams.
- Absolute population round `r` and post-branch horizon `h` are separate.
  With `L=2`, checkpoint `h=0` is after absolute round 2; continuation rounds
  have `h=1..10` and absolute rounds 3..12.
- Scientific identities never depend on SLURM task IDs or shard boundaries.
- The existing MI/CMI, bootstrap, null, and support engines remain
  authoritative. No replacement CMI implementation is added.
- No study-specific SLURM job file is introduced. Execution uses the generic
  study launcher and shared provider load control.
- A real provider pilot or main run requires separate explicit authorization.

## 3. Phase 1 — forkable checkpoint and continuation engine

### Outcome

At the end of Phase 1, a deterministic or mock-provider population can run for
two rounds, save one immutable checkpoint, restore it into all nine branches,
and run each branch for ten additional rounds with correct boundary timing and
identity metadata.

Phase 1 is complete only when save/restore continuation is demonstrably
equivalent to the corresponding uninterrupted mock execution.

### 3.1 Define a versioned parent-checkpoint contract

Add a dedicated immutable checkpoint artifact rather than repurposing the
single replaceable episode-resume checkpoint. The artifact must contain or
bind:

- serialized `RelationalGameState`;
- individual votes and semantic answer IDs;
- active and historical evidence, fact provenance, and agent memory;
- complete blackboard history, message authorship/order, creation round, and
  expiry round;
- absolute round and micro-update boundary;
- task, initialization, prompt, model, and resolved preparation-config hashes;
- parent ID, checkpoint ID, and canonical checkpoint hash;
- parent random-stream provenance;
- controller history, explicitly empty for uncontrolled preparation;
- any runtime-local state needed to reproduce later report eligibility,
  cooldown, or adaptive communication behavior.

The checkpoint must be content-addressed, written atomically, and never
overwritten by a child branch.

### 3.2 Add validated state restoration

Add a round-trip loader for the relational state. Restoration must validate:

- schema version;
- task and initialization hashes;
- population and agent identities;
- semantic answer alphabet;
- board-message integrity and ages;
- active facts being a subset of historically known facts;
- checkpoint hash;
- preparation coordinates `q` and `rho`.

Changing controller policy, target, or posting budget is allowed at the branch
boundary. Changing the prepared population dynamics, task, `q`, or `rho` is
not allowed.

### 3.3 Refactor the runtime into preparation and continuation paths

Extend the existing relational blackboard runtime so it can either initialize
a new episode or accept a restored checkpoint plus:

- `start_round`;
- continuation length `M`;
- branch policy and target;
- continuation copy ID;
- continuation seed/stream identity.

The first continuation iteration must use absolute round `L+1`. The normal
message-expiry and evidence-persistence work for that round must execute once,
not during checkpoint creation and not twice after restoration.

Keep provider-failure recovery separate from scientific branching. A branch
may still use the existing recovery mechanism after it starts.

### 3.4 Define deterministic branch identities and random streams

Derive a continuation stream from the full scientific identity:

```text
(q, rho, parent_id, policy, posting_budget, copy_id)
```

Record the derivation version and resolved seed. The `none` branch has a
canonical budget coordinate of zero/none and must not acquire separate stream
identities merely because it is compared with two budgets.

The implementation must not claim replayability of remote LLM responses.
Deterministic equivalence tests use a deterministic/mock provider.

### 3.5 Add a parent-bundle worker contract

Represent one parent and all its descendants as one scientific block. The
worker contract is:

```text
prepare parent once
  -> seal immutable checkpoint
  -> execute/resume nine named continuations
  -> seal parent bundle only when expected branch coverage is complete
```

Successful sibling branches must survive a retry of a failed branch. Physical
execution may run branches sequentially or with bounded concurrency, but it
must not change branch seeds or scientific identities.

The generic study planner/worker may be extended with this reusable bundle
topology. Do not create a blackboard-checkpoint-specific `.job` file.

### 3.6 Retain checkpoint and branch provenance

Every continuation observation must contain:

- `parent_id`, `checkpoint_id`, and `checkpoint_hash`;
- `branch_policy`, `posting_budget`, and `copy_id`;
- `absolute_round` and `post_branch_horizon`;
- `q`, `rho`, `N`, `L`, and `M`;
- controller target and correct answer as semantic IDs;
- preparation and continuation seeds;
- branch completion/recovery status.

Retain the `h=0` checkpoint vote vector and option counts in canonical data.

### 3.7 Phase 1 tests

Implement tests before enabling real-provider execution:

1. State serialization round trip preserves the checkpoint hash.
2. An uninterrupted deterministic run and save/restore continuation produce
   identical state and observations under the same continuation stream.
3. All nine branches begin from the same checkpoint hash.
4. Expiry and persistence at the first continuation round occur exactly once.
5. `always` records `U=1, e=1`; `none` records `U=0, e=0`; sensing preserves
   the existing logistic gate.
6. The controller target resolves to `ALLOCATION_0` for truth and
   `ALLOCATION_2` for false under the frozen task hash.
7. Posting budget is enforced as a maximum, not a required post count.
8. The no-control branch is generated once.
9. Distinct branch/copy identities produce distinct continuation streams.
10. A failed branch resumes without regenerating the parent or completed
    siblings.

### Phase 1 acceptance gate

- All Phase 1 tests pass locally with no credentials.
- A downscaled parent with all nine mock branches seals successfully.
- Checkpoint and child hashes are inspectable in machine-readable artifacts.
- Canonical retained fields survive the intended compact artifact profile.
- No provider call, SLURM submission, or paid execution is needed for this
  gate.

## 4. Phase 2 — study integration, paired analysis, and pilot readiness

### Outcome

At the end of Phase 2, the repository contains resolved scientific configs,
study and analysis manifests, strict validation, paired estimators, figures,
and a credential-free costed execution plan. If separately authorized, the
same implementation can run the 10-parent pilot and use its measurements to
freeze the main protocol.

### 4.1 Add typed experiment configuration

Add only the schema fields required to express the ensemble design. The
resolved configuration must distinguish:

- parent count `K`;
- preparation rounds `L`;
- continuation rounds `M`;
- continuation copies `B`;
- explicit branch policy list;
- parent/checkpoint retention policy;
- branch completeness requirements.

Map the remaining notation to existing keys:

| Quantity | Configuration |
|---|---|
| `N` | `game.population_size` |
| `q` | `game.options.social_group_size` |
| `rho` | `game.options.epistemic_persistence` |
| `b` | `control.options.intervention_budget` |
| `s` | `control.options.sensor_sample_size` |
| `theta` | `control.options.threshold` |
| `beta` | `control.options.beta` |

Do not encode the scientific design in shell scripts.

### 4.2 Create the study folder

Create one checkpoint-study folder containing:

- ordinary experiment YAML configuration(s);
- `study.yaml` with stable config order and generic `execution.mode: auto`;
- `analysis.yaml` containing estimators, parent-block resampling, diagnostics,
  and plots;
- a notation-to-schema document;
- README/preflight handoff material.

The full design has four prepared settings:

```text
(q, rho) in {(3, 0.70), (3, 1.00), (12, 0.70), (12, 1.00)}
```

Within each setting are nine branch conditions. Thus there are 36 branch
conditions, while the independent block remains the parent and all its
descendants.

### 4.3 Extend canonical validation

Strict aggregation must validate:

- expected/found parents and checkpoints;
- exactly one shared baseline per parent/copy;
- expected/found branch coverage;
- identical checkpoint hashes across siblings;
- unique continuation stream identities;
- complete `h=0..10` trajectories;
- semantic target consistency;
- absence of duplicate baselines in pooled data;
- failures, interruptions, and resumed branches;
- required round, micro-slot, sensor, exposure, evidence, token, and cost
  fields.

Missing branches are never replaced with zeros. Partial output requires the
existing explicit incomplete-analysis path and remains visibly provisional.

### 4.4 Implement paired response analysis

Build a canonical parent-by-branch endpoint table. For each fixed `(q,rho)`
setting, policy, budget, target, and horizon, compute the controlled-minus-none
paired difference and average it over parents.

Primary horizons are:

- `h=1`: immediate full-population-round response;
- `h=10`: longitudinal endpoint effect.

Also retain trajectories for `h=0..10`. Report correct-answer support and
false-target support separately in fractions and percentage points.

Bootstrap complete parents, keeping every branch, copy, and horizon together.
The default is 2,000 parent-block bootstrap replicates. Report effective `K`
and missingness for every comparison.

### 4.5 Implement assigned-policy information analysis

For each controlled branch versus its paired `none` branch, construct a
balanced label `A` and estimate:

```text
I(A; m_h | n_0)
```

Adapt these observations to the existing conditional-information estimator.
Do not implement a second CMI engine. Report unsmoothed and specified
uniform-smoothing sensitivity variants with support diagnostics.

Add the cross-fitted classifier score as a separate estimator procedure:

- five outer folds grouped by parent;
- grouped inner validation;
- constant `1/2` classifier candidate;
- preregistered low-order features;
- recorded penalties, clipping, calibration, and held-out log loss;
- repeated grouped splits as sensitivity, not confidence intervals;
- no clipping of negative held-out scores.

Add a paired whole-record label-swap diagnostic that preserves parent
dependence and refits the classifier. Validate the complete analysis on an
LLM-free model with known information before making recovery claims.

### 4.6 Add figures and resource reporting

Produce, separately by `q` and `rho`:

- truth-support paired effects at `h=1` and `h=10`;
- false-target paired effects at `h=1` and `h=10`;
- `h=0..10` trajectories with parent-block uncertainty;
- assigned-policy information and estimator-sensitivity comparisons;
- actual activation, posts, message reads, retries, tokens, cost, and latency;
- the contrast between `b=3` and `b=12`, without fitting a continuous dose
  curve from two points.

### 4.7 Credential-free preflight

Preflight must report:

- four prepared settings and 36 branch conditions;
- parent, checkpoint, continuation, and total episode counts;
- expected and conservative provider calls;
- input/output token and monetary estimates;
- provider request concurrency and RPM-safe array throttle;
- CPUs, memory, time limit, shard duration, and expected wall time;
- result root and absolute log paths;
- full-design workload and budget-constrained alternative.

For `K=120`, the main design has 480 parents, 4,320 continuations, and
1,059,840 ordinary updates before initialization/controller/retry overhead.
The separate `K=10` pilot has 40 parents, 360 continuations, and 88,320
ordinary updates.

### 4.8 Authorized pilot and protocol freeze

The pilot is an execution step, not automatic completion of the implementation
phase. Run it only after explicit authorization.

The proposed pilot uses `K=10` per `(q,rho)`, `L=2`, `M=10`, and `B=1` with
the complete nine-branch structure. Inspect:

- checkpoint vote distributions and ceiling/floor effects;
- whether `q=12` is near consensus after two preparation rounds;
- actual controller activity, posts, and reads;
- validation failures and missingness;
- latency, requests, tokens, and cost;
- parent-level variability and expected interval width.

If necessary, compare `L=1` descriptively, then freeze one common `L` before
the main ensemble. Pilot results do not authorize outcome-dependent filtering,
sample-size stopping, or omission of difficult branches.

### Phase 2 acceptance gate

- All configs resolve and pass credential-free preflight.
- The downscaled mock study aggregates strictly and produces canonical Parquet
  tables, estimator summaries, diagnostics, plots, reports, and provenance.
- Parent-block bootstrap and classifier fold-leakage tests pass.
- Existing estimator equivalence tests pass; no replacement CMI exists.
- Execution planning uses a generic launcher and provider load control.
- The resource report clearly separates full `K=120`, possible `K=60`, and
  pilot `K=10` designs.
- No paid pilot or main study is submitted without explicit authorization.

## 5. Recommended implementation order

1. Write failing Phase 1 checkpoint round-trip and boundary-timing tests.
2. Implement immutable checkpoint serialization/restoration.
3. Refactor preparation/continuation runtime entry points.
4. Implement the parent-bundle worker and branch-resume contract.
5. Verify compact retained data with a complete nine-branch mock bundle.
6. Add typed ensemble configuration and study planning.
7. Extend canonical validation and parent-child identities.
8. Implement paired response analysis and parent bootstrap.
9. Adapt assigned-policy observations to the existing information engine.
10. Add the grouped classifier, label-swap diagnostic, and known-truth tests.
11. Create scientific configs, analysis recipe, figures, and preflight report.
12. Stop at the pilot-authorization gate.

## 6. Definition of implementation complete

Implementation is complete when both phase acceptance gates pass and a user
can generate a credential-free, fully resolved execution plan for the pilot
and main designs. Completion does not mean that either paid study has run.

The later operational sequence is:

```text
explicit pilot authorization
  -> pilot execution and aggregation
  -> protocol/cost freeze
  -> explicit main-run authorization
  -> main execution and strict aggregation
```

