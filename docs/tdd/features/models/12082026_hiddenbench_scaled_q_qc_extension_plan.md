# HiddenBench Imitation — Scaled Population, Social Group, and Controller-Sensing Extension

## Purpose

Implement the next empirical extension of the HiddenBench imitation game while preserving the current controller semantics and leaving a clean interface for the theoretical feedback-control model to be derived next.

The extension should add:

1. **larger HiddenBench populations using validated paraphrased replication**;
2. a configurable **social interaction group size** \(q\);
3. a configurable **controller sensing sample size** \(q_c\);
4. preservation of the current `threshold_target` and `soft_target` feedback policies;
5. two new post-hoc analysis quantities based on the **truth current and its fluctuations**.

The implementation should remain backward compatible with the current experiment. In particular, the configuration

\[
q=1,\qquad q_c=2
\]

must recover the current interaction/controller protocol.

This document is an **implementation specification for today's empirical extension**. Do not redesign the stochastic-thermodynamic theory or the classical continuous-time model here. The code should, however, expose the variables needed to build that theory cleanly afterward.

---

# 1. Current behavior that must be preserved

The current reasoning-mode episode works as follows.

Initial state:

- all population agents make an initial HiddenBench vote;
- the controller target is resolved, currently typically to the task's `correct_answer`;
- the controller is **external to the population**:
  - it has no vote;
  - it is not counted in \(N_t\);
  - it has no private HiddenBench evidence;
  - it does not directly overwrite an agent action.

For each elementary interaction:

1. select one focal population agent;
2. select one ordinary peer;
3. the controller independently samples `sensor_sample_size = 2` population agents;
4. the controller reads only their current votes;
5. it computes sampled target support;
6. it chooses `ADVOCATE_Z` or `NO_OP`;
7. under `NO_OP`, the ordinary peer exchange occurs;
8. under `ADVOCATE_Z`, the ordinary peer is replaced by a controller-generated peer-style message;
9. only the focal agent may update its committed vote.

The controller therefore has the feedback structure

\[
N_t \longrightarrow Y_t \longrightarrow U_t \longrightarrow X^{f}_{t+1},
\]

where:

- \(N_t\) is the population occupation vector;
- \(Y_t\) is the controller's finite poll;
- \(U_t\in\{\mathrm{ADVOCATE}_Z,\mathrm{NO\_OP}\}\);
- \(X^f_{t+1}\) is the focal agent's next opinion.

The currently implemented policies are:

### `threshold_target`

\[
U_t=\mathrm{ADVOCATE}_Z
\quad	ext{iff}\quad
\hat p_Z(Y_t)<	heta.
\]

### `soft_target`

\[
P(U_t=\mathrm{ADVOCATE}_Z\mid Y_t)
=
\sigma\!\left[eta(	heta-\hat p_Z(Y_t))ight].
\]

For the current configuration:

\[
q_c=2,\qquad 	heta=0.5,\qquad eta=4.
\]

Do not change these policy definitions in this extension.

---

# 2. Separate the two different group-size parameters

The implementation must use two distinct concepts and names.

## 2.1 Social interaction size: \(q\)

Define

```text
q = number of ordinary social influence slots presented to the focal agent
```

Suggested config name:

```yaml
game:
  options:
    social_group_size: 1
```

The current game is the special case \(q=1\).

For \(q>1\), one focal agent is selected and \(q\) distinct ordinary peers are sampled from the remaining \(N-1\) population agents.

Validation:

\[
1\le q\le N-1.
\]

Peers must be sampled **without replacement** within that interaction.

## 2.2 Controller sensing size: \(q_c\)

The already existing

```yaml
control:
  options:
    sensor_sample_size: 2
```

is the controller sensing size:

\[
q_c=	ext{number of population agents polled by the controller}.
\]

Keep this existing config field for backward compatibility.

Validation:

\[
1\le q_c\le N.
\]

The controller sensor sample remains independent from the social-peer sample. Therefore a sensor agent may also be one of the social peers, and the sensor may include the focal agent if the current implementation allows this. These overlaps must be logged, not forbidden.

This separation is scientifically important:

\[
q=	ext{social interaction order},\qquad
q_c=	ext{controller measurement budget}.
\]

---

# 3. Generalized reasoning interaction for \(q>1\)

The generalized interaction must preserve the current semantics that the controller occupies a social influence slot rather than becoming a voting population agent.

## 3.1 `NO_OP`

At one interaction:

1. choose focal agent \(i\);
2. sample \(q\) distinct peers from the population excluding \(i\);
3. run the ordinary communication protocol for those \(q\) peers;
4. provide their peer messages to the focal update;
5. the focal performs one vote-update call;
6. only the focal's committed vote may change.

The \(q=1\) case must reproduce the current peer exchange.

For \(q>1\), implement the smallest natural generalization of the current protocol:

- each of the \(q\) peers contributes one peer message;
- the focal receives all \(q\) peer messages in a deterministic, logged order;
- the focal then performs one vote update;
- do not introduce extra rounds of deliberation inside one interaction.

## 3.2 `ADVOCATE_Z`

Keep **one controller actuation slot** for now. Define \(b=1\), where \(b\) is the number of social slots occupied by the controller when it advocates.

Do **not** expose `b` as an experimental sweep yet unless implementation architecture requires a field. If a field is added, default and validate it as `1`.

When the controller chooses `ADVOCATE_Z`:

1. choose the same focal agent;
2. sample the same \(q\)-peer social context;
3. replace exactly **one** of the \(q\) peer influence slots by the controller;
4. retain the remaining \(q-1\) ordinary peer inputs;
5. generate the current controller peer-style advocacy message;
6. present exactly \(q\) influence inputs total to the focal: \(q-1\) ordinary peer messages and 1 controller message;
7. the focal performs one vote-update call.

Thus the immediate social exposure remains fixed at \(q\) inputs under both actions.

```text
NO_OP:
    [peer_1, peer_2, ..., peer_q]

ADVOCATE_Z:
    [peer_1, ..., controller_Z, ..., peer_q]
```

For \(q=1\):

```text
NO_OP:       [peer]
ADVOCATE_Z:  [controller_Z]
```

which is exactly the current protocol.

For \(q=1\), no additional replacement-slot random draw should be introduced. For \(q>1\), the replaced peer slot may be selected uniformly using the episode-seeded controller/runtime RNG. Log `replaced_peer_id` and `replaced_peer_slot`.

---

# 4. Do not turn the controller into population agent \(N+1\)

The controller must remain an **external feedback device**. It must not receive a HiddenBench private-information assignment, hold a persistent population vote, enter the occupation vector \(N_t\), be selected as the focal population agent, change the denominator used for order parameters, or become part of the paraphrased population.

Its local actuation should *look like a peer input* to the focal LLM, but mathematically the controller remains external.

This gives the tractable feedback decomposition needed later:

\[
P(n',u,y\mid n)
=
S_{q_c}(y\mid n)\,\pi(u\mid y)\,P_q(n'\mid n,u).
\]

Today we only need to expose the ingredients cleanly in the event logs.

---

# 5. Larger populations through paraphrased replication

Use the existing HiddenBench paraphrase/factorization infrastructure rather than implementing a new population-construction mechanism.

Relevant existing workflow:

```text
scripts/local_llms/hiddenbench_population_pipeline/
scripts/generate_semantic_annotations.py
scripts/prepare_hiddenbench.py
scripts/run_information_sufficiency_audit.py
```

The empirical population extension should use:

```text
method = paraphrased_replication
```

and **not** `factorized_evidence` in this experiment.

Reason: paraphrased replication changes population size while preserving the semantic information carried by each evidence type; factorization changes the informational fragmentation of the task and is therefore a separate scientific intervention.

## 5.0 Current local paraphrase status and recommended starting task

The repository currently has enough validated paraphrase material to implement
and exercise this extension on a selected task, but it does **not** have a
complete 65-task paraphrased release.

The relevant local artifact is:

```text
data/hidden_bench/annotations/paraphrases.json
```

Its current contents are:

- tasks 1--42: complete, with 10 accepted paraphrases for every hidden evidence
  type;
- task 43 (`The Artifact Delivery`): partial -- evidence type 0 has 10 accepted
  variants, evidence type 1 has none, and the remaining evidence types were not
  reached;
- tasks 44--65: absent;
- top-level `status`: unset, so the file is intentionally not a frozen release;
- `data/hidden_bench/scaled/paraphrased_replication/N_*.json`: not built.

These annotation and generated-data files are Git-ignored local artifacts.
Implementing code and CI tests must therefore fail clearly when they are absent
and use small committed fixtures for unit tests rather than assuming that every
checkout contains the local pool.

The recommended first task is task 2, `evacuation_north_hill`. It is already the
task selected by the main reasoning and classical imitation configs and has:

```text
4 hidden evidence types x 10 accepted paraphrases per type
```

With balanced allocation and no reuse, the largest requested condition,
`N = 32`, needs only 8 variants per evidence type. The same task-scoped release
therefore supports the whole proposed `N in {4, 8, 16, 32}` grid without any new
LLM generation. Tasks 1 (`evacuation_west_city`) and 22
(`Antarctic Storm Safe Haven`) also have four complete 10-variant pools and are
available as later replication tasks.

For task 2, the stored per-variant metadata identifies
`microsoft/gpt-5-mini` as both generator and verifier. Treat this as a two-pass
self-verification workflow, not an independent cross-model validation. Retain
the generation and verification provenance in the frozen subset; the later
information-sufficiency audit is still required before scientific use.

Do **not** add `"status": "frozen"` to the incomplete global file. Instead,
create a reproducible task-subset artifact containing only the selected complete
task, validate that every canonical evidence type is present and has sufficient
accepted capacity, and mark that subset release as frozen. Preserve provenance
linking the subset to the source annotation file and selected task IDs. The
existing population builder already accepts `--task-ids`, for example:

```bash
python scripts/prepare_hiddenbench.py \
  --agents 4 8 16 32 \
  --method paraphrased_replication \
  --task-ids 2 \
  --annotations <frozen-task-2-paraphrase-release.json> \
  --data-root data/hidden_bench
```

The builder checks `status == "frozen"`, balances evidence types
deterministically, rejects insufficient unique variants unless reuse is
explicitly enabled, and retains `evidence_type`, `variant_id`, source indices,
source text, and `transformation: "validated_paraphrase"` in each agent record.
Keep those fields intact when connecting the scaled dataset to the imitation
game.

Finally, do not confuse these private-evidence paraphrases with the imitation
controller's version-2 message template bank. The controller bank contains four
task-independent advocacy phrasings based only on shared facts. It is already
used to vary controller wording and does not scale or assign agents' private
HiddenBench evidence. This extension requires the annotation artifact described
above.

## 5.1 Preserve evidence-type balance

For a task with \(E\) original hidden evidence types and population size \(N\):

- distribute evidence types as evenly as possible;
- each agent receives one complete hidden fact;
- agents assigned the same evidence type should receive independently validated paraphrases where capacity permits;
- retain source evidence IDs and paraphrase IDs in the agent record.

Reuse the existing balancing and deterministic assignment logic.

## 5.2 Annotation capacity

Do not silently reuse paraphrases unless explicitly configured.

For a desired maximum population \(N_{\max}\), each evidence type should have at least approximately

\[
\left\lceil rac{N_{\max}}{E}ightceil
\]

accepted variants.

The existing pool targets 10 variants per evidence type. This is enough for many tasks at \(N=32\) when \(E=4\), but not necessarily when \(E=3\).

Before building the final population grid:

1. determine the minimum number of evidence types across the selected task subset;
2. compute the required paraphrase capacity;
3. generate additional accepted paraphrases where necessary rather than enabling reuse by default.

## 5.3 Keep the frozen-pool safeguard

Do not weaken the current rule that the population builder refuses incomplete/unfrozen annotation files.

If today's experiment uses only a selected subset of HiddenBench tasks, create a **task-subset annotation artifact** that is complete and explicitly frozen for that subset rather than treating the globally incomplete annotation pool as valid.

Then run `run_information_sufficiency_audit.py` on every scaled dataset before scientific use.

---

# 6. Population sizes for the first empirical grid

Target a modest finite-size grid first, for example:

```text
N = 4, 8, 16, 32
```

subject to validated paraphrase capacity.

`N=4` should retain the canonical/small-population reference condition. Do not change task semantics, answer options, or correct-answer labels when scaling \(N\).

---

# 7. Time convention and horizon scaling

The reasoning experiment is a **discrete event process**. One interaction updates at most one focal agent.

Keep the raw event index \(t=0,1,\ldots,T\). For comparisons across population sizes, additionally define normalized sweep time

\[
	au=rac{t}{N}.
\]

Interpretation:

```text
N elementary focal-update opportunities = 1 population sweep
```

This is not continuous physical time and must not be labeled as such.

## 7.1 Match exposure across population sizes

Do not compare different \(N\) using the same raw number of interactions. Instead choose a fixed number of sweeps \(S\) and set

\[
T=SN.
\]

For example, 25 sweeps imply 100, 200, 400, and 800 interactions for \(N=4,8,16,32\), respectively.

Whether this is implemented as a config convenience or calculated when preparing the grid is secondary. The resolved config must record both \(N\) and \(T\).

---

# 8. Controller exposure diagnostics to log

Do not add many new headline metrics today, but make sure the event data can recover controller resource usage.

At cell level we should be able to compute

\[
c_{m decision}=rac{T}{N},
\qquad
c_{m adv}=rac{\#\{t:U_t=\mathrm{ADVOCATE}_Z\}}{N}.
\]

Also retain enough information to derive:

```text
unique_focal_agents_exposed_to_controller
fraction_population_ever_exposed_to_controller
controller_advocacy_count
controller_noop_count
```

These may initially be report diagnostics rather than registered scientific metrics.

The purpose is to distinguish:

```text
controller sensing resource  ~ q_c
controller social leverage   ~ 1/q when b=1
controller repeated exposure ~ ADVOCATE count / N
```

Do not call any of these "energy" or "dissipation" yet.

---

# 9. Event-schema additions

Preserve all existing feedback fields and add enough information to reconstruct the generalized interaction.

Each event should contain at least:

```text
population_size
social_group_size
social_peer_ids
social_peer_votes_before
controller_sensor_sample_size
controller_sensor_ids
controller_sensor_votes
controller_target_support
controller_policy
controller_advocacy_probability
controller_action
controller_message_template
controller_message
replaced_peer_id
replaced_peer_slot
focal_agent_id
focal_vote_before
focal_vote_after
population_counts_before
population_counts_after
correct_answer
controller_target
```

If peer messages are already persisted elsewhere, store stable references instead of duplicating large text blobs unnecessarily.

For tomorrow's theoretical model, it must be possible to reconstruct

\[
(n_t,\;Y_t,\;U_t,\;	ext{social context},\;n_{t+1}).
\]

---

# 10. New analysis metric 1 — truth current

Add a post-hoc episode-level quantity:

```text
truth_current
```

For each elementary interaction define

\[
j^{m truth}_t=
egin{cases}
+1,&X^f_t
eq Y^*\ 	ext{and}\ X^f_{t+1}=Y^*,\
-1,&X^f_t=Y^*\ 	ext{and}\ X^f_{t+1}
eq Y^*,\
0,&	ext{otherwise}.
\end{cases}
\]

The episode truth current is

\[
J_{m truth}
=
\sum_{t=0}^{T-1}j^{m truth}_t
=
\#(	ext{switches toward truth})-
\#(	ext{switches away from truth}).
\]

Store per episode:

```text
truth_current
truth_switches_toward
truth_switches_away
```

Only `truth_current` is the new headline metric; the directional counts are audit components.

## Important interpretation caveat

Because only one focal opinion changes per event,

\[
J_{m truth}=n_{m truth}(T)-n_{m truth}(0).
\]

Therefore this **net truth current telescopes**. It measures net directional displacement toward truth, but by itself it does **not** distinguish clean monotone convergence from many toward/away switches that cancel when both have the same initial and final truth headcount.

This is acceptable for the requested first current metric and is directly useful for current-fluctuation statistics, but the event log must preserve the separate toward/away counts so that a later activity/reaction-resolved current can be introduced if required.

Do not claim that this net current alone measures path volatility.

---

# 11. New analysis metric 2 — truth-current Fano/precision ratio

The stochastic-thermodynamics reference defines, for a stationary current \(I\),

\[
F(\dot I)=rac{|\langle\dot Iangle|}{\dot\sigma^2(I)},
\]

where the denominator is the current dispersion rate.

For equal-duration episodes, the common time normalization cancels:

\[
\langle\dot Jangle=rac{\langle Jangle}{	au},
\qquad
\dot\sigma^2(J)=rac{\operatorname{Var}(J)}{	au},
\]

so

\[
F_{m truth}
=
rac{|\langle J_{m truth}angle|}{\operatorname{Var}(J_{m truth})}.
\]

Add the cell-level quantity:

```text
truth_current_fano
```

with exactly this convention.

This follows Eq. 27 of *Stochastic Thermodynamics of Social Imitation beyond Energetics*, where the authors use the mean-current-to-dispersion ratio as a signal-to-noise / current-precision quantity.

### Required report fields

For every cell report:

```text
truth_current_mean
truth_current_variance
truth_current_fano
episodes
fixed_horizon
```

Optionally also report `truth_current_mean_per_agent = mean(J_truth) / N` as a scaling diagnostic, but do not make it a third headline metric.

### Variance convention

Use the across-episode sample variance:

\[
\widehat{\operatorname{Var}}(J_{m truth})
=
rac{1}{E-1}\sum_e(J^{(e)}_{m truth}-ar J_{m truth})^2.
\]

Bootstrap **whole episodes**, consistent with the existing information analysis.

### Edge cases

If \(\operatorname{Var}(J_{m truth})=0\) and \(|\langle J_{m truth}angle|>0\), the mathematical ratio is \(+\infty\). Report this explicitly with a `zero_dispersion` flag rather than silently clipping it.

If both mean and variance are zero, report the ratio as undefined / `NaN`.

Do not replace undefined values by zero.

### No action-shuffle null

An action-label permutation null is not meaningful for this metric because \(J_{m truth}\) is computed directly from the population trajectory and does not depend on relabeling \(U_t\).

Use episode bootstrap uncertainty and matched comparison across experimental cells/no-control controls where relevant.

---

# 12. Add the two metrics to the HiddenBench analysis configuration

The new quantities belong to **post-hoc analysis**, not the per-round generic metric shelf.

Extend the HiddenBench imitation analysis configuration so the relevant run YAML can request:

```yaml
analysis:
  # existing MI / entropy / overlap quantities...
  truth_current: true
  truth_current_fano: true
```

or the repository's established equivalent syntax.

Do not remove or rename the already implemented information, entropy, information-fraction, signed-actuation, or overlap diagnostics.

---

# 13. Keep the current classical model intact today

The existing reasoning-OFF model currently uses:

```yaml
classical:
  kernel: irisarri_multi_opinion
  forward_rate: 1.0
  reverse_rate: 1.0
  interaction_factor: destination_count_plus_offset
  interaction_offset: 1.0
  control_strength: 2.0
```

Do **not** silently reinterpret this as a \(q>1\) q-voter kernel in today's extension.

The current classical interaction factor is effectively the \(q=1\) / linear member of the family.

Today's code should:

1. expose `social_group_size = q` at the shared runtime/protocol level;
2. sample and log the \(q\) social peers in both reasoning and classical modes;
3. allow reasoning mode to consume the \(q\)-peer context;
4. leave a clear hook/context object containing the sampled peer opinions for the classical transition code;
5. preserve the existing classical transition kernel until the theoretical \(q>1\) controlled model is specified.

If classical mode ignores the sampled peer identities under the current kernel, document that explicitly.

Tomorrow's theoretical work can then replace/extend the classical transition kernel without redesigning the scheduler or event schema.

---

# 14. Backward-compatibility tests

With:

```text
N = 4
q = 1
q_c = 2
same task
same initial votes
same controller config
same seed
```

the generalized runtime should reproduce the current interaction semantics.

At minimum verify:

- one focal;
- one ordinary peer;
- sensor sample of two;
- sensor may include focal;
- soft/threshold action probability unchanged;
- `ADVOCATE_Z` replaces the single peer;
- `NO_OP` performs the current ordinary exchange;
- only focal vote changes;
- controller remains outside the population.

Where deterministic RNG stream compatibility is technically feasible, require exact event-sequence reproducibility. If adding the generalized scheduler necessarily changes random-number consumption, document the break and add a compatibility test based on semantics/distributions.

---

# 15. Unit and integration tests

## Social sampling

```text
q <= N - 1
len(peer_ids) == q
focal not in peer_ids
peer_ids unique
```

## Controller sensing

```text
q_c <= N
len(sensor_ids) == q_c
sensor_ids unique if current sampling is without replacement
focal may appear
sensor/social overlap allowed
```

## Controller replacement

For `ADVOCATE_Z`:

```text
number of influence slots == q
number of controller slots == 1
number of ordinary peer slots == q - 1
```

For `NO_OP`:

```text
number of influence slots == q
number of controller slots == 0
number of ordinary peer slots == q
```

## Paraphrased population

For a scaled task test:

- population size exactly \(N\);
- evidence types balanced to within one holder;
- every agent has a valid source evidence ID;
- paraphrase IDs resolve to frozen accepted variants;
- no answer leakage;
- no unintended paraphrase reuse unless explicitly enabled;
- information-sufficiency audit passes.

## Truth current

Construct a toy trajectory and verify:

```text
incorrect -> truth           +1
truth -> incorrect           -1
incorrect A -> incorrect B    0
truth -> truth                0
```

Verify \(J_{m truth}=n_{m truth}(T)-n_{m truth}(0)\).

## Truth-current Fano

Using toy episode currents such as `[1, 2, 1, 0]`, verify mean, sample variance, \(|	ext{mean}|/	ext{variance}\), episode bootstrap, zero-variance and zero-mean edge cases.

---

# 16. First scaled experiment after implementation

Do not immediately launch a large expensive grid.

## Stage A — mock/classical/runtime validation

Use cheap or provider-free runs to validate:

```text
N in {4, 8, 16}
q in {1, 2}
q_c in {1, 2, 4} subject to q_c <= N
```

Check event schema, peer/sensor sampling, controller action probabilities, replacement behavior, exposure diagnostics, and truth-current analysis.

## Stage B — small reasoning smoke test

Use one validated HiddenBench task and paraphrased populations:

```text
N in {4, 8}
q in {1, 2}
q_c = 2
```

with very few episodes. The purpose is software validation, not scientific inference.

## Stage C — first scientific finite-size run

After the above passes, use something like:

```text
N in {4, 8, 16, 32}
q in {1, 2}
q_c in {2, 4}
```

with a fixed number of population sweeps. Do not expand all axes aggressively until runtime/cost is measured.

The final scientific grid can be decided after the theoretical feedback model is derived.

---

# 17. What should be deliberately postponed

Do **not** add these today unless required for compatibility:

- network topology / sparse connectivity;
- multiple external controllers;
- controller actuation budget \(b>1\);
- factorized evidence;
- a new continuous-time/Gillespie clock;
- a new \(q>1\) classical thermodynamic kernel;
- entropy production for the reasoning LLM;
- a TUR claim for the LLM dynamics;
- information-normalized "thermodynamic efficiency";
- free-text InfoNCE estimation.

Today's goal is to make the empirical protocol large-\(N\), \(q\)-aware, \(q_c\)-aware, and fully logged.

---

# 18. Deliverables

The implementing agent should finish with:

1. **Code**
   - configurable social group size \(q\);
   - existing configurable controller sensor size \(q_c\);
   - generalized \(q\)-peer reasoning interaction;
   - single-slot controller replacement;
   - scaled paraphrased population loading/building;
   - truth-current analysis;
   - truth-current Fano/precision analysis.

2. **Tests**
   - all unit/integration tests above;
   - explicit \(q=1, q_c=2\) backward-compatibility test.

3. **Configs**
   - one cheap validation config;
   - one scaled paraphrased reasoning config/grid;
   - analysis section including `truth_current` and `truth_current_fano`.

4. **Documentation**
   - update the HiddenBench imitation README/specification with \(N\), \(q\), \(q_c\), controller replacement semantics, discrete sweep time \(t/N\), paraphrased population preparation, and truth-current definitions.

5. **Post-implementation report**
   - exact files changed;
   - test results;
   - generated scaled dataset paths;
   - annotation/sufficiency-audit status;
   - any backward-compatibility caveat;
   - one small example event showing \(q>1\) and `ADVOCATE_Z`;
   - one example analysis report containing the two new current quantities.

---

# 19. Theory interface to preserve for tomorrow

After today's implementation, the empirical protocol should admit the following abstract description without further runtime redesign:

\[
n_t
\overset{S_{q_c}(y\mid n_t)}{\longrightarrow}
Y_t
\overset{\pi(u\mid y)}{\longrightarrow}
U_t
\overset{P_q(n_{t+1}\mid n_t,u)}{\longrightarrow}
n_{t+1}.
\]

Here:

- \(N\) is the system size;
- \(q\) is the size/order of social influence;
- \(q_c\) is the controller measurement budget;
- \(U_t\) is the stochastic feedback action;
- the controller occupies one of \(q\) social influence slots when active;
- the reasoning system implements \(P_q\) implicitly through an LLM;
- the classical system will implement \(P_q\) explicitly.

That is the interface the theoretical feedback-control model should use.

---

# 20. Scientific interpretation of the two new current quantities

For each episode,

\[
J_{m truth}
=
\#(	ext{switches toward truth})-
\#(	ext{switches away from truth}).
\]

Across repeated matched episodes,

\[
F_{m truth}
=
rac{|\langle J_{m truth}angle|}
{\operatorname{Var}(J_{m truth})}.
\]

This is intended as an empirical analogue of the current signal-to-noise quantity used in the stochastic-thermodynamics paper.

For now interpret it only as:

```text
net truth-directed current and its across-episode precision/fluctuation ratio
```

Do **not** claim that \(F_{m truth}\) satisfies a thermodynamic uncertainty relation in the LLM system.

That theoretical question is explicitly deferred until the classical feedback model and its appropriate time/reversibility structure are derived.
