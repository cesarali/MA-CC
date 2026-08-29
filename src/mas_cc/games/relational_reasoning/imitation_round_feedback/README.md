# `relational_imitation_round_feedback`

Distributed relational reasoning under one-focal imitation dynamics, with one
controller decision per population round and **explicit, machine-readable
evidence transmission** between agents.

The dynamics are the same two-clock structure as
`hidden_bench_imitation_round_feedback`. The new thing is a second state
variable: alongside what each agent *votes*, the game tracks exactly what each
agent *knows*, and it tracks every interaction that moved a piece of knowledge.

---

## 1. The relational task

Tasks come from the standalone generator in
`src/mas_cc/relational_task_generator/`. They are **frozen JSON**: the game
reads them and never runs, redistributes, or repairs anything.

One task is an exact symbolic 2-D spatial world. A fact

```json
{"id": "f1", "subject": "Bavi", "relation": "NORTHEAST", "object": "Zora", "role": "supporting"}
```

means `position(Bavi) - position(Zora) = (1, 1)` and renders deterministically as

```text
Bavi is northeast of Zora.
```

A task carries a query, an answer chain of length `L` (the *supporting* facts),
some distractors in a disconnected component, `K` labelled answer options, and
the initial allocation of facts across the population. `task_0001` looks like:

| | |
|---|---|
| question | *Where is Bavi relative to Ralo?* |
| supporting | `f1` Bavi is northeast of Zora. `f2` Zora is northwest of Ralo. |
| distractors | `f3`…`f6` (a disjoint component) |
| options | `A) NORTHEAST` `B) SOUTHWEST` `C) NORTH` |
| answer | `C` (`NORTH`) |
| population | 24 agents, each supporting fact held by 6 of them, no agent holding both |

### Votes are semantic, letters are per call

The population state is the **compass relation** (`NORTH`, `NORTHEAST`, …), not
a letter. A globally shared `A`/`B`/`C` labelling would be its own attractor:
agents could converge on "B" for reasons having nothing to do with what B
means, and the run would record consensus that is really option-position bias.

So every LLM call gets its **own seeded shuffle** of the answers:

```text
call for agent 3   A) NORTH      B) SOUTHWEST   C) NORTHEAST
call for agent 7   A) NORTHEAST  B) NORTH       C) SOUTHWEST
```

The model answers with a letter; `parse_action` resolves it to the relation
immediately, and a letter never reaches persistent or socially visible state. A
peer's vote is re-rendered into the *reading* agent's letters, so one prompt is
internally consistent. The map is derived from `(episode seed, stage, agent,
update index)`, so it replays exactly and is stable across retries of one
decision. The frozen `option_labels` / `correct_option` survive only as
provenance; scoring is against `correct_relation`.

The loader is `games/relational_reasoning/data.py`. See §9 for what it refuses.

---

## 2. Knowledge state

Every agent carries three independent variables:

```text
X_i(t)  the currently voted option label       -> agent.committed_action
H_i(t)  every fact id ever received            -> agent.known_fact_ids
K_i(t)  facts currently available to reasoning -> agent.active_fact_ids
```

`X_i` moves whenever the agent votes. `H_i` grows when another participant
exposes a new fact to *this* agent. `K_i` can also shrink at the end of a
population round. Each active fact survives independently with probability
`game.options.epistemic_persistence`, written as rho. The default is `1.0`,
which exactly preserves the earlier behavior and consumes no persistence
random numbers. A valid later exposure reactivates an inactive historical fact.

The full ballot is `(X_i, R_i, S_i)`: vote, reason, and the one fact id the
agent chose to expose. Only `X_i` and `S_i` are ever **rendered** — `R_i` is
recorded for analysis and shown to nobody, not to peers and not back to its own
author on a later turn (§4). Active `K_i` is private: it reaches that agent's
own prompt and nowhere else. Historical `H_i` remains available only for
provenance and historical diagnostics.

Every acquired fact also carries its provenance:

```json
{"f2": {"source": "controller", "round_index": 0, "within_round_index": 7, "from": null}}
```

with `source` ∈ `initial` | `peer` | `controller`.

---

## 3. Initial distribution

`K_i(0)` is the task file's `agents[<id>].fact_ids`, verbatim. Agent identities
are the task's own (`agent_001` … `agent_024`), so the assignment is auditable
by reading the two side by side. Many agents start with **no facts at all** —
that is a normal state, and their prompt says so explicitly.

At startup the game checks that the union `∪ᵢ Kᵢ(0)` contains every supporting
fact. A task whose population could not collectively solve it does not run.

---

## 4. How facts propagate

A ballot may name at most one fact:

```json
{"vote": "C", "reason": "The two relations compose to due north.", "shared_fact_id": "f2"}
```

The cited fact must be in `K_i(t)` — the knowledge set *before* this update.
A citation of anything else fails validation and the normal retry loop asks
again; it is never silently downgraded to "shared nothing". `"none"` is always
legal: an agent is never forced to disclose.

When focal agent `i` is shown a source whose ballot exposes fact `f`:

```text
f ∈ K_i(t+1)
```

Only agents actually shown that ballot acquire it. There is no broadcast, and
no other code path writes to `K`.

### `shared_fact_id` is the only channel

A ballot's `reason` is written, parsed, stored and available for analysis — but
it is **never rendered into another agent's prompt**. If prose were shown to
peers, an agent could pass a fact, or a conclusion drawn from one, while
reporting `shared_fact_id: none`; information would then move without appearing
in any `K_i`, the knowledge state would be a lower bound rather than an exact
record, and every epistemic observable built on it would be unfalsifiable.

So a visible participant shows exactly three things — who it is, what it votes,
and the fact it chose to expose. The exposed fact is **rendered text with no
identifier**: the symbols stay in the log, the experiment stays a language
task.

```text
Agent 7
Vote: B
Evidence they are sharing:
Kavi is east of Tero.
```

With `shared_fact_id: none`, no evidence lines appear at all:

```text
Agent 7
Vote: B
```

The prompt says so plainly, so the instructions never invite a channel the
runtime does not provide: *"Other participants will see the same of you: your
vote and any fact you choose to share, and nothing else. Your reason is your
own record: it is not shown to anyone."*

An agent does not see its own previous reason either. Its standing position
block carries the vote and nothing else:

```text
YOUR CURRENT POSITION

Vote: C
```

Feeding prose back to its own author would make it an uncontrolled *internal*
memory channel — conclusions carried forward in text that nothing in the state
records — and `K_i` would stop being the only thing that explains what an agent
knows. The previously exposed fact is not repeated here either: it is already
listed under YOUR CURRENT KNOWLEDGE.

An agent's *own* facts are shown *with* their ids, because it needs them to
cite one:

```text
YOUR CURRENT KNOWLEDGE

- f1: Bavi is northeast of Zora.
```

and the JSON instruction advertises exactly those ids plus `none` — never a
fact the agent does not hold:

```json
"shared_fact_id": "<f1 | none>"
```

An agent that knows nothing is shown `"<none>"`. The response contract enforces
that list, and `Game.validate_action` independently re-checks the citation
against `K_i(t)` from the state, so evidence honesty has two safeguards rather
than one.

---

## 5. How votes are updated

One microscopic update:

1. sample one focal agent and `q` distinct peers (one draw, `q+1` agents);
2. render each social slot from its current `(vote, exposed fact)`;
3. **one** provider call, returning one ballot;
4. apply the vote immediately, publish the ballot, and grow `K_focal` by
   whatever this prompt exposed to it.

Exactly one agent changes at each microscopic position. A round is `N`
consecutive updates, so `game.horizon` counts rounds and the elementary-step
horizon is `rounds × N`.

---

## 6. The round-feedback controller

```text
population state
      |
      v
controller senses q_c votes          <- votes only, never K_i
      |
      v
one controller decision {NO_OP, ADVOCATE_Z}
      |
      v
if ADVOCATE_Z: preallocate b controlled positions (uniform, without replacement)
      |
      v
N microscopic focal updates
```

The sensor/policy machinery is inherited unchanged from the HiddenBench round
controller: a hypergeometric vote sensor of size `q_c`, a soft (logistic) policy
`P(ADVOCATE_Z) = σ(β(θ − p_Z))`, and an exact budget `b`.

The controller's input is built by stripping the state down to
`{committed_action}` per agent plus the option alphabet, so it *cannot* read a
knowledge set — that is enforced by construction, not by convention.

At a controlled position the controller **replaces one ordinary peer slot**
rather than adding a speaker. With `q=1` the focal sees the controller *instead
of* its peer. Controlled and uncontrolled updates therefore cost exactly one
provider call each and show exactly `q` sources.

The controller is one persistent, ordinary-looking participant, labelled
`Agent N+1`. Nothing in any prompt identifies it as a controller.

---

## 7. Controller evidence

`control.options.message_mode`:

The controller goes through the same renderer as a peer, so it gets no prose
channel a peer does not have either. Its recommendation **is** its vote.

**`recommendation_only`** (default) — the controller's slot carries a vote and
nothing else:

```text
Agent 25
Vote: C
```

No fact ever enters anybody's `K_i` from the controller.

**`recommendation_plus_fact`** — the same vote, plus exactly one fact of the
frozen task, transmitted through the **same** `shared_fact_id` channel a peer
would use:

```text
Agent 25
Vote: C
Evidence they are sharing:
Zora is northwest of Ralo.
```

The fact is a real fact of the task, rendered by the generator, never
paraphrased or invented. Its id is recorded as `controller_fact_id`, and any
focal agent exposed at a controlled position acquires it with
`source: "controller"` — which is what keeps *social information diffusion*
distinguishable from *externally injected information*.

Selection is deterministic and never delegated to a model:

* `controller_fact_id: f2` — an explicit id, validated against the task;
* `controller_fact_selector: supporting` — resolves to the task's first
  supporting fact in task order.

Exactly one of the two must be given, and the fact is resolved **once per
episode**, before anything runs. Varying the citation per slot would add an
uncontrolled stochastic channel to an experiment whose point is measuring what
one message does.

A `NO_OP` round transmits nothing, whatever the mode.

> **Deviation worth knowing about.** The task brief sketched the controller's
> evidence as prose (`"Relevant information: …"`). It is implemented instead as
> the same structured evidence field a peer uses. That keeps the controller
> indistinguishable from an ordinary participant — an invariant the HiddenBench
> game treats as load-bearing — and makes `recommendation_plus_fact` vs. a
> peer-shared fact a comparison of *content* rather than of *format*. It is also
> what §4's single-channel rule requires: a prose slot for the controller would
> be exactly the untracked channel that rule exists to close.
>
> `render_control_reason` still produces `"I recommend option C."`, but only for
> the `controller_message` field in the trajectory — it is never rendered. The
> controller's recommendation reaches the population as its **vote**.

---

## 8. Vote state vs. knowledge state

They are measured separately and neither is derived from the other:

| | vote state | knowledge state |
|---|---|---|
| symbol | `X_i(t)` | `K_i(t)` |
| changes when | the agent votes | a source exposes a fact to it |
| visible to others | yes, with the exposed fact; the reason is not | no |
| visible to the controller | yes | **no** |
| round observables | `truth_vote_share`, `m_truth`, `m_ctrl`, `m_order`, `H_vote` | `mean_supporting_fact_coverage`, `full_proof_agent_share`, `supporting_fact_reach` |

The interesting questions live in the gap: does coverage rise before the vote
share does; can the controller move votes without moving knowledge; does an
injected fact spread further than a peer-shared one.

---

## 9. Configuration

```yaml
prompt:
  prompt_family: relational_public_ballot
  prompt_version: 1
  response_contract:
    type: relational_public_ballot
    allowed_values: [A, B, C]

game:
  type: relational_imitation_round_feedback
  population_size: 24        # must equal the task's own agent count
  horizon: 10                # population rounds
  options:
    task_dataset_dir: src/mas_cc/relational_task_generator/relational_task_generator/examples
    task_id: task_0001
    dynamics_mode: reasoning
    rounds: 10
    social_group_size: 1     # q
    vote_visibility: public
    prompt_version: 1
    epistemic_prompt_class: strategic_uncertainty
    stop_on_consensus: false
    invalid_response_retries: 1
    expected_validation_failure_rate: 0.05
    initialization:
      mode: local_vote       # local_vote | uniform_random | explicit

control:
  mechanism: relational_round_budgeted
  options:
    target: correct          # correct | random_incorrect | a label | an index
    sensor_sample_size: 6    # q_c
    policy: soft_target
    threshold: 0.5
    beta: 4.0
    intervention_budget: 4   # b
    advocacy_schedule: soft  # or: always (open loop)
    message_mode: recommendation_plus_fact
    controller_fact_id: f2   # or controller_fact_selector: supporting
```

| field | meaning |
|---|---|
| `game.options.task_dataset_dir` | directory of frozen `task_*.json` files. Two are shipped: the generator's `examples/` (24 agents, the default) and `datasets/n12_L2_k3` (12 agents), both under `src/mas_cc/relational_task_generator/relational_task_generator/` |
| `game.options.task_id` | which task; omitted takes the first, which is only for smoke runs |
| `game.options.rounds` | population rounds; the elementary horizon is `rounds × N` |
| `game.options.social_group_size` | `q`, the number of visible social slots |
| `game.options.dynamics_mode` | only `reasoning` is implemented; `classical` is refused explicitly |
| `game.options.vote_visibility` | only `public` is implemented; `hidden` is reserved |
| `game.options.epistemic_prompt_class` | `naive` \| `distributed_information` \| `strategic_uncertainty` (default) \| `evidence_calibrated`. This is a fixed prompt block and therefore changes the prompt definition hash. The deprecated `social_distrust` boolean remains accepted only as an unambiguous adapter (`true` → `strategic_uncertainty`, `false` → `distributed_information`); contradictory dual settings are refused. |
| `game.options.initialization.mode` | `local_vote` (one provider call per agent from `K_i(0)` alone), `uniform_random` (provider-free), `explicit` (with `initial_votes`) |
| `game.options.initialization.initial_distribution` | optional weights for `uniform_random` |
| `game.options.stop_on_consensus` | checked only at round boundaries |
| `control.options.intervention_budget` | `b`, controlled positions per advocating round |
| `control.options.message_mode` | `recommendation_only` \| `recommendation_plus_fact` |
| `control.options.advocacy_schedule` | `soft` (default) closes the loop through the sensed target share; `always` advocates every round regardless. Open loop is what a controllability study wants — under `soft` the actuation a population gets is a function of its own state, confounding "did control move it" with "did it need moving". Sensing still runs and is still logged either way |
| `control.options.controller_fact_id` / `controller_fact_selector` | the deterministic citation; exactly one, and only with `recommendation_plus_fact` |

Startup **refuses**, rather than repairs: a missing task or dataset directory,
an unknown `schema_version`, a population size that differs from the task's, an
agent referencing a nonexistent fact, a supporting fact no agent holds, an
unrendered fact, a `correct_option` outside the option set or disagreeing with
`correct_relation`, duplicate options, and a `controller_fact_id` that is not a
fact of the task.

---

## 10. Logged observables

**Per microscopic update** (`trajectory.jsonl`):

```text
round_index  within_round_index  global_update_index  focal_agent_id
sampled_peer_ids  effective_peer_ids  replaced_peer_id  replaced_peer_slot
controlled_slot  controller_action  controller_target  intervention_budget
vote_before  vote_after  focal_reason_before  focal_reason_after
focal_shared_fact_id  social_sources
focal_known_fact_ids_before  focal_known_fact_ids_after
peer_exposed_fact_ids  controller_fact_id
new_peer_fact_ids  new_controller_fact_ids
peer_fact_exposures  controller_fact_exposures
focal_supporting_fact_coverage_before / _after
m_truth  m_ctrl  m_order  H_vote  delta_*  truth_current_increment
```

**Per round** (`round_trajectory.jsonl`):

```text
occupation_counts_before / _after   population_state_before / _after
truth_vote_share  controller_target_share
m_truth_*  m_ctrl_*  m_order_*  H_vote_*  delta_*
mean_supporting_fact_coverage (and _before)
full_proof_agent_share (and _before)
supporting_fact_reach (and _before)   mean_known_fact_count
peer_fact_exposures  controller_fact_exposures
new_peer_facts  new_controller_facts
controller_action  controller_message_mode  controller_fact_id  controller_fact_text
controlled_positions  controlled_positions_seed  controlled_positions_hash_or_id
sensor_agent_ids  sensor_observed_opinions  sensor_count_vector  sensor_target_share
```

Definitions:

```text
supporting fact coverage(i) = |K_i ∩ S| / |S|
mean_supporting_fact_coverage = mean over agents
full_proof_agent_share = fraction of agents whose K_i ⊇ S
supporting_fact_reach = per supporting fact, how many agents hold it
```

`metrics/streaming.csv` carries the vote observables *and*
`mean_supporting_fact_coverage`, `full_proof_agent_share`,
`peer_fact_exposures`, `controller_fact_exposures`, `new_peer_facts`,
`new_controller_facts`.

---

## 11. A worked example

Task `task_0001`. `S = {f1, f2}`, answer `C` (`NORTH`).

**Initial distribution.** `agent_003` holds `{f1}`, `agent_005` holds
`{f2, f4}`, `agent_017` holds `{}`. No agent holds both supporting facts.

```text
K_003(0) = {f1}      K_005(0) = {f2, f4}      K_017(0) = {}
```

**A peer exposure.** Update `(round 0, position 3)` draws focal `agent_003`
with peer `agent_005`. `agent_005`'s standing ballot exposes `f2`, so the
prompt shows:

```text
Agent 5
Vote: B
Evidence they are sharing:
Zora is northwest of Ralo.
```

Whatever `agent_005` wrote in its own `reason` is on the record and in the
trajectory, but `agent_003` does not see it: `f2` is the only thing that
crossed.

**Knowledge update.** `agent_003` answers, and afterwards:

```text
K_003(t+1) = {f1, f2}        new_peer_fact_ids = ["f2"]
```

`agent_003` now holds the whole proof — `full_proof_agent_share` ticks up by
`1/24`. Nobody else's `K` moved.

**Vote update.** With `f1` and `f2` in hand, `agent_003` can compose
`NORTHEAST + NORTHWEST = NORTH` and votes `C`. `truth_vote_share` rises. Its
own prompt at the next update will show `YOUR CURRENT POSITION / Vote: C` and
offer `"shared_fact_id": "<f1 | f2 | none>"` — its reason for that vote is in
the trajectory and nowhere else.

**Controller exposure.** At round 1 the controller advocates `C` under
`recommendation_plus_fact` with `controller_fact_id: f2`. Position 7 is one of
its `b` controlled positions and draws focal `agent_017` (`q=1`), so the
controller *replaces* its single peer:

```text
Agent 25
Vote: C
Evidence they are sharing:
Zora is northwest of Ralo.
```

**Knowledge update.**

```text
K_017(t+1) = {f2}    new_controller_fact_ids = ["f2"]
provenance[f2] = {source: "controller", round_index: 1, within_round_index: 7}
```

**New vote.** `agent_017` now has one link of the chain plus a recommendation.
Whether it follows the recommendation, reasons from the single fact, or holds
its position is exactly what the experiment measures — and because the fact's
source is recorded as `controller` rather than `peer`, injected information
stays separable from diffused information in every downstream analysis.

---

## 12. Current limitations

* **`dynamics_mode: classical` is not implemented** and fails with an explicit
  message. A provider-free kernel would have to define what an exposed fact
  does to a q-voter jump; inventing that silently would produce numbers nobody
  could interpret.
* **`vote_visibility: hidden` is reserved.** The renderer supports it, but the
  surrounding prompt text still says the vote is what others see — and now that
  prose is not shown either, a hidden-vote source would carry only its identity
  and its exposed fact, which is close to no signal at all.
* **At most one fact per ballot** (version 1), and at most one controller fact
  per episode.
* **The prompt definition hash varies per agent.** The contract advertises each
  agent's own citable fact ids, and the contract is part of the definition, so
  two agents with different knowledge get different definition hashes. Agents
  with the same knowledge get the same one. This is the price of not offering a
  model a fact it cannot legally cite.
* **No information-theoretic analysis.** `analysis.enabled` must be `false`:
  the configured MI/CMI estimators are HiddenBench-specific. This version logs
  the exact state they would need and stops there.
* **The controller cites the same fact all episode.** Per-round or per-slot
  variation is deliberately not offered yet.
* **Answer sets larger than 8 options are not presentable** — the per-call
  shuffle draws from `A`…`H`.
* **Only `spatial_relational_task_v1`.** Newer generator schemas will need an
  explicit loader change, by design.
* The generator is *not* importable (it is deliberately not a package); the
  game reads its output files only. Regenerate datasets with the generator's
  own `generate_dataset.py`.
