# Relational Reasoning Round-Feedback Game — Implementation Handoff

Date: 2026-08-18
Task source: frozen relational tasks from `src/mas_cc/relational_task_generator/`
Sibling game it is modelled on: [`hidden_bench_imitation_round_feedback`](../../src/mas_cc/games/hidden_bench/imitation_round_feedback/README.md)

> **Amended 2026-08-18** by
> [`18082026_relational_single_information_channel_fix_handoff.md`](18082026_relational_single_information_channel_fix_handoff.md):
> a peer's free-form `reason` is no longer rendered into another agent's prompt,
> so `shared_fact_id` is the only task-information channel between participants.
> Everywhere below that shows a rendered `Reason:` line inside a social block
> (§9, §10, §16) is superseded by that document; everything else stands.

---

## 1. Status

`relational_imitation_round_feedback` is implemented and runnable through the
normal MAS-CC experiment stack: `experiment preflight`, `experiment run`,
budget guard, recorder, streaming metrics, plots, checkpoint/resume.

Both shipped smoke configs execute end to end on the mock provider from a clean
results tree. No real-model run was launched.

The scientific content of `hidden_bench_imitation` and
`hidden_bench_imitation_round_feedback` is **unchanged**. No file belonging to
either game was edited.

The one genuinely new idea is a **second state variable**. Alongside the vote,
every agent carries an exact knowledge set, and every fact that moves between
agents is recorded with its source and the interaction that moved it:

```text
X_i(t)  currently voted option label      -> agent.committed_action
K_i(t)  exact set of known fact ids       -> agent.known_fact_ids
```

`X_i` moves whenever the agent votes. `K_i` moves only when a participant
actually shown to that agent exposed a fact. Neither is derived from the other.

---

## 2. Files created

New game package, `src/mas_cc/games/relational_reasoning/`:

```text
relational_reasoning/
├── __init__.py
├── data.py                          frozen-task loading and startup validation
└── imitation_round_feedback/
    ├── __init__.py
    ├── state.py                     rules, typed records, round record
    ├── prompts.py                   the relational_public_ballot family
    ├── game.py                      the Game ABC implementation
    ├── controller.py                round-level budgeted controller
    ├── runtime.py                   two-clock runtime
    ├── metrics.py                   streaming metrics + knowledge observables
    └── README.md                    the game manual, incl. a worked example
```

Configs:

```text
configs/runs/relational_reasoning/
├── relational_imitation_round_feedback_no_control_smoke.yaml
└── relational_imitation_round_feedback_controlled_smoke.yaml
```

Tests:

```text
tests/mas_cc/test_relational_task_data.py                  18 tests
tests/mas_cc/test_relational_imitation_round_feedback.py   70 tests
```

---

## 3. Files modified

Three registry/dispatch sites, all **purely additive**:

| File | Change |
|---|---|
| `src/mas_cc/games/registry.py` | register the game; register the `relational_public_ballot` prompt factory |
| `src/mas_cc/control/registry.py` | register the `relational_round_budgeted` mechanism |
| `src/mas_cc/experiments/orchestrator.py` | one dispatch branch to the new runtime |

Nothing under `src/mas_cc/games/hidden_bench/` was touched.

---

## 4. Registered names

```text
game.type            relational_imitation_round_feedback
control.mechanism    relational_round_budgeted
prompt.prompt_family relational_public_ballot   (version 1)
response contract    relational_public_ballot
```

---

## 5. Task data: exact path and schema

Tasks are consumed **frozen**. The generator is never run, imported, or
regenerated during an experiment — it is deliberately not a Python package, so
the game reads its output files only.

```text
src/mas_cc/relational_task_generator/relational_task_generator/examples/
```

Note the doubled directory name: `src/mas_cc/relational_task_generator/` is the
drop location and `relational_task_generator/` inside it is the generator's own
self-contained folder. This is one level deeper than the path sketched in the
original task description, and is what `DEFAULT_TASK_DATASET_DIR` points at.

Accepted schema: `spatial_relational_task_v1`. The fields the game reads:

```text
schema_version, task_id, seed
generation.population_size
world.facts[]            {id, subject, relation, object, role}
query.supporting_fact_ids, query.reasoning_depth
answer.options[]         {label, relation}
answer.correct_option, answer.correct_relation
agents.<agent_id>.fact_ids
rendered.question, rendered.facts.<id>, rendered.reasoning_chain
```

### Vote alphabet

Votes are **option labels** (`A`/`B`/`C`), not compass relations, because the
frozen schema stores the answer as `correct_option` (a label) and the
label→relation map is randomized per task by the generator. Voting on labels
therefore gives one stable `K`-symbol alphabet across tasks.

A model may answer with the label (`"C"`, `"c"`, `"Option C"`) or with the
relation name (`"NORTH"`); both resolve to the label. There is deliberately
**no substring fallback** — with single-letter labels, "a bit north of B" would
otherwise match almost anything, and a vote silently resolved to the wrong
option is worse than a retry.

### Natural language

The game never re-renders a fact. `rendered.facts[<id>]` *is* the generator's
deterministic rendering and is used verbatim, so there is exactly one source of
truth for the language.

### Agent identity

Agent ids are the task's own keys (`agent_001` … `agent_024`), so the frozen
assignment and `K_i(0)` can be audited by reading the two side by side. Prompt
labels follow the task's 1-based numbering: `agent_007` → `Agent 7`. The
controller is `Agent N+1`.

---

## 6. Startup validation (refuse, never repair)

`load_relational_task` raises `RelationalTaskError` — it never patches. Refused:

- missing dataset directory or task file (the error lists what is available);
- unknown `schema_version`;
- `game.population_size` differing from the task's own agent count;
- an agent referencing a fact that does not exist, or listing one twice;
- a supporting fact no agent holds (the population could not solve the task);
- a fact with no rendered text, or rendered text for a fact that does not exist;
- `correct_option` outside the option set, or disagreeing with `correct_relation`;
- duplicate option labels or duplicate option relations;
- `query.supporting_fact_ids` disagreeing with the facts marked `supporting`;
- `generation.population_size` disagreeing with the number of assigned agents.

The controller adds one more, resolved once before the episode runs: a
`controller_fact_id` that is not a fact of the selected task.

---

## 7. Response schema

```json
{
  "vote": "<A | B | C>",
  "reason": "<brief public reason>",
  "shared_fact_id": "<f1 | f2 | ... | none>"
}
```

Validation is split across two layers on purpose:

| Layer | Checks | Why there |
|---|---|---|
| `RelationalBallotContract.validate` | JSON present; vote resolves; reason non-empty and ≤ 600 chars; `shared_fact_id` present, well-formed, and a fact **of this task** | task-constant, so the prompt definition hash stays stable within a task |
| `Game.validate_action` | `shared_fact_id ∈ K_i(t)` — **evidence honesty** | the only place the focal's knowledge set is in scope |

Both feed the same retry loop. A citation of a fact the agent does not know is
**rejected and retried**, never silently downgraded to "shared nothing" — that
would turn a hallucination into an invisible non-event.

`"none"` is always legal. An agent is never forced to disclose, or the
distributed-information problem would dissolve on the first round.

---

## 8. Dynamics preserved from the HiddenBench round-feedback game

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
N microscopic focal updates, one focal agent each:
    sample focal + q social slots
        -> a controlled slot REPLACES one ordinary peer
        -> ONE focal provider call
        -> {vote, reason, shared_fact_id}
        -> apply the vote, publish the ballot, grow K_focal
      |
      v
next population round
```

Preserved exactly: `N` population size, `q` social group size, `q_c` sensor
sample size, `b` intervention budget; exactly one focal agent updating per
microscopic position; one provider call per update whether controlled or not;
`game.horizon` counting **population rounds**, with the elementary-step horizon
`rounds × N`.

The controller's input is built by stripping state down to
`{committed_action}` per agent plus the option alphabet, so it *cannot* read a
knowledge set — enforced by construction, not by convention.

---

## 9. Knowledge propagation

When focal agent `i` is shown a source whose ballot exposes fact `f`:

```text
f ∈ K_i(t+1)
```

Only agents actually shown that ballot acquire it. There is no broadcast, and
no other code path writes to `K`. Peer- and controller-sourced acquisitions are
recorded separately, which is what keeps *social information diffusion*
separable from *externally injected information*.

Every fact carries provenance:

```json
{"f2": {"source": "controller", "round_index": 1, "within_round_index": 7, "from": null}}
```

with `source ∈ {initial, peer, controller}`.

If two slots expose the same fact at one update, both count as *exposures* but
only the first is an *acquisition*, and slot order is the deterministic
tie-break that decides which one owns the provenance entry.

### What a peer shows vs. what an agent sees of itself

A peer's fact is shown as **rendered text with no identifier** — the symbols
stay in the log, the experiment stays a language task:

```text
Agent 7
Vote: B
Reason: The second relation turns the direction east.
Evidence they are sharing:
Kavi is east of Tero.
```

An agent's *own* facts are shown **with** their ids, because it needs them to
cite one:

```text
YOUR CURRENT KNOWLEDGE

- f1: Bavi is northeast of Zora.
```

An agent that knows nothing — a normal state in these tasks, most agents start
empty — gets an explicit line saying so rather than an empty block.

---

## 10. Controller message modes

`control.options.message_mode`:

**`recommendation_only`** (default) — a recommendation and nothing else. No
fact ever enters any agent's `K_i` from the controller.

**`recommendation_plus_fact`** — the same recommendation plus exactly one real
fact of the frozen task, rendered by the generator, never paraphrased or
invented, with its id recorded as `controller_fact_id`.

Fact selection is deterministic and never delegated to a model:

- `controller_fact_id: f2` — an explicit id, validated against the task;
- `controller_fact_selector: supporting` — the task's first supporting fact in
  task order.

Exactly one of the two must be given, and only with `recommendation_plus_fact`.
The fact is resolved **once per episode**, before anything runs: varying the
citation per slot would add an uncontrolled stochastic channel to an experiment
whose point is measuring what one message does. A `NO_OP` round transmits
nothing, whatever the mode.

---

## 11. Two design decisions worth reviewing

### 11.1 Controller evidence travels the peer channel

The original task description sketched the controller's evidence as prose:

```text
I recommend option B.

Relevant information:
Kavi is east of Tero.
```

It is implemented instead as the **same structured evidence field a peer
uses** — the controller's slot carries a `shared_fact_id`, rendered through the
one shared source renderer:

```text
Agent 25
Vote: C
Reason: I recommend option C.
Evidence they are sharing:
Zora is northwest of Ralo.
```

Reasoning: the HiddenBench round-feedback game treats "the controller is
indistinguishable from an ordinary participant" as load-bearing, and the task
brief asked for that replacement semantics to be preserved. A distinct prose
format would make controlled slots identifiable by shape alone. It also makes
`recommendation_plus_fact` versus a peer-shared fact a comparison of *content*
rather than of *format*.

Everything §10 of the brief actually required is satisfied:
`recommendation_only` transmits no fact, `recommendation_plus_fact` transmits
one existing fact using the generator's own rendering, and the exact
`controller_fact_id` is recorded.

The recommendation text is the single module constant
`CONTROL_RECOMMENDATION` in `prompts.py` if this should change.

### 11.2 The game does not subclass `HiddenBenchImitationGame`

It shares its *shape*, not its state. HiddenBench's rules
(`ImitationRules.from_config` hard-requires `game.type ==
"hidden_bench_imitation"`), corpus loading, evidence allocation, and
`disclosed_facts` string matching have no meaning on a frozen symbolic task —
inheriting them would have meant overriding almost every one of them and
leaving ~30 dead configuration fields in the rules object.

What **is** reused rather than reimplemented:

| Reused | From |
|---|---|
| ask/validate/retry decision loop | `mas_cc.runtime.run_validated_decision` |
| seed derivation, budget guard, recorder, plots, checkpointing | shared infrastructure |
| prompt kernel (blocks, contracts, definition/instance hashes) | `mas_cc.llm_runtime.prompts` |
| `population_observables` (`m_truth`, `m_ctrl`, `m_order`, `H_vote`) | `hidden_bench.imitation.metrics` |
| `ADVOCATE_Z` / `NO_OP`, sensor + soft policy + budget semantics | `hidden_bench.imitation[_round_feedback].controller` |
| JSON extraction | `hidden_bench.vanilla.prompts.extract_json_object` |
| streaming metric base classes, `RoundView` | `mas_cc.metrics` |

`RelationalRoundBudgetedControl` **does** subclass the HiddenBench round
controller, so the sensor, the soft policy and the budget are literally the same
code. It replaces only the option validation (`message_mode` instead of
HiddenBench's `evidence_mode`, which it now rejects explicitly rather than
silently ignoring) and adds deterministic fact resolution.

The runtime resolves the controller fact by duck typing
(`getattr(control, "resolve_fact_id", None)`), so a Control that predates these
options — or a test double — simply advocates without evidence instead of
failing.

---

## 12. Configuration reference

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
    message_mode: recommendation_plus_fact
    controller_fact_id: f2   # or controller_fact_selector: supporting
```

| Field | Meaning |
|---|---|
| `task_dataset_dir` | directory of frozen `task_*.json` (defaults to the generator's `examples/`) |
| `task_id` | which task; omitted takes the first, which is only for smoke runs |
| `rounds` | population rounds; elementary horizon is `rounds × N` |
| `social_group_size` | `q`, visible social slots |
| `dynamics_mode` | only `reasoning`; `classical` is refused explicitly |
| `vote_visibility` | only `public`; `hidden` is reserved |
| `initialization.mode` | `local_vote` (one call per agent from `K_i(0)` alone), `uniform_random` (provider-free), `explicit` (with `initial_votes`) |
| `initialization.initial_distribution` | optional weights for `uniform_random` |
| `stop_on_consensus` | checked only at round boundaries |
| `intervention_budget` | `b`, controlled positions per advocating round |
| `message_mode` | `recommendation_only` \| `recommendation_plus_fact` |
| `controller_fact_id` / `controller_fact_selector` | the deterministic citation; exactly one, and only with `recommendation_plus_fact` |

Initial votes are recorded separately as in the existing game
(`result.initial_state.initial_votes`, and `initial_votes` in the serialized
state).

---

## 13. Logged observables

### Per microscopic update (`trajectory.jsonl`)

New knowledge fields:

```text
focal_known_fact_ids_before      focal_known_fact_ids_after
peer_exposed_fact_ids            controller_fact_id
new_peer_fact_ids                new_controller_fact_ids
peer_fact_exposures              controller_fact_exposures
focal_supporting_fact_coverage_before / _after
focal_shared_fact_id_before      focal_shared_fact_id
```

Retained round-feedback fields:

```text
round_index  within_round_index  global_update_index  microscopic_event_index
focal_agent_id  sampled_peer_ids  effective_peer_ids
replaced_peer_id  replaced_peer_slot  controlled_slot
controller_action  controller_target  intervention_budget
vote_before  vote_after  focal_vote_before  focal_vote_after
focal_reason_before  focal_reason_after  social_sources
m_truth  m_ctrl  m_order  H_vote  delta_*  truth_current_increment
```

### Per round (`round_trajectory.jsonl`)

```text
occupation_counts_before / _after     population_state_before / _after
truth_vote_share (and _before)        controller_target_share (and _before)
m_truth_*  m_ctrl_*  m_order_*  H_vote_*  delta_*
mean_supporting_fact_coverage (and _before)
full_proof_agent_share (and _before)
supporting_fact_reach (and _before)   mean_known_fact_count
peer_fact_exposures  controller_fact_exposures
new_peer_facts  new_controller_facts
controller_action  controller_message_mode
controller_fact_id  controller_fact_text
controlled_positions  controlled_positions_seed  controlled_positions_hash_or_id
sensor_agent_ids  sensor_observed_opinions  sensor_count_vector  sensor_target_share
```

Definitions:

```text
supporting fact coverage(i)   = |K_i ∩ S| / |S|
mean_supporting_fact_coverage = mean over agents
full_proof_agent_share        = fraction of agents whose K_i ⊇ S
supporting_fact_reach         = per supporting fact, how many agents hold it
```

### `metrics/streaming.csv`

Carries the vote observables **and** six epistemic ones — verified present in a
real run:

```text
mean_supporting_fact_coverage   full_proof_agent_share
peer_fact_exposures             controller_fact_exposures
new_peer_facts                  new_controller_facts
truth_vote_share
```

No MI/CMI/transfer-entropy estimator was added. Version 1 logs the exact state
those would need and stops there.

---

## 14. Tests

**88 new tests, all passing.**

`test_relational_task_data.py` (18): dataset loads and every shipped task
validates; exact symbolic and rendered content; assignment preserved verbatim;
supporting-fact union invariant; and eleven refusal cases (missing task, schema
version, population mismatch, unknown fact reference, uncovered supporting
fact, unrendered fact, bad `correct_option`, `correct_option`/`correct_relation`
disagreement, supporting-list disagreement, distractor listed as supporting,
declared population mismatch, duplicate options).

`test_relational_imitation_round_feedback.py` (70), grouped by invariant:

| Group | Covers |
|---|---|
| initial information | `K_i(0)` equals the frozen assignment; no unassigned fact; population union covers `S`; population-size mismatch refused; `local_vote` and `uniform_random` initialization |
| scheduling | `N` updates per round with one focal each; distinct focal/peer sampling; whole-trajectory replay under a fixed seed; deterministic controlled schedule matching the budget |
| ballots | returned vote is the committed action; label/relation parsing incl. no-substring-guessing; eight contract rejection cases; fenced JSON |
| evidence honesty | only a fact in `K_i(t)` may be exposed; hallucinated citation raises after retries; validator names the knowledge set |
| propagation | a peer fact reaches exactly the focal that saw it; knowledge only grows and only via a recorded exposure; repeat exposure is not an acquisition; provenance recorded for every fact |
| controller | slot replacement at `q=1` and `q=2`; controller senses votes only; `recommendation_only` transmits nothing; `recommendation_plus_fact` injects one exact fact; a controller fact reaches only controlled focals; `NO_OP` transmits nothing; **no-control transmits nothing ever**; persistent ordinary identity; no leak words in any prompt |
| controller config | seven incoherent configurations refused; selector resolution; out-of-task fact refused |
| rendering | peer fact shown as text without its id; own facts shown with ids; fixed social-environment block byte-identical with one definition hash; prompt snapshot |
| round boundaries | round record sums agree with the microscopic trajectory; every declared §16/§17 field present; coverage rises monotonically as evidence circulates; nothing moves when nobody shares |
| config guards | `classical` refused; `hidden` reserved; prompt version 2 refused; wrong prompt family refused; `horizon = rounds × N`; shipped configs resolve |
| backward compatibility | new registrations are additive; every pre-existing game/control/prompt entry still resolves; the relational prompt family is genuinely distinct from HiddenBench's |

### Regression check

Full suite: **53 FAILED + 4 ERROR — byte-identical to the pre-existing `main`
baseline.** Verified by running the suite on a `git stash`-ed tree and diffing
the sorted failure lists. Every failure lives in a HiddenBench test file
(missing `hidden_bench` corpus configs, plus three stale config-comment
assertions); none is relational.

```bash
# new tests
conda run -n MA-CC --no-capture-output python -m pytest \
  tests/mas_cc/test_relational_task_data.py \
  tests/mas_cc/test_relational_imitation_round_feedback.py -q

# HiddenBench unchanged
conda run -n MA-CC --no-capture-output python -m pytest \
  tests/mas_cc/test_hidden_bench_imitation_round_feedback.py \
  tests/mas_cc/test_hidden_bench_round_feedback_public_ballot.py \
  tests/mas_cc/test_hidden_bench_imitation_round_feedback_classical.py \
  tests/mas_cc/test_hidden_bench_imitation_round_feedback_analysis.py -q
```

---

## 15. Smoke and example commands

All commands run from the repository root. Both smoke configs use the **mock
provider** and set `logging.comet: false` *and*
`observability.comet.sweep_experiment: false`, so neither can contact a model or
open a remote experiment.

Provider-free structural check:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_no_control_smoke.yaml \
  --output-dir inspection/relational_no_control_smoke_preflight
```

No-control smoke run (q = 1, no controller, 2 rounds, 1 repetition):

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_no_control_smoke.yaml
```

Controlled smoke run (same task and settings, `b = 4`,
`recommendation_plus_fact` with the explicit supporting fact `f2`):

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_controlled_smoke.yaml
```

Results land under
`results/relational_imitation_round_feedback/<experiment-name>/…`. A completed
run is re-entered as `skipped (resumed)`; delete the directory to re-run.

### Towards a real experiment

Copy the controlled config and change the provider block to a live one (see
`configs/runs/hidden_bench/hidden_bench_imitation_round_feedback_qwen_N6_qc6_b_2x2_smoke.yaml`
for the university-provider shape), keeping `max_output_tokens` generous — the
ballot contract allows a 600-character reason plus JSON scaffolding, so a tight
ceiling turns an expensive call into a dead episode. Leave
`analysis.enabled: false`. A useful first grid axis is
`control.options.message_mode: [recommendation_only, recommendation_plus_fact]`
at matched `b`, which isolates injected evidence from bare advocacy.

---

## 16. A worked example

Task `task_0001`: `S = {f1, f2}`, answer `C` (`NORTH`), 24 agents, no agent
holding both supporting facts.

**Initial distribution.**

```text
K_003(0) = {f1}      K_005(0) = {f2, f4}      K_017(0) = {}
```

**Peer exposure.** Update `(round 0, position 3)` draws focal `agent_003` with
peer `agent_005`, whose standing ballot exposes `f2`:

```text
Agent 5
Vote: B
Reason: The relation I have points northwest.
Evidence they are sharing:
Zora is northwest of Ralo.
```

**Knowledge update.**

```text
K_003(t+1) = {f1, f2}        new_peer_fact_ids = ["f2"]
```

`agent_003` now holds the whole proof; `full_proof_agent_share` ticks up by
`1/24`. Nobody else's `K` moved.

**Vote update.** With `f1` and `f2` in hand, `agent_003` can compose
`NORTHEAST + NORTHWEST = NORTH` and vote `C`. `truth_vote_share` rises.

**Controller evidence exposure.** At round 1 the controller advocates `C` under
`recommendation_plus_fact` with `controller_fact_id: f2`. Position 7 is one of
its `b` controlled positions and draws focal `agent_017`; at `q = 1` the
controller *replaces* its single peer:

```text
Agent 25
Vote: C
Reason: I recommend option C.
Evidence they are sharing:
Zora is northwest of Ralo.
```

**Knowledge update.**

```text
K_017(t+1) = {f2}    new_controller_fact_ids = ["f2"]
provenance[f2] = {source: "controller", round_index: 1, within_round_index: 7}
```

**New vote.** `agent_017` now holds one link of the chain plus a
recommendation. Whether it follows the recommendation, reasons from the single
fact, or holds its position is exactly what the experiment measures — and
because the source is recorded as `controller` rather than `peer`, injected
information stays separable from diffused information in every downstream
analysis.

---

## 17. Known limitations

- **`dynamics_mode: classical` is not implemented** and fails with an explicit
  message rather than approximating. A provider-free kernel would have to
  define what an exposed fact does to a q-voter jump; inventing that silently
  would produce numbers nobody could interpret. Left for a later task.
- **`vote_visibility: hidden` is reserved.** The renderer supports dropping a
  source's vote, but the surrounding prompt text still says the vote becomes
  public, so enabling it is a text decision that has not been made.
- **At most one fact per ballot**, and at most one controller fact per episode.
- **No information-theoretic analysis.** `analysis.enabled` must be `false`;
  the configured MI/CMI estimators are HiddenBench-specific. This version logs
  the exact state they need and stops there.
- **The controller cites the same fact all episode.** Per-round or per-slot
  variation is deliberately not offered.
- **Only `spatial_relational_task_v1`.** A newer generator schema needs an
  explicit loader change, by design.
- The generator is not importable; regenerate datasets with its own
  `generate_dataset.py`.

### One pre-existing red test, flagged not fixed

`tests/mas_cc/test_games.py::test_default_registry_constructs_a_generic_game_lazily`
pins a hardcoded registry name tuple that went stale when the four HiddenBench
games landed; it has been failing on `main` since. The new game is now the
fifth missing name. It was left alone rather than quietly rewriting an
assertion outside this task's scope — it is a one-line fix whenever you want it.

---

## 18. Where to read next

- The game manual, with the full config table and the same worked example:
  [`src/mas_cc/games/relational_reasoning/imitation_round_feedback/README.md`](../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/README.md)
- The task generator's own contract:
  [`src/mas_cc/relational_task_generator/relational_task_generator/README.md`](../../src/mas_cc/relational_task_generator/relational_task_generator/README.md)
- The game this one is modelled on:
  [`src/mas_cc/games/hidden_bench/imitation_round_feedback/README.md`](../../src/mas_cc/games/hidden_bench/imitation_round_feedback/README.md)
