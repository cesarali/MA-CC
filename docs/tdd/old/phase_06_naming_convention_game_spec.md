# Phase 6 Companion Specification: Ashery–Aiello–Baronchelli Naming Convention Game

**Status:** implementation contract for `mas_cc.games.naming_convention`  
**Primary sources:**

- Ariel Flint Ashery, Luca Maria Aiello, and Andrea Baronchelli, *Emergent social conventions and collective bias in LLM populations*, Science Advances 11, eadu9368 (2025).
- Supplementary Materials for the same article, especially the sections **Prompting**, **Prompt Structure**, **Output Structure**, **Example Prompt**, and Table S4.

**Scope of this document:** the repeated LLM coordination game used in the paper. It is not the canonical speaker/hearer inventory Naming Game described in the supplementary theoretical-model section.

---

## 1. Purpose

Implement the first scientifically meaningful game in `mas_cc` as a faithful, inspectable version of the repeated naming-convention experiment in Ashery et al.

The game must isolate three layers:

1. **Game dynamics:** pair sampling, simultaneous actions, payoff, memory update, and termination.
2. **Prompt/decision layer:** conversion of one agent's private memory into a provider-independent decision request.
3. **Experiment/analysis layer:** replication grids, committed minorities, convergence estimators, plots, and later committee interventions.

Phase 6 implements the **base convention-emergence game**. Committed minorities and committees may be represented by extension points, but ordinary game execution must not depend on them.

---

## 2. Terminology and non-equivalence warning

The source article calls the experiment a naming game because agents repeatedly select names and local coordination can create a global convention. Operationally, however, this is a **repeated symmetric coordination game with private finite memory**.

It is not the canonical minimal Naming Game with:

- a speaker and a hearer;
- mutable lexical inventories;
- invention of a word by an empty speaker;
- success/failure inventory-collapse rules.

In the Ashery et al. LLM game:

- both selected agents choose simultaneously;
- both choose exactly one name from the same finite action pool;
- neither action changes an inventory;
- matching actions give both players `+100`;
- different actions give both players `-50`;
- adaptation occurs only through the LLM seeing its own bounded interaction history.

Use a distinct implementation namespace and state model. Suggested name:

```text
mas_cc.games.naming_convention
```

Do not reuse inventory-game state types merely because both systems use the phrase “naming game.”

---

## 3. Scientific game definition

### 3.1 Population

A trial contains a homogeneous population of `N` LLM agents:

```text
V = {0, 1, ..., N - 1}
```

The paper studies an unstructured, well-mixed population. Therefore, the source-faithful topology is the complete graph:

```text
E = {(i, j) : i != j}
```

A general topology can be supported by the package interface, but the paper-replication profile must use complete mixing.

### 3.2 Convention/action pool

All agents share a finite set of `W` legal names:

```text
A = {a_1, ..., a_W}
```

The paper usually uses unique letters from the English alphabet. It also studies two-name pools such as `{Q, M}` and robustness variants using other labels or random strings.

The labels are strategically symmetric in the payoff function. Any observed asymmetry must therefore arise from the model, prompt, memory-state distribution, or sampling dynamics—not from game payoffs.

### 3.3 Global interaction clock

The game evolves through random-sequential pair interactions:

```text
t = 0, 1, ..., T - 1
```

At every global interaction `t`, select two distinct agents uniformly at random from the population.

Recommended source-faithful sampler:

```python
pair = rng.sample(range(N), k=2)
```

The two sampled identities may be stored as an ordered audit tuple `(i_t, j_t)`, but the scientific interaction is symmetric. Role labels such as “Player 1” and “Player 2” are local prompt labels, not persistent social roles.

### 3.4 Simultaneous decisions

For selected agents `i_t` and `j_t`, construct both decision requests from the same pre-interaction state:

```text
M_i(t^-), M_j(t^-)
```

Then obtain both actions concurrently:

```text
x_i(t) in A
x_j(t) in A
```

Neither player may observe the other player's current action before deciding. The transition is applied only after both valid actions are available.

This is a hard behavioral invariant.

### 3.5 Payoff

The pair payoff is symmetric:

```text
r_i(t) = r_j(t) = +100,  if x_i(t) == x_j(t)
r_i(t) = r_j(t) =  -50,  otherwise
```

Define:

```text
success(t) = 1[x_i(t) == x_j(t)]
```

There is no direct population-level reward and no bonus for global consensus.

### 3.6 Private memory

Each ordinary agent has a private chronological history of interactions in which that agent participated.

A source-faithful private memory entry contains at least:

```yaml
local_interaction_index: int
own_action: Action
partner_action: Action
payoff: 100 | -50
success: bool
```

The next prompt exposes only the most recent `H` entries:

```text
visible_memory_i(t) = tail_H(full_private_history_i(t))
```

The evaluator may retain the complete history. The model may not see the complete history when its length exceeds `H`.

The source paper says the prompt gives the agent its accumulated score **within the memory range**. Therefore define:

```text
visible_score_i(t) = sum(entry.payoff for entry in visible_memory_i(t))
```

The engine may additionally store a lifetime score for auditing:

```text
lifetime_score_i(t) = sum(all past payoffs of agent i)
```

Do not silently substitute lifetime score for visible-window score in the paper-faithful prompt profile. If legacy code does so, record that as an explicit parity difference rather than hiding it.

### 3.7 Empty-memory initialization

All ordinary agents begin with empty private histories:

```text
full_private_history_i(0) = []
```

**Critical fidelity rule:** an empty-memory action is still requested from the LLM. Do not replace it with a uniform engine-side random action.

The article studies the distribution of first actions under empty memory to measure individual model bias. A uniform local draw would erase precisely the phenomenon the experiment is designed to observe.

“Random” in the source description should be operationalized as stochastic generation from the configured model, not as a forced uniform distribution over labels.

---

## 4. Information boundary

For an ordinary agent, a decision request may reveal only:

- the legal action labels;
- the simultaneous-choice structure;
- the matching and mismatching payoffs;
- the objective of maximizing its own score conditional on the other player's behavior;
- its own most recent `H` interaction outcomes;
- a local round number derived from that visible history;
- its score over the visible history window;
- the required output schema.

It must not reveal:

- `N`, unless explicitly required by a non-paper experimental variant;
- that the agent belongs to a larger population;
- the global interaction index `t`;
- the selected partner's persistent identity;
- how partners are sampled;
- the partner's private history or score;
- action frequencies in the population;
- global success rate;
- consensus status;
- strong/weak-convention labels;
- committed-minority or committee membership;
- evaluator-derived macrostates;
- future stopping conditions.

Every interaction should re-label the focal agent as local **Player 1** and the anonymous co-player as local **Player 2**. Past partners are likewise represented generically as Player 2.

This privacy boundary is part of the experimental intervention, not a presentation preference.

---

## 5. Prompt contract

### 5.1 Prompt versioning

Create a versioned prompt contract, for example:

```text
ashery_2025_v1
```

The prompt implementation must be testable independently of providers and must expose a stable hash in artifacts.

### 5.2 Semantic system-prompt blocks

The source prompt has three components:

1. **Fixed game block**
   - repeated two-player partnership framing;
   - both players choose simultaneously;
   - legal action labels;
   - `+100` for a match and `-50` for a mismatch;
   - objective: maximize own accumulated points conditional on the other player's behavior.

2. **Dynamic memory block**
   - up to `H` prior local interactions;
   - locally numbered from oldest visible item to newest visible item;
   - focal agent shown as Player 1;
   - past co-player shown as Player 2;
   - both actions and the focal payoff shown for each entry;
   - current local round and visible-window score shown explicitly.

3. **Output instruction block**
   - ask for the action first and the reason second;
   - legal action must be machine-extractable;
   - no strategy examples and no recommended decision rule.

The user message should contain only a short request to choose Player 1's action.

### 5.3 Local round number

For an agent with visible memory length `h`:

```text
local_round = h + 1
```

When the full private history is longer than `H`, the visible records are re-numbered locally unless an exact legacy fixture demonstrates otherwise. The global simulation index must never appear.

### 5.4 Action-order randomization

For each player in each interaction, independently shuffle the presented list of legal names:

```text
presented_actions_i(t) = permutation_i,t(A)
presented_actions_j(t) = permutation_j,t(A)
```

The underlying legal-action set remains identical.

Store both permutations in the audit record. They are prompt variables, not game state.

### 5.5 Answer-first format

The supplement deliberately uses an answer-first, reason-later structure. The canonical provider-independent parsed object should be:

```json
{
  "value": "Q",
  "reason": "..."
}
```

The exact historical prompt uses a JSON-like representation rather than guaranteed standards-compliant JSON. The new package should distinguish:

- `raw_text`: exact provider output;
- `parsed_action`: one legal action;
- `parsed_reason`: optional text;
- `parser_mode`: strict JSON, tolerant paper-style object, constrained choice, etc.

The scientific action is `parsed_action`. Reasons are audit data and must never alter the transition.

### 5.6 No strategy injection

The prompt must not tell the agent to:

- imitate its partner's last action;
- play a majority action;
- explore;
- remain consistent;
- seek population consensus;
- use tit-for-tat;
- infer a “strong” convention.

The source prompt asks the agent to examine its history but does not prescribe how to transform history into a choice.

---

## 6. Provider and sampling profile

The supplement reports the following generation settings for the source experiments:

```yaml
temperature: 0.5
top_k: 10
max_tokens: 6
```

These values belong in a paper-replication provider profile, not inside the pure transition function.

Provider adapters differ in whether they support `top_k`, token limits, constrained choice, or deterministic seeds. The resolved run configuration must record:

- requested sampling parameters;
- parameters actually sent;
- unsupported parameters;
- returned provider/model identifier;
- provider retry counts;
- token usage when available.

Do not silently claim exact paper replication when a provider cannot implement the source sampling contract.

---

## 7. Validation and retries

The paper describes a consistent output format but does not fully specify an application-level retry protocol. Phase 6 requires one for robust execution. Treat it as infrastructure, not as a scientific rule from the paper.

### 7.1 Valid action

A response is valid only if exactly one legal action can be extracted under the configured parser contract.

A reason may be required by the paper-faithful `json_reason` format, but the transition depends only on the action.

### 7.2 Validation retry semantics

On invalid output:

1. do not mutate either agent;
2. do not advance the global interaction clock;
3. do not reveal the partner's attempted action;
4. retry only the invalid focal decision request;
5. use the identical pre-interaction memory and legal-action set;
6. record every validation attempt.

After `invalid_response_retries + 1` total validation attempts, raise a typed game error unless a separately versioned failure policy is configured.

Do not silently infer an action from free-form reasoning. Do not choose a random fallback in the paper-faithful profile.

### 7.3 Provider retries are different

Maintain separate counters for:

- logical decisions;
- validation attempts;
- provider/transport attempts;
- provider retries;
- forced decisions;
- permanent failures.

A provider retry is not a new game decision. A validation retry is a repeated attempt to realize one logical decision.

---

## 8. Pure transition

The pure transition receives two valid actions and the immutable pre-interaction state.

```python
def apply_transition(
    state: NamingConventionState,
    pair: tuple[AgentId, AgentId],
    actions: tuple[Action, Action],
) -> tuple[NamingConventionState, NamingConventionTransition]:
    ...
```

Required behavior:

1. verify the two agents are distinct and legal participants;
2. verify both actions belong to the configured pool;
3. compute `success` and common payoff;
4. append one perspective-correct memory entry to each agent;
5. update evaluator histories and scores;
6. increment the global interaction index exactly once;
7. return a complete immutable transition record.

Perspective-correct entries:

```text
agent i entry: own_action=x_i, partner_action=x_j
agent j entry: own_action=x_j, partner_action=x_i
```

The provider response, parser, latency, and retry metadata belong in the interaction audit record, not in the mathematical transition rule.

---

## 9. Reference interaction algorithm

```python
async def step(state, spec, provider, rng):
    # 1. Select pair from the current topology.
    i, j = sample_two_distinct_agents(state, spec, rng)

    # 2. Freeze both private views before either decision is requested.
    view_i = build_private_view(state, focal=i, partner=j, spec=spec, rng=rng)
    view_j = build_private_view(state, focal=j, partner=i, spec=spec, rng=rng)

    # 3. Compile provider-independent requests.
    request_i = build_decision_request(view_i, spec.prompt_contract)
    request_j = build_decision_request(view_j, spec.prompt_contract)

    # 4. Obtain simultaneous logical decisions.
    result_i, result_j = await gather(
        execute_validated_decision(provider, request_i, spec.validation),
        execute_validated_decision(provider, request_j, spec.validation),
    )

    # 5. Apply one pure transition after both valid actions exist.
    next_state, transition = apply_transition(
        state,
        pair=(i, j),
        actions=(result_i.action, result_j.action),
    )

    # 6. Attach provider/audit information outside the pure transition.
    interaction_record = assemble_interaction_record(
        pre_state=state,
        views=(view_i, view_j),
        requests=(request_i, request_j),
        decisions=(result_i, result_j),
        transition=transition,
        post_state=next_state,
    )

    return next_state, interaction_record
```

Population interactions are sequential. Only the two decisions inside one pair are concurrent.

---

## 10. State and record schemas

### 10.1 `NamingConventionGameSpec`

Minimum fields:

```yaml
population_size: int                 # N >= 2
actions: list[str]                   # unique, W >= 2
memory_size: int                     # H >= 0
success_payoff: 100
failure_payoff: -50
max_interactions: int                # fixed horizon T
topology: complete | adjacency
pair_sampling: uniform_two_distinct
simultaneous_pair_decisions: true
randomize_presented_action_order: true
prompt_contract: ashery_2025_v1
response_contract: json_reason
invalid_response_retries: int
stop_on_convergence: bool            # false in the strict fixed-horizon profile
convergence_rule: optional
```

### 10.2 `ConventionAgentState`

```yaml
agent_id: int
private_history: list[PrivateMemoryEntry]
lifetime_score: int
committed_action: null | Action      # extension point; null in base game
```

A bounded deque may be used for prompt memory only if complete evaluator history is stored elsewhere. Do not lose full histories needed for audit and analysis.

### 10.3 `PrivateMemoryEntry`

```yaml
agent_local_interaction_index: int
own_action: str
partner_action: str
payoff: int
success: bool
```

Persistent partner identity should not be required by the agent-visible memory object. It may be retained in evaluator-only metadata.

### 10.4 `ConventionGameState`

```yaml
global_interaction_index: int
agents: map[AgentId, ConventionAgentState]
action_pool: tuple[Action, ...]
topology: immutable topology state
evaluator_history: list[TransitionSummary]
terminated: bool
termination_reason: null | str
```

### 10.5 `ConventionDecisionRequest`

```yaml
agent_id: evaluator-only
local_role: Player 1
anonymous_partner_role: Player 2
visible_memory: list[PromptMemoryEntry]
visible_score: int
local_round: int
presented_actions: list[Action]
messages: list[ChatMessage]
prompt_contract: str
prompt_hash: str
```

### 10.6 `ConventionInteractionRecord`

At minimum:

```yaml
interaction_index: int
selected_agents: [int, int]
pre_interaction_private_memory:
  player_1: [...]
  player_2: [...]
presented_action_orders:
  player_1: [...]
  player_2: [...]
compiled_messages:
  player_1: [...]
  player_2: [...]
provider_results:
  player_1: {...}
  player_2: {...}
parsed_actions: [str, str]
parsed_reasons: [str | null, str | null]
validation:
  player_1: {...}
  player_2: {...}
success: bool
payoff: 100 | -50
post_interaction_private_memory:
  player_1: [...]
  player_2: [...]
visible_scores_before: [int, int]
lifetime_scores_after: [int, int]
forced_decisions: [bool, bool]
```

The record must be sufficient to reconstruct and independently verify the transition.

---

## 11. Time and stopping semantics

### 11.1 Fixed-horizon base profile

The safest source-faithful Phase 6 default is a fixed number `T` of pair interactions:

```text
terminate when global_interaction_index == T
```

The paper reports trajectories, convergence, stability, and critical-mass outcomes. It does not provide a complete generic software termination API for every base trial. Therefore, early stopping must be optional and explicitly versioned.

### 11.2 Population rounds

The article plots time in “population rounds,” but the prose sources used for this specification do not formally define the conversion. The existing project has used:

```text
1 population round = N pair interactions
```

If retained, encode this as an explicit project convention:

```python
population_round = global_interaction_index / population_size
```

Do not present it as a direct quotation or fully specified definition from the article without checking the original released code.

### 11.3 Observational consensus

A global convention is an evaluator-level property, never agent-visible.

For a finite stochastic trajectory, consensus should be defined through a documented windowed rule rather than a single matched pair. One paper criterion used in the committed-minority experiment is:

```text
at least 95% successful interactions over the previous 3N interactions
```

That criterion is specifically reported for detecting a consensus flip after introducing committed agents. Do not automatically reuse it as the sole termination rule for every base-game experiment.

Recommended Phase 6 behavior:

- run to the configured fixed horizon;
- compute convergence/consensus diagnostics offline or as evaluator-only state;
- optionally expose the `95% over 3N interactions` rule as a named criterion for later critical-mass experiments.

---

## 12. Paper configurations and Phase 6 smoke configuration

### 12.1 Main convention-emergence reference profile

The main paper's default setting is:

```yaml
population_size: 24          # N
memory_size: 5               # H
action_pool_size: 10         # W
success_payoff: 100
failure_payoff: -50
topology: complete
pair_sampling: uniform_two_distinct
empty_initial_memory: true
```

Use the exact action pool and model-specific configuration only when running a declared replication.

### 12.2 Two-convention profile

For collective-bias and later committed-minority work:

```yaml
population_size: 24
actions: [Q, M]
memory_size: 5
```

Some source experiments use different population or memory sizes for particular models. Those are experiment configurations, not core game rules.

### 12.3 Recommended Phase 6 smoke profile

A small smoke run may use:

```yaml
population_size: 6
actions: [Q, M]
memory_size: 3
max_interactions: 12
provider: mock
seed: 1
stop_on_convergence: false
```

This is an architecture test, not a scientific replication. The generated report and manifest must say so explicitly.

---

## 13. Resource-demand/call plan

For a base game with fixed horizon `T` and no forced decisions:

```text
logical decisions per interaction = 2
fixed logical decisions           = 2T
```

Let:

```text
R_v = invalid_response_retries
A_v = 1 + R_v  # maximum validation attempts per logical decision
```

Then:

```text
minimum provider completions = 2T
expected provider completions = derived from configured validation-failure assumption
maximum provider completions = 2T * A_v
```

Provider/transport retries are not additional logical decisions and should be costed by the provider layer separately.

The call plan should report:

```yaml
stages:
  - stage: pair_decision
    repetitions: T
    requests_per_stage: 2
    concurrency_within_stage: 2
    state_barrier_after_stage: true
forced_decisions: 0
validation_retry_bound_per_request: R_v
stopping_assumption: fixed_horizon
prompt_context_scenarios:
  - empty_memory: 0 entries
  - representative_memory: min(H, configured representative h)
  - maximum_memory: H entries
```

Because memory grows agent-locally and participation counts are random, token estimation must query prompt scenarios at multiple history lengths. Do not extrapolate all calls from the first empty-memory prompt.

The call plan must not import provider prices or select a provider.

---

## 14. Metrics retained by the game

The game engine should emit raw observables, not only plotted summaries.

Per interaction:

```text
selected pair
both actions
success indicator
payoff
both pre/post private memories
both visible scores
provider and validation metadata
```

Evaluator-level derived series may include:

```text
coordination indicator success(t)
action production count/share over both player outputs
per-agent participation count
per-agent lifetime score
rolling success rate
rolling action shares
```

Keep raw trajectories so that binning and smoothing choices can be changed offline. A figure's smoothing window is analysis configuration, not a transition rule.

---

## 15. Determinism and random streams

Use explicit seeded random streams for:

- pair sampling;
- action-order permutation for each focal prompt;
- deterministic mock-provider behavior;
- audit-trace sampling.

Provider stochasticity is a separate source of randomness and may not be exactly seedable across APIs.

A deterministic mock rerun with the same configuration and seed must reproduce:

- selected pairs;
- presented action orders;
- mock outputs;
- parsed actions;
- transitions;
- artifact hashes, excluding explicitly non-deterministic metadata such as wall-clock timestamps.

Switching from mock to a live provider must not change game code or the provider-independent call plan.

---

## 16. Required tests

### 16.1 Transition tests

- matching actions give both agents `+100`;
- mismatching actions give both agents `-50`;
- each memory entry has correct own/partner perspective;
- only the selected pair changes;
- transition increments the global clock once;
- invalid actions cannot enter the transition.

### 16.2 Simultaneity tests

- both prompts are compiled from the same pre-interaction state;
- neither prompt contains the other player's current output;
- provider completion order does not change the transition.

### 16.3 Privacy tests

Assert that ordinary prompts contain none of:

```text
persistent agent IDs
global interaction index
population size/global membership
population action counts
consensus status
committee/committed metadata
partner memory
```

### 16.4 Memory tests

- empty memory is represented correctly;
- at most `H` entries are visible;
- full evaluator history can exceed `H`;
- visible score equals the sum of visible payoffs;
- local round derives from visible history;
- empty-memory decisions still invoke the provider.

### 16.5 Prompt-order tests

- legal action order can differ between the two players;
- the order is seeded and auditable;
- shuffling does not alter the legal action set.

### 16.6 Validation tests

- malformed output does not mutate state;
- only the invalid focal decision is retried;
- retry exhaustion raises a typed error;
- free-form reason cannot be used to invent a fallback action.

### 16.7 Resource-plan tests

- fixed horizon `T` reports `2T` base logical calls;
- maximum calls include validation bounds;
- the plan is provider-independent;
- prompt scenarios include memory lengths `0` and `H`;
- planned and actual logical calls agree in a deterministic valid mock run.

---

## 17. Required audit checks for `selected_audit_traces.jsonl`

Every selected trace should make the following review possible without reading application code:

1. identify the selected pair;
2. inspect both bounded private memories before the interaction;
3. verify that no current action appears in the other prompt;
4. inspect independently shuffled action lists;
5. read both compiled message lists;
6. inspect raw responses and parser outputs;
7. inspect validation and retry status;
8. recompute match/mismatch payoff manually;
9. inspect both perspective-correct post-interaction memories;
10. confirm no population or committee information leaked.

---

## 18. Artifact-specific guidance

### `agents_initial.json`

Must show:

- all agents present;
- empty ordinary-agent histories for the spontaneous-emergence profile;
- zero lifetime scores;
- no committed actions in the base game.

### `interactions.jsonl`

One full audit row per global pair interaction.

### `selected_audit_traces.jsonl`

A deterministic subset with complete prompts, responses, memories, parsing, payoff, and post-state.

### `game_call_plan.json`

Provider-independent stages, fixed/expected/maximum logical demand, validation bounds, stopping assumptions, and prompt scenarios.

### `prompt_token_scenarios.csv`

At minimum:

```text
empty memory
representative partial memory
full H-entry memory
```

For each scenario, include both players' possible action-order permutations only if the tokenizer changes under label ordering; otherwise one representative permutation plus a conservative bound is sufficient.

### `trajectory.csv`

Recommended raw columns:

```text
interaction_index
population_round_project_convention
agent_i
agent_j
action_i
action_j
success
payoff
rolling_coordination_rate
share_<action> for each action
```

### `action_share.png`

Clearly state whether shares are instantaneous, rolling, cumulative, or binned. Prefer plotting a documented rolling or binned statistic from raw actions.

### `coordination_rate.png`

Clearly state the window. Do not label a single binary interaction outcome as a probability.

---

## 19. Explicit source ambiguities and implementation decisions

The paper and supplement provide the scientific mechanism but not every software-level behavior. The implementation must mark the following distinctions:

| Topic | Source-supported statement | Phase 6 implementation decision |
|---|---|---|
| First action | Empty-memory action is generated under model stochasticity and is used to measure individual bias. | Always call provider; never force a uniform action in paper-faithful profile. |
| Pair order | Two agents are randomly selected. | Sample two distinct agents uniformly; store an audit ordering only. |
| Population round | Used as plot time. | Existing project convention may define one round as `N` pair interactions; label it as project convention pending released-code verification. |
| Generic early stopping | Not fully specified for all base trials. | Fixed horizon by default; convergence is evaluator-only. |
| Invalid-output retry | Not fully specified. | Versioned, bounded validation retries with no state mutation. |
| Score shown to model | Article states accumulated score within memory range. | Compile visible score from the last `H` payoffs; store lifetime score separately. |
| Exact JSON compliance | Supplement shows a JSON-like answer-first format. | Preserve semantic format and raw output; use a versioned tolerant parser or strict modern JSON profile. |
| Top-k portability | Source uses top-k sampling. | Record unsupported provider parameters; do not claim exact replication when unavailable. |

---

## 20. Acceptance criterion for the scientific game

Phase 6 is complete when:

1. a population of ordinary agents can run the repeated convention game through the generic game interface;
2. every pair decision is simultaneous with respect to the pre-interaction state;
3. each agent sees only its own bounded local history and anonymous partner actions;
4. action ordering is independently randomized per player and interaction;
5. empty-memory behavior remains a provider decision;
6. the engine applies the symmetric `+100/-50` transition exactly;
7. one trace can be followed from private memory to prompt, raw response, parsed action, payoff, and post-memory;
8. mock runs are deterministic under a fixed seed;
9. the base game runs without committee logic;
10. the call plan is stage-aware, memory-aware, and provider-independent;
11. the report distinguishes architecture smoke tests from paper replications;
12. any deviation from the paper profile is named, versioned, and visible in the resolved configuration.

---

## 21. Minimal implementation checklist for the coding agent

- [ ] Create a dedicated repeated-convention game module, separate from inventory Naming Game code.
- [ ] Implement complete-mixing pair sampling for the paper profile.
- [ ] Freeze two private views before launching concurrent decisions.
- [ ] Implement bounded private prompt memory and separate complete evaluator history.
- [ ] Compute visible-window score explicitly.
- [ ] Keep first action as an LLM call under empty memory.
- [ ] Shuffle action order independently for each player and interaction.
- [ ] Implement versioned Ashery-style prompt blocks.
- [ ] Parse answer first, reason second; retain raw response.
- [ ] Implement bounded validation retries without state mutation.
- [ ] Apply a pure symmetric payoff/memory transition.
- [ ] Emit full interaction audit records.
- [ ] Implement fixed-horizon termination and evaluator-only convergence diagnostics.
- [ ] Generate memory-length-aware token scenarios.
- [ ] Verify provider-independent call-plan equality across provider choices.
- [ ] Add paper-profile, smoke-profile, privacy, simultaneity, and transition tests.

---

## 22. Source pointers for human verification

Consult these locations before changing the scientific contract:

- Main article, **Experimental setup**: population, random pair selection, finite name pool, payoff, private memory, empty-memory initialization, and committed agents.
- Main article, **Prompting**: external-observer framing, hidden identities, step-by-step instruction, randomized action-list order, and explicit payoff/score presentation.
- Main article, **Materials and Methods – Prompt**: fixed/dynamic/instructional blocks, zero-shot decision extraction, self-interested objective, and `+100/-50` payoff.
- Main article, **Committed minorities**: `95%` successful interactions over the preceding `3N` interactions as a consensus-flip criterion.
- Supplement, **Prompt Structure** and **Output Structure**: prompt decomposition and answer-first/reason-later rationale.
- Supplement, **Example Prompt**: concrete message organization and local Player 1/Player 2 representation.
- Supplement, **Table S4**: temperature `0.5`, top-k `10`, max tokens `6`.

