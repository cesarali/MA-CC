# Implement and Benchmark Sequential and Synchronous LLM Naming Games

## Objective

Create a small Python implementation of a binary Naming Game that uses the university LLM API.

The immediate objective is to compare the runtime of two genuinely different game dynamics:

1. **Sequential asynchronous Naming Game**
   - one speaker-listener pair interacts at a time;
   - the pair is updated immediately;
   - the next pair observes the updated population.

2. **Synchronous parallel Naming Game**
   - disjoint speaker-listener pairs interact concurrently within a round;
   - all pairs read the same start-of-round state;
   - all state updates are applied simultaneously after the round finishes.

Test population sizes:

```text
N = 5, 10, 20
```

Test equivalent interaction budgets corresponding to:

```text
5, 10, 20 synchronous rounds
```

The comparison must be normalized by the total number of pair interactions, not by the raw number of rounds or steps.

The repository is mostly empty, so create the package structure, API wrapper, game implementations, CLI, tests, and benchmark output.

---

## Required university API models and integration guide

Use exactly these two model identifiers in the benchmark:

```text
gwdg/qwen3-30b-a3b-instruct-2507
microsoft/gpt-4o
```

Run the complete matched benchmark separately for each model.

Do not guess the university API endpoint, authentication method, headers, payload format, environment-variable names, response schema, rate-limit handling, or model-routing conventions.

Before implementing `api_client.py`, read and follow:

```text
docs/university_llm_api.md
```

Treat that document as the authoritative integration guide for the repository.

The implementation must:

- use the endpoint and authentication mechanism defined in `docs/university_llm_api.md`;
- use the documented environment-variable names;
- follow the documented request and response schemas;
- preserve the exact model identifiers shown above;
- implement retries and error handling consistently with the guide;
- document any incompatibility between the guide and the actual API response;
- avoid introducing an alternative SDK or endpoint unless the guide explicitly permits it.

The benchmark summary must include the model identifier for every run.


---

## Core parameters

Introduce:

```text
reasoning_fraction = alpha, with 0 <= alpha <= 1
```

Semantics:

- `alpha = 0`: every interaction follows the basic binary Naming Game;
- `alpha = 1`: every interaction follows the reasoning Naming Game;
- `0 < alpha < 1`: independently for each pair interaction, use the reasoning interaction with probability `alpha`; otherwise use the basic interaction.

For the current benchmark, use only:

```text
alpha = 0
```

Also introduce:

```text
update_mode = sequential | synchronous_parallel
```

These are different stochastic processes and must not be conflated.

---

## Basic binary Naming Game: `alpha = 0`

Each agent has one inventory:

```text
{A}
{B}
{A, B}
```

Initialize the population approximately evenly between `{A}` and `{B}`. For odd `N`, assign the remaining agent using the configured random seed.

### One pair interaction

For a speaker-listener pair:

1. The speaker receives its current inventory.
2. The speaker selects one name from its inventory:
   - from `{A}`, select `A`;
   - from `{B}`, select `B`;
   - from `{A, B}`, select either `A` or `B`.
3. The listener receives:
   - its current inventory;
   - the transmitted name.
4. The listener reports whether that name was already in its inventory.
5. The game engine validates the response and applies the Naming-Game rule:
   - **success:** if the listener already had the name, both agents retain only that name;
   - **failure:** if the listener did not have the name, the listener adds it and the speaker remains unchanged.

Use the LLM API for both the speaker action and the listener response. The purpose of this first experiment is to measure API execution time under the two update schemes.

Require short JSON outputs.

Speaker:

```json
{
  "selected_name": "A"
}
```

Listener:

```json
{
  "already_known": true
}
```

The game engine is the source of truth for whether the listener already knew the transmitted name. Log disagreements as invalid responses and repair or retry them.

No arithmetic, factual, or reasoning question is needed when `alpha = 0`.

---

## Reasoning Naming Game: `alpha = 1`

Create a separate interface for this mode, but do not run it in the current benchmark.

A reasoning task must be loaded explicitly from a file or configuration. Do not hardcode trivial questions.

In this mode:

1. the speaker selects `A` or `B` and generates a reason;
2. the listener receives the claim, reason, its own evidence, and current state;
3. the listener returns a new state from `{A}`, `{B}`, or `{A, B}`;
4. the listener update is determined by reasoning rather than by the deterministic basic Naming-Game success rule.

Use the same two-call pair structure:

- one speaker API call;
- one listener API call.

If `alpha > 0` is requested without a reasoning-task specification, stop with a clear configuration error.

---

# Update mode 1: Sequential asynchronous Naming Game

This is the canonical random-sequential version.

For each interaction step:

1. sample one ordered speaker-listener pair uniformly from all distinct agents;
2. execute the speaker API call;
3. execute the listener API call;
4. apply the resulting state update immediately;
5. record the new population state;
6. sample the next pair from the updated population.

Formally:

```text
X_(m+1) = Phi_(i_m, j_m)(X_m)
```

where `m` counts pair interactions.

Within one trajectory, pair interactions are sequential because each new interaction depends on the preceding update.

To use API concurrency without changing the game dynamics, run multiple independent sequential trajectories concurrently. Do not run multiple pair interactions from the same sequential trajectory concurrently.

### Sequential-mode concurrency

Concurrency may occur across:

- independent random seeds;
- independent replicates;
- independent benchmark configurations.

Concurrency must not occur between dependent interaction steps in the same trajectory.

---

# Update mode 2: Synchronous parallel Naming Game

This is a parallel-round approximation and is not identical to the canonical process.

For each round:

1. snapshot the complete population state;
2. shuffle all agents using the configured random generator;
3. divide agents into disjoint ordered speaker-listener pairs;
4. if `N` is odd, leave one agent idle;
5. execute all pair interactions concurrently;
6. every pair must read the same start-of-round snapshot;
7. wait for every pair in the round to complete;
8. apply all pair updates simultaneously;
9. record the new population state.

Formally:

```text
X_(r+1) = Phi_(P_r)(X_r)
```

where `P_r` is the set of disjoint pairs in round `r`.

No agent may participate in more than one pair in a round.

API response order must not affect the resulting state.

---

## Fair comparison between modes

Do not compare:

```text
5 sequential steps
```

with:

```text
5 synchronous rounds
```

because they contain different numbers of pair interactions.

Define:

```text
M = total number of pair interactions
```

For a synchronous run with population size `N` and `R` rounds:

```text
M = floor(N / 2) * R
```

The matching sequential run must use exactly the same `M`.

For example:

| N | Synchronous rounds R | Pair interactions M | Sequential steps |
|---:|---:|---:|---:|
| 5 | 5 | 10 | 10 |
| 10 | 5 | 25 | 25 |
| 20 | 5 | 50 | 50 |
| 5 | 10 | 20 | 20 |
| 10 | 10 | 50 | 50 |
| 20 | 10 | 100 | 100 |
| 5 | 20 | 40 | 40 |
| 10 | 20 | 100 | 100 |
| 20 | 20 | 200 | 200 |

Because every pair interaction uses two API calls, the expected number of API calls is:

```text
2 * M
```

Thus both modes must use the same total number of pair interactions and the same expected number of API calls for each matched benchmark.

---

## Benchmark matrix

Run:

```text
model in {
  gwdg/qwen3-30b-a3b-instruct-2507,
  microsoft/gpt-4o
}
N in {5, 10, 20}
R in {5, 10, 20}
alpha = 0
update_mode in {sequential, synchronous_parallel}
```

For every `(N, R)` condition:

1. compute `M = floor(N / 2) * R`;
2. run the synchronous mode for exactly `R` rounds;
3. run the sequential mode for exactly `M` pair interactions;
4. use the same initial state distribution;
5. use matched random seeds where meaningful;
6. record runtime and dynamics separately.

Use at least one replicate initially. Keep the number of replicates configurable.

---

## Repository structure

Create:

```text
repo/
├── pyproject.toml
├── README.md
├── .env.example
├── configs/
│   └── speed_test.yaml
├── src/
│   └── naming_game/
│       ├── __init__.py
│       ├── api_client.py
│       ├── models.py
│       ├── agent.py
│       ├── interaction.py
│       ├── sequential_game.py
│       ├── synchronous_game.py
│       ├── reasoning_game.py
│       ├── runner.py
│       ├── benchmark.py
│       └── cli.py
├── tests/
│   ├── test_interaction.py
│   ├── test_sequential_game.py
│   ├── test_synchronous_game.py
│   ├── test_mock_api.py
│   └── test_benchmark.py
└── results/
    └── .gitkeep
```

Do not add extra agent types, committees, network topologies, dashboards, or control logic.

---


## Strict agent and pair isolation

Each agent must be represented by a distinct `Agent` instance with its own:

- `agent_id`;
- current inventory;
- private interaction history;
- model configuration, if agent-specific;
- random-number-generator state, if needed.

All agents may use the same `AsyncLLMClient` object for HTTP connection pooling, authentication, rate limiting, and concurrency control. However, the API client must be stateless with respect to conversations.

Sharing the API client does **not** permit sharing:

- message histories;
- prompts containing another agent's private state;
- provider conversation IDs;
- cached model responses;
- hidden summaries;
- mutable prompt buffers;
- outputs from unrelated pairs.

It is not necessary to define a different Python class for every agent. Use multiple independent instances of the same `Agent` class:

```python
agents = [
    Agent(agent_id=i, inventory=initial_inventory[i])
    for i in range(num_agents)
]
```

The important requirement is independent state, not different class definitions.

### One API call belongs to one acting agent

Every speaker action must be generated by a separate API request constructed only from that speaker's permitted information.

Every listener action must be generated by a separate API request constructed only from that listener's permitted information plus the message produced by its current speaker.

For one pair `(speaker_i, listener_j)`, the call sequence is:

```text
speaker_i API call
        |
        v
speaker message
        |
        v
listener_j API call
```

The listener call waits for its paired speaker call. Different pairs may execute these two-call chains concurrently.

For example, a synchronous round with pairs

```text
(0, 3), (1, 4), (2, 5)
```

should execute three independent tasks concurrently:

```text
Task 1: agent 0 speaker call -> agent 3 listener call
Task 2: agent 1 speaker call -> agent 4 listener call
Task 3: agent 2 speaker call -> agent 5 listener call
```

No task may inspect the prompts, responses, inventories, or histories of another task.

### Information allowed in the basic game

For `alpha = 0`, the speaker request may contain only:

- the fixed game instruction;
- the speaker's own inventory;
- the speaker's own identifier only if needed for logging.

The listener request may contain only:

- the fixed game instruction;
- the listener's own inventory;
- the name transmitted by the paired speaker;
- the listener's own identifier only if needed for logging.

Do not provide either agent with:

- the population state;
- the inventories of other agents;
- the identities or messages of other pairs;
- the number of successful or failed interactions elsewhere;
- the global transcript;
- the round outputs before the update barrier;
- the correct result computed by the game engine.

The complete experiment log is for the evaluator only and must never be inserted automatically into later agent prompts.

### Provider-side conversation state

Prefer stateless chat-completion requests where the complete permitted message list is supplied explicitly on every call.

If the university API requires persistent conversation or session identifiers:

- create a unique session for each agent;
- never reuse one session across agents;
- never reuse one session across independent benchmark replicates;
- document exactly what server-side state is retained;
- provide an option to disable persistent sessions.

A pair must not share one provider conversation session. The speaker response is passed explicitly as text into the listener's independent request.

### Sequential-mode isolation

In sequential mode:

1. select one pair;
2. snapshot only the two selected agents' permitted states;
3. call the speaker;
4. call the listener;
5. update only those selected agents;
6. proceed to the next interaction.

The next pair may observe changes only through the updated state of an agent that it actually contains. It must not receive the previous pair's transcript.

### Synchronous-mode isolation

In synchronous parallel mode:

1. construct an immutable start-of-round snapshot;
2. create disjoint pairs;
3. create one independent coroutine per pair;
4. give each coroutine only the two relevant agent snapshots;
5. buffer all proposed updates;
6. apply updates only after every pair coroutine finishes.

The API completion order must not affect prompts or state updates.

Because pairs are disjoint, every agent makes at most one speaker or listener action in a round.

### Required isolation tests

Add tests proving that:

1. two agents never share the same mutable history object;
2. modifying one agent's history does not change another agent's history;
3. one pair task cannot access another pair's prompt or output;
4. synchronous pair prompts are constructed solely from the immutable round snapshot;
5. sequential interactions do not append transcripts to non-participating agents;
6. provider conversation or session IDs are unique per agent when sessions are used;
7. independent replicates use fresh agent instances and fresh sessions;
8. evaluator logs are never included in agent prompts;
9. a shared API client contains no per-agent message history;
10. changing API completion order does not change a seeded synchronous update result.


---

## API client

In `api_client.py`, wrap the university API already used by the project.

Provide an asynchronous interface such as:

```python
class AsyncLLMClient:
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        ...
```

Requirements:

- use `asyncio`;
- limit simultaneous requests with `asyncio.Semaphore`;
- read credentials and endpoints from environment variables;
- do not hardcode secrets;
- include request timeouts;
- retry transient errors with bounded exponential backoff;
- record latency, retries, status, and token usage when available;
- provide a mock implementation with configurable artificial latency;
- remain stateless with respect to agent conversations;
- accept a complete agent-specific message list for every request;
- never store or merge message histories between calls.

---

## Configuration

Create `configs/speed_test.yaml`:

```yaml
models:
  - gwdg/qwen3-30b-a3b-instruct-2507
  - microsoft/gpt-4o
temperature: 0.0
max_tokens_speaker: 20
max_tokens_listener: 20
timeout_seconds: 60
max_retries: 2
concurrency: 20
seed: 1

reasoning_fraction: 0.0
update_modes:
  - sequential
  - synchronous_parallel

agent_sizes: [5, 10, 20]
synchronous_round_counts: [5, 10, 20]
replicates: 1
```

The concurrency value must be configurable from the CLI.

---

## Required runtime measurements

For every run, record:

- model identifier;
- `N`;
- `alpha`;
- update mode;
- synchronous-round equivalent `R`;
- total pair interactions `M`;
- expected API calls;
- actual API calls;
- random seed;
- concurrency limit;
- total wall-clock time;
- time per pair interaction;
- time per synchronous-round equivalent;
- time per actual synchronous round, when applicable;
- successful calls;
- failed calls;
- retries;
- mean request latency;
- median request latency;
- p90 request latency;
- maximum request latency;
- achieved API calls per second;
- token usage when available;
- number of successful Naming-Game interactions;
- number of failed Naming-Game interactions;
- counts of `{A}`, `{B}`, and `{A, B}` over time;
- whether consensus was reached;
- interaction index at consensus, if reached.

For synchronous mode, also record:

- number of parallel pairs per round;
- round wall-clock time;
- slowest pair latency per round.

For sequential mode, also record:

- total trajectory time;
- whether multiple independent trajectories were run concurrently;
- number of concurrent trajectories.

---

## Output files

Write:

```text
results/interactions_<run_id>.jsonl
results/states_<run_id>.csv
results/rounds_<run_id>.csv
results/benchmark_summary.csv
results/config_<run_id>.json
```

For sequential runs, `states_<run_id>.csv` should contain one row per pair interaction.

For synchronous runs, `rounds_<run_id>.csv` should contain one row per round.

---

## CLI

Support one sequential run:

```bash
python -m naming_game.cli run   --update-mode sequential   --num-agents 10   --num-interactions 50   --reasoning-fraction 0   --seed 1
```

Support one synchronous run:

```bash
python -m naming_game.cli run   --update-mode synchronous_parallel   --num-agents 10   --rounds 10   --reasoning-fraction 0   --concurrency 20   --seed 1
```

Support the matched benchmark:

```bash
python -m naming_game.cli benchmark   --agent-sizes 5 10 20   --synchronous-round-counts 5 10 20   --update-modes sequential synchronous_parallel   --reasoning-fraction 0   --replicates 1   --concurrency 20
```

Also support:

```bash
python -m naming_game.cli benchmark --mock
```

The benchmark command must automatically convert every synchronous round count `R` into the matched sequential interaction count:

```text
M = floor(N / 2) * R
```

---

## Tests

Tests must not call the real API.

### Shared interaction tests

1. only `{A}`, `{B}`, and `{A, B}` are valid inventories;
2. the success update is correct;
3. the failure update is correct;
4. malformed JSON is repaired or reported;
5. expected API-call counts are correct.

### Sequential-mode tests

6. exactly one pair is active per trajectory step;
7. each update is applied immediately;
8. the next interaction observes the preceding update;
9. exactly `M` pair interactions are executed;
10. seeded pair sampling is reproducible;
11. no internal concurrency is used within one trajectory.

### Synchronous-mode tests

12. every agent appears in at most one pair per round;
13. odd `N` leaves exactly one idle agent;
14. all pairs read the same start-of-round snapshot;
15. no update is visible before the round finishes;
16. updates are independent of API completion order;
17. seeded pairing is reproducible.

### Benchmark tests

18. matched sequential and synchronous runs use the same `M`;
19. matched runs expect the same number of API calls;
20. mock benchmark creates all result files;
21. requesting `alpha > 0` without a reasoning task raises a clear error.

---

## Execution sequence

1. Read `docs/university_llm_api.md` completely.
2. Create the package structure.
3. Implement typed models and one shared pair-interaction function.
4. Implement the mock asynchronous API client.
5. Implement the sequential game.
6. Implement the synchronous parallel game.
7. Implement the university API adapter exactly as documented.
8. Add logging and matched benchmark summaries.
9. Add the CLI.
10. Run `pytest`.
11. Run the complete matched benchmark in mock mode.
12. Run a real sequential test with `N=5` and `M=10` for each model.
13. Run the matched real synchronous test with `N=5` and `R=5` for each model.
14. If both models succeed, run the complete real benchmark grid separately for each model.

---

## Final report

Report:

1. files created;
2. commands executed;
3. test results;
4. mock benchmark results;
5. real-API timing results;
6. API errors or rate limits;
7. the fastest stable concurrency setting for synchronous rounds;
8. the useful number of concurrent sequential trajectories;
9. wall-clock speed-up of synchronous execution over matched sequential execution;
10. whether synchronous batching changes the observed consensus trajectory;
11. whether `N=20`, `R=20` is practical in both modes.

Do not treat the two update modes as equivalent dynamics. Report runtime differences and game-trajectory differences separately.

Do not claim a result about phase transitions, consensus physics, or reasoning quality from this experiment. It is an implementation and runtime benchmark comparing two update schemes.
