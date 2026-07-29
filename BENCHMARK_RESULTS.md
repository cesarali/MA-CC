# LLM Naming Game Benchmark Results

## 1. Purpose of the benchmark

This benchmark compares the runtime and observed population trajectories of two
different binary Naming Game update schemes:

1. **Sequential asynchronous:** one ordered speaker-listener pair interacts at
   a time. Its update is applied immediately, so the next pair observes the
   changed population.
2. **Synchronous parallel:** agents are shuffled into disjoint ordered pairs at
   the beginning of each round. All pairs read the same start-of-round state,
   their API call chains run concurrently, and all updates are applied together
   after every pair has finished.

The modes use the same number of pair interactions in every matched condition,
but they are not the same stochastic process. Runtime differences and Naming
Game trajectory differences must therefore be interpreted separately.

## 2. Variables and terminology

| Variable | Meaning |
|---|---|
| `N` | Number of agents in the population. The benchmark used `5`, `10`, and `20`. |
| `R` | Number of actual rounds in the synchronous game. It is also the synchronous-round equivalent used to label the matched sequential run. The benchmark used `5`, `10`, and `20`. |
| `floor(N / 2)` | Number of disjoint pairs in one synchronous round. For odd `N`, one agent is idle during that round. |
| `M` | Total number of speaker-listener pair interactions in one game run. `M = floor(N / 2) * R`. |
| `alpha` | Reasoning fraction. Each pair uses reasoning with probability `alpha`. This benchmark used `alpha = 0`, so every interaction followed the basic binary Naming Game. |
| `update_mode` | Either `sequential` or `synchronous_parallel`. |
| `seed` | Seed controlling initial inventories, pair selection or pairing, and local deterministic repairs. The benchmark used seed `1`. |
| `replicate` | Independent repetition of one configuration. The benchmark used one replicate. |
| `concurrency` | Maximum number of simultaneous chat-completion requests. The full benchmark used a limit of `20`. A later focused sweep compared limits `5`, `10`, and `20`. |
| `{A}`, `{B}`, `{A, B}` | The only valid agent inventories. `{A, B}` means the agent currently knows both names. |
| Naming-Game success | The listener already had the transmitted name. Both agents retain only that name. |
| Naming-Game failure | The listener did not have the transmitted name. The listener adds it and the speaker remains unchanged. |
| Consensus | Every agent has the same singleton inventory, either `{A}` or `{B}`. A population in which every agent has `{A, B}` is not counted as consensus. |

## 3. API calls per pair interaction

Every pair interaction always contains exactly two logical chat-completion API
calls:

```text
speaker API call
        -> selected name
listener API call
        -> listener response
```

Therefore:

```text
expected chat-completion calls per game = 2 * M
```

The listener call waits for its own speaker call. In synchronous mode, the
two-call chains belonging to different disjoint pairs may overlap. In
sequential mode, the next pair cannot begin until the current pair has been
updated.

The reported call counts refer to chat-completion calls. Model-discovery
requests such as `GET /models` are not included. A retry would increase the
actual call count above the expected logical call count, but no retries occurred
in the completed real benchmark.

## 4. Exact interaction and call budgets

For a synchronous run:

```text
pairs per round = floor(N / 2)
M = floor(N / 2) * R
```

The matched sequential run executes exactly `M` sequential steps. Consequently,
one sequential run and one synchronous run each use `M` pair interactions and
`2 * M` expected chat-completion calls.

| `N` | `R` | Pairs per synchronous round | `M` interactions per game | Sequential steps | API calls per game (`2M`) | API calls in matched two-mode pair (`4M`) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 2 | 10 | 10 | 20 | 40 |
| 5 | 10 | 2 | 20 | 20 | 40 | 80 |
| 5 | 20 | 2 | 40 | 40 | 80 | 160 |
| 10 | 5 | 5 | 25 | 25 | 50 | 100 |
| 10 | 10 | 5 | 50 | 50 | 100 | 200 |
| 10 | 20 | 5 | 100 | 100 | 200 | 400 |
| 20 | 5 | 10 | 50 | 50 | 100 | 200 |
| 20 | 10 | 10 | 100 | 100 | 200 | 400 |
| 20 | 20 | 10 | 200 | 200 | 400 | 800 |

Totals for the nine conditions for one model are:

| Scope | Pair interactions | Chat-completion calls |
|---|---:|---:|
| All sequential games for one model | 595 | 1,190 |
| All synchronous games for one model | 595 | 1,190 |
| Both modes for one model | 1,190 | 2,380 |
| Both modes for both required models | 2,380 | 4,760 |

The full real benchmark produced exactly 4,760 successful chat-completion calls,
which equals the expected count. It had zero failed attempts and zero retries.

## 5. Benchmark configuration

The full mock and real grids used:

```text
models:
  gwdg/qwen3-30b-a3b-instruct-2507
  microsoft/gpt-4o
N: 5, 10, 20
R: 5, 10, 20
alpha: 0
temperature: 0
seed: 1
replicates: 1
full-grid concurrency limit: 20
speaker max tokens: 20
listener max tokens: 20
```

The same seeded initial population was used for the two modes in each matched
condition. Pair selection differs between modes because sequential sampling and
synchronous disjoint pairing are different processes.

## 6. Automated test and artifact audit

The final automated test run was:

```text
34 passed in 0.17 seconds
```

Tests covered inventory validation, success and failure updates, malformed JSON
repair, engine-authoritative listener validation, pair isolation, private agent
histories, immediate sequential updates, synchronous update barriers, seeded
pairing, completion-order independence, matched interaction budgets, expected
API calls, result-file creation, and reasoning-task configuration errors.

An additional artifact audit checked all 82 recorded runs. For every run it
confirmed that:

- the interaction JSONL line count equals `M`;
- a sequential state CSV contains exactly one row per interaction;
- a synchronous round CSV contains exactly one row per round;
- the expected per-run files exist;
- the config model and backend agree with the summary.

## 7. Mock benchmark

The complete mock benchmark contained 36 games: 18 games for each model label,
covering nine conditions and two update modes. It made 4,760 simulated
chat-completion calls in total.

The mock client used deterministic JSON responses and 1 ms configured artificial
latency per request. The model labels do not select different mock behavior, so
small timing differences between the two labelled mock grids are execution
noise.

| Model label | Runs | Simulated calls | Sequential wall-time sum | Synchronous wall-time sum | Aggregate speed-up |
|---|---:|---:|---:|---:|---:|
| Qwen3 30B A3B | 18 | 2,380 | 1.632 s | 0.310 s | 5.26x |
| GPT-4o | 18 | 2,380 | 1.625 s | 0.315 s | 5.16x |

These mock timings verify scheduling behavior and output generation; they are
not estimates of real provider latency.

## 8. Real API pilot runs

Before the full grid, each model was tested with the matched `N = 5`, `R = 5`,
`M = 10` condition. Each game made 20 chat-completion calls.

| Model | Sequential time | Synchronous time | Pilot speed-up | Calls per game |
|---|---:|---:|---:|---:|
| Qwen3 30B A3B | 11.189 s | 2.766 s | 4.05x | 20 |
| GPT-4o | 17.700 s | 8.498 s | 2.08x | 20 |

All 80 pilot chat-completion calls succeeded without a retry. Both required
model identifiers were present in the live University proxy model list.

## 9. Complete real benchmark timing results

Speed-up is calculated separately for every matched condition as:

```text
speed-up = sequential wall time / synchronous wall time
```

Both modes in a row made the same number of pair interactions and API calls.

| `N` | `R` | `M` | Qwen sequential | Qwen synchronous | Qwen speed-up | GPT-4o sequential | GPT-4o synchronous | GPT-4o speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 10 | 5.773 s | 2.719 s | 2.12x | 17.211 s | 9.536 s | 1.80x |
| 5 | 10 | 20 | 27.127 s | 6.118 s | 4.43x | 37.729 s | 17.632 s | 2.14x |
| 5 | 20 | 40 | 24.610 s | 10.638 s | 2.31x | 68.871 s | 39.160 s | 1.76x |
| 10 | 5 | 25 | 20.163 s | 3.180 s | 6.34x | 40.497 s | 8.703 s | 4.65x |
| 10 | 10 | 50 | 50.650 s | 13.326 s | 3.80x | 80.642 s | 18.892 s | 4.27x |
| 10 | 20 | 100 | 81.594 s | 27.869 s | 2.93x | 161.301 s | 35.739 s | 4.51x |
| 20 | 5 | 50 | 38.018 s | 10.991 s | 3.46x | 90.709 s | 9.671 s | 9.38x |
| 20 | 10 | 100 | 100.986 s | 20.546 s | 4.91x | 177.538 s | 21.617 s | 8.21x |
| 20 | 20 | 200 | 163.532 s | 34.384 s | 4.76x | 357.665 s | 47.450 s | 7.54x |

Aggregate results across the nine conditions:

| Model | Sequential wall-time sum | Synchronous wall-time sum | Aggregate speed-up | Mean matched speed-up | Matched speed-up range |
|---|---:|---:|---:|---:|---:|
| Qwen3 30B A3B | 512.453 s | 129.770 s | 3.95x | 3.90x | 2.12x-6.34x |
| GPT-4o | 1,032.162 s | 208.399 s | 4.95x | 4.92x | 1.76x-9.38x |

Real full-grid API accounting:

| Model | Games | Pair interactions | Expected calls | Actual calls | Failed attempts | Retries | Total tokens reported |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 30B A3B | 18 | 1,190 | 2,380 | 2,380 | 0 | 0 | 190,595 |
| GPT-4o | 18 | 1,190 | 2,380 | 2,380 | 0 | 0 | 189,967 |

No malformed or semantically inconsistent speaker or listener response occurred
in the full real grid. The game engine nevertheless remains authoritative and
would log and locally repair an invalid response.

## 10. Observed Naming Game trajectories

The two update modes produced different trajectories in all nine matched
conditions for both model runs. The two models produced the same trajectory
results in this experiment because their short, temperature-zero JSON actions
selected the same names. This should not be generalized to other prompts,
temperatures, seeds, or reasoning tasks.

`@k` below means that consensus was first recorded at cumulative pair
interaction index `k`.

| `N` | `R` | Sequential result | Synchronous result |
|---:|---:|---|---|
| 5 | 5 | No consensus; final `3 A, 1 B, 1 AB` | `{A}` consensus `@6` |
| 5 | 10 | No consensus; final `3 A, 1 B, 1 AB` | `{A}` consensus `@6` |
| 5 | 20 | `{A}` consensus `@31` | `{A}` consensus `@6` |
| 10 | 5 | No consensus; final `9 A, 0 B, 1 AB` | `{A}` consensus `@25` |
| 10 | 10 | `{A}` consensus `@28` | `{A}` consensus `@25` |
| 10 | 20 | `{A}` consensus `@28` | `{A}` consensus `@25` |
| 20 | 5 | No consensus; final `10 A, 6 B, 4 AB` | `{A}` consensus `@50` |
| 20 | 10 | `{A}` consensus `@97` | `{A}` consensus `@50` |
| 20 | 20 | `{A}` consensus `@97` | `{A}` consensus `@50` |

Four of the nine conditions differed in final consensus status or final
population composition. The other five reached the same final `{A}` consensus
but did so at different interaction indices. This is expected because
synchronous batching changes the dynamics, not just the runtime.

## 11. Concurrency sweep

A focused synchronous sweep used `N = 20`, `R = 5`, and therefore `M = 50`
pair interactions and 100 chat-completion calls per game. Every round contained
10 disjoint pairs.

| Model | Concurrency 5 | Concurrency 10 | Concurrency 20 repeat | Fastest controlled-sweep setting |
|---|---:|---:|---:|---:|
| Qwen3 30B A3B | 4.859 s | 2.913 s | 4.286 s | 10 |
| GPT-4o | 18.796 s | 11.583 s | 15.694 s | 10 |

All six sweep games completed without failed attempts or retries. A limit of 10
is also the largest concurrency that can be filled by one `N = 20` round,
because only 10 pair chains exist. Increasing the semaphore limit to 20 cannot
create more disjoint pairs. Shared provider load introduces timing variability,
so these values should be treated as measurements of these runs rather than a
permanent service guarantee.

Sequential games used one trajectory at a time because the configured replicate
count was one. Concurrency across multiple independent sequential trajectories
was not measured, and no claim is made about its optimal value.

## 12. Total calls made during evaluation

The full-grid results above should be used for the matched comparison. Additional
pilot and concurrency runs were executed as validation:

| Evaluation component | Games | Pair interactions | Chat-completion calls |
|---|---:|---:|---:|
| Full mock grid | 36 | 2,380 | 4,760 simulated |
| Real pilots | 4 | 40 | 80 real |
| Full real grids | 36 | 2,380 | 4,760 real |
| Real concurrency sweep | 6 | 300 | 600 real |
| **All real evaluation** | **46** | **2,720** | **5,440 real** |

The University proxy's model-list requests are excluded from these
chat-completion totals.

## 13. Practicality and limitations

- `N = 20`, `R = 20` completed successfully for both models and both modes.
- The Qwen maximum condition took 163.532 seconds sequentially and 34.384
  seconds synchronously.
- The GPT-4o maximum condition took 357.665 seconds sequentially and 47.450
  seconds synchronously.
- The maximum condition is practical for one replicate, but sequential GPT-4o
  execution takes approximately six minutes per trajectory.
- Only one replicate and one seed were used, so timing variation and stochastic
  trajectory variation are not estimated.
- The benchmark used `alpha = 0`; it does not evaluate reasoning quality.
- The service is shared infrastructure. Runtime can change with provider load,
  even when the interaction and call budgets are identical.
- No conclusion about phase transitions or consensus physics should be drawn
  from this implementation and runtime benchmark.

Generated per-run CSV and JSONL artifacts remain local under `results/`, which
is intentionally ignored by Git. This Markdown report is stored at the
repository root so the benchmark methodology and principal results can be
version controlled without committing generated artifacts.
