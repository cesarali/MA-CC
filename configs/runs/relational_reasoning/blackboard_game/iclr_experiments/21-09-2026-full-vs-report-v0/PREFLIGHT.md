# Preflight: 21-09-2026-full-vs-report-v0

Credential-free study preflight on 2026-09-21: **PASS / permitted**.

## Workload

| Quantity | Value |
|---|---:|
| Configurations | 2 |
| Scientific cells | 20 |
| Episodes | 100 |
| Rounds per episode | 10 |
| Controller sensing | previous public board, sample size 12 |
| Controller budgets | 3, 12 |
| Nominal provider calls | 24,800 |
| Expected provider calls | 29,100 |
| Conservative provider calls | 122,400 |

The nominal count consists of 240 ordinary-agent calls per episode plus up to
10 controller-authoring calls in each of the 80 controlled episodes.

## Token and cost estimate

| Scenario | Input tokens | Output tokens | Estimated cost |
|---|---:|---:|---:|
| Lower / nominal | 26,968,800 | 24,800 | USD 1.00 |
| Representative expected | 31,784,800 | 119,193,600 | USD 21.44 |
| Conservative retry/context bound | 134,666,400 | 501,350,400 | USD 90.21 |

Cost uses the repository's offline DeepInfra snapshot retrieved 2026-09-16:
USD 0.037 per million ordinary input tokens and USD 0.17 per million output
tokens for `openai/gpt-oss-120b`. Token counts use the repository's regex
estimator rather than the provider tokenizer. The expected value is a planning
estimate; the conservative value is a bound under configured retry and maximum
context assumptions.

The public DeepInfra model page was checked on 2026-09-21 and showed the same
rates: <https://deepinfra.com/openai/gpt-oss-120b>. Authenticated account limits
were not queried during this credential-free preflight.

## Execution envelope

- repetitions / in-process parallelism: 5 / 5
- one cell per shard
- array throttle / maximum active cell shards: 20 / 20
- provider request concurrency: 10 per shard
- configured study-wide initial / maximum coordinator concurrency: 100 / 100
- simultaneous episode slots / CPU allocation: 100 / 100 CPUs
- target RPM: 500
- assumed provider latency: 90 seconds
- requested resources per shard: 5 CPUs, 12 GiB, 12 hours

The preflight's summed per-config rough-runtime estimate is approximately 4.85
hours, but this is not a scheduler wall-time guarantee and does not incorporate
a live model-specific latency probe. Before any paid submission, refresh the
DeepInfra model price and authenticated account limits and regenerate the
submission execution plan.

No provider calls or scheduler submission were made by this preflight.
