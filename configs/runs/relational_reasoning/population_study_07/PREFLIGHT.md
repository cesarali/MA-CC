# Study 07 preflight and resource report

Generated 2026-08-25 from live-metadata, no-experiment preflights. Both configs
were `permitted`.

| block | cells | reps/cell | episodes | nominal calls | expected calls | conservative calls |
|---|---:|---:|---:|---:|---:|---:|
| fine-beta `(b,beta)` | 144 | 10 | 1,440 | 380,160 | 401,760 | 760,320 |
| truth-aligned `(b,theta)` | 120 | 10 | 1,200 | 316,800 | 334,800 | 633,600 |
| **total** | **264** | **10** | **2,640** | **696,960** | **736,560** | **1,393,920** |

## Token and cost estimates

| block | expected input | expected output bound | conservative input | conservative output bound |
|---|---:|---:|---:|---:|
| fine-beta | 183,254,400 | 1,645,608,960 | 346,913,280 | 3,114,270,720 |
| truth-aligned | 152,712,000 | 1,371,340,800 | 289,094,400 | 2,595,225,600 |
| **total** | **335,966,400** | **3,016,949,760** | **636,007,680** | **5,709,496,320** |

Output values are `max_output_tokens=4096` reservation bounds, not predicted
realized completions. Token counts use the repository's deterministic regex
approximation. Live input/output prices were 0.00 proxy accounting units for
`gwdg/openai-gpt-oss-120b`; this is not asserted to be a currency.

## Execution and runtime

The Study 06 execution policy is unchanged: automatic original-cell sharding,
array throttle 18, experiment parallelism 8, provider request concurrency 8,
144 simultaneous request slots, and an estimated 864 RPM at the configured
10-second planning latency. Each shard requests 8 CPUs, 8 GB, and four hours.

The scheduler record for Study 06 spans approximately 7 h 48 m from first task
start to last task end; its cell tasks averaged about 52 minutes. Scaling that
observed 156-cell workload to 264 cells at the same throttle gives an expected
Study 07 wall time of about **13 hours**, subject to queueing, provider latency,
and retries. The static single-process preflight totals 76 h 43 m and is not the
cell-array wall-time prediction.

Preflight IDs:

- fine-beta: `92c142fbbc2b35d69407f5c1cefe65ee30d2755618bb88715198220e0a185083`
- truth-aligned: `e981d58547dfa0d744085989f77cb809490851aaf77aa9cbd4bc92e75ccd4f83`

