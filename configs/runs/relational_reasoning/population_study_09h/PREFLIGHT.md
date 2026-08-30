# Study 09h provider-free preflight

Preflight date: 2026-08-30. No completion, pricing, account, or scheduler call
was made. The standard static estimator used a clearly marked zero-price
planning quote; live availability, pricing, account limits, and the dedicated
Potsdam `MA-CC` imports must be refreshed before any authorized submission.

## Resolved scientific contract

- Status: **PERMITTED**
- Strict contract: `relational_persistence_high_statistics_false_v1`
- Frozen task: `n12_L3_r03_k3/task_0002`; truth `NORTH`; target `NORTHWEST`
- Fixed: `N=12`, `L=3`, rounds `30`, `q_c=6`, receiver `naive`,
  evidence `strategic`, `recommendation_plus_fact`, soft schedule, `beta=4.0`,
  `theta=0.75`
- Exact axes, in stable order:
  `q={1,2}` x `rho={0.80,0.85}` x `b={3,4,6,8,9,12}`
- Structural cells: `24`; target repetitions per cell: `15`; episodes: `360`
- Theory comparison: disabled; permanent analysis tables: CSV
- Production slot semantics: one controller slot; q=1 has zero ordinary peers
  on controlled updates, q=2 has exactly one ordinary peer

## Reuse audit

Scientifically eligible sources are only Study 09d's eight q=2/strategic cells
at `rho={0.80,0.85}` and `b={3,6,9,12}`, with ten repetitions per cell.
Study 09f is excluded because L=2. No neutral, q=1, new-budget, different-rho,
or different-target episode is eligible.

The audit key is exact equality of task, q, L, rho, budget, evidence strategy,
target semantics, receiver disposition, message mode, beta, theta, rounds, and
controller semantics; seed-level duplicates count only once.

The Potsdam result root could not be read from this machine (`Permission denied
(publickey)`), so **zero episodes are currently certified reusable**. Therefore
the safe new-episode count is 360. If all sealed Study 09d candidates pass
artifact, schema, seed, prompt, task, and resolved scientific-condition checks,
the unique reuse becomes 80 and the new count becomes 280. Submission stays
blocked until this audit is completed; YAML similarity alone is not reuse.

## Provider, token, cost, and runtime estimate

| Quantity | Unreused | With all 80 candidates validated |
|---|---:|---:|
| New episodes | 360 | 280 |
| Nominal provider calls | 133,920 | 104,160 |
| Expected provider calls | 141,120 | 109,760 |
| Conservative provider calls | 267,840 | 208,320 |
| Expected input tokens | 59,502,600 | 46,037,240 |
| Conservative input tokens | 112,950,720 | 87,389,760 |
| Expected output-token bound | 578,027,520 | 449,576,960 |
| Conservative output-token bound | 1,097,072,640 | 853,278,720 |
| Static rough runtime at 200 RPM | 11.76 h | 9.15 h |

Token figures are deterministic planning bounds based on the prompt estimator
and configured 4,096-token output cap, not provider tokenizer measurements.
The planning monetary result is 0.0 `proxy_accounting_unit`; this is not a
currency quote and must be replaced by a fresh live quote before launch.

## Execution plan and safe split

- 24 scientific-cell shards; generic `run_study_cell_array.job`; no
  study-specific job file
- Array throttle 2; 10 episode slots and 10 request permits per shard
- Effective ceilings: 20 episode slots and 20 provider requests across two
  active nodes
- Declared target 500 RPM; latency-plan sustained rate 266.7 RPM
- 10 CPUs, 12 GB, and 16 hours per shard; adaptive shared provider controller
- Expected call time is about 8.82 hours at 266.7 RPM before queueing/outages
  (6.86 hours if all 80 candidates are later certified); the static 200-RPM
  planning estimate is 11.76 hours (9.15 hours with full eligible reuse)

The safe primary split is the already separate target-condition pair: submit
Study 09h and Study 09i independently and sequentially, not concurrently,
unless the provider-wide RPM allocation is recalculated. The throttle remains
unchanged; no higher-throttle assumption is included in these estimates.
