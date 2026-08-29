# Study 09c preflight

Preflight date: 2026-08-29.

## Result

- Status: **PERMITTED**
- Frozen task: `task_0002` only
- Population: `N=12`
- Rounds: `30`
- Social group: `q=2`
- Sensor size: `q_c=6`
- Supporting chain: `L=3`
- Supporting-fact redundancy: `r=3`
- Persistence values: `rho={0.6,0.8,0.9}`
- Budgets: `b={3,6,9,12}`
- Receiver: `naive`
- Evidence: `strategic`
- Message: `recommendation_plus_fact`
- Schedule: `soft`
- `beta=4.0`, `theta=0.75`
- Repetitions: `1`
- Cells and episodes: `12` and `12`
- Matched revised theory: disabled (`theoretical_reference: none`)

## Resolved cells

| Cell | rho | b | b/N |
|---|---:|---:|---:|
| `cell-0000` | 0.6 | 3 | 0.25 |
| `cell-0001` | 0.6 | 6 | 0.50 |
| `cell-0002` | 0.6 | 9 | 0.75 |
| `cell-0003` | 0.6 | 12 | 1.00 |
| `cell-0004` | 0.8 | 3 | 0.25 |
| `cell-0005` | 0.8 | 6 | 0.50 |
| `cell-0006` | 0.8 | 9 | 0.75 |
| `cell-0007` | 0.8 | 12 | 1.00 |
| `cell-0008` | 0.9 | 3 | 0.25 |
| `cell-0009` | 0.9 | 6 | 0.50 |
| `cell-0010` | 0.9 | 9 | 0.75 |
| `cell-0011` | 0.9 | 12 | 1.00 |

Each episode contains 12 local-initialization decisions and `12 x 30 = 360`
microscopic update decisions.

## Provider and token estimates

| Quantity | Nominal | Expected | Conservative |
|---|---:|---:|---:|
| Provider calls | 4,464 | 4,704 | 8,928 |
| Input tokens | 1,917,072 | 2,019,804 | 3,834,144 |
| Output tokens | 4,464 | 19,267,584 | 36,569,088 |
| Cost | 0.00 | 0.00 | 0.00 |

Cost is reported in `proxy_accounting_unit` from live University provider
metadata. It is not a monetary price.

The expected count includes the configured five-percent validation-failure
model: 13 initialization attempts and 379 update attempts per episode. The
conservative count assumes every logical request uses its one allowed retry.
Persistence adds no provider calls.

## Runtime and execution assumptions

The ordinary experiment preflight reports a 1,764-second serial-equivalent
estimate, or 29.4 minutes. That estimate assumes eight-way request concurrency
throughout and three seconds per request, so it is optimistic for an episode's
state-dependent microscopic updates.

The study plan uses 12 cell shards with a throttle of 12, eight configured
request permits per shard, a shared adaptive provider coordinator starting at
24 permits, and a 900-RPM target. Its configured capacity bound is 576 RPM at
the declared ten-second planning latency. During microscopic updates, each
one-episode shard normally has one request in flight, so approximately 12
concurrent requests, or about 72 RPM at ten seconds, is more realistic.

A state-barrier-aware estimate at ten seconds per attempt is:

- expected: about 3,810 seconds per parallel shard, or 63.5 minutes;
- conservative: about 7,230 seconds, or 120.5 minutes.

The SLURM time limit is three hours. These are planning estimates, not service
guarantees.

## Frozen task audit

- Truth: `NORTH`
- False target: `NORTHWEST`
- Strategic true fact: `f1`, “Vero is northwest of Garo.”
- Task fingerprint: `edf1d1271acd4b71ffca9a8a920dc4fc4a01e70a95d8dec1ac64c7b3c909dbe0`

Machine-readable evidence is under `results/inspection/study09c_preflight/`.
No provider completion and no scheduler submission was performed.
