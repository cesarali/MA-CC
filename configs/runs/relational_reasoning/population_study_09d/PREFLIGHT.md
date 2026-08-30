# Study 09d preflight

Preflight date: 2026-08-29.

## Result

- Status: **PERMITTED**
- Strict contract: `relational_persistence_refinement_v1`
- Frozen task: `task_0002` only
- Population: `N=12`; rounds: `30`; `q=2`; `q_c=6`; `L=3`; `r=3`
- Truth: `NORTH`; false target: `NORTHWEST`
- Receiver/evidence/message: `naive` / `strategic` / `recommendation_plus_fact`
- Soft controller: `beta=4.0`, `theta=0.75`
- Persistence: `rho={0.70,0.75,0.80,0.85,0.90}`
- Budgets: `b={3,6,9,12}` (`b/N={0.25,0.50,0.75,1.00}`)
- Repetitions: `10`; structural cells: `20`; episodes: `200`
- Matched non-persistence theory: disabled (`theoretical_reference: none`)

## Resolved structural cells

| Cell | rho | b | b/N | Episodes |
|---|---:|---:|---:|---:|
| cell-0000 | 0.70 | 3 | 0.25 | 10 |
| cell-0001 | 0.70 | 6 | 0.50 | 10 |
| cell-0002 | 0.70 | 9 | 0.75 | 10 |
| cell-0003 | 0.70 | 12 | 1.00 | 10 |
| cell-0004 | 0.75 | 3 | 0.25 | 10 |
| cell-0005 | 0.75 | 6 | 0.50 | 10 |
| cell-0006 | 0.75 | 9 | 0.75 | 10 |
| cell-0007 | 0.75 | 12 | 1.00 | 10 |
| cell-0008 | 0.80 | 3 | 0.25 | 10 |
| cell-0009 | 0.80 | 6 | 0.50 | 10 |
| cell-0010 | 0.80 | 9 | 0.75 | 10 |
| cell-0011 | 0.80 | 12 | 1.00 | 10 |
| cell-0012 | 0.85 | 3 | 0.25 | 10 |
| cell-0013 | 0.85 | 6 | 0.50 | 10 |
| cell-0014 | 0.85 | 9 | 0.75 | 10 |
| cell-0015 | 0.85 | 12 | 1.00 | 10 |
| cell-0016 | 0.90 | 3 | 0.25 | 10 |
| cell-0017 | 0.90 | 6 | 0.50 | 10 |
| cell-0018 | 0.90 | 9 | 0.75 | 10 |
| cell-0019 | 0.90 | 12 | 1.00 | 10 |

## Provider and token estimates

| Quantity | Nominal | Expected | Conservative |
|---|---:|---:|---:|
| Provider calls | 74,400 | 78,400 | 148,800 |
| Input tokens | 31,951,200 | 33,663,400 | 63,902,400 |
| Output tokens | 74,400 | 321,126,400 | 609,484,800 |
| Cost | 0.00 | 0.00 | 0.00 |

Cost uses the provider's live `proxy_accounting_unit`; it is not a currency
price. Token counts are deterministic planning estimates rather than provider
tokenizer measurements.

## Execution plan

- 20 cell shards; array throttle 2.
- Ten episode slots and ten request permits per active shard.
- About 20 genuinely active episode requests across two cells.
- Shared adaptive provider controller: 20 initial/max concurrency, 500 RPM cap.
- At the Study 09c observed 4--5 second latency, expected throughput is roughly
  240--300 RPM; the declared 4.5-second plan reports about 267 RPM.
- Resources per active shard: 10 CPUs, 12 GB RAM, 16-hour wall-time.
- Rough preflight runtime estimate: 23,520 seconds (6.53 hours). The Study 09c
  observed latency suggests approximately 5--7 hours if service remains stable.

No provider completion or scheduler submission was performed by the standalone
preflight. Machine-readable evidence is under
`results/inspection/study09d_launch_preflight/` on `/work`.
