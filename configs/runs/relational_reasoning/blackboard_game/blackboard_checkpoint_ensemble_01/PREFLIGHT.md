# Credential-free resource plan

Static demand is computed from the same typed ensemble config used at runtime.
One parent contains `2 + 9*10 = 92` population-round paths and therefore
`92*24 = 2,208` ordinary focal updates. It also contains 24 initialization
calls and at most 80 structured controller-communication calls before retry
overhead. With the configured validation-failure expectation and retry bounds,
the provider-request estimates per parent are 2,312 lower, 2,707 expected, and
11,400 conservative maximum.

| Design | Prepared settings | Branch conditions | Parents / checkpoints | Continuations | Ordinary updates | Calls lower / expected / max |
|---|---:|---:|---:|---:|---:|---:|
| Main `K=120` | 4 | 36 | 480 | 4,320 | 1,059,840 | 1,109,760 / 1,299,360 / 5,472,000 |
| Alternative `K=60` | 4 | 36 | 240 | 2,160 | 529,920 | 554,880 / 649,680 / 2,736,000 |
| Pilot `K=10` | 4 | 36 | 40 | 360 | 88,320 | 92,480 / 108,280 / 456,000 |

The scheduler sees 480, 240, or 40 resumable parent-bundle episodes,
respectively. Scientific analysis sees all descendant continuations.

The generic launcher requests 8 CPUs, 12 GiB memory, a 24-hour shard limit,
one cell per shard, at most four active nodes, and a shared adaptive provider
load controller capped at 600 RPM. At 600 RPM, request-only lower bounds are
approximately 30.8, 15.4, and 2.6 hours for the three designs; provider
latency, serialization within branches, retries, and queue time increase wall
time. Results and logs resolve under
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_01`.

The University model has no authoritative offline monetary quote in the
repository. Preflight therefore reports calls and tokens but marks currency
cost unknown under an explicit preflight-only override; live launch remains
bounded by the configured request/token/proxy-accounting limits and requires a
fresh provider quote. This is deliberately not represented as a zero-dollar
estimate.

No pilot or main job has been submitted.
