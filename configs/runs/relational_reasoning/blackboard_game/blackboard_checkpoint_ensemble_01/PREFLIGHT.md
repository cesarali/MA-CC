# Credential-free resource plan

Static demand is computed from the same typed ensemble config used at runtime.
One parent contains `2 + 9*10 = 92` population-round paths and therefore
`92*24 = 2,208` ordinary focal updates. It also contains 24 initialization
calls. The report-only controller does not make a separate LLM communication
choice. With the configured validation-failure expectation and retry bounds,
the static provider-request estimates per parent are approximately 2,232
lower, 2,626 expected, and 11,160 conservative maximum. Fresh cluster
preflight remains authoritative.

| Design | Prepared settings | Branch conditions | Parents / checkpoints | Continuations | Ordinary updates | Calls lower / expected / max |
|---|---:|---:|---:|---:|---:|---:|
| Configured `K=40` | 4 | 36 | 160 | 1,440 | 353,280 | 357,120 / 420,109 / 1,785,600 |
| Reference `K=120` | 4 | 36 | 480 | 4,320 | 1,059,840 | 1,071,360 / 1,260,328 / 5,356,800 |
| Alternative `K=60` | 4 | 36 | 240 | 2,160 | 529,920 | 535,680 / 630,164 / 2,678,400 |
| Pilot `K=10` | 4 | 36 | 40 | 360 | 88,320 | 89,280 / 105,027 / 446,400 |

The configured scheduler sees 160 resumable parent-bundle episodes. Scientific
analysis sees their 1,440 descendant continuations.

The generic launcher requests 8 CPUs, 12 GiB memory, a 24-hour shard limit,
one cell per shard, at most four active nodes, and a shared adaptive provider
load controller capped at 600 RPM. At 600 RPM, the configured `K=40`
request-only lower bound is approximately 9.9 hours; provider
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
