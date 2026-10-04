# Study 08 preflight

- Status: permitted for both configs
- Configs: 2
- Cells: 96 wrong + 96 truth = 192
- Episodes per cell: 10
- Episodes: 960 wrong + 960 truth = 1,920
- Provider calls: 506,880 nominal; 535,680 expected; 3,041,280 conservative
- Expected cost: approximately 647.26 USD across both blocks
- Conservative cost: approximately 3,674.80 USD across both blocks
- Provider: NeuralWatt `deepseek-v4-flash`, five global in-flight slots
- Response protocol: the NeuralWatt adapter's `json_object` default, with a
  4,096-token output ceiling; no game- or study-level transport override
- Validation robustness: six attempts per decision. At the declared 5%
  invalid-response rate this leaves fewer than 0.01 expected terminal malformed
  decisions over the complete study; it does not change the 1.0526 expected
  attempts per decision, but expands the deliberately pessimistic bound.
- Transport robustness: nine bounded HTTP attempts use full-jitter exponential
  backoff for retryable 429/5xx/connection faults. The shared coordinator opens
  its global breaker at a 5% rolling failure ratio and locally pauses a node
  after two failures, preventing a short synchronized 503 burst from killing an
  episode. It starts at two requests and waits five healthy minutes before each
  additive increase toward the five-request maximum, avoiding rapid
  pause/recovery oscillation during a longer incident.
- Planned load: 5 active cell shards × 1 local request slot = 5 global
  requests; approximately 200 requests/minute at the 1.5-second planning
  latency, beneath the 500-RPM rolling gate
- Scheduler source defaults: 192 single-cell shards, 8 CPUs/shard, 1 episode
  slot/shard, 8 GB/shard, array throttle 5, Potsdam `all`/`normal`
- NERSC preparation replaces only the result boundary and scheduler adapter;
  it preserves the same throttle and shared provider coordinator under
  interactive CPU allocations
- Measured production-shaped throughput on 2026-08-29 was approximately 211
  successful RPM. At 535,680 expected requests, the rough full-study provider
  time is therefore 42 hours, requiring repeated four-hour NERSC allocations.

The standalone preflight's summed serial runtime and maximum-output-token cost
are conservative planning quantities. Actual observed responses averaged about
115--120 output tokens; final cost and runtime depend on realized outputs and
provider latency.
