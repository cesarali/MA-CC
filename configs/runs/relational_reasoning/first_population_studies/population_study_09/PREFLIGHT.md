# Study 09a preflight

Preflight date: 2026-08-28.

- Status: permitted for both configs
- Configs: 2
- Cells: 8 false + 8 truth = 16
- Episodes per cell: 10
- Episodes: 80 false + 80 truth = 160
- Provider calls: 42,240 nominal; 44,640 expected; 84,480 conservative
- Input tokens: 18,364,800 nominal; 19,401,840 expected;
	36,729,600 conservative
- Output tokens: 42,240 nominal; 182,845,440 expected;
	346,030,080 conservative
- Reported price: 0 proxy accounting units from live University model metadata
- Per-config serial-equivalent runtime estimate: 8,370 seconds
- Scheduler: 16 single-cell shards, 8 CPUs/shard, 8 episode slots/shard,
	8 GB/shard, array throttle 2, `all` partition, `normal` quality of service
- Planned maximum load: 2 active shards × 5 request permits = 10 concurrent requests
- Planning rate ceiling: 200 requests/minute at the configured 3-second
	planning latency; observed throughput may be lower when requests take longer

The token counts are deterministic estimates, not counts from the provider's
tokenizer. The zero price is the provider's current proxy-accounting result,
not a currency-valued prediction. The summed serial runtime is not expected
wall time because the automatic launcher runs two cell shards concurrently. At
the 200 RPM planning ceiling, the 44,640-call expected workload is about 3.7
hours; provider latency and adaptive pauses may extend it.
