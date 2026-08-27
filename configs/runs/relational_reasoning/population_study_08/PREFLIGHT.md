# Study 08 preflight

- Status: permitted for both configs
- Configs: 2
- Cells: 96 wrong + 96 truth = 192
- Episodes per cell: 10
- Episodes: 960 wrong + 960 truth = 1,920
- Provider calls: 506,880 nominal; 535,680 expected; 1,013,760 conservative
- Expected input tokens: 246,615,840 across both blocks
- Conservative input tokens: 466,859,520 across both blocks
- Reported price: 0 proxy accounting units (live University model metadata)
- Provider limit reported by live metadata: 2,000 requests/minute
- Planned load: 18 active shards × 8 requests = 144 concurrent requests;
  approximately 864 requests/minute at the 10-second planning latency
- Scheduler: 192 single-cell shards, 8 CPUs/shard, 8 episode slots/shard,
  8 GB/shard, array throttle 18, `all` partition, `normal` QoS
- Relative runtime: approximately `192/156 = 1.23×` the completed Study 06
  workload at the same throttle, or roughly 5 hours from Study 06's observed
  four-hour completion. Queue/provider variability may extend this; the
  requested overnight 12-hour window is reasonable.

The standalone preflight's summed serial runtime is deliberately not used as
the wall-time estimate because it ignores the established 18-shard cell-array
parallelism.
