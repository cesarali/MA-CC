# Running MA-CC on Cygnus (Slurm on Kubernetes)

Site notes for the `cygnus` cluster, the Slurm cluster the Cerebral Work Institute runs on its
Kubernetes estate. Everything here was measured on the live cluster on 2026-09-17/18.

## Access

- `ssh <user>@slurm-login.tail769bd2.ts.net` over the tailnet. Users `allen`, `darius`, `todie`,
  `cesar` with fixed uids; homes under `/shared/home/<user>` on the cluster NFS volume.
- Partition `main`: five 16-CPU nodes (`cygnus-principes-1..5`) and one 40-CPU node (`big-0`),
  Slurm 26.05. Per-job memory limits are enforced by cgroup v2: a step over its `--mem` is
  killed by Slurm and the node stays up. Accounting is on (`sacct` works).
- The login image has no git/pip/editor; user tooling lives in `~/.local/share/mamba` (micromamba)
  or `/shared/opt`, `/shared/bin`.

## LLM access: the gateway

All model calls go through the cluster's LiteLLM gateway, never to a vendor directly. The
launchers in `scripts/Cygnus/SLURM/` export:

| variable | value | meaning |
|---|---|---|
| `LLM_BASE` | `http://llm-batch.llm.svc.cluster.local:4000` | the batch pool, reserved for experiment namespaces (the shared pool is `llm.llm.svc.cluster.local:4000`) |
| `LLM_KEY` | from `$HOME/.llm.key`, else `/shared/secrets/llm.key` | per-user gateway key (alias `slurm-<user>`) so spend is attributed per person |
| `MAS_CC_PROVIDER_CONTROL_REDIS_URL` | `redis://provider-control-redis.slurm-cluster.svc.cluster.local:6379/0` | the cluster Valkey for `provider_load_control.mode: redis_adaptive` |

Provider components: `configs/components/cygnus/gateway_*.yaml` (type `university`, i.e.
OpenAI-compatible, `credentials_env: LLM_KEY`, `base_url_env: LLM_BASE`). Model ids on the
gateway: `deepseek-v4-flash` and `gpt-oss-120b` (both DeepInfra), `glm-5.2-fast`,
`kimi-k3-fast`, `qwen3.6-35b-fast`. Spend, requests per minute and per-key limits show on the
gateway's Grafana dashboard.

Measured ceilings (DeepSeek-V4-Flash, ~4.5k-token prompts, from a login pod):

| concurrent in flight | ok req/min | p50 | errors |
|---|---|---|---|
| 150 | 3,400-4,600 | 1.6-2.1 s | 0 |
| 400 through the gateway | 14,400 | 1.4 s | 0 |

The provider is guaranteed 400 concurrent requests for this account. Size
`array throttle x llm_provider.request_concurrency` to about 400 in total. `gpt-oss-120b` needs
`options.reasoning_effort: low`: default effort spends 2,500-3,000 hidden reasoning tokens per
vote (25-70 s per call, truncating at 4,096 output tokens); low is about 6 s.

Do not use the file-based `shared_adaptive` coordinator on `/shared`: at 150-390 waiters every
lock transaction cost 30-70 ms of NFS directory scans and the harness never exceeded ~150 req/min
(2026-09-15/16 runs). Use `redis_adaptive` against the cluster Valkey, or `mode: off` with
per-shard `request_concurrency` summing to the ceiling.

## Submitting a study

```bash
cd ~/LanguageGames/MA-CC            # the checkout the launchers cd into (MA_CC_REPOSITORY_ROOT)
mkdir -p results/slurm-logs         # once; Slurm does not create log directories
mas-cc study preflight --config-dir configs/runs/<study> --output-dir results/inspection/<study>
mas-cc study initialize --config-dir configs/runs/<study> --output-dir results/studies/<study>_initializations   # blackboard studies
mas-cc study submit --config-dir configs/runs/<study> --job-script scripts/Cygnus/SLURM/run_config_array.job
squeue -u $USER; sacct -j <job>; tail -f results/slurm-logs/mas-cc-study-<job>_<task>.out
mas-cc study aggregate --study-dir results/studies/<study>
```

`study.yaml` carries the Cygnus execution block (partition `main`, `results_root` under the
checkout, `provider_load_control` with `mode: redis_adaptive`); see
`configs/runs/relational_reasoning/first_population_studies/population_study_04_cygnus/` for a
complete example. For cell-sharded studies use `scripts/Cygnus/SLURM/run_study_cell_array.job`.

## Provenance

The launcher and the study-04 Cygnus row were developed by Allen Hackley in
`cerebral-work/ma-cc-cygnus` (branch `cygnus-adapt`, 2026-09-11..13) and brought into this
repository on 2026-09-18 with the gateway routing and the Redis coordinator wiring that landed the
same week (PR #7). The cluster itself is `cerebral-work/slurm-cygnus`.
