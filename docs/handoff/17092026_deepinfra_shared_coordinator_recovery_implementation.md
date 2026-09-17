# DeepInfra shared-coordinator recovery implementation handoff

Implementation snapshot: 2026-09-17. The user separately authorized the paid
false-arm resume after reviewing the implementation and launch summary. The
truth arm and aggregation were not submitted.

## Redis integration follow-up

Remote `main` commit `41dfcb7` was merged after the initial recovery. Its
optional `redis_adaptive` coordinator removes the network-filesystem lock queue
and JSON rewrite from acquire, renew, release, and snapshot operations. The
false-arm pilot now selects this backend explicitly; its execution-only policy
hash is
`9d3b22dbd55f42a9df23184d30f23918e5ecec4b8c2d4fc2a3064b28d8568ba6`.

The Redis backend was brought under the same recovery guarantees as the file
backend: independent admission deadlines and diagnostics, cancellation-safe
late-grant cleanup, immutable policy identity, live-limit range validation,
fresh heartbeat deadlines, and fail-fast worker validation when
`MAS_CC_PROVIDER_CONTROL_REDIS_URL` is absent. A generic shared-cluster Redis
service launcher is available at
`scripts/SLURM/run_provider_redis_micromamba.job`; it publishes a mode-0600
authenticated endpoint file and runs Redis with `maxmemory-policy noeviction`.

The merged coordinator/adapter suite passes 66 tests. A real Redis 8.10.1
smoke granted, renewed, and released 150 simultaneous leases, recorded 150
distinct outcomes with zero expirations, and rejected a mismatched policy.
This validates a local real server; a multi-node scheduler smoke remains the
final infrastructure check before paid work.

## Implemented repair

- First shared-lease admission now has the independent bounded policy
  `admission_max_elapsed_seconds`; provider recovery timing begins only after
  admission and is still bounded during retry reacquisition.
- Admission expiry normalizes to the distinct resumable code
  `provider_admission_timeout` and does not create an adaptive provider event.
- Saturated workers optimistically read shared state and back off without
  creating FIFO lock tickets. Every actual grant still rechecks concurrency,
  pause, lease expiry, and RPM state under the exclusive lock.
- Acquisition backoff persists across failed checks. Renewals and releases
  continue to bypass the per-process acquisition gate.
- Heartbeats use a fresh short coordination deadline rather than the provider
  retry deadline, so a healthy long transport continues renewing its lease.
- The complete resolved policy has a stable SHA-256 identity in coordinator
  state and immutable per-job settings. Mixed worker policies and live limits
  outside the policy range fail closed. Runtime policy mutation is rejected.

The false-arm pilot policy is now 15 active cell shards, two episode/request
slots per shard, and a shared 30/10/30 initial/minimum/maximum ceiling. It uses
1,000 target RPM, 180-second leases, 60-second heartbeats, separate 1,800-second
retry and admission windows, 12 global samples, a 0.25 failure threshold, a
0.5 decrease factor, and recovery by two slots every 60 seconds. Its resolved
policy hash is
`da3ca0ba580557fa6d908f833b7fd1f32b4251258a5b2fb27efb3d894ac3802a`.

This tuning lets an initial failure wave reduce 30 to 15 and, after another
cooldown with continued failures, to the floor of 10. Stable traffic recovers
two slots per minute. The 30-request ceiling equals the natural 15 shards × two
local request slots; a larger shared ceiling would add no throughput.

## Scientific and checkpoint compatibility

The scientific YAML is byte-identical to job 442:

- config SHA-256:
  `f6da4b653444b3d027fee13719cfecf0b454fdaae0a5e3a01a81d64b42cb22ab`
- resolved study config SHA-256:
  `e3d629c5e2d37e55ea95b4c10930eec5e9d67981dd24ce29456eaea2381480cd`
- cells/episodes: 15/30; execution seed: 20260907
- output mapping: the same 15 `cell-0000` through `cell-0014` shard paths

Provider load control remains in `study.yaml` and the execution plan, outside
episode-resolved scientific configuration. Rebuilding submission entries
reproduced the job-442 config hash, resolved hash, cell count, episode count,
seed, and result root exactly.

Initialization artifact hashes:

- seed `2943338873358119247`:
  `bd32341f3fc3df8ef9212f6012c1f943083fc9da5d5632851d8d7ba59d74bf27`
- seed `3680160250853535454`:
  `6ee9f6aba4f4c6250441264e5f7e6d7d7f49614a4f20b13cb8d24e4ad2fc40eb`

All 30 failure checkpoints remain present in place. Their SHA-256 inventory is:

- `cell-0000-0000`: `078e599ced8398b2c3683f10b3136c24b5d1c85d084e47a508b81b66ae05c821`
- `cell-0000-0001`: `f68d96ae36b02d5f7283492e3c7cfac6a07665f7d9e696cd7b2467cee9ac4fe5`
- `cell-0001-0000`: `2fbe196906dddb2dfada128d787b0d703e272dcae03977e54061baf0e1fe570d`
- `cell-0001-0001`: `48a7abc560b689a68b12e8bdf60aaa2e3c1fbc2d4dd9c91145b87eaea36425c9`
- `cell-0002-0000`: `1b580c111396ff42d55a97ab2b11cb18529c53bddf92577abc7a889bec2657b0`
- `cell-0002-0001`: `07763928598ffc1e5e090492c1b519e368e427b0e710df416f0ea9acc1865c15`
- `cell-0003-0000`: `632f9fafc82040273cb5d79de9f057c093aea74eea7413018041570c11b6aa1a`
- `cell-0003-0001`: `474380b516bec99ac9697d319ff84b9ccb85f3aaf864638d3555c34601404b79`
- `cell-0004-0000`: `c3f760397082ac75a80696e98f5e48e5da5c5f2a9b8dc2c4d5b063c15c27b5ea`
- `cell-0004-0001`: `bd43545326e35d2af4ff2892d6e663b6c2386c5de1f416f21ffacca1af52bc8e`
- `cell-0005-0000`: `eed9b7cc666f9eba6153c28f67a0569fa0651f0ee356e9d853bdeb9df3d8393b`
- `cell-0005-0001`: `fb14c06a95757ac4fbf784cf888b30a510a3bb4e79ce0ab4b1effc2dcb4b8213`
- `cell-0006-0000`: `c8f0fa11c513d1051b35e561d7efff1cd7199f51c4530bb4f4aa5c06125657a8`
- `cell-0006-0001`: `0a9f329c1153640b0cbebb401b80dd6a3280c3aa8c2fd6bacb4438e1fdbb6a09`
- `cell-0007-0000`: `5447030dec67b9a01a41eb8bf4e61af07d17046e982dec392a18fd5dd31e6bc4`
- `cell-0007-0001`: `8d17c5de0ba2d308e5962587d8c5fea9e4d442e1b74753c9d8724cdfc1f9da8e`
- `cell-0008-0000`: `109078a23a1a7b2c1a806fe95df34db17f0a4e16bd6acb411b63331c1062b30d`
- `cell-0008-0001`: `ae84e25c4cac1ae8299337633955ae7a06b3382c8be66cdd9edaf06879e18933`
- `cell-0009-0000`: `cab536f6ae1793b9c3c12ab169a62fc81b17ca117b7ab8a8f97193943201fac4`
- `cell-0009-0001`: `bace5bce15e57d9a06562809e0d7e49afd641fdd9f79c5a0e77f3a351a6f1edb`
- `cell-0010-0000`: `5d6484a251eed3b61323647ed0064b9360cda22b7473e1a310958c54cead979d`
- `cell-0010-0001`: `ff52a623e01d43b3819c8c78bc5da7c1bcc894c21c6b5a120dda234b69a0d5da`
- `cell-0011-0000`: `8b30e70ee49f0aa8d7b61fa98cdcf17eaae4ca03d4ceee4dedfdfdc4ac135566`
- `cell-0011-0001`: `e5cb6f8118d01fa596dd0b96b2aabf00c19e6c5ad5c139b7a811238c9e3dc917`
- `cell-0012-0000`: `7bcfc58ec826dc5cad7657931ad56c5f7b55f3cbad67da3bdd0a5f6662ce0dd0`
- `cell-0012-0001`: `d4d78ed2c6e890b033f88b199167fee2711c5ae539109573eca184aa2367986f`
- `cell-0013-0000`: `fb3c00e18aa0992bb88ecc6faca3bd72cc654ea90a280c25a2d700cb06c7ae83`
- `cell-0013-0001`: `6cabca7c2b369bb660ad08454d8803787c3399c18dc8ed700b34a90fe5853cbb`
- `cell-0014-0000`: `8ef82e85e4e37a6e65bfe8a8369d56679f1b1f117d3b42472fd2d093a9ecc060`
- `cell-0014-0001`: `0081d4580b960e8448b5af5fc204276fdecec1c78445143e38986bb3299fd917`

The ordered inventory digest is
`2b8bad552ada8e67417c70da173d91c0d60846dc203e27dd2dd902de28a93cea`.

Mock resume tests prove that validated decisions before a failed call are
replayed locally, the missing call is resumed, the final trajectory matches an
uninterrupted baseline, sealed episodes are skipped before provider creation,
and completion leaves one durable cell seal. No scientific tree was copied.

## Preflight and remaining-work estimate

Credential-free preflight passed with ID
`d8f3f21cedeb92ea2bba3a05ead4ca0b2975cca02ee47fdf4d8698b76a16b8d7`:
one config, 15 cells, two repetitions, 30 episodes. Full-run offline estimates
are 26,340 expected / 110,700 conservative provider requests, 28,007,580 /
118,775,700 input tokens, 107,888,640 / 453,427,200 output tokens, and
19.37734926 / 81.47732490 USD. These are prediction/bound pairs, not observed
usage.

Already recorded: 3,269 calls, 7,206,081 input tokens, 3,979,351 output tokens,
0.943114667 USD, and 115 of 900 rounds. Scaling observed usage over the 785
remaining rounds predicts about 22,314 calls, 49.19 million input tokens,
27.16 million output tokens, and 6.44 USD remaining. The preflight expected
remaining cost by round fraction is 16.90 USD; the conservative proportional
bound is 71.07 USD. These differ because the deterministic preflight token
model is deliberately conservative and later prompts may grow.

The preflight's 39,510-second aggregate runtime becomes about 38 minutes for
the remaining fraction with 15 simultaneous shards, assuming the configured
latency and no prolonged provider outage. This is a planning estimate, not a
service guarantee. The proposed 36-hour shard limit remains the outer bound.

Current storage is 44 MiB across 411 files. Linear round scaling projects about
345 MiB at completion, below the 1 GiB review threshold. Preflight independently
estimates 74,387,520 bytes of dashboard-semantic episode data. Monitor both
size and file count during a real resume.

## Verification

The focused coordinator, adapter, persistence, relational resume, and worker
policy set passes all 223 collected tests, including transport cancellation
and cancellation during a locked admission recheck without a leaked lease.
The 30-caller, renewal-progress,
and multi-process ceiling stress cases also passed three consecutive runs. The
full `test_studies.py` run reaches the new test but has seven unrelated existing
failures because this checkout lacks the referenced Study 06/07/08 and legacy
relational smoke YAML fixtures. An additional study-support run passed apart
from three PDF-report tests whose external `latexmk` executable is absent.

No compute-node shared-filesystem smoke was run because the local deterministic
multi-process and 30-caller tests covered the repair without scheduler
mutation. A compute-node mock smoke remains a recommended first rollout check;
it needs no provider credential.

## Authorized launch

The supported generic resume command was executed with the shared-cluster
launcher:

```bash
/shared/home/cesar/.local/share/mamba/envs/MA-CC/bin/mas-cc study submit \
  --config-dir /shared/home/cesar/LanguageGames/MA-CC/configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_q12_deepinfra_gptoss120b_rho3_pilot_r2 \
  --results-dir /shared/home/cesar/work/results/studies/astra_task003_false_control_q12_deepinfra_gptoss120b_rho3_pilot_r2 \
  --job-script /shared/home/cesar/LanguageGames/MA-CC/scripts/SLURM/run_study_cell_array_micromamba.job
```

It resolves to the existing generic
`scripts/SLURM/run_study_cell_array_micromamba.job`, array `0-14%15`, two CPUs
and 6 GiB per shard, 30 episode slots, 30 local request slots, and the same
absolute result root. Logs remain under `<result-root>/logs`. The truth arm and
aggregation are not included.

SLURM accepted job `459` at `2026-09-17T10:19:56Z`. At
`2026-09-17T10:23:04Z`, all 15 tasks were running, all 30 retained trajectory
files had been touched by the new workers, and the completed-round total had
increased from the incident baseline of 115 to 125. Coordinator state matched
policy hash `da3ca0ba580557fa6d908f833b7fd1f32b4251258a5b2fb27efb3d894ac3802a`,
with limit 30, 30 active leases, 60 dispatches and 58 completed events in the
latest 60 seconds, mean event latency 30.85 seconds, zero retryable failures,
zero HTTP 402 events, zero expired leases, and no transaction error. All 30
failure checkpoints remained present with the same ordered digest.

No study-specific SLURM file or estimator implementation was added. Strict
aggregation remains deferred until all 30 episodes and 15 cells seal.
