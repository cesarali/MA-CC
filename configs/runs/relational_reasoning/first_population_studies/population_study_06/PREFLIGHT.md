# Study 06 preflight and 48-hour resource report

Generated on 2026-08-22 with the repository preflight commands. No provider
experiment was launched. All three experiment configs were `permitted`; the
Qwen relational-support benchmark planned 960 requests and passed validation.

| block | configs | cells | worlds | reps/cell | episodes | rounds | nominal calls | expected calls | conservative calls | rough wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main GPT-OSS atlas | 1 | 120 | 4 | 10 | 1,200 | 10 | 316,800 | 334,800 | 633,600 | 34 h 52 m 30 s |
| GPT-OSS beta ablation | 1 | 36 | 4 | 10 | 360 | 10 | 95,040 | 100,440 | 190,080 | 10 h 27 m 45 s |
| **default study total** | **2** | **156** | **4 matched** | **10** | **1,560** | **10** | **411,840** | **435,240** | **823,680** | **~2 h 40 m ideal at 18 active cell shards** |
| Qwen validation gate | benchmark | 4 parameter cells | 20 generated tasks | — | 960 items | — | 960 | — | at most 1,000 | about 40 m at concurrency 4 |
| optional Qwen anchor | 1 | 36 | 4 | 10 | 360 | 10 | 95,040 | 100,440 | 190,080 | 10 h 27 m 45 s |

The original one-config-at-a-time estimate was 45 h 20 m. The automatic
execution planner now distributes the 156 original cells with up to 18 active
shards; ideal workload division is about 2 h 40 m before queueing, latency
variance, and retries. Every shard requests four hours so its approximately
17-minute expected cell workload has substantial headroom.

## Token and cost estimates

| block | expected input tokens | expected output-token bound | conservative input | conservative output bound | live estimated cost |
|---|---:|---:|---:|---:|---:|
| main | 152,712,000 | 1,371,340,800 | 289,094,400 | 2,595,225,600 | 0.00 proxy units |
| beta | 45,813,600 | 411,402,240 | 86,728,320 | 778,567,680 | 0.00 proxy units |
| **default total** | **198,525,600** | **1,782,743,040** | **375,822,720** | **3,373,793,280** | **0.00 proxy units** |

Output figures are reservation bounds based on `max_output_tokens=4096`, not
predicted realized completions. Token counts use the repository's deterministic
regex approximation, not the provider tokenizer. The live pricing snapshot
reported zero ordinary-input and output rates in proxy accounting units.

## Execution topology and provider safety

`study.yaml` selects automatic original-cell sharding. Each task has experiment
parallelism 8 and provider request concurrency 8. The 900-RPM target and
10-second planning latency yield an array throttle of 18: at most **144**
simultaneous requests and an estimated **864 RPM**. The live snapshot reported
a 2,000-RPM ceiling. Each shard requests 8 CPUs, 8 GB, and four hours. The
submitter rejects a CLI throttle above the calculated bound.

## Preflight artifacts

- Main: `results/inspection/relational_study06_main_preflight`
- Beta: `results/inspection/relational_study06_beta_preflight`
- Qwen validation: `results/inspection/relational_study06_qwen_validation_preflight`
- Qwen anchor: `results/inspection/relational_study06_qwen_anchor_preflight`

These inspection paths are diagnostics, not authoritative study results.
