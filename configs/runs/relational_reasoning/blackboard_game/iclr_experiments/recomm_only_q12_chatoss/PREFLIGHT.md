# Credential-free preflight

Prepared 2026-09-17. No study or SLURM job was submitted.

## Design

| Arm | Cells | Episodes |
|---|---:|---:|
| No control | 3 | 180 |
| Truth control | 9 | 540 |
| False control | 9 | 540 |
| **Total** | **21** | **1,260** |

Controlled cells span intervention budgets 6, 12, and 18 crossed with
epistemic persistence 0.70, 0.85, and 1.00. The autonomous baseline spans only
the persistence axis; duplicating it across a scientifically inactive budget
coordinate would not add independent evidence.

All configurations resolve to DeepInfra `openai/gpt-oss-120b`, q=12, 30
rounds, 60 repetitions per cell, seed 20260907, prompt-v4 ballot responses,
explicit `reasoning_effort: low`, and the same Task-003 dataset. Controlled
arms resolve `message_mode: recommendation_only`.

## Offline estimates

| Estimate | Lower | Expected | Conservative |
|---|---:|---:|---:|
| Provider requests | 939,600 | 1,100,880 | 4,633,200 |
| Cost (USD) | 37.12 | 810.07 | 3,410.70 |

Expected token estimates are 1,175,718,960 input and 4,509,204,480 output
tokens. These are deterministic preflight scenarios using the versioned
offline DeepInfra price catalog, not observed usage or a provider quote.

Preflight IDs:

- no control: `4508a912537c460a040b43033424b918ecee151ec1413d58935c8d3ab1f71f03`
- truth control: `ae85498811feb3ca589da223d6ea7980159ad26d142fed8a30938a97bfaeff9f`
- false control: `ab2ab033f6dddbc5bb4d74650d93ca89542dd11e72d5ececa01e40b6f574c23f`

## Execution preparation

The declared topology uses one cell per shard, five active shards, 8 CPUs and
12 GiB per shard, a 1,000 RPM target, and a Redis-adaptive ceiling of 150
requests. Live provider limits and healthy cluster capacity must be checked
again before launch.

Sixty paired initialization artifacts remain a prerequisite. They are not
materialized here because their generation calls the configured provider.
