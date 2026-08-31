# Study 09j provider-free preflight

Preflight date: 2026-08-31. **No study or provider job was submitted.** The
strict static study contract and focused mock-provider tests passed. The real
20-state initialization bundle has not been generated, so submission is
intentionally blocked until those provider calls are separately approved and
the realized pairing audit passes.

## Resolved scientific design

- Status: **PERMITTED for static design; BLOCKED for submission pending initialization artifacts**
- Baseline: actual validated Study 09f package and its resolved config
- Deliberate change: `naive_strategic` -> `vigilant_strategic`; this is not an exact replication
- Fixed: `N=24`, `q=1`, `L=2`, 30 rounds, complete topology
- Grid: `rho={0.75,0.85,1.00}` x `b={6,8,12,16,18,24}`
- Size: 18 structural cells x 20 repetitions = 360 episodes
- Controller: false target `SOUTH`; `q_c=12`; soft-target policy; `beta=4.0`;
  `theta=0.75`; soft schedule; `recommendation_plus_fact`; strategic evidence
- Strategic fact: real frozen fact `f1`, “Jeni is south of Zani.” No fact is fabricated.
- Production q=1 semantics: one ordinary peer on an ordinary update; on a
  controlled update the controller replaces that slot and zero ordinary peers remain.
- Theory: no theory curves in the empirical report.

## Dataset and initial epistemic difficulty

Use frozen `pop24_L2_r06/task_0002`, fingerprint
`f8bc6f9a358840445c26730559a62fc3dc675004d5cb15b725c63c52d5e68fa2`.
It has `N=24`, `L=2`, `r=6`, four distractors, `K=3`, and
`no_single_agent_solution=true`. Truth is `SOUTHWEST`.

Study 09f used `r/N=4/12=1/3`. The available N=24 redundancies are 1, 3, 6,
and 12. `r=6` gives `6/24=1/4`, the closest available support coverage.
An exact density match would require unavailable `r=8`. The share of agents
with one supporting fact changes from `8/12=2/3` to `12/24=1/2`; no agent has
both. Distractors double from two to four, preserving placements per agent at
`1/6`. This unavoidable difficulty change is explicit.

## Paired initialization

Common random numbers alone were tested and found insufficient: they match
request seeds, not remote-model responses. The new `paired_local_vote` path:

1. generates one complete natural initial state for each repetition;
2. stores all initial actions, reasons, exposed facts, active/historical facts,
   task, agent order, and assignment in an atomic hashed artifact;
3. validates and replays that artifact in every rho/b cell without another
   initialization provider call;
4. records the physical-state hash in every round; and
5. makes strict aggregation fail if any repetition differs across cells.

Provider-free tests construct three distinct repetition states and prove that
each is identical across several rho/b cells and both target conditions. This
proves the mechanism. It does **not** claim that the real model-generated
states already vary: their `x_0` values can be reported only after the 20
initialization artifacts are generated. Submission remains blocked until then.

False/truth sharing is valid because controller target, sensing, budget, and
evidence are absent from `local_vote` prompts. The two configs resolve to the
same initialization compatibility key for all 20 episode seeds.

After explicit approval, generate the shared bundle with `mas-cc study
initialize` using both Study 09j and Study 09k config directories and the
configured shared artifact directory. This sends only the 20 x 24 local-vote
requests. It does not start dynamics cells or submit SLURM jobs.

## Provider and runtime estimate

The exact dynamics preflight excludes initialization because every dynamics
episode loads an artifact:

| Quantity | Study 09j dynamics | Shared 09j/09k initialization |
|---|---:|---:|
| Nominal calls | 259,200 | 480 |
| Expected calls | 272,880 | 520 |
| Conservative calls | 518,400 | 960 |
| Nominal input tokens | 120,528,000 | 192,960 |
| Expected input tokens | 126,889,200 | 209,040 |
| Conservative input tokens | 241,056,000 | 385,920 |
| Nominal output-token bound | 259,200 | 480 |
| Expected output-token bound | 1,117,716,480 | 2,129,920 |
| Conservative output-token bound | 2,123,366,400 | 3,932,160 |

Token values are deterministic planning estimates. Expected and conservative
output values use the configured 4,096-token cap and are bounds, not predicted
actual generations. The current live provider metadata reports zero
`proxy_accounting_unit`; this is not a currency quote.

Static serial-equivalent runtime is 81,864 seconds (22.74 hours) per target.
At the declared 500 requests/minute target, expected dynamics call time is
about 9.10 hours per target before queueing, retries, and adaptive pauses. At
the latency-plan ceiling of 266.7 requests/minute, it is about 17.06 hours.

## Proposed SLURM shape

- 18 scientific-cell shards using generic `run_study_cell_array.job`
- Array `0-17%2`; no study-specific job file
- 10 CPUs, 12 GB, 20 hours per shard
- 10 episode slots and 10 request permits per shard
- At most 20 episode slots / requests across two active shards
- Shared adaptive provider controller, target 500 RPM
- Submit 09j and 09k sequentially unless provider capacity is recalculated

The initialization bundle is a separate prerequisite with 20 artifacts. It
must finish and pass validation before the dynamics array can be submitted.