# Santa Fe v3 theory validation: local stage

The canonical scientific protocol is
[`SANTA_FE_V3_THEORY_VALIDATION_AGENT_PROMPT.md`](../tdd/santa_fe/SANTA_FE_V3_THEORY_VALIDATION_AGENT_PROMPT.md).
This handoff records which pieces are implemented and which claims the current
outputs support.

## Reproducible inputs and commands

The existing two-cell pilot uses
[`configs/santa_fe/v3_pilot.yaml`](../../configs/santa_fe/v3_pilot.yaml).
The planned 12-cell extension is
[`configs/santa_fe/v3_theory_pilot.yaml`](../../configs/santa_fe/v3_theory_pilot.yaml).
It fixes N=24, F+=7, F-=3, q=3, q_c=12, redundancy=3, policy threshold
0.55, 60 rounds, 64 episodes per cell, rho in {0.4, 0.7, 0.95, 1}, and
b in {0, 6, 24}. It has 768 episodes and 1,105,920 scheduled focal slots.
It is a configuration, not a completed study. A dry run reports all realized
integer coordinates:

```bash
.venv/bin/python -m santa_fe.cli --config configs/santa_fe/v3_theory_pilot.yaml --dry-run
```

Validate existing retained trajectories without generating new episodes:

```bash
.venv/bin/python -m santa_fe.v3_validation \
  --config configs/santa_fe/v3_pilot.yaml \
  --max-events 30 --max-drift-states 3 \
  --max-paired-states 3 --paired-repetitions 16
```

The validator writes `protocol_manifest.json`, `protocol_alignment.md`,
`validation_report.md`, Parquet comparison tables and local vector plots under
`results/santa_fe_v3_pilot/validation`. The manifest checks that the saved
config and episode seed schedule match the requested config. It records realized
F+/F-, b, q_c, semantic switches, stages, code hashes, and commit. A post-hoc
run reads the saved 40 episodes; it does not simulate a new pilot.

## What the local stage checks

- The exact identity-resolved MICRO kernel and event ledger: persistence,
  distinct-message sampling, post-acquisition voting, aligned emission,
  peer-board sensing, action probability, and factual next-day replay.
- Message-category sampling under the simulator's multivariate
  hypergeometric law and the paper's multinomial approximation.
- Identity-level acquisition with exact distinct-message encounter probability,
  an identity-level with-replacement comparison, and the sign-exchangeable
  approximation. Duplicate fact IDs on different messages remain possible.
- Exact MICRO one-slot drift and raw/centered covariance, with Poisson and
  fixed-slot clock conventions explicitly separated. One observed slot at a
  saved state is not a covariance estimate.
- Independently computed **local** HMF and REDUCED class predictions at the
  same saved after-night states and frozen board categories. Their identity
  exchangeability and reduced independent-binomial assumptions are explicit.
- Forced U=0/U=1 next-day branches. Each active branch redraws controller fact
  identities uniformly. Action propensity averages over the closed peer board's
  hypergeometric sensor. Per-snapshot branch distributions and ratios of
  weighted information sums are retained. At b=0, both branches are an exact
  population null.

The existing shared information engine continues to compute observational
MI/CMI, episode bootstrap and null summaries. Its coarse observational
conditioning is different from the complete-state paired JSD.

## Small saved-pilot result, 2026-09-26

The local command above checked 30 one-step events, 3 local after-night states,
and 3 paired snapshots with 16 branch pairs each. All 17 semantic invariants
passed, and factual replay matched. Across the saved pilot's 40 episodes and
28,800 focal events, the observed per-missing-fact acquisition frequency was
0.1960, versus 0.1964 from the exact finite-board encounter formula and 0.2399
from the exchangeable approximation. The mean finite-board category-law total
variation between hypergeometric and multinomial sampling was 0.0307. Raw
closure total variation was 0.3503, while its same-N binomial sampling baseline
was 0.3473; the raw value alone is not evidence of closure failure. These
are descriptive pilot diagnostics, not a tested validity boundary. The three
paired snapshots and 16 branch pairs are too few for stable information
estimation; plug-in JSD can be upward biased.

## Reduced-theory integration completed after this local check

The supplied reduced Langevin runner is now integrated through
[`santa_fe_theory_integration.md`](../documentation/games/santa_fe/santa_fe_theory_integration.md).
It evolved matched round-0 states for the two saved pilot cells, generated all
three sampling/covariance variants, made trajectory overlays and late-window
residual tables, and compared forced next-day branches. The completed pilot
report is `results/santa_fe_v3_theory_comparison_pilot/pilot_report.md`. The
reduced runner is a numerical three-variable closure, **not** full class-density
HMF. This earlier local validation remains a separate diagnostic layer.

The larger 12-cell configuration has not been launched. Full stochastic HMF,
a broad rho/b regime map, more independent MICRO initial blocks, and a
partitioned, bias-controlled information comparison of continuous branch
outputs remain outstanding. The pilot does not establish a phase boundary or
thermodynamic efficiency.

The saved pilot trajectories occupy about 4.3 MB for 1,200 post-initialization
rounds. At similar compression and state density, the 12-cell design would
produce roughly 160–170 MB of trajectory Parquet before information and
validation outputs. This is a storage estimate, not a runtime guarantee.
