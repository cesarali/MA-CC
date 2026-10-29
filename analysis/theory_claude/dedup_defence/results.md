# Deduplicated reading: stage 1 results (finite fact-flow model)

29 October 2026. Pre-registration: `00_preregistration.md`. Code: `stage1_finite.py` and the `read_mode` option of
`../phenomenological_fact_flow/finite_model.py`. Outputs: `stage1_summary.csv`, `stage1_episodes.csv` (400 paired
episodes per cell; about 1.5 minutes on 7 workers).

| q | ρ | G standard | G dedup | change | silent A2 standard → dedup | native facts circulating (silent) |
|---|---|---:|---:|---:|---|---|
| 3 | 0.25 | 0.319 | 0.094 | −71% | 0.148 → **0.265** | 3.15 → 3.35 |
| 3 | 0.50 | 0.297 | 0.194 | −35% | 0.103 → 0.102 | 4.77 → 6.56 |
| 3 | 0.75 | 0.175 | 0.118 | −33% | 0.048 → 0.040 | 8.37 → 9.48 |
| 6 | 0.25 | 0.388 | 0.292 | −25% | 0.090 → 0.085 | 3.94 → 4.27 |
| 6 | 0.50 | 0.307 | 0.220 | −28% | 0.062 → 0.043 | 5.64 → 5.92 |
| 6 | 0.75 | 0.147 | 0.108 | −27% | 0.019 → 0.014 | 8.87 → 8.48 |

## Predictions

- **D1 (G falls, by ≥ 25% at ρ ≤ 0.5): met in direction everywhere.** The size threshold is met in 3 of 4 low-memory
  cells; q 6, ρ 0.25 falls 24.7%, narrowly short.
- **D2 (more native facts survive at ρ ≤ 0.5): met** in all four cells.
- **D3 (silent A2 share rises by no more than 0.02): failed at q 3, ρ 0.25** (+0.117). Elsewhere the silent arm
  improves. The cause is a modelling artifact of the pre-registered rule: each background post counts as its own item,
  so with few distinct facts and many background posts, deduplicated reading reads mostly background. A native fact
  then has 1.9 holders instead of 7.6. Agents are less informed, so the −71% in that cell is **not** a valid defence
  result. In all q = 6 cells, holders are equal or higher under deduplication.
- **D4 (retained persistence no larger): failed.** δ(130)/δ(30) is 1.06 and 1.04 under deduplication against 0.89 and
  1.01 under standard reading. Deduplication lowers the effect's level but does not make it fade faster.
- **D5 (the reversal survives): met.** D(3→6) is +0.198, +0.026, −0.010 under deduplication (standard: +0.069, +0.010,
  −0.028).

## Reading

In this engine-free model, deduplicated reading is a genuine partial defence where agents read enough (q = 6): the
controller effect falls by about a quarter, the silent arm is slightly more accurate, and more native facts survive. It
does not remove persistence. Its weak point is any reading pool dominated by many distinct low-value items. In the
simulator that is the real question, because there every post carries a task fact.

Stage 2 (Simulation 2) waits for the 15-agent setups and will be pre-registered separately.
