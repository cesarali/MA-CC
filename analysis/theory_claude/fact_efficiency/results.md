# Fact efficiency of the controller: results (8 November 2026)

Pre-registration: `../../theory_blackboard/2026-11-08_claude.md`, section 3. Code: `scan.py` (models), `data_check.py`
(campaign posts). Outputs: `mf.csv`, `finite.csv`, `data_check.csv` (`finite_episodes.parquet` is regenerable and not
versioned). Fact-flow model: P0 except N = 15, W = 30, F_A = 15, ε = 1. Finite model: 400 paired episodes per cell.

## Predictions

| | prediction | outcome |
|---|---|---|
| E1 | mean field, fixed β_C: G(30) increases with F_C | **met** in 75/75 cells |
| E2 | finite, fixed β_C = 0.5: interior optimum F_C\* in ≥ half the cells, rising with q τ | **failed**: G rises monotonically to F_C = 16 in all 15 cells |
| E3 | mean field: retention varies < 0.1 across F_C; finite: retention decreases with F_C in ≥ 70% of cells | **mean-field part failed** (range up to 1.6); **finite part met** (12/15) |
| E4 | doubling the rate less than doubles G(30) | **met** everywhere; median ratio 1.33 (mean field), 1.18 (finite) |
| E5 | fixed total evidence F_C β_C = 2: mean field favours concentration; finite even more | **mean field failed** (nearly flat; best F_C is 16 in 14 cells, 1 in 11); **finite partly met** (concentration wins at short memory, spreading at long memory; below) |
| D1 | real controller uses ≤ 6 distinct facts; controller facts alive at t = 30 within a factor 2 of F_C\* | **partly met**: 3.7–5.4 distinct facts in task003, but up to 7.6 in OP |

## Why E2 failed: the design confounded facts with evidence

At a fixed strength per fact, adding facts also adds total evidence (F_C β_C grows from 0.5 to 8). The extinction cost
of spreading never outweighs that. The question "how efficiently does a controller use facts" must hold total evidence
fixed. That is scan B.

## Main result: at fixed total evidence, the optimal number of facts grows with reads per memory lifetime

Finite model, F_C β_C = 2, λ0 = 1, G(30):

| q τ | 1 | 2 | 4 | 8 | 16 | 31–63 |
|---|---|---|---|---|---|---|
| best F_C (finite) | 1 | 1 | 1–2 | 1–4 | 4 | 8 |

The rank correlation between F_C\* and q τ is 0.75 (p = 0.001, 15 cells).
- **Few reads per memory lifetime:** agents can hold only a little, so one strong fact is best. Spreading divides a
  small budget over facts that die out.
- **Many reads per memory lifetime:** agents saturate on a few facts, so repeated reads are wasted, and spreading the
  same evidence over more facts lets each agent accumulate more.

The mean field shows no clear optimum (differences ≤ 0.04). **The trade-off is a finite-size effect.**

## Persistence

At fixed total evidence (finite, q = 4), the retained *effect* G(130)/G(30) barely depends on F_C (0.77–1.22). The
controller's *facts*, however, survive much better when concentrated:

| ρ | facts alive at t = 130 / alive at t = 30, F_C = 1 | F_C = 16 |
|---|---:|---:|
| 0.135 | 0.87 | 0.30 |
| 0.61 | 0.83 | 0.34 |
| 0.88 | 0.97 | 0.69 |

Spread evidence loses individual facts without losing much of the effect.

## Data check (campaign, false arm, standard reading)

- **Facts used:** from a pool of 10–12 controller-only facts, the simulator's controller uses on average 3.7–5.4
  distinct facts per episode in task003 and 4.0–7.6 in OP.
- **Posts:** 11–27 per episode, so about 4–5 per fact used.
- **Facts alive at t = 30:** 2.6–7.2.

The real controller picks facts by evidence, not by a fixed F_C, and it concentrates more at short memory and spreads
more at long memory. That is the same direction as the finite optimum. This last point is post-hoc and qualitative.

## Correction after the ChatGPT agent's review (8 November 2026; `2026-11-08_chatgpt_02_efficiency_review.md`)

1. **E2 stays failed.** Scan B (fixed total evidence) is an exploratory result chosen after E2 failed, not a rescue
   of E2.
2. **My mechanism for the scan-B pattern was wrong.** At fixed total strength B = F_C β_C and fixed board share, the
   mean held controller evidence is B m_F, which *decreases* with F_C. So "spreading lets agents accumulate more" does
   not hold at fixed total evidence; it holds only at fixed strength per fact. The long-memory preference for spreading
   must come from something else. Candidates:
   - the sigmoid's curvature acting on the variance of held evidence (concentration makes it more all-or-nothing);
   - endogenous changes of the controller's board share;
   - memory correlations.

   These have not yet been separated. "Finite-model effect" does not mean "caused by fact extinction".
3. **The optimum is a broad preference region, not a sharp F_C\*.** ChatGPT's simultaneous bootstrap resolves a unique
   best F_C in only 2 of 15 cells. The robust statement:
   - concentration (1–2 facts) wins at short memory and little reading, where differences are large and resolved
     (e.g. q = 4, ρ = 0.135: G = 0.48 at F_C = 1 against 0.23 at F_C = 16);
   - at long memory several F_C values are statistically tied, with the observed best at 4–8.

   The rank association with q τ (0.72–0.75) is descriptive.
4. **Persistence wording:** concentration preserves the *fraction* of the controller's facts that survive (0.87
   against 0.30). The *number* of distinct controller facts still alive is higher with spreading (2.0 against 0.87 at
   q = 4, ρ = 0.135). Verified in `finite.csv`.
