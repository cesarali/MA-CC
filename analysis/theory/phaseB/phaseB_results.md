# Phase B — results (24 October 2026)

Plan and attribution rule: [`00_plan.md`](00_plan.md) (commit `1a32d53`, before any M3s run). Post-hoc: all data were
known; M3s chooses nothing. Scoring: `score_attribution.py` (existing scorers; table `attribution.csv`).

**Answer: both changes contribute.** The stochastic controller (M3 → M3s) gives 41% (task003) and 52% (Object
Placements) of the M3 → M4 reduction in G error. The shared realised board (M3s → M4) gives the rest. The board
alone also brings the Object Placements crossover and the post-stop ordering.

| | M3 | M3s | M4 |
|---|---|---|---|
| task003 held-out G error (mean of setups) | 0.121 | 0.082 | 0.026 |
| Object Placements G error (reversal) | 0.139 | 0.080 | 0.025 |
| task003 held-out signs, symmetric / no-proof | 13/17, 13/16 | 14/17, 11/16 | 13/17, 13/16 |
| Object Placements signs, symmetric / no-proof | 5/16, 3/15 | 6/16, 9/15 | 11/16, 12/15 |
| task003 symmetric 3→6 crossover (data 0.5–0.625) | 0.25–0.375 | 0.375–0.5 | 0.625–0.75 |
| task003 no-proof 3→6 crossover (data 0.75–1) | 0.625–0.75 | 0.625–1 | 0.75–1 |
| Object Placements symmetric 3→6 crossover (data 0.25–0.375) | none | none | 0.25–0.375 |
| post-stop ordering, task003 / Object Placements (of 8) | 3, 1 | 5, 1 | 7, 7 |
| Object Placements no-communication half-time exact (of 4) | 0 | 0 | 2 |

f_controller = (err_M3 − err_M3s)/(err_M3 − err_M4) = 0.41 (task003), 0.52 (Object Placements).

**Correction.** M4's Object Placements no-communication half-times are exact in 2 of 4 cells: 1.0 and 2.0 match;
symmetric ρ 0.25 gives 1.25 against 1.0; symmetric ρ 0.75 has no effect to halve. Two earlier statements called them
exact: the 23 October progress line and the 23 October addenda. Both are corrected in the 24 October addenda.

**Interpretation.** Two finite-size ingredients, each about half of the gain:
- **A finite controller.** The controller posts whole facts at random times. An expected controller spreads
  fractional posts over all the facts it might choose.
- **A shared finite board.** All readers see the same fluctuating board.

The board is the part that decides *when* reading starts to protect the swarm (the Object Placements crossover) and how
the effect behaves after the controller stops. This fits Phase A, A4: all coupling between agents passes through the
board, so Phase C must model the realised board, not its average.
