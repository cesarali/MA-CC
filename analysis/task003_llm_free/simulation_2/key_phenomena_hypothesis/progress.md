# progress.md: theory programme (plan of 21 October 2026)

One line per decision: hypothesis, experiment, outcome, next decision. Newest last.

| date | phase | hypothesis / action | outcome | next decision |
|---|---|---|---|---|
| 2026-10-22 | 1 rules | extract exact transition rules from code | `theory/00_microscopic_rules.md`; posting depends on whole memory and avoids just-read facts; controller deterministic given what it reads; no-proof controller-only facts are jointly neutral (0.37, 0.26, 0.37) | use as model inputs |
| 2026-10-22 | 1 data | benchmark targets and split | `calibration/build_benchmarks.py`; split registered in `calibration/holdout_preregistration.md` before any model | start phase 2 after the researcher's go |
| 2026-10-22 | 2 M1, M2 | deterministic mean-field closures (fact-by-fact M1; exact single-agent memory M2), same simulator rules | **fail**: held-out resolved signs M1 9/17 sym, 6/16 nos; M2 8/17, 8/16 (chance); G about 3x too small | diagnose |
| 2026-10-22 | 2 diag | same code with a real 24-agent board | reproduces the simulator (G 0.32 vs 0.33, 0.29 vs 0.29): not a bug | test N |
| 2026-10-22 | 2 diag | scale N = 24 R (W, rate, budget x R) | G falls toward mean field (sym 0.32 to 0.14 at N 192): finite-size effect; reversal survives at N 96, its symmetric amplifying side shrinks 5x | test extinction |
| 2026-10-22 | 2 diag | pin own facts (no agent-fact extinction) at N 24 | sym G 0.44 to 0.09, 0.32 to 0.07; nos 0.30 to 0.20, 0.29 to 0.21 | propose M3; researcher approved |
| 2026-10-22 | 2 scaling note | researcher: scaling agents without facts raises copies per fact | the N series measures redundancy, not size; agreed design: K projects with one split, K = 2, 4 | build after M3 |
| 2026-10-22 | 2 M3 | M2 + integer holder counts per fact in a 24-agent swarm, absorbing extinction | **partial success**: held-out signs 13/17 sym, 13/16 nos (from 8/17, 8/16); post-stop mechanisms and no-communication half-times right; magnitudes ~60%; symmetric amplifying side missed | build K = 2, 4 worlds; then consider M4 (shared finite-board noise) |
