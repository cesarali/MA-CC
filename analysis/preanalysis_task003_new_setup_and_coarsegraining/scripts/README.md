# scripts — reproduce everything here

Run from the repository root with the project venv (`.venv/bin/python`).
Depends on `../../game_analysis/engine.py` for the exact posterior over worlds.

| order | script | produces | time |
|---|---|---|---|
| 1 | `build_states.py` | `data/states_all.parquet` — per-round epistemic state, all three arms | ~2 min |
| 2 | `add_coordinates.py` | `data/states_full.parquet` — adds spread, dec_frac, pool_frac, mean_nfacts | ~2 min |
| 3 | `run_four_tests.py` | `02_markov_tests/four_tests.csv`, `lumpability.csv` | ~2 min |
| 4 | `augmented_states.py` | `02_markov_tests/augmented_states.csv` | ~1 min |
| 5 | `estimate.py` | `03_estimates/*.csv` | ~2 min |

Supporting modules, not run directly:

| module | provides |
|---|---|
| `four_tests.py` | the four Markov tests (order, Chapman-Kolmogorov, conditional independence, lumpability) |
| `estimate.py` | chain fitting, exact path KL, label MI, entropy-production rate, cluster bootstrap |

Earlier scripts kept for provenance: `markov_test.py` (cross-arm transfer test —
invalid, see its report), `markov_test2.py` (same plus exposure),
`markov_order.py` (superseded by `four_tests.py`), `bin_sweep.py` (bin count sweep).

## Definitions used throughout

- **e** — population mean of `P(ALLOCATION_0 | K_j)`, the exact posterior given
  agent *j*'s currently active facts. Range roughly 0.11-0.91 in this study.
- **spread** — standard deviation of that posterior *across the 24 agents*.
  Distinguishes concentrated from dispersed knowledge at equal mean.
- **dec_frac** — fraction of agents holding ≥1 of the 6 decisive facts (the
  population's only complete proof route).
- **pool_frac** — fraction holding ≥1 controller-pool fact.
- **mean_nfacts** — mean number of active facts per agent.
