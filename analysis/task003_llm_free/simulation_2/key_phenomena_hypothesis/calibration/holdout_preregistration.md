# Pre-registered development / held-out split

**Registered 22 October 2026, before any model was written** (commit of this file precedes every file in `theory/`
except `00_microscopic_rules.md`). Applies to every model version.

## Honest caveat

All the data below were already seen in the step 0–3 reports. The split cannot make the held-out data unseen. What it
does: **model choices (closures, approximations, any constant) may be made by looking only at development data**;
held-out data are evaluated once per model version, logged in `../progress.md`, and never used to choose between
versions. The simulator rules themselves (`../theory/00_microscopic_rules.md`) are exact and need no fitting.

## Development data (may be used to build and choose models)

- Both worlds (symmetric, nosolution); run `2026-10-20_sim2_reading_mechanism` (q_c 12, λc 1, gate on, B 30, W 48).
- ρ ∈ {0.25, 0.75, 1}, q ∈ {3, 6}, t ≤ 30, false and silent arms:
  vote susceptibility G and G(t); per-fact memory share m_f and board share b_f; agent memories (bitmasks); posts.
- Reading increment D(3→6) at ρ ∈ {0.25, 0.75, 1} (6 values). These include an amplifying cell (ρ = 0.25) and a
  protective one (ρ = 1) in each world, so a model can see both signs.

## Held-out data (evaluated, never tuned on)

1. **Other memories:** ρ ∈ {0.125, 0.375, 0.5, 0.625, 0.875}, every q.
2. **Larger reading:** q ∈ {9, 12}, every ρ; hence every D(6→9), D(9→12), and D(3→6) at the held-out ρ
   (42 of the 48 reading increments).
3. **Persistence:** everything after t = 30: the four post-stop rules to t = 130 (`2026-10-20_sim2_persistence_branches`)
   and G(t) for 30 < t ≤ 40 in the reading run.
4. **Variants:** W = 24, W = 240, B = 100 (light records; G and D at q_c 12).
5. **Defence** (later): the deduplicated reading rule, predicted before it is run.

## Scorecard (computed for every model version)

- Sign agreement on resolved reading increments (95% interval excludes 0), held-out cells, per world, with counts of
  positive and negative cells and the list of misses.
- Crossover brackets per reading step: model against data, in units of the ρ grid (0.125).
- Absolute error of G at t = 30 on held-out (ρ, q).
- Post-stop: ordering of the four rules; δ at t = 31, 40, 80, 130 against data; time to half the t = 30 value.
- Mechanistic: error of m_f and b_f (controller-only and agent-origin facts separately) at held-out (ρ, q).

Targets are in `benchmark_*_development.csv` (here) and `../held_out_validation/benchmark_*_held_out.csv`; the full
tables (G(t), m_f, b_f, post-stop δ) are built by `build_benchmarks.py` into `results/key_phenomena/theory_benchmarks/`.
