# Phase 5 inspection report

- Status: **PASS**
- Command: `mas-cc game run --config configs/runs/toy_game_smoke_test_v3.yaml --output-dir inspection/phase_05_v3`
- Code paths exercised: resolved configuration loading, lazy game/provider registries, generic game protocol, compositional prompt rendering, normalized provider calls, local response/action validation, pure state transitions, provider-neutral demand planning, Phase 4 pricing composition, runtime budget enforcement, and deterministic trajectory rendering.
- Input: `/home/cesarali/LanguageGames/MA-CC/configs/runs/toy_game_smoke_test_v3.yaml`
- Expected behavior: two agents choose A or B through the configured provider for exactly 3 interactions; matching actions earn one point; the same seed and resolved inputs reproduce all scientific trajectory artifacts.
- Deviations or warnings: provider timing is intentionally omitted from `interactions.jsonl` so deterministic mock traces remain byte reproducible; runtime timing/audit events enter in Phase 7.

## Results

- Interactions completed: 3
- Normalized provider requests: 6
- Matching interactions: 3
- Termination reason: `finite_horizon_reached`
- Provider-independent expected request demand: 6
- Static pricing composition status: `known` / launch `permitted`
- Final scores: agent-000=3, agent-001=3

## Files to inspect manually

- `resolved_config.yaml` — all component references expanded without secrets.
- `initial_state.json` — immutable state before any provider call.
- `observations.jsonl`, `bound_prompts.jsonl`, and `compiled_prompts.jsonl` — the narrow prompt chain for every decision.
- `interactions.jsonl` — one complete observation/prompt/response/action/transition chain per line.
- `final_state.json` — terminated state and cumulative scores.
- `game_call_plan.json` and `prompt_scenarios.json` — provider-independent demand and bound prompt scenarios.
- `trajectory.csv` and `trajectory.png` — tabular and visual score trajectories.
- `manifest.json` — artifact hashes and machine-readable acceptance checks.
