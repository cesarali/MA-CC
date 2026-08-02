# Phase 6 Naming Convention Game inspection report

- Status: **PASS**
- Command: `mas-cc game run --config /home/cesarali/LanguageGames/MA-CC/configs/runs/naming_convention_smoke_test.yaml --output-dir inspection/phase_06`
- Scientific scope: repeated symmetric Ashery–Aiello–Baronchelli convention game, not the speaker/hearer inventory Naming Game.
- Run classification: **architecture smoke test; not a paper replication**.
- Code paths exercised: complete-mixing pair sampling, two frozen private views, concurrent validated provider decisions, answer-first parsing, isolated validation retries, pure +100/-50 transition, bounded prompt memory, complete evaluator history, stage-aware provider-neutral planning, Phase 4 pricing/budget composition, and deterministic plotting.
- Expected behavior: 6 ordinary agents play 12 sequential pair interactions with two simultaneous decisions per pair; empty-memory choices remain provider calls.
- Deviations from the paper profile: population and horizon are reduced for inspection; the deterministic mock provider replaces source-model stochasticity; `top_k=10` is requested and recorded but the current normalized adapters omit it; one project population round is labeled as N pair interactions.

## Results

- Pair interactions: 12
- Logical decisions: 24
- Validation attempts/provider requests: 24
- Provider transport retries: 0
- Successful coordination interactions: 12
- Maximum full private history: 6
- Visible memory bound: 3
- Planned requests lower/expected/maximum: 24/24/72
- Pricing composition: `known`; launch `permitted`
- Legacy fixed-fixture parity: passed

## Files to inspect manually

- `agents_initial.json` and `agents_final.json` — complete evaluator-side agent histories and lifetime scores.
- `interactions.jsonl` — every selected pair, private view, prompt, response, parser result, retry count, transition, and post-memory.
- `selected_audit_traces.jsonl` — deterministic full traces for interactions 1, 6, 12.
- `game_call_plan.json` — pair stage, concurrency barrier, logical decisions, validation bounds, and memory scenarios without provider prices.
- `prompt_token_scenarios.csv` — empty, representative, and full-memory prompt estimates and hashes.
- `trajectory.csv` — raw outcomes plus rolling action shares and coordination over up to N interactions.
- `action_share.png` and `coordination_rate.png` — rolling evaluator-only diagnostics using window N=6.
- `manifest.json` — hashes and machine-readable acceptance checks.
