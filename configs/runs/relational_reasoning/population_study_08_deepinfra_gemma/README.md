# Study 08 — DeepInfra Gemma provider variant

This directory preserves the 192-cell, 1,920-episode scientific design from
`../population_study_08` while changing only provider/execution identity:

- provider: `deepinfra`;
- model: `google/gemma-4-E4B-it`;
- credential name: `DEEPINFRA_API_KEY`;
- provider-scoped JSON-object mode inherited from `DeepInfraProvider`;
- independent experiment/study names and result root;
- DeepInfra capacity plan calibrated from the production-shaped smoke, a
  successful 64-request burst, and the authenticated 200-request account
  limit: 100 initial / 200 maximum outstanding requests with a hard 1,200 RPM
  rolling gate. All 192 cells may remain runnable; the shared gate determines
  actual provider traffic.

The wrong/truth axes, controller semantics, prompts, root seed, tasks,
repetitions, metrics, and strict aggregation recipe are unchanged. The copied
analysis recipe is intentionally byte-identical to Study 08 so the provider
comparison uses the same estimators.

NERSC production must use `mas-cc study prepare` followed by the generic
interactive launchers in `scripts/nersc/`. Never use `sbatch`, regular QoS, or
write results inside this config directory.

Do not launch production until all of the following pass:

1. authenticated account-limit lookup;
2. one JSON/system-message completion;
3. the N6/R3 smoke;
4. one production-shaped N=24, ten-round Study 08 episode;
5. an explicitly sized concurrent capacity probe informed by measured token
   usage and the account's 1.1M TPM ceiling.

All five checks passed on 2026-08-30. The 64-request burst returned only HTTP
200 responses. A production startup then returned 948/948 HTTP successes at
49 outstanding requests, 948 RPM, and 3.94 seconds mean latency (10.74 seconds
maximum). The full-shaped episode averaged about 690 total tokens per request,
making the 1,200 RPM gate approximately 828,000 TPM before workload variation,
below the authenticated 1.1M TPM account ceiling.
