# DeepInfra provider profile

Read this reference whenever `llm_provider.type` is `deepinfra`, whether the
workers run on Potsdam, another cluster, or a standalone server.

## Configuration and credentials

- Use the repository's `deepinfra` adapter and OpenAI-compatible runtime.
- Credentials resolve from `DEEPINFRA_API_KEY`; the optional base URL override
  resolves from `DEEPINFRA_BASE_URL`. Never print, embed in YAML, or commit
  either value.
- Validate the configured model against DeepInfra's live model endpoint.
- Query authenticated account limits from
  `https://api.deepinfra.com/v1/me/rate_limit` during setup or preflight.
  Treat concurrency and token-per-minute values as mutable account state.
- Keep DeepInfra coordinator state, credentials, limits, and failures isolated
  from the university provider.

## Planning provider load

Separate CPU processes, episode slots, provider request concurrency, token
throughput, and RPM. More CPUs do not by themselves increase provider RPM.
Plan from current account concurrency and TPM limits, expected prompt volume,
measured model latency, and a declared RPM safety target.

For the current Task003 q=12 anchor, the last proven topology was:

- model `deepseek-ai/DeepSeek-V4-Flash`;
- one cell per shard, five active shards;
- 20 episode/request slots per shard;
- study-wide concurrency ceiling 100;
- target 1,000 RPM;
- 8 CPUs and 12 GiB per shard;
- 20-hour shard limit.

These are dated acceptance-anchor settings, not defaults for new models,
accounts, or studies. Re-query limits and preflight every real launch. Observed
throughput can remain below target because of response latency even when the
concurrency plan is correct; inspect ready work, occupied leases, response
latency, token throughput, and successful RPM before changing resources.

Keep the shared adaptive provider coordinator enabled for multi-worker runs.
Begin below the discovered account ceiling, reduce quickly on retryable
failures, and recover gradually after stable service.

## Failure semantics

- HTTP 429/5xx, timeouts, and connection loss are retryable service
  conditions within the bounded logical-request window.
- HTTP 402 is a payment or credit refusal. Lowering concurrency does not fix
  it; stop new paid work, preserve checkpoints, and report the account blocker.
- Malformed model decisions are semantic-validation failures, not provider
  throttling. Use the contract-authored correction flow and never coerce an
  invalid identifier.
- Resume only missing or interrupted episode identities. Reuse sealed episodes
  and validated trajectory prefixes.

When DeepInfra runs on Potsdam SLURM, also read `references/potsdam.md` for the
compute environment, output-root, and generic-launcher rules.
