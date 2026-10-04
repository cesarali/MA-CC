# NeuralWatt LLM API

These notes describe the NeuralWatt provider integration checked on
2026-08-28. Model availability, limits, and prices can change, so recheck the
linked provider pages before a large run.

## Authoritative sources

- [NeuralWatt API overview](https://portal.neuralwatt.com/docs/api/overview)
  documents the OpenAI-compatible API and fixed base URL.
- [DeepSeek V4 Flash](https://portal.neuralwatt.com/models/deepseek-v4-flash)
  documents the model ID, Chat Completions example, capabilities, and pricing.
- [Models API](https://portal.neuralwatt.com/docs/api/models) documents the
  authenticated customer catalog and its model metadata.
- [Authentication](https://portal.neuralwatt.com/docs/authentication) documents
  bearer authentication and recommends `NEURALWATT_API_KEY`.

## Repository configuration

Keep the key only in the repository-root `.env` or the process environment:

```dotenv
NEURALWATT_API_KEY=replace-with-your-neuralwatt-key
```

Use a distinct provider type in experiment YAML:

```yaml
llm_provider:
  type: neuralwatt
  model: deepseek-v4-flash
  credentials_env: NEURALWATT_API_KEY
  timeout_seconds: 120
  max_retries: 2
  request_concurrency: 4
  temperature: 0.0
  max_output_tokens: 256
```

The adapter fixes its base URL to:

```text
https://api.neuralwatt.com/v1
```

It deliberately does not reuse `type: openai`, `OPENAI_API_KEY`, or
`https://api.openai.com/v1`. NeuralWatt and official OpenAI experiments can
therefore coexist without one provider's URL or credential changing the
other's behavior.

The runtime uses its existing OpenAI-compatible HTTP transport rather than
adding the OpenAI Python package. This preserves the repository's normalized
responses, bounded retries, per-provider request semaphore, shared cluster
load control, usage accounting, and lazy-import contract.

`type: neuralwatt` defaults to the provider's fast
`response_format: {type: json_object}` transport mode. This default belongs to
the NeuralWatt adapter, not to a game or study, so NeuralWatt experiments do
not repeat transport policy in scientific YAML. Official OpenAI and the
University of Potsdam adapter are unchanged. A NeuralWatt experiment that
genuinely requires free-form text must opt out explicitly at the provider
boundary:

```yaml
llm_provider:
  type: neuralwatt
  model: deepseek-v4-flash
  options:
    response_format: null
```

Forced tool choice remains available as an explicit provider diagnostic, but
it is not the default because the measured Study 08 route was approximately
5--9 times slower than JSON-object mode.

## Verified model metadata

An authenticated `GET /v1/models` request returned the customer catalog and
listed `deepseek-v4-flash` with:

- system messages, JSON mode, streaming, tools, and reasoning support;
- a 1,048,560-token context limit;
- a 65,536-token maximum output;
- USD pricing of $0.14 per million input tokens, $0.028 per million cached
  input tokens, and $0.28 per million output tokens.

Those dated token prices and limits are present in the repository's offline
pricing catalog so credential-free preflight can estimate a run. Recheck live
metadata before increasing the scale materially.

## Tests and smoke run

Credential-free provider tests use an injected HTTP session and must run in
the ordinary suite:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC python -m pytest tests/mas_cc/test_neuralwatt_provider.py
```

The live test is intentionally opt-in because it makes one billable request:

```bash
module load python/3.11-24.1.0
MAS_CC_RUN_NEURALWATT_SMOKE=1 \
  conda run -n MA-CC python -m pytest tests/mas_cc/test_neuralwatt_provider.py \
  -k live_chat_completion_smoke
```

The downscaled relational experiment config is:

```text
configs/runs/relational_reasoning/relational_imitation_round_feedback_neuralwatt_N6_R3_smoke.yaml
```

It contains one episode with six agents and three population rounds. Preflight
it before launch and keep the output on external `/pscratch` storage:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_neuralwatt_N6_R3_smoke.yaml \
  --output-dir /pscratch/sd/d/dfarough/MA-CC-results/inspection/neuralwatt-n6-r3-preflight

conda run -n MA-CC mas-cc experiment run \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_neuralwatt_N6_R3_smoke.yaml \
  --output-dir /pscratch/sd/d/dfarough/MA-CC-results/neuralwatt \
  --approve-preflight /pscratch/sd/d/dfarough/MA-CC-results/inspection/neuralwatt-n6-r3-preflight/preflight_id.txt \
  --no-progress
```
