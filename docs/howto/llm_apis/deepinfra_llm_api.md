# DeepInfra LLM API

These notes describe the DeepInfra integration checked on 2026-08-30.
Model availability, account limits, and prices can change, so refresh the
provider metadata before a large production study.

## Authoritative sources

- [Chat Completions](https://docs.deepinfra.com/chat/overview) documents the
  OpenAI-compatible base URL and chat endpoint.
- [Structured Outputs](https://docs.deepinfra.com/chat/structured-outputs)
  documents `json_object` and `json_schema` response modes.
- [Rate Limits](https://docs.deepinfra.com/account/rate-limits) documents the
  default per-model outstanding-request limit and its relationship to RPM.
- [Account Rate Limit](https://docs.deepinfra.com/api-reference/account/account-rate-limit)
  documents the authenticated concurrency and TPM metadata endpoint.
- [DeepSeek V4 Flash metadata](https://api.deepinfra.com/models/deepseek-ai/DeepSeek-V4-Flash)
  is the dated public model record used by the offline pricing catalogue.
- [Gemma 4 E4B API page](https://deepinfra.com/google/gemma-4-E4B-it/api)
  is the dated public record used by the Study 08 provider variant.
- [Billing checklist](https://docs.deepinfra.com/api-reference/billing/get-checklist),
  [billing portal](https://docs.deepinfra.com/api-reference/billing/billing-portal),
  and [automatic top-up](https://docs.deepinfra.com/api-reference/billing/setup-topup)
  document the account's read-only billing state and funding controls.

## Repository configuration

Keep the key only in the repository-root `.env` or process environment:

```dotenv
DEEPINFRA_API_KEY=replace-with-your-deepinfra-key
```

Use a distinct provider type in experiment YAML:

```yaml
llm_provider:
  type: deepinfra
  model: deepseek-ai/DeepSeek-V4-Flash
  credentials_env: DEEPINFRA_API_KEY
  timeout_seconds: 120
  max_retries: 2
  request_concurrency: 32
  temperature: 0.0
  max_output_tokens: 4096
```

The adapter fixes its chat base URL to:

```text
https://api.deepinfra.com/v1/openai
```

It deliberately does not reuse `type: openai`, `OPENAI_API_KEY`,
`NEURALWATT_API_KEY`, or either provider's URL. DeepInfra, NeuralWatt,
official OpenAI, and the University of Potsdam proxy can coexist in the same
checkout without credential or route fallback.

DeepInfra exposes OpenAI-compatible chat below `/v1/openai`, but its model
catalogue is exposed separately at `/v1/models`. The adapter therefore sends
chat directly to `/v1/openai/chat/completions` and does not run the shared
`base_url/models` discovery algorithm, which would probe a nonexistent route.

The runtime uses its existing OpenAI-compatible HTTP transport rather than
adding the OpenAI Python package. Normalized responses, bounded retries,
per-client request concurrency, shared NERSC load control, usage accounting,
model profiles, and lazy imports therefore behave exactly as they do for the
other remote providers.

`type: deepinfra` defaults to the provider's fast
`response_format: {type: json_object}` mode. This is a DeepInfra adapter
default, not game or study policy. A genuinely free-form DeepInfra workload
can opt out explicitly:

```yaml
llm_provider:
  type: deepinfra
  model: deepseek-ai/DeepSeek-V4-Flash
  options:
    response_format: null
```

## Model metadata and offline pricing

The public model record checked on 2026-08-30 lists
`deepseek-ai/DeepSeek-V4-Flash` as active, with OpenAI chat, JSON, structured
output, tools, reasoning, seed, and configurable-temperature support. It
reports:

- 1,048,576-token context and 65,536-token maximum output;
- $0.09 per million ordinary input tokens;
- $0.018 per million cached input tokens;
- $0.18 per million output tokens.

Those dated values are in the repository's offline pricing catalogue for
credential-free preflight. They do not prove current account access, so the
live smoke remains required before production.

The Study 08 Gemma variant uses `google/gemma-4-E4B-it`. Its public
record, checked on 2026-08-30, reports OpenAI compatibility, tools, reasoning,
temperature, and seed support. It reports:

- a 131,072-token context;
- $0.02 per million ordinary input tokens;
- $0.10 per million output tokens.

DeepInfra does not publish a distinct maximum-output value for this model, so
the experiment retains its explicit 4,096-token output cap. The catalogue
also advertises a 0.8 multiplier for the optional flex service tier, but the
checked-in configuration uses the ordinary tier. Do not budget at flex rates
unless that tier is explicitly configured and validated.

DeepInfra's generic input metadata lists a `response_format` field, but a live
E4B request on 2026-08-30 returned HTTP 405 with the explicit message that
`json_object` is unsupported for this model. `DeepInfraProvider` therefore
keeps JSON-object mode as the provider default while disabling it for this
exact model ID. This is provider/model compatibility logic, not game or study
policy; the relational response contract is still enforced by prompting and
the normal parser/validation path, which accepts a valid object inside a
provider-added Markdown JSON fence.

The previous `google/gemma-4-31B-it-turbo` record was $0.09 per million input
tokens and $0.34 per million output tokens on the same date. E4B is therefore
materially cheaper at base rates, but production budgeting must still rely on
fresh preflight totals.

## Balance and top-up

A `402 payment_required` response is non-retryable. Check
`GET /payment/checklist` before allocating compute: a zero `stripe_balance`
with `suspend_reason: balance` means the account needs funds even when its
payment method, billing address, and team role are valid. DeepInfra documents
`GET /payment/billing-portal` for obtaining its hosted billing URL and
`POST /payment/topup` for configuring automatic top-up. Funding changes are
account mutations and must be made deliberately by the account owner; MA-CC
never performs them during preflight or launch.

## Concurrency and RPM

DeepInfra documents a default ceiling of 200 outstanding requests **per
model**. It is a concurrency limit, not a fixed requests-per-minute limit. At
steady state, the latency-derived upper bound is approximately:

```text
RPM = concurrent requests * 60 / average request latency in seconds
```

For example, 200 concurrent requests averaging 10 seconds correspond to about
1,200 RPM. Token-per-minute capacity, transient model saturation, and response
length can lower actual throughput. The authenticated endpoint
`GET https://api.deepinfra.com/v1/me/rate_limit` returns the account's actual
per-model outstanding-request and TPM limits. `DeepInfraProvider` exposes this
as `discover_account_limits()` so tests and launch tooling do not have to
handle the key themselves.

The authenticated MA-CC account returned 200 outstanding requests/model and
1,100,000 tokens/minute on 2026-08-30. Production concurrency must satisfy
both ceilings; the TPM limit can bind before the concurrency limit for long
Study 08 prompts.

Do not jump directly from the documented ceiling to 200 production workers.
Query the account limit, run the opt-in burst at increasing sizes, measure
latency/429s, and configure the NERSC study's shared provider coordinator below
the first unstable point. The component default of 32 is a conservative local
cap, not a claim about measured production capacity.

## Tests and NERSC smoke run

Credential-free adapter, isolation, routing, profile, pricing, and config tests
run in the ordinary suite:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC python -m pytest tests/mas_cc/test_deepinfra_provider.py
```

After `DEEPINFRA_API_KEY` is present, the live flag performs one free metadata
lookup and one billable chat completion:

```bash
module load python/3.11-24.1.0
MAS_CC_RUN_DEEPINFRA_SMOKE=1 \
MAS_CC_DEEPINFRA_SMOKE_MODEL=google/gemma-4-E4B-it \
  conda run -n MA-CC python -m pytest tests/mas_cc/test_deepinfra_provider.py \
  -k 'live_account_limit_smoke or live_chat_completion_smoke' -s
```

The separate burst flag is billable and should run on an NERSC interactive CPU
node. Its size defaults to 16 and can be changed explicitly:

```bash
module load python/3.11-24.1.0
MAS_CC_RUN_DEEPINFRA_BURST_SMOKE=1 \
MAS_CC_DEEPINFRA_BURST_REQUESTS=16 \
  conda run -n MA-CC python -m pytest tests/mas_cc/test_deepinfra_provider.py \
  -k live_concurrent_burst_smoke -s
```

The downscaled relational experiment is:

```text
configs/runs/relational_reasoning/relational_imitation_round_feedback_deepinfra_N6_R3_smoke.yaml
```

It contains one episode with six agents and three population rounds. Preflight
may run on the login node; the live episode must run in an allocation obtained
with `--qos=interactive --constraint=cpu`, and all output remains under
`/pscratch/sd/d/dfarough/MA-CC-results`.

For Gemma, the NERSC smoke payload runs the live contract check, N6/R3 episode,
and one production-shaped N=24, ten-round Study 08 episode in the same separate
interactive allocation:

```bash
scripts/nersc/run_command.sh --account m4539 --time 00:45:00 \
  --cpus-per-task 64 -- \
  scripts/nersc/smoke_deepinfra_gemma.sh \
  /pscratch/sd/d/dfarough/MA-CC-results/smoke/deepinfra-gemma-<timestamp>
```

The full paired provider variant is in
`configs/runs/relational_reasoning/population_study_08_deepinfra_gemma`.
