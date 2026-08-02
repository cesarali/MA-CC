# OpenAI API model pricing

These notes capture the official OpenAI API pricing published on 2026-08-02.
Pricing and model availability can change; use the linked live pages before a
large or expensive run.

## Authoritative sources

- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) is the
  authoritative live price table.
- [OpenAI API models](https://developers.openai.com/api/docs/models) describes
  current model families and capabilities.
- [`GET /v1/models`](https://developers.openai.com/api/reference/resources/models/methods/list)
  returns the models available to an API account.

Unlike the University proxy's `/v1/model/info`, OpenAI's documented
`GET /v1/models` response does not include pricing. Its model objects contain
basic identity/ownership metadata, so pricing must be joined from the official
pricing page or maintained in a dated local catalog.

## Credentials and endpoint

The repository-root `.env` contains only the credential used by the official
OpenAI adapter:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-key
```

Never print or save this value. The official base URL is:

```text
https://api.openai.com/v1
```

To inspect account-specific model availability without making a completion:

```python
from openai import OpenAI

client = OpenAI()
for model in client.models.list():
    print(model.id)
```

## Standard text-token pricing

Prices below are USD per 1 million tokens for standard processing. A dash means
the official table does not publish that rate. For GPT-5.4/5.5/5.6 models,
"long" is the official long-context tier; consult the live pricing page for the
model-specific threshold.

| Model | Input | Cached input | Cache write | Output | Long input | Long cached | Long cache write | Long output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.6-sol` | $5.00 | $0.50 | $6.25 | $30.00 | $10.00 | $1.00 | $12.50 | $45.00 |
| `gpt-5.6-terra` | $2.00 | $0.20 | $2.50 | $12.00 | $4.00 | $0.40 | $5.00 | $18.00 |
| `gpt-5.6-luna` | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |
| `gpt-5.5` | $5.00 | $0.50 | — | $30.00 | $10.00 | $1.00 | — | $45.00 |
| `gpt-5.5-pro` | $30.00 | — | — | $180.00 | $60.00 | — | — | $270.00 |
| `gpt-5.4` | $2.50 | $0.25 | — | $15.00 | $5.00 | $0.50 | — | $22.50 |
| `gpt-5.4-mini` | $0.75 | $0.075 | — | $4.50 | — | — | — | — |
| `gpt-5.4-nano` | $0.20 | $0.02 | — | $1.25 | — | — | — | — |
| `gpt-5.4-pro` | $30.00 | — | — | $180.00 | $60.00 | — | — | $270.00 |
| `gpt-5.2` | $1.75 | $0.175 | — | $14.00 | — | — | — | — |
| `gpt-5.2-pro` | $21.00 | — | — | $168.00 | — | — | — | — |
| `gpt-5.1` | $1.25 | $0.125 | — | $10.00 | — | — | — | — |
| `gpt-5` | $1.25 | $0.125 | — | $10.00 | — | — | — | — |
| `gpt-5-mini` | $0.25 | $0.025 | — | $2.00 | — | — | — | — |
| `gpt-5-nano` | $0.05 | $0.005 | — | $0.40 | — | — | — | — |
| `gpt-5-pro` | $15.00 | — | — | $120.00 | — | — | — | — |
| `gpt-4.1` | $2.00 | $0.50 | — | $8.00 | — | — | — | — |
| `gpt-4.1-mini` | $0.40 | $0.10 | — | $1.60 | — | — | — | — |
| `gpt-4.1-nano` | $0.10 | $0.025 | — | $0.40 | — | — | — | — |
| `gpt-4o` | $2.50 | $1.25 | — | $10.00 | — | — | — | — |
| `gpt-4o-2024-05-13` | $5.00 | — | — | $15.00 | — | — | — | — |
| `gpt-4o-mini` | $0.15 | $0.075 | — | $0.60 | — | — | — | — |
| `o1` | $15.00 | $7.50 | — | $60.00 | — | — | — | — |
| `o1-pro` | $150.00 | — | — | $600.00 | — | — | — | — |
| `o3-pro` | $20.00 | — | — | $80.00 | — | — | — | — |
| `o3` | $2.00 | $0.50 | — | $8.00 | — | — | — | — |
| `o4-mini` | $1.10 | $0.275 | — | $4.40 | — | — | — | — |
| `o3-mini` | $1.10 | $0.55 | — | $4.40 | — | — | — | — |
| `gpt-4-turbo-2024-04-09` | $10.00 | — | — | $30.00 | — | — | — | — |
| `gpt-4-0613` | $30.00 | — | — | $60.00 | — | — | — | — |
| `gpt-3.5-turbo` | $0.50 | — | — | $1.50 | — | — | — | — |
| `gpt-3.5-turbo-0125` | $0.50 | — | — | $1.50 | — | — | — | — |
| `gpt-3.5-turbo-1106` | $1.00 | — | — | $2.00 | — | — | — | — |
| `gpt-3.5-turbo-instruct` | $1.50 | — | — | $2.00 | — | — | — | — |
| `davinci-002` | $2.00 | — | — | $2.00 | — | — | — | — |
| `babbage-002` | $0.40 | — | — | $0.40 | — | — | — | — |

The `gpt-4o-mini` entry is the one currently represented in MAS-CC's Phase 4
static catalog: $0.15/M input and $0.60/M output. Cached input is cheaper, but
the current estimator conservatively prices all estimated input as uncached.

## Specialized model pricing

### Chat, Codex, search, embedding, and moderation

| Category | Model | Input/M | Cached input/M | Output/M |
|---|---|---:|---:|---:|
| ChatGPT | `chat-latest` | $5.00 | $0.50 | $30.00 |
| ChatGPT | `gpt-5.3-chat-latest` | $1.75 | $0.175 | $14.00 |
| ChatGPT | `gpt-5.2-chat-latest` | $1.75 | $0.175 | $14.00 |
| Codex | `gpt-5.3-codex` | $1.75 | $0.175 | $14.00 |
| Cyber | `gpt-5.5-cyber` | $12.50 | $1.25 | $75.00 |
| Search | `gpt-5-search-api` | $1.25 | $0.125 | $10.00 |
| Embedding | `text-embedding-3-small` | $0.02 | — | — |
| Embedding | `text-embedding-3-large` | $0.13 | — | — |
| Embedding | `text-embedding-ada-002` | $0.10 | — | — |
| Moderation | `omni-moderation-latest` | Free | — | — |

### Image generation

Image models price text and image tokens separately. These are standard rates
per million tokens.

| Model | Modality | Input | Cached input | Output |
|---|---|---:|---:|---:|
| `gpt-image-2` | Image | $8.00 | $2.00 | $30.00 |
| `gpt-image-2` | Text | $5.00 | $1.25 | — |
| `gpt-image-1.5` | Image | $8.00 | $2.00 | $32.00 |
| `gpt-image-1.5` | Text | $5.00 | $1.25 | $10.00 |
| `gpt-image-1-mini` | Image | $2.50 | $0.25 | $8.00 |
| `gpt-image-1-mini` | Text | $2.00 | $0.20 | — |
| `gpt-image-1` | Image | $10.00 | $2.50 | $40.00 |
| `gpt-image-1` | Text | $5.00 | $1.25 | — |

### Video generation

Video is priced per generated second, not per million ordinary text tokens.

| Model | Resolution | Standard price/second |
|---|---:|---:|
| `sora-2` | 720p | $0.10 |
| `sora-2-pro` | 720p | $0.30 |
| `sora-2-pro` | 1024p | $0.50 |
| `sora-2-pro` | 1080p | $0.70 |

### Realtime and audio examples

Prices are per million modality-specific tokens unless a per-minute unit is
shown.

| Model | Modality | Input | Cached input | Output/cost |
|---|---|---:|---:|---:|
| `gpt-realtime-2.1` | Audio | $32.00 | $0.40 | $64.00 |
| `gpt-realtime-2.1` | Text | $4.00 | $0.40 | $24.00 |
| `gpt-realtime-2.1-mini` | Audio | $10.00 | $0.30 | $20.00 |
| `gpt-realtime-2.1-mini` | Text | $0.60 | $0.06 | $2.40 |
| `gpt-audio-1.5` | Audio | $32.00 | — | $64.00 |
| `gpt-audio-1.5` | Text | $2.50 | — | $10.00 |
| `gpt-audio-mini` | Audio | $10.00 | — | $20.00 |
| `gpt-audio-mini` | Text | $0.60 | — | $2.40 |
| `gpt-transcribe` | Transcription | — | — | $0.0045/minute |
| `gpt-4o-transcribe` | Transcription | $2.50 | — | $10.00; about $0.006/minute |
| `gpt-4o-mini-transcribe` | Transcription | $1.25 | — | $5.00; about $0.003/minute |

## Processing tiers and additional charges

- Batch and Flex rates are usually lower than standard rates, but availability
  and exact numbers are model-specific. The live pricing page publishes full
  Batch and Flex tables.
- Fast mode costs more than standard processing. It can be requested through
  `service_tier: "fast"` or the compatibility spelling `"priority"` where the
  selected model supports it.
- Eligible regional-processing endpoints for models released on or after
  2026-03-05 receive the 10% uplift documented on the pricing page.
- Built-in tools add separate charges. For example, ordinary web search is
  $10 per 1,000 calls plus model-priced search-content tokens; file search tool
  calls are $2.50 per 1,000 calls, and file-search storage is $0.10/GB/day after
  the free allowance.
- Responses, Chat Completions, Realtime, Batch, and Assistants are not given a
  separate API-endpoint fee; usage is billed from model tokens, modalities,
  tools, storage, or processing tiers.

## Cost calculation

For a standard text request below any long-context threshold:

```text
estimated cost in USD =
    uncached input tokens × input price / 1,000,000
  + cached input tokens × cached input price / 1,000,000
  + cache-write tokens × cache-write price / 1,000,000
  + output tokens × output price / 1,000,000
  + tool or modality charges
```

[Reasoning tokens](https://developers.openai.com/api/docs/guides/reasoning#how-reasoning-works)
are billed as output tokens. Use the `usage` fields returned by the API for
actual token counts, and retain the date/source beside every static price
snapshot used for experiment planning.
