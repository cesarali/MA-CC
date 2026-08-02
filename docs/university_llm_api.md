# University of Potsdam LLM Proxy

This repository uses the University of Potsdam LLM Proxy through its
OpenAI-compatible HTTP API. These notes describe the working setup observed
through 2026-08-02 and are intended for agents working in this repository.

## Credentials and environment

The repository-root `.env` file already contains:

- `POTSDAM_API_KEY`
- `BASE_POTSDAM_LLM_URL`
- optionally, `POTSDAM_MODEL`

Never print, log, commit, or include the API key in an error message. Do not
modify `.env`. It is ignored by Git.

Use the `MA-CC` Conda environment for all Python commands:

```bash
conda run -n MA-CC python ...
```

It contains Python 3.11, `requests`, and `python-dotenv`. Do not depend on
`conda activate` persisting between tool calls.

## WSL and the Windows VPN

The repository automatically starts a restricted Windows CONNECT bridge for
the known Potsdam API host when a supported client runs under WSL. This lets
the shared `AsyncLLMClient` and the HiddenBench OpenAI client use the Windows
VPN without exposing credentials or terminating TLS on the bridge. The bridge
listens only on Windows localhost and rejects destinations other than the
configured Potsdam host on TCP 443.

Mirrored networking remains recommended on Windows 11 22H2 or newer. Configure
it globally in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

Ensure `/etc/wsl.conf` does not disable generated DNS configuration. A suitable
network section is:

```ini
[network]
generateResolvConf=true
```

Apply these settings from Windows PowerShell after saving all WSL work:

```powershell
wsl --shutdown
```

Restart Ubuntu, connect the Windows VPN, and verify the same bridge path used
by API-heavy pipelines:

```bash
conda run --live-stream -n MA-CC python \
  scripts/Potsdam/check_university_api.py
```

Use `--direct` only to diagnose the raw WSL route; Cisco VPN configurations may
block that path even in mirrored mode. Use `--windows` to run the complete
diagnostic in PowerShell as a final fallback. The ordinary command is the
relevant preflight for the HiddenBench pipeline.

## Known working API shape

Load credentials with `python-dotenv` and send the key as a bearer token:

```python
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

repo_root = Path.cwd().resolve()
while not (repo_root / ".env").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

if not (repo_root / ".env").exists():
    raise FileNotFoundError("Could not find the repository-root .env file.")

load_dotenv(repo_root / ".env")

api_key = os.environ["POTSDAM_API_KEY"]
base_url = os.environ["BASE_POTSDAM_LLM_URL"].rstrip("/")
headers = {"Authorization": f"Bearer {api_key}"}
```

The following endpoints currently work:

| Purpose | Method and path |
|---|---|
| List models | `GET {base_url}/models` |
| Model prices and limits | `GET {base_url}/v1/model/info` |
| Account budget information | `GET {base_url}/user/info` |
| Chat completion | `POST {base_url}/chat/completions` |

If `GET /models` returns 404, try `GET /v1/models` once and then use the same
`/v1` prefix for `/v1/chat/completions`. Do not try alternate endpoints after
a 401 or 403 response.

Always retrieve the current model list instead of assuming that a previously
available model is still enabled:

```python
response = requests.get(f"{base_url}/models", headers=headers, timeout=60)
response.raise_for_status()

available_models = sorted(
    entry["id"]
    for entry in response.json().get("data", [])
    if isinstance(entry, dict) and isinstance(entry.get("id"), str)
)
```

## Available models

On 2026-08-02, `GET /models` returned the following 45 entries:

```text
all-proxy-models
gwdg/openai-gpt-oss-120b
gwdg/qwen3-30b-a3b-instruct-2507
gwdg/qwen3.5-397b-a17b
microsoft/Kimi-K2.6
microsoft/Mistral-Codestral-2501
microsoft/Mistral-Large-3
microsoft/claude-fable-5
microsoft/claude-haiku-4-5
microsoft/claude-opus-4-5
microsoft/claude-opus-4-6
microsoft/claude-opus-4-7
microsoft/claude-opus-4-8
microsoft/claude-opus-5
microsoft/claude-sonnet-4-5
microsoft/claude-sonnet-4-6
microsoft/claude-sonnet-5
microsoft/gpt-4.1
microsoft/gpt-4o
microsoft/gpt-5
microsoft/gpt-5-codex
microsoft/gpt-5-mini
microsoft/gpt-5.1
microsoft/gpt-5.1-codex-max
microsoft/gpt-5.2
microsoft/gpt-5.2-codex
microsoft/gpt-5.3-codex
microsoft/gpt-5.4
microsoft/gpt-5.4-mini
microsoft/gpt-5.4-nano
microsoft/gpt-5.4-pro
microsoft/gpt-5.5
microsoft/gpt-5.6-luna
microsoft/gpt-5.6-sol
microsoft/gpt-5.6-terra
microsoft/gpt-chat-latest
microsoft/gpt-image-1.5
microsoft/gpt-image-2
microsoft/sora-2
microsoft/text-embedding-3-large
up/e5-mistral-7b
up/gemma4-31b
up/minimax-m2-5
up/qwen3-6-35b
up/translategemma-27b-it
```

This is a dated snapshot, not a permanent allowlist. Query `/models` before a
run because the proxy's available deployments can change. The list includes
the `all-proxy-models` routing alias as well as image, video, and embedding
models; those entries are not concrete chat models.

## Current pricing and limits snapshot

The following table was generated on 2026-08-02 by joining the exact model IDs
returned by `GET {base_url}/models` with the records returned by
`GET {base_url}/v1/model/info`. The endpoint returned metadata for 44 of the 45
available IDs. `all-proxy-models` is the only ID without its own model-info
record because it is a routing alias rather than a concrete deployment.

The endpoint reports costs per token. This table multiplies those values by
1,000,000 for readability. Values are in the proxy's budget/accounting unit;
do not assume a currency independently of `GET /user/info`. `0` means the
endpoint explicitly reported zero token cost, while `—` means it supplied no
value. RPM and TPM are published ceilings, not guaranteed throughput.

| Model | Input/M | Cached input/M | Output/M | Long input/output/M | RPM | TPM |
|---|---:|---:|---:|---:|---:|---:|
| `all-proxy-models` | — | — | — | — | — | — |
| `gwdg/openai-gpt-oss-120b` | 0 | — | 0 | — | 2,000 | — |
| `gwdg/qwen3-30b-a3b-instruct-2507` | 0 | — | 0 | — | 2,000 | — |
| `gwdg/qwen3.5-397b-a17b` | 0 | — | 0 | — | 2,000 | — |
| `microsoft/Kimi-K2.6` | 0.95 | — | 4 | — | 250 | 250,000 |
| `microsoft/Mistral-Codestral-2501` | 0.30 | — | 0.90 | — | 250 | 250,000 |
| `microsoft/Mistral-Large-3` | 0.50 | — | 1.50 | — | 250 | 250,000 |
| `microsoft/claude-fable-5` | 10 | — | 50 | — | 170 | 170,000 |
| `microsoft/claude-haiku-4-5` | 1 | 0.10 | 5 | — | 250 | 250,000 |
| `microsoft/claude-opus-4-5` | 5 | 0.50 | 25 | — | 250 | 250,000 |
| `microsoft/claude-opus-4-6` | 5 | 0.50 | 25 | — | 250 | 250,000 |
| `microsoft/claude-opus-4-7` | 5 | 0.50 | 25 | — | 250 | 250,000 |
| `microsoft/claude-opus-4-8` | 5 | — | 25 | — | 120 | 120,000 |
| `microsoft/claude-opus-5` | 5 | — | 25 | — | 250 | 250,000 |
| `microsoft/claude-sonnet-4-5` | 3 | 0.30 | 15 | — | 250 | 250,000 |
| `microsoft/claude-sonnet-4-6` | 3 | 0.30 | 15 | — | 100 | 100,000 |
| `microsoft/claude-sonnet-5` | 2 | — | 10 | — | 250 | 250,000 |
| `microsoft/gpt-4.1` | 2 | 0.50 | 8 | — | 250 | 250,000 |
| `microsoft/gpt-4o` | 2.50 | 1.25 | 10 | — | 1,500 | 250,000 |
| `microsoft/gpt-5` | 1.25 | — | 10 | — | 2,500 | 250,000 |
| `microsoft/gpt-5-codex` | 1.25 | — | 10 | — | 250 | 250,000 |
| `microsoft/gpt-5-mini` | 0.25 | 0.025 | 2 | — | 250 | 250,000 |
| `microsoft/gpt-5.1` | 1.25 | 0.125 | 10 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.1-codex-max` | 1.25 | — | 10 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.2` | 1.75 | 0.175 | 14 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.2-codex` | 1.75 | 0.175 | 14 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.3-codex` | 1.75 | 0.175 | 14 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.4` | 2.50 | 0.25 | 15 | 5 / 22.50 | 2,500 | 250,000 |
| `microsoft/gpt-5.4-mini` | 0.75 | 0.075 | 4.50 | — | 2,500 | 250,000 |
| `microsoft/gpt-5.4-nano` | 0.20 | 0.02 | 1.25 | — | 250 | 250,000 |
| `microsoft/gpt-5.4-pro` | 30 | 3 | 180 | 60 / 270 | 150 | 150,000 |
| `microsoft/gpt-5.5` | 5 | 0.50 | 30 | 10 / 45 | 250 | 250,000 |
| `microsoft/gpt-5.6-luna` | 1 | — | 6 | — | 250 | 250,000 |
| `microsoft/gpt-5.6-sol` | 5 | — | 30 | — | 250 | 250,000 |
| `microsoft/gpt-5.6-terra` | 2.50 | — | 15 | — | 250 | 250,000 |
| `microsoft/gpt-chat-latest` | 5 | — | 30 | — | 2,500 | 250,000 |
| `microsoft/gpt-image-1.5` | 5 | 1.25 | 32 | — | 90 | — |
| `microsoft/gpt-image-2` | 5 | — | 30 | — | 2 | — |
| `microsoft/sora-2` | 0 | — | 0 | — | 50 | — |
| `microsoft/text-embedding-3-large` | 0.14 | — | 0 | — | 1,500 | 250,000 |
| `up/e5-mistral-7b` | 0.05 | — | 0.30 | — | — | — |
| `up/gemma4-31b` | 0.10 | — | 0.70 | — | — | — |
| `up/minimax-m2-5` | 0.20 | — | 1.50 | — | — | — |
| `up/qwen3-6-35b` | 0.10 | — | 0.70 | — | — | — |
| `up/translategemma-27b-it` | 0.10 | — | 0.70 | — | — | — |

### Context and modality details

- `microsoft/gpt-5.4`, `microsoft/gpt-5.4-pro`, and `microsoft/gpt-5.5`
  expose long-context overrides. Their model-info records also expose the exact
  threshold-specific fields; query the endpoint when planning requests near or
  above 128K/272K tokens instead of relying only on the compact table.
- Known maximum input/output token pairs include: GPT-5.4 family and GPT-5.5,
  1,050,000/128,000; GPT-5.1/5.2/5.3 and GPT-5 mini,
  272,000/128,000; GPT-4.1, 1,047,576/32,768; GPT-4o,
  128,000/16,384; Claude Sonnet 4.6, 1,000,000/64,000; and the other
  fully described Claude chat deployments, 200,000 input tokens.
- The Claude records include separate cache-write charges. On this snapshot,
  ordinary/over-one-hour cache-write costs per million tokens were 1.25/2 for
  Haiku 4.5, 3.75/6 for Sonnet 4.5/4.6, and 6.25/10 for Opus 4.5/4.6/4.7.
- `microsoft/gpt-image-1.5` additionally reports 8 per million image-input
  tokens and 32 per million image-output tokens. The ordinary text-input rate
  is the 5 shown in the main table.
- `microsoft/sora-2` is priced by generated duration rather than ordinary text
  tokens: the endpoint reports 0.10 per output-video second.
- A missing mode, context limit, RPM, or TPM means the endpoint returned null;
  it is not evidence that the model has no such limit or capability.

### Cost calculation

For ordinary token-billed requests below any long-context threshold:

```text
estimated cost =
    input tokens × input_cost_per_token
  + cached input tokens × cache_read_input_token_cost
  + output tokens × output_cost_per_token
```

The `usage` returned by the proxy is authoritative for actual token counts.
The static MAS-CC preflight catalog is a dated fallback and does not currently
replace this live endpoint snapshot.

A safe refresh starts from the current model list and joins on `model_name`:

```python
models_response = requests.get(
    f"{base_url}/models", headers=headers, timeout=60
)
models_response.raise_for_status()
available = {
    item["id"]
    for item in models_response.json().get("data", [])
    if isinstance(item, dict) and isinstance(item.get("id"), str)
}

info_response = requests.get(
    f"{base_url}/v1/model/info", headers=headers, timeout=60
)
info_response.raise_for_status()
info_by_name = {
    item["model_name"]: item
    for item in info_response.json().get("data", [])
    if isinstance(item, dict) and isinstance(item.get("model_name"), str)
}

missing_info = sorted(available - set(info_by_name))
```

Do not serialize `headers`, and do not publish internal `api_base` or deployment
fields from the raw response when only price/limit metadata is needed.

## Recommended test model

For inexpensive parallel connectivity tests, prefer:

```text
gwdg/qwen3-30b-a3b-instruct-2507
```

On 2026-08-02, `/v1/model/info` reported zero input and output token cost and a
2,000 requests-per-minute limit for this model. Treat these as dynamic proxy
settings: query `/v1/model/info` again before a large run.

If an OpenAI model is required, use:

```text
microsoft/gpt-5.4-nano
```

The proxy reported costs of 0.20 per million input tokens and 1.25 per million
output tokens, with a 250 requests-per-minute limit. The currency/accounting
unit is controlled by the proxy; `/user/info` is authoritative for the
account's remaining budget.

Do not use `all-proxy-models` as a concrete test model, and do not send chat
requests to embedding, image, or video models.

## Minimal chat request

```python
model = os.getenv(
    "POTSDAM_MODEL",
    "gwdg/qwen3-30b-a3b-instruct-2507",
)

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Reply with exactly: API works."}
    ],
    "max_tokens": 16,
}

response = requests.post(
    f"{base_url}/chat/completions",
    headers={**headers, "Content-Type": "application/json"},
    json=payload,
    timeout=120,
)
response.raise_for_status()

assistant_text = response.json()["choices"][0]["message"]["content"]
print(assistant_text)
```

For the complete connectivity and endpoint-fallback implementation, use
`notebooks/test_potsdam_llm_proxy.ipynb`.

## Parallel-call guidance

- Start with 10 to 20 concurrent requests rather than immediately targeting
  the published RPM limit.
- Bound concurrency with a semaphore or worker pool.
- Keep test prompts and `max_tokens` small.
- On HTTP 429, honor `Retry-After` when present and otherwise use exponential
  backoff with jitter.
- Retry timeouts, connection failures, and transient 5xx responses a limited
  number of times. Do not blindly retry 400, 401, or 403 responses.
- Record status codes, latency, and model IDs, but never request headers or the
  API key.
- Shared deployment capacity may cause throttling below the published limit.

The University's service documentation is available at:

<https://www.uni-potsdam.de/de/gptup/llm-proxy>
