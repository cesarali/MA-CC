# University of Potsdam LLM Proxy

This repository uses the University of Potsdam LLM Proxy through its
OpenAI-compatible HTTP API. These notes describe the working setup observed on
2026-07-28 and are intended for agents working in this repository.

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

## Recommended test model

For inexpensive parallel connectivity tests, prefer:

```text
gwdg/qwen3-30b-a3b-instruct-2507
```

On 2026-07-28, `/v1/model/info` reported zero input and output token cost and a
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
